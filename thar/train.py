"""Train the early-warning models, pick alert thresholds, backtest, and save predictions.

Models (scikit-learn HistGradientBoosting = LightGBM-style trees, handles missing values natively):
  - P(dry within 2 / 4 / 6 weeks)          -> classifiers
  - level in 4 weeks at 10% / 50% / 90%    -> quantile regressors (the confidence band)
  - weeks until dry (capped at 12)         -> regressor (for "~5 weeks" in the SMS)

Honest evaluation:
  - 2001-2004 (the 2002 drought replay + buffer) are never trained on.
  - 2019-2023 is a strict forward test.
  - Alert thresholds come from leave-one-year-out predictions on training years,
    chosen to minimise  COST_MISS * misses + COST_FALSE_ALARM * false alarms.
  - Baseline: straight-line extrapolation of the last sensor trend.
  - Stress test: re-run the test with every in-situ sensor dead.

Run:  python -m thar.train
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from thar import features as F
from thar.config import (COST_FALSE_ALARM, COST_MISS, DRY_LEVEL, HELDOUT_YEARS, HORIZONS,
                         METRICS_FILE, PRED_FILE, REPLAY_END, REPLAY_START, SIM_FILE, TEST_YEARS)

X_COLS = F.FEATURES
CLF_KW = dict(max_iter=200, learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=40,
              l2_regularization=1.0, categorical_features=[X_COLS.index("type_code")], random_state=0)
REG_KW = {k: v for k, v in CLF_KW.items()}


def fit_clf(d, h):
    m = d[f"y_dry_{h}w"].notna()
    return HistGradientBoostingClassifier(**CLF_KW).fit(d.loc[m, X_COLS], d.loc[m, f"y_dry_{h}w"])


def fit_reg(d, target, **kw):
    m = d[target].notna()
    return HistGradientBoostingRegressor(**REG_KW, **kw).fit(d.loc[m, X_COLS], d.loc[m, target])


def best_threshold(y, p):
    """Threshold that minimises expected cost (a miss costs COST_MISS, a false alarm COST_FALSE_ALARM)."""
    grid = np.linspace(0.02, 0.9, 89)
    costs = [COST_MISS * ((p < t) & (y == 1)).sum() + COST_FALSE_ALARM * ((p >= t) & (y == 0)).sum()
             for t in grid]
    return float(grid[int(np.argmin(costs))])


def scores(y, p, t):
    y = np.asarray(y)
    a = np.asarray(p) >= t
    tp, fp, fn = int((a & (y == 1)).sum()), int((a & (y == 0)).sum()), int((~a & (y == 1)).sum())
    return {
        "auc": round(float(roc_auc_score(y, p)), 3) if len(set(y)) > 1 else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "alerts": tp + fp, "positive_weeks": int(y.sum()),
    }


def baseline_prob(d, h):
    """Naive: extend the last 4-week sensor trend; 'dry' if the line crosses the dry level."""
    proj = d.lvl_last + d.lvl_slope4.clip(upper=0) * h
    return (proj < DRY_LEVEL).astype(float).where(d.lvl_last.notna(), 0.0)


def lead_times(pred: pd.DataFrame, tier_col="tier"):
    """For every 'source goes dry' event, how many weeks before did the first alert fire?"""
    out = []
    for sid, g in pred.sort_values("date").groupby("source_id"):
        dry = g.true_level.to_numpy() < DRY_LEVEL
        alert = g[tier_col].to_numpy() > 0
        for i in range(1, len(g)):
            if dry[i] and not dry[i - 1]:  # a new failure starts at week i
                lead = 0
                for k in range(1, 9):  # look back up to 8 weeks for a continuous alert run
                    if i - k >= 0 and alert[i - k]:
                        lead = k
                    else:
                        break
                out.append(lead)
    return out


# Each tier trades sensitivity for trust differently:
#   Savdhaan (watch)  : cost-optimal threshold -> catches almost everything, cheap to be wrong
#   Chetavni (warning): needs at least 25% risk within 4 weeks
#   Turant (act now)  : needs at least 50% risk within 2 weeks -> rarely wrong, protects trust
TIER_FLOOR = {1: 0.0, 2: 0.25, 3: 0.5}
TIER_HORIZON = {1: 6, 2: 4, 3: 2}


def assign_tiers(p, thr, quality):
    """0 none, 1 Savdhaan, 2 Chetavni, 3 Turant.
    When data is poor (sensor dead, satellite cloudy) we lower the watch/warning bar, because a
    miss costs more than a false alarm, but we never say 'act now' when flying blind."""
    tier = np.zeros(len(quality), int)
    blind = np.asarray(quality) < 0.34
    for lvl in (1, 2, 3):
        h = TIER_HORIZON[lvl]
        t = np.maximum(thr[h], TIER_FLOOR[lvl])
        if lvl < 3:
            t = np.where(blind, t * 0.6, t)
        hit = np.asarray(p[h]) >= t
        if lvl == 3:
            hit &= ~blind
        tier = np.where(hit, lvl, tier)
    return tier


BAND_TARGET = 0.8  # we promise "80% of the time the level lands inside the band"


def band_group(frame):
    blind = np.where(frame.data_quality < 0.34, "blind", "seeing")
    return pd.Series(frame["type"].to_numpy(), index=frame.index) + "/" + blind


def conformal_q(scores):
    """Finite-sample conformal quantile of the out-of-band misses."""
    s = np.sort(np.asarray(scores))
    k = min(len(s) - 1, int(np.ceil((len(s) + 1) * BAND_TARGET)) - 1)
    return float(s[k])


def main():
    raw = pd.read_csv(SIM_FILE, parse_dates=["date"])
    d = F.build(raw)
    d = d[d.date >= "2001-01-01"]  # first year is warm-up for 52-week rolling features

    train = d[~d.year.isin(HELDOUT_YEARS + TEST_YEARS)]
    print(f"Train rows {len(train):,} | years {sorted(train.year.unique())}")

    # --- 1. thresholds from out-of-fold predictions (4 folds of whole years) on training years ---
    thr = {}
    years = sorted(train.year.unique())
    folds = [years[i::4] for i in range(4)]
    for h in HORIZONS:
        oof = pd.Series(np.nan, index=train.index)
        for fy in folds:
            tr, va = train[~train.year.isin(fy)], train[train.year.isin(fy)]
            m = va[f"y_dry_{h}w"].notna()
            oof[va.index[m]] = fit_clf(tr, h).predict_proba(va.loc[m, X_COLS])[:, 1]
        m = oof.notna() & ~train.dry_now
        thr[h] = best_threshold(train.loc[m, f"y_dry_{h}w"].to_numpy(), oof[m].to_numpy())
    print("Alert thresholds (cost-optimal):", thr)

    # --- 1b. conformal calibration of the 10-90% level band (same year folds) ---
    # Raw quantile trees are over-confident. Conformalized quantile regression (CQR):
    # measure how far out-of-fold truths fell outside the raw band, then widen the band by the
    # 80th percentile of that miss. Done per (source type, flying-blind) group, since a pond with
    # a dead sensor is far less predictable than a tubewell with a fresh reading.
    oof_lo = pd.Series(np.nan, index=train.index)
    oof_hi = pd.Series(np.nan, index=train.index)
    for fy in folds:
        tr, va = train[~train.year.isin(fy)], train[train.year.isin(fy)]
        oof_lo[va.index] = fit_reg(tr, "y_level_4w", loss="quantile", quantile=0.1).predict(va[X_COLS])
        oof_hi[va.index] = fit_reg(tr, "y_level_4w", loss="quantile", quantile=0.9).predict(va[X_COLS])
    cal = train.assign(lo=oof_lo, hi=oof_hi)
    cal = cal[cal.y_level_4w.notna() & ~cal.dry_now]
    cal["score"] = np.maximum(cal.lo - cal.y_level_4w, cal.y_level_4w - cal.hi)
    cal["group"] = band_group(cal)
    widen = {g: conformal_q(s) for g, s in cal.groupby("group").score}
    widen["_all"] = conformal_q(cal.score)
    print("Band widening (storage points) by group:", {k: round(v, 1) for k, v in widen.items()})

    # --- 2. final models on all training years ---
    clfs = {h: fit_clf(train, h) for h in HORIZONS}
    q = {a: fit_reg(train, "y_level_4w", loss="quantile", quantile=a) for a in (0.1, 0.5, 0.9)}
    wtd = fit_reg(train, "y_weeks_to_dry")

    def predict(frame):
        p = frame[["date", "source_id", "village", "type", "lat", "lon", "true_level", "true_tds",
                   "sensor_level", "sar_area", "ndvi", "rain_mm", "herder_reports", "tds_last",
                   "data_quality", "dry_now", "year"] + [f"y_dry_{h}w" for h in HORIZONS]].copy()
        for h in HORIZONS:
            p[f"p_dry_{h}w"] = clfs[h].predict_proba(frame[X_COLS])[:, 1]
        for a, m in q.items():
            p[f"level_4w_q{int(a * 100)}"] = m.predict(frame[X_COLS])
        p["level_4w_q10_raw"], p["level_4w_q90_raw"] = p.level_4w_q10, p.level_4w_q90
        w = band_group(frame).map(widen).fillna(widen["_all"]).to_numpy()
        p["level_4w_q10"] = (p.level_4w_q10_raw - w).clip(0, 100)
        p["level_4w_q90"] = (p.level_4w_q90_raw + w).clip(0, 100)
        p["level_4w_q50"] = p.level_4w_q50.clip(p.level_4w_q10, p.level_4w_q90)
        p["weeks_to_dry"] = wtd.predict(frame[X_COLS]).clip(0, 12)
        p["tier"] = assign_tiers({h: p[f"p_dry_{h}w"] for h in HORIZONS}, thr, p.data_quality)
        return p

    pred = predict(d)
    pred["split"] = np.select([pred.year.isin(TEST_YEARS), pred.year.isin(HELDOUT_YEARS)],
                              ["test", "replay"], "train")
    pred.to_csv(PRED_FILE, index=False)

    # --- 3. evaluation ---
    metrics = {"thresholds": thr, "cost_miss": COST_MISS, "cost_false_alarm": COST_FALSE_ALARM}

    def evaluate(name, frame_pred, frame_feat):
        res = {}
        live = ~frame_pred.dry_now  # only score warnings issued while the source still had water
        for h in HORIZONS:
            m = live & frame_pred[f"y_dry_{h}w"].notna()
            y = frame_pred.loc[m, f"y_dry_{h}w"].to_numpy()
            res[f"model_{h}w"] = scores(y, frame_pred.loc[m, f"p_dry_{h}w"], thr[h])
            res[f"baseline_{h}w"] = scores(y, baseline_prob(frame_feat.loc[m], h), 0.5)
        lt = lead_times(frame_pred)
        res["dry_events"] = len(lt)
        res["events_warned_2w_ahead"] = round(float(np.mean([x >= 2 for x in lt])), 3) if lt else None
        res["median_lead_weeks"] = float(np.median(lt)) if lt else None
        # How often the true level 4 weeks later fell inside our 10-90% band (target: ~0.8).
        # Levels can't go below 0, so a band that reaches 0 counts an empty source as covered.
        y4 = frame_feat.loc[live, "y_level_4w"]
        ok = y4.notna()
        for suffix, key in (("_raw", "band_80_coverage_raw"), ("", "band_80_coverage")):
            lo = frame_pred.loc[live, f"level_4w_q10{suffix}"]
            lo = lo.where(lo > 1, 0)
            hi = frame_pred.loc[live, f"level_4w_q90{suffix}"]
            res[key] = round(float(((y4[ok] >= lo[ok]) & (y4[ok] <= hi[ok])).mean()), 3)
            res[key.replace("coverage", "width")] = round(float((hi[ok] - lo[ok]).mean()), 1)
        metrics[name] = res

    test_idx = pred.split == "test"
    evaluate("test_2019_2023", pred[test_idx], d[test_idx])
    rp = (pred.date >= REPLAY_START) & (pred.date <= REPLAY_END)
    evaluate("replay_2002_03", pred[rp], d[rp])

    # Stress test: every sensor dead -> model must fall back to satellite + herder reports.
    d_blind = F.build(raw, drop_sensors=True)
    d_blind = d_blind[d_blind.date >= "2001-01-01"]
    pb = predict(d_blind)
    evaluate("test_all_sensors_dead", pb[pb.year.isin(TEST_YEARS)], d_blind[d_blind.year.isin(TEST_YEARS)])

    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "thresholds"}, indent=2))
    print(f"Saved predictions -> {PRED_FILE}\nSaved metrics -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
