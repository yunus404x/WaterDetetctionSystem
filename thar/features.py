"""Turn raw (gappy) weekly signals into model features and targets.

Rule: features only use what a real system would know at week t
(observed signals up to t). The hidden `true_level` is used ONLY for targets.
"""
import numpy as np
import pandas as pd

from thar.config import DRY_LEVEL, HORIZONS

TYPE_CODE = {"nadi": 0, "beri": 1, "tubewell": 2}

FEATURES = [
    "type_code", "woy_sin", "woy_cos", "weeks_to_monsoon",
    "lvl_last", "lvl_age", "lvl_slope4", "lvl_slope12",
    "sar_last", "sar_age", "sar_slope4",
    "ndvi_last", "ndvi_age", "ndvi_chg8",
    "rain_4", "rain_12", "rain_26", "rain_52",
    "temp_4", "grace_cm", "grace_chg13",
    "reports_4", "tds_last", "tds_age", "data_quality",
]


def _last_and_age(s: pd.Series):
    """Last observed value (forward-filled) and how many weeks old it is."""
    idx = pd.Series(np.where(s.notna(), np.arange(len(s)), np.nan), index=s.index).ffill()
    age = np.arange(len(s)) - idx.to_numpy()
    return s.ffill(), pd.Series(age, index=s.index)


def build_source(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    woy = g.date.dt.isocalendar().week.astype(int)
    g["type_code"] = g["type"].map(TYPE_CODE)
    g["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    g["woy_cos"] = np.cos(2 * np.pi * woy / 52)
    g["weeks_to_monsoon"] = (27 - woy) % 52

    g["lvl_last"], g["lvl_age"] = _last_and_age(g.sensor_level)
    g["lvl_slope4"] = (g.lvl_last - g.lvl_last.shift(4)) / 4
    g["lvl_slope12"] = (g.lvl_last - g.lvl_last.shift(12)) / 12
    # A reading that is many weeks old is not a reading. Hide it so the model leans on other signals.
    stale = g.lvl_age > 8
    g.loc[stale, ["lvl_last", "lvl_slope4", "lvl_slope12"]] = np.nan

    g["sar_last"], g["sar_age"] = _last_and_age(g.sar_area)
    g["sar_slope4"] = (g.sar_last - g.sar_last.shift(4)) / 4
    g["ndvi_last"], g["ndvi_age"] = _last_and_age(g.ndvi)
    g["ndvi_chg8"] = g.ndvi_last - g.ndvi_last.shift(8)

    for w in (4, 12, 26, 52):
        g[f"rain_{w}"] = g.rain_mm.rolling(w, min_periods=1).sum()
    g["temp_4"] = g.temp_c.rolling(4, min_periods=1).mean()
    g["grace_chg13"] = g.grace_cm - g.grace_cm.shift(13)
    g["reports_4"] = g.herder_reports.rolling(4, min_periods=1).sum()
    g["tds_last"], g["tds_age"] = _last_and_age(g.tds_obs)

    fresh_sensor = (g.lvl_age <= 1).astype(float)
    fresh_sat = np.where(g.type == "nadi", (g.sar_age <= 2), (g.ndvi_age <= 3)).astype(float)
    g["data_quality"] = (2 * fresh_sensor + fresh_sat) / 3  # 1 = all signals fresh, 0 = flying blind

    # ---- targets (use hidden truth) ----
    lvl = g.true_level.to_numpy()
    n = len(lvl)
    g["dry_now"] = lvl < DRY_LEVEL
    for h in HORIZONS:
        fut_min = np.array([lvl[i + 1:i + 1 + h].min() if i + h < n else np.nan for i in range(n)])
        g[f"y_dry_{h}w"] = np.where(np.isnan(fut_min), np.nan, (fut_min < DRY_LEVEL).astype(float))
    g["y_level_4w"] = g.true_level.shift(-4)
    # weeks until first dry week (capped at 12)
    below = lvl < DRY_LEVEL
    wtd = np.full(n, 12.0)
    nxt = None
    for i in range(n - 1, -1, -1):
        if below[i]:
            nxt = i
        if nxt is not None and nxt > i:
            wtd[i] = min(12, nxt - i)
    g["y_weeks_to_dry"] = wtd
    return g


def build(df: pd.DataFrame, drop_sensors: bool = False) -> pd.DataFrame:
    df = df.copy()
    if drop_sensors:  # stress test: every in-situ sensor is dead
        df["sensor_level"] = np.nan
    out = pd.concat([build_source(g) for _, g in df.groupby("source_id")], ignore_index=True)
    out["year"] = out.date.dt.year
    return out
