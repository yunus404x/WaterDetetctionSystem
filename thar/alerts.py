"""Turn predictions into SMS warnings people can act on.

Design for the desert:
  - Short messages for basic phones. Hindi in Devanagari costs 70 chars per SMS segment,
    so we keep each message to 1-2 segments and also send a Latin-script fallback.
  - Only send when the tier CHANGES (or every 2 weeks while it stays high): no alert spam.
  - Every warning names the nearest source that is predicted to stay safe.
  - When data is poor we say so ("kam bharosa" / low confidence) rather than going silent.
  - Two audiences: herders (SMS) and the water department / panchayat (ranked list).

Run:  python -m thar.alerts          (writes outputs/alerts.csv from the replay)
"""
import math

import pandas as pd

from thar.config import ALERTS_FILE, PRED_FILE, REPLAY_END, REPLAY_START

TIERS = {
    0: ("none", "", ""),
    1: ("Savdhaan", "सावधान", "WATCH"),
    2: ("Chetavni", "चेतावनी", "WARNING"),
    3: ("Turant", "तुरंत", "ACT NOW"),
}
TYPE_HI = {"nadi": "नाडी", "beri": "बेरी", "tubewell": "ट्यूबवेल"}
DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def km_and_dir(a, b):
    """Distance (km) and compass direction from source a to source b."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    km = 6371 * 2 * math.asin(math.sqrt(h))
    bearing = (math.degrees(math.atan2(math.sin(dlon) * math.cos(lat2),
                                       math.cos(lat1) * math.sin(lat2) -
                                       math.sin(lat1) * math.cos(lat2) * math.cos(dlon))) + 360) % 360
    return km, DIRS[int((bearing + 22.5) // 45) % 8]


def nearest_safe(row, week_df):
    """Closest other source whose 6-week risk is low."""
    safe = week_df[(week_df.source_id != row.source_id) & (week_df.p_dry_6w < 0.1) & (~week_df.dry_now)]
    if safe.empty:
        return None
    best = min(((km_and_dir(row, s), s) for s in safe.itertuples()), key=lambda t: t[0][0])
    (km, d), s = best
    return {"village": s.village, "type": s.type, "km": round(km), "dir": d}


HORIZON = {1: 6, 2: 4, 3: 2}
MAX_ALT_KM = 60  # beyond this, walking herds there is not realistic -> ask for a tanker instead


def message(row, alt):
    tier = int(row.tier)
    name, hi, en = TIERS[tier]
    h = HORIZON[tier]
    p = int(round(100 * getattr(row, f"p_dry_{h}w")))
    # "~N weeks" from the regressor, but never later than the tier's own horizon.
    wk = max(1, min(h, int(round(row.weeks_to_dry))))
    low_conf = row.data_quality < 0.34
    hi_msg = f"{hi}: {row.village} {TYPE_HI[row.type]} ~{wk} हफ्ते में सूख सकती है ({p}%)."
    en_msg = f"{en}: {row.village} {row.type} may run dry in ~{wk} wks ({p}%)."
    if alt and alt["km"] <= MAX_ALT_KM:
        hi_msg += f" पानी: {alt['village']} {alt['km']}km {alt['dir']}."
        en_msg += f" Water at {alt['village']} {alt['km']}km {alt['dir']}."
    else:
        hi_msg += " टैंकर हेतु पंचायत को सूचना दी."
        en_msg += " No safe source nearby: panchayat asked for tanker."
    if low_conf:
        hi_msg += " (कम भरोसा)"
        en_msg += " (low confidence: sensor down)"
    en_msg += " Reply LOW/OK to report."
    return hi_msg, en_msg


def build_alerts(pred: pd.DataFrame, resend_every=2) -> pd.DataFrame:
    pred = pred.sort_values("date")
    out, last = [], {}
    for date, wk in pred.groupby("date"):
        for r in wk.itertuples():
            prev_tier, prev_date = last.get(r.source_id, (0, None))
            tier = int(r.tier)
            if r.dry_now:  # already dry: switch to the water department's tanker list, not herder SMS
                last[r.source_id] = (tier, prev_date)
                continue
            changed = tier != prev_tier
            stale = prev_date is not None and (date - prev_date).days >= 7 * resend_every
            if tier > 0 and (changed or stale):
                alt = nearest_safe(r, wk)
                hi, en = message(r, alt)
                out.append({"date": date, "source_id": r.source_id, "village": r.village, "type": r.type,
                            "tier": tier, "tier_name": TIERS[tier][0], "p_dry_4w": round(r.p_dry_4w, 2),
                            "weeks_to_dry": round(r.weeks_to_dry, 1), "data_quality": round(r.data_quality, 2),
                            "sms_hi": hi, "sms_en": en, "sms_hi_chars": len(hi),
                            "alt_source": alt["village"] if alt else None})
                last[r.source_id] = (tier, date)
            elif changed:
                last[r.source_id] = (tier, prev_date)
    return pd.DataFrame(out)


def authority_ranking(week_df: pd.DataFrame) -> pd.DataFrame:
    """For the water department: which villages need a tanker first."""
    r = week_df.copy()
    r["priority"] = r.dry_now.astype(int) * 2 + r.p_dry_2w + 0.5 * r.p_dry_4w
    cols = ["village", "type", "tier", "p_dry_2w", "p_dry_4w", "weeks_to_dry", "data_quality", "dry_now"]
    return r.sort_values("priority", ascending=False)[cols]


def main():
    pred = pd.read_csv(PRED_FILE, parse_dates=["date"])
    rp = pred[(pred.date >= REPLAY_START) & (pred.date <= REPLAY_END)]
    alerts = build_alerts(rp)
    alerts.to_csv(ALERTS_FILE, index=False)
    print(f"{len(alerts)} SMS sent across {rp.source_id.nunique()} sources over {rp.date.nunique()} weeks")
    print(alerts.tier_name.value_counts().to_string())
    print("\nSample messages:")
    for r in alerts.sort_values("tier", ascending=False).head(4).itertuples():
        print(f"  [{r.date.date()}] {r.sms_hi}\n               {r.sms_en}")


if __name__ == "__main__":
    main()
