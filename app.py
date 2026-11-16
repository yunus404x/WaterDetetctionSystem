"""Thar Water Early Warning: demo dashboard.

Run:  streamlit run app.py
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from thar.alerts import authority_ranking, build_alerts
from thar.config import DRY_LEVEL, METRICS_FILE, PRED_FILE, REPLAY_END, REPLAY_START

st.set_page_config(page_title="Thar Water Early Warning", page_icon="💧", layout="wide")

TIER_NAME = {0: "OK", 1: "Savdhaan (watch)", 2: "Chetavni (warning)", 3: "Turant (act now)", 9: "Dry now"}
TIER_RGBA = {0: "#2e9e5b", 1: "#e3b505", 2: "#f07c1b", 3: "#d7263d", 9: "#5b2333"}


@st.cache_data
def load(version):  # version = file timestamp, so retraining refreshes the cache
    p = pd.read_csv(PRED_FILE, parse_dates=["date"])
    m = json.loads(METRICS_FILE.read_text())
    return p, m


@st.cache_data
def alerts_for(version, start, end):
    p, _ = load(version)
    return build_alerts(p[(p.date >= start) & (p.date <= end)])


VERSION = PRED_FILE.stat().st_mtime
pred, metrics = load(VERSION)

# ---------------- sidebar ----------------
st.sidebar.title("💧 Thar Water Early Warning")
st.sidebar.caption("Jaisalmer district, Rajasthan · 14 village water sources")
period = st.sidebar.radio("Period", ["2002 drought replay (never trained on)", "2019-2023 forward test"])
if period.startswith("2002"):
    start, end = pd.Timestamp(REPLAY_START), pd.Timestamp(REPLAY_END)
else:
    start, end = pd.Timestamp("2019-01-01"), pd.Timestamp("2023-12-31")
weeks = sorted(pred[(pred.date >= start) & (pred.date <= end)].date.unique())
default = pd.Timestamp("2002-04-15") if period.startswith("2002") else pd.Timestamp("2019-05-06")
default = min(weeks, key=lambda d: abs(pd.Timestamp(d) - default))
week = pd.Timestamp(st.sidebar.select_slider("Week", options=weeks, value=default,
                                             format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y")))
st.sidebar.warning("Data is **simulated** (realistic physics, real village locations, rainfall years "
                   "anchored to history). Swap in real data with `thar/gee_pull.py`.")

wk = pred[pred.date == week].copy()
wk["status"] = wk.tier.where(~wk.dry_now, 9)

# ---------------- header KPIs ----------------
st.title(f"Week of {week:%d %B %Y}")
c = st.columns(5)
for col, code in zip(c, [9, 3, 2, 1, 0]):
    col.metric(TIER_NAME[code], int((wk.status == code).sum()))

left, right = st.columns([3, 2])

# ---------------- map ----------------
with left:
    st.subheader("Risk map")
    wk["color"] = wk.status.map(TIER_RGBA)
    wk["size"] = 2500 + 6000 * wk.p_dry_4w  # metres
    st.map(wk, latitude="lat", longitude="lon", color="color", size="size", zoom=7)
    st.caption(" · ".join(f"<span style='color:{TIER_RGBA[k]}'>●</span> {v}" for k, v in TIER_NAME.items()),
               unsafe_allow_html=True)

# ---------------- phone ----------------
with right:
    st.subheader("📱 What herders received")
    al = alerts_for(VERSION, start, end)
    recent = al[(al.date <= week) & (al.date > week - pd.Timedelta(days=21))].sort_values("date", ascending=False)
    lang = st.radio("Language", ["हिन्दी", "English"], horizontal=True, label_visibility="collapsed")
    bubbles = ""
    for r in recent.head(6).itertuples():
        txt = r.sms_hi if lang == "हिन्दी" else r.sms_en
        bubbles += (f"<div style='background:#e9f7ef;color:#111;border-radius:10px;padding:8px 10px;margin:6px 0;"
                    f"border-left:5px solid {TIER_RGBA[r.tier]};font-size:14px'>"
                    f"<div style='font-size:11px;color:#666'>JalSuchna · {r.date:%d %b}</div>{txt}</div>")
    if not bubbles:
        bubbles = "<div style='color:#777;padding:20px;text-align:center'>No warnings in the last 3 weeks</div>"
    st.markdown(f"<div style='border:10px solid #222;border-radius:28px;padding:14px;background:#fafafa;"
                f"height:440px;overflow:auto;max-width:360px'>{bubbles}</div>", unsafe_allow_html=True)
    st.caption("Messages go out only when a tier changes (or every 2 weeks while it stays high). "
               "Devanagari SMS = 70 chars/segment, so messages are kept to 1-2 segments.")

# ---------------- water department ----------------
st.subheader("🚚 Water department / panchayat: tanker priority this week")
rank = authority_ranking(wk)
rank["status"] = [TIER_NAME[9] if d else TIER_NAME[t] for t, d in zip(rank.tier, rank.dry_now)]
rank = rank.rename(columns={"p_dry_2w": "risk ≤2 wk", "p_dry_4w": "risk ≤4 wk", "weeks_to_dry": "est. weeks left",
                            "data_quality": "data quality"})
st.dataframe(rank[["village", "type", "status", "risk ≤2 wk", "risk ≤4 wk", "est. weeks left", "data quality"]],
             hide_index=True, use_container_width=True,
             column_config={"risk ≤2 wk": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                            "risk ≤4 wk": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                            "data quality": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                            "est. weeks left": st.column_config.NumberColumn(format="%.0f")})

# ---------------- source detail ----------------
st.subheader("🔎 One source up close")
village = st.selectbox("Source", sorted(pred.village.unique()),
                       index=sorted(pred.village.unique()).index(rank.village.iloc[0]))
s = pred[(pred.village == village) & (pred.date >= start) & (pred.date <= end)].copy()
s["band_date"] = s.date + pd.Timedelta(weeks=4)  # the 4-week-ahead band is plotted where it applies
base = alt.Chart(s).encode(x=alt.X("date:T", title=None))
band = alt.Chart(s).mark_area(opacity=0.25, color="#4c78a8").encode(
    x="band_date:T", y=alt.Y("level_4w_q10:Q", title="storage (0-100)"), y2="level_4w_q90:Q")
truth = base.mark_line(color="#b279a2", strokeWidth=2.5).encode(y="true_level:Q")
sensor = base.mark_circle(size=18, color="#4c78a8").encode(y="sensor_level:Q")
dry = alt.Chart(pd.DataFrame({"y": [DRY_LEVEL]})).mark_rule(color="#d7263d", strokeDash=[4, 4]).encode(y="y:Q")
now = alt.Chart(pd.DataFrame({"x": [week]})).mark_rule(color="#888").encode(x="x:T")
risk = base.mark_area(color="#f07c1b", opacity=0.6).encode(
    y=alt.Y("p_dry_4w:Q", title="P(dry ≤4 wk)", scale=alt.Scale(domain=[0, 1])))
st.altair_chart(alt.layer(band, truth, sensor, dry, now).properties(height=260), use_container_width=True)
st.altair_chart(alt.layer(risk, now).properties(height=110), use_container_width=True)
st.caption("Purple line = true level (hidden from the model). Dots = sensor readings (gaps = sensor dead). "
           "Blue band = our 10-90% forecast for 4 weeks ahead. Red dashed = dry line. Orange = risk.")

# ---------------- evaluation ----------------
with st.expander("📊 How good is it? (backtest)", expanded=False):
    rows = []
    for split, label in [("replay_2002_03", "2002 drought replay"), ("test_2019_2023", "2019-23 forward test"),
                         ("test_all_sensors_dead", "2019-23, ALL sensors dead")]:
        m = metrics[split]
        for h in (2, 4, 6):
            for who in ("model", "baseline"):
                r = m[f"{who}_{h}w"]
                rows.append({"period": label, "horizon": f"{h} wk", "method": who, "AUC": r["auc"],
                             "precision": r["precision"], "recall": r["recall"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    k = st.columns(3)
    for col, (split, label) in zip(k, [("replay_2002_03", "2002 replay"), ("test_2019_2023", "2019-23"),
                                       ("test_all_sensors_dead", "Sensors dead")]):
        m = metrics[split]
        col.metric(f"{label}: failures warned ≥2 wk ahead",
                   f"{m['events_warned_2w_ahead']:.0%}" if m["events_warned_2w_ahead"] is not None else "n/a",
                   f"median lead {m['median_lead_weeks']} wk · {m['dry_events']} failures")
    k = st.columns(3)
    for col, (split, label) in zip(k, [("replay_2002_03", "2002 replay"), ("test_2019_2023", "2019-23"),
                                       ("test_all_sensors_dead", "Sensors dead")]):
        m = metrics[split]
        col.metric(f"{label}: level inside 80% band", f"{m['band_80_coverage']:.0%}",
                   f"{m['band_80_coverage'] - m['band_80_coverage_raw']:+.0%} vs uncalibrated · "
                   f"width {m['band_80_width']:.0f} pts")
    st.markdown(f"""
**Confidence band** is conformally calibrated: held-out misses decide how much to widen it, separately
for each source type and for "seeing" vs "flying blind". It gets honestly wider when sensors are down.

**Baseline** = extend the last 4-week sensor trend in a straight line. With all sensors dead it goes blind;
our model falls back to radar, NDVI, rainfall, GRACE and herder reports.

**Balancing false alarms vs misses:** a miss is costed at **{metrics['cost_miss']:.0f}×** a false alarm.
The *Savdhaan* (watch) threshold is the cost-optimal one, so it catches nearly every failure.
*Chetavni* needs ≥25% risk and *Turant* needs ≥50% risk, and *Turant* is never sent when
data quality is poor, so the loudest alarm stays trustworthy.
""")
