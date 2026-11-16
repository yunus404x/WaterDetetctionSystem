# JalSuchna: early warnings for desert water sources (Thar, Rajasthan)

Warn herders and village councils in Jaisalmer district **weeks before** a village pond (*nadi*),
seepage well (*beri*) or tubewell fails, and tell them where water still is.

**Chain:** satellite + sensor + herder reports → P(dry in 2/4/6 weeks) with a confidence band →
tiered SMS in Hindi/English + a tanker-priority list for the water department (PHED).

> ⚠️ The data is **simulated**: realistic physics, real village locations, and rainfall years
> anchored to history (2002 monsoon failure, 2006 Barmer floods). See `thar/simulate.py`.
> `thar/gee_pull.py` pulls the real satellite signals.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m thar.simulate   # 1. make data      -> data/sim_weekly.csv
.venv/bin/python -m thar.train      # 2. train + test   -> outputs/predictions.csv, metrics.json  (~30 s)
.venv/bin/python -m thar.alerts     # 3. SMS from replay -> outputs/alerts.csv
.venv/bin/streamlit run app.py      # 4. dashboard
```

## Files

| File | What it does |
|---|---|
| `thar/simulate.py` | Hidden "true" storage per source + the noisy, gappy signals a real system sees |
| `thar/gee_pull.py` | Real CHIRPS / ERA5 / Sentinel-2 NDVI / Sentinel-1 radar / GRACE from Google Earth Engine |
| `thar/features.py` | Features from **observed** signals only (last value + age, slopes, rain sums, data quality) |
| `thar/train.py` | Gradient-boosted classifiers + quantile regressors, cost-based thresholds, backtest, stress test |
| `thar/alerts.py` | Tiered SMS (Savdhaan / Chetavni / Turant), nearest safe source, no alert spam |
| `app.py` | Streamlit demo: replay week by week, map, phone, tanker list, source chart, backtest |

## How we handle the desert rules

| Rule | What we do |
|---|---|
| Sensor dies | Readings older than 8 weeks are hidden; model falls back to radar, NDVI, rain, GRACE, herder reports. Stress test: **all sensors dead → still warns 93% of failures ≥2 weeks ahead** (baseline goes blind) |
| Cloudy image | Sentinel-1 radar sees through cloud and dust; cloudy NDVI = NaN, never a made-up value |
| Basic phones | Short SMS (≤2 Devanagari segments) + English fallback; reply `LOW`/`OK` feeds back as herder reports |
| False alarm vs miss | Miss costed 8× a false alarm. *Savdhaan* = cost-optimal (sensitive), *Chetavni* ≥25%, *Turant* ≥50%, and *Turant* is never sent when flying blind |
| Alert fatigue | SMS only when the tier changes, or every 2 weeks while it stays high |

## Current results (simulated data)

| | 2002 drought replay (never trained on) | 2019–23 forward test | 2019–23, all sensors dead |
|---|---|---|---|
| AUC, 4-week risk (model / straight-line baseline) | 0.99 / 0.80 | 0.997 / 0.85 | 0.99 / 0.50 |
| Failures warned ≥2 weeks ahead | 100% (12 events) | 93% (14 events) | 93% |
| Median lead time | 8 weeks | 7 weeks | 8 weeks |
| 10–90% band coverage, target 0.80 (raw → calibrated) | 0.62 → **0.80** | 0.66 → **0.83** | 0.74 → **0.90** |
| Average band width (storage points, 0–100) | 12.6 | 12.1 | 49.2 (honestly wide when blind) |

**Confidence band:** conformalized quantile regression (CQR). Out-of-fold misses on training years
set how much to widen the raw 10–90% band, separately per source type and per
"seeing / flying blind" (sensor dead and satellite stale). A pond with a dead sensor gets ±4 points;
a tubewell with a dead sensor gets ±6 points; a fresh reading gets only ±1.

Simulated data is cleaner than reality, so treat these numbers as an upper bound. Say that in the pitch.

## Next steps (in priority order)
1. **Real ground truth:** India-WRIS groundwater levels for Jaisalmer/Barmer (only ~4 per year: use them to calibrate, not train).
2. **Recalibrate on real data:** the band widening in `train.py` must be re-learned once real readings are in.
3. **Salinity:** `true_tds` is simulated already; add a "too salty for animals" target (`SALT_LIMIT` in `config.py`).
4. **Real SMS:** send via Twilio / Exotel / a GSM modem; add a missed-call IVR voice message for people who can't read.
5. **Pitch:** the replay at **15 Apr 2002** is a strong opening frame (3 dry, 1 Turant, 2 Chetavni).
