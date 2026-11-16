"""Realistic SIMULATED weekly data for water sources in Jaisalmer district (Thar desert).

Why simulate: real well readings (CGWB / India-WRIS) are only taken ~4 times a year,
and there are no public weekly sensors at village ponds. We simulate the hidden
"true" storage of each source with simple physics, then generate the noisy, gappy
signals a real system would actually see:

  - in-situ sensor level   (dies for weeks at a time -> NaN)
  - Sentinel-1 radar water-area   (ponds only; sees through cloud and dust)
  - Sentinel-2 NDVI around the source (missing under monsoon cloud)
  - CHIRPS rainfall, ERA5 temperature (always available)
  - GRACE regional groundwater anomaly (monthly, coarse)
  - herder SMS reports ("NADI LOW")
  - TDS salinity (wells only)

Rainfall years are anchored to known history: 2002 = severe monsoon failure in
western Rajasthan, 2006 = Barmer floods, 2009/2018 = deficit years.
Swap this file for thar/gee_pull.py output when you have real data.

Run:  python -m thar.simulate
"""
import numpy as np
import pandas as pd

from thar.config import SIM_FILE, SOURCES_FILE, DRY_LEVEL

RNG = np.random.default_rng(42)

# Approximate locations of villages in Jaisalmer district. Data at each is simulated.
SOURCES = [
    # id, village, type, lat, lon
    ("S01", "Ramgarh", "nadi", 27.37, 70.49),
    ("S02", "Sam", "nadi", 26.83, 70.51),
    ("S03", "Khuri", "beri", 26.61, 70.69),
    ("S04", "Mohangarh", "tubewell", 27.28, 71.21),
    ("S05", "Pokaran", "tubewell", 26.92, 71.92),
    ("S06", "Fatehgarh", "nadi", 26.46, 71.12),
    ("S07", "Devikot", "beri", 26.70, 71.19),
    ("S08", "Lathi", "tubewell", 26.88, 71.53),
    ("S09", "Chandan", "nadi", 26.94, 71.33),
    ("S10", "Myajlar", "beri", 26.24, 70.37),
    ("S11", "Tanot", "tubewell", 27.75, 70.28),
    ("S12", "Nachna", "nadi", 27.50, 71.73),
    ("S13", "Basanpi", "beri", 26.98, 70.98),
    ("S14", "Bhaniyana", "nadi", 26.76, 71.80),
]

# Monsoon strength relative to normal (~170 mm/yr in Jaisalmer).
RAIN_YEAR_OVERRIDES = {2000: 0.6, 2002: 0.2, 2006: 1.9, 2009: 0.45, 2015: 1.2,
                       2017: 1.4, 2018: 0.5, 2021: 0.9}


def weekly_calendar(start="2000-01-03", end="2023-12-31"):
    return pd.date_range(start, end, freq="W-MON")


def make_rain_and_temp(dates):
    years = sorted(set(dates.year))
    strength = {y: RAIN_YEAR_OVERRIDES.get(y, float(np.clip(RNG.lognormal(0, 0.35), 0.4, 1.8)))
                for y in years}
    onset = {y: int(26 + RNG.integers(-1, 3) + (2 if strength[y] < 0.5 else 0)) for y in years}

    rain = np.zeros(len(dates))
    temp = np.zeros(len(dates))
    for i, d in enumerate(dates):
        w = d.isocalendar().week
        y = d.year
        # Monsoon: ~12 weeks after onset, bell-shaped, bursty.
        if onset[y] <= w <= onset[y] + 12:
            shape = np.sin(np.pi * (w - onset[y]) / 12.0)
            burst = RNG.gamma(0.8, 1.0)  # many dry weeks, a few big storms
            rain[i] = 170 * strength[y] * shape * burst / 7.6
        elif RNG.random() < 0.04:  # rare winter western-disturbance shower
            rain[i] = RNG.gamma(1.0, 4.0)
        # Temperature: hottest late May / June.
        temp[i] = 26 + 9 * np.sin(2 * np.pi * (w - 9) / 52) + RNG.normal(0, 1.2)
        if strength[y] < 0.5 and 18 <= w <= 30:
            temp[i] += 1.0
    return rain, temp, strength


def simulate_source(src, dates, rain, temp):
    sid, village, kind, lat, lon = src
    n = len(dates)
    S = np.zeros(n)
    tds = np.zeros(n)
    week = dates.isocalendar().week.to_numpy()

    if kind == "nadi":  # village pond: fills fast from runoff, evaporates fast
        catch, loss_base, draw = RNG.uniform(0.9, 1.4), RNG.uniform(1.0, 1.5), RNG.uniform(0.5, 0.9)
        S[0] = 60
    elif kind == "beri":  # shallow seepage well: slow, lagged recharge
        catch, loss_base, draw = RNG.uniform(0.42, 0.58), RNG.uniform(0.25, 0.4), RNG.uniform(0.6, 0.85)
        S[0] = 65
    else:  # tubewell: deep aquifer, slow recharge, pumped hard, declining over decades
        catch, loss_base, draw = RNG.uniform(0.2, 0.28), RNG.uniform(0.05, 0.08), RNG.uniform(0.36, 0.48)
        S[0] = 80
    trend = RNG.uniform(0.002, 0.005) if kind == "tubewell" else 0.0
    base_tds = {"nadi": 600, "beri": 1800, "tubewell": 2800}[kind] * RNG.uniform(0.8, 1.3)

    # Beri / tubewell recharge is delayed and smeared out (water takes time to percolate).
    if kind == "nadi":
        recharge = rain
    else:
        lag = 6 if kind == "beri" else 14
        k = np.exp(-np.arange(lag * 2) / lag)
        k /= k.sum()
        recharge = np.convolve(rain, k)[:n]

    for i in range(1, n):
        evap = loss_base * (1 + 0.05 * (temp[i] - 26))
        # More herds gather in the hot months (migrants arrive as other sources fail).
        herd = draw * (1.3 if 14 <= week[i] <= 27 else 1.0)
        S[i] = S[i - 1] + catch * recharge[i] - evap - herd - trend * i / 52
        S[i] = float(np.clip(S[i], 0, 100))
    # Salinity rises as the water column shrinks (concentration effect).
    tds = base_tds * (1 + 1.6 * np.clip((60 - S) / 60, 0, 1) ** 1.5) * RNG.normal(1, 0.03, n)
    return S, tds


def blocky_dropout(n, p_start=0.01, mean_len=6):
    """Sensors die and stay dead for a few weeks: returns a boolean 'missing' mask."""
    miss = np.zeros(n, bool)
    i = 0
    while i < n:
        if RNG.random() < p_start:
            L = max(1, int(RNG.exponential(mean_len)))
            miss[i:i + L] = True
            i += L
        i += 1
    return miss


def main():
    dates = weekly_calendar()
    rain, temp, strength = make_rain_and_temp(dates)
    week = dates.isocalendar().week.to_numpy()

    # GRACE: regional groundwater anomaly, monthly, smooth.
    grace = pd.Series(np.cumsum(rain - rain.mean()) * 0.02, index=dates)
    grace = grace - np.linspace(0, 6, len(dates))  # long-term depletion
    grace = grace.resample("MS").mean().reindex(dates, method="ffill").to_numpy()

    rows = []
    for src in SOURCES:
        sid, village, kind, lat, lon = src
        S, tds = simulate_source(src, dates, rain, temp)
        n = len(dates)

        sensor = S + RNG.normal(0, 2.5, n)
        sensor[blocky_dropout(n, 0.012, 7)] = np.nan

        if kind == "nadi":  # radar water area tracks pond storage nonlinearly
            sar = 100 * (np.clip(S, 0, 100) / 100) ** 0.7 + RNG.normal(0, 5, n)
            sar[RNG.random(n) < 0.15] = np.nan  # missed passes
        else:
            sar = np.full(n, np.nan)

        ndvi = 0.08 + 0.18 * pd.Series(S / 100).ewm(span=8).mean().to_numpy() \
            + 0.002 * pd.Series(rain).rolling(4, min_periods=1).sum().to_numpy() + RNG.normal(0, 0.015, n)
        cloudy = (RNG.random(n) < np.where((week >= 26) & (week <= 38), 0.55, 0.08))
        ndvi[cloudy] = np.nan

        p_report = np.clip((40 - S) / 40, 0, 1) * 0.6
        reports = RNG.binomial(3, p_report)

        tds_obs = tds.copy() if kind != "nadi" else np.full(n, np.nan)
        if kind != "nadi":
            tds_obs[RNG.random(n) < 0.75] = np.nan  # field kit tests are occasional

        rows.append(pd.DataFrame({
            "date": dates, "source_id": sid, "village": village, "type": kind,
            "lat": lat, "lon": lon,
            "true_level": S.round(2), "true_tds": tds.round(0),
            "sensor_level": sensor.round(2), "sar_area": np.round(sar, 2),
            "ndvi": ndvi.round(4), "rain_mm": rain.round(2), "temp_c": temp.round(2),
            "grace_cm": grace.round(3), "herder_reports": reports, "tds_obs": np.round(tds_obs, 0),
        }))

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(SIM_FILE, index=False)
    pd.DataFrame(SOURCES, columns=["source_id", "village", "type", "lat", "lon"]).to_csv(SOURCES_FILE, index=False)

    dry = df.assign(year=df.date.dt.year, dry=df.true_level < DRY_LEVEL).groupby("year").dry.mean()
    print(f"Wrote {len(df):,} rows for {len(SOURCES)} sources -> {SIM_FILE}")
    print("Share of source-weeks below the dry line, by year:")
    print((dry * 100).round(1).to_string())


if __name__ == "__main__":
    main()
