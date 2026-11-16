"""Pull REAL satellite signals for each water source from Google Earth Engine.

Produces data/gee_weekly.csv with the same column names the simulator uses
(rain_mm, temp_c, ndvi, sar_area, grace_cm), so you can swap real data in.

Setup (one time):
    pip install earthengine-api
    earthengine authenticate          # opens a browser, sign in with Google
    # create / pick a Cloud project with Earth Engine enabled, then:
    export EE_PROJECT=your-project-id

Run:
    python -m thar.gee_pull --start 2018-01-01 --end 2023-12-31

Notes / desert rules:
  - Sentinel-2 NDVI is cloud-masked with the SCL band; cloudy weeks come back empty (NaN).
    The model treats NaN as "unknown", it never pretends a value.
  - Sentinel-1 (radar) sees through cloud and dust. Water is dark in VV backscatter;
    we count pixels below -18 dB inside a 300 m buffer as open water.
  - GRACE is monthly and ~300 km wide: it is a regional trend signal, not a per-well one.
  - Ground-truth well levels: download from India-WRIS (indiawris.gov.in) -> Groundwater Level,
    filter State=Rajasthan, District=Jaisalmer/Barmer. Those are ~4 readings a year.
"""
import argparse
import os

import pandas as pd

from thar.config import DATA_DIR, SOURCES_FILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--buffer_m", type=int, default=300)
    args = ap.parse_args()

    import ee  # imported here so the rest of the project runs without it
    ee.Initialize(project=os.environ.get("EE_PROJECT"))

    sources = pd.read_csv(SOURCES_FILE)
    weeks = pd.date_range(args.start, args.end, freq="W-MON")

    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("temperature_2m")
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .select("VV"))
    grace = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON").select("lwe_thickness")

    def s2_ndvi(img):
        scl = img.select("SCL")
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))  # no shadow/cloud/cirrus
        return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(clear)

    rows = []
    for _, s in sources.iterrows():
        pt = ee.Geometry.Point([s.lon, s.lat])
        area = pt.buffer(args.buffer_m)
        big = pt.buffer(5000)
        print(f"Pulling {s.source_id} {s.village} ...")
        for wk in weeks:
            a, b = ee.Date(str(wk.date())), ee.Date(str((wk + pd.Timedelta(days=7)).date()))
            r = {"date": wk, "source_id": s.source_id}

            def reduce(img, geom, band, scale, reducer=ee.Reducer.mean()):
                v = img.reduceRegion(reducer, geom, scale, maxPixels=1e9).get(band)
                return v

            stats = ee.Dictionary({
                "rain_mm": reduce(chirps.filterDate(a, b).sum(), big, "precipitation", 5000),
                "temp_c": reduce(era5.filterDate(a, b).mean(), big, "temperature_2m", 11000),
                "ndvi": reduce(s2.filterDate(a, b).filterBounds(area).map(s2_ndvi).median(), area, "ndvi", 10),
                "sar_water_frac": reduce(s1.filterDate(a, b).filterBounds(area).median().lt(-18).rename("VV"),
                                         area, "VV", 10),
                "grace_cm": reduce(grace.filterDate(a.advance(-20, "day"), b).mean(), big, "lwe_thickness", 50000),
            }).getInfo()
            r.update(stats)
            rows.append(r)

    df = pd.DataFrame(rows)
    df["temp_c"] = df["temp_c"] - 273.15
    df["sar_area"] = df.pop("sar_water_frac") * 100
    out = DATA_DIR / "gee_weekly.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print("Tip: getInfo() per week is slow. For many wells, switch to ee.batch.Export.table.toDrive.")


if __name__ == "__main__":
    main()
