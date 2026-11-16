"""Shared settings for the Thar early-warning pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

SIM_FILE = DATA_DIR / "sim_weekly.csv"
SOURCES_FILE = DATA_DIR / "sources.csv"
PRED_FILE = OUT_DIR / "predictions.csv"
METRICS_FILE = OUT_DIR / "metrics.json"
ALERTS_FILE = OUT_DIR / "alerts.csv"

# A source is "failed" when its storage index drops below this (0-100 scale).
DRY_LEVEL = 15.0
# Too salty for livestock (TDS, mg/L). Rough guide: >5000 stresses sheep/goats.
SALT_LIMIT = 5000.0

HORIZONS = [2, 4, 6]  # weeks ahead

# Cost of a missed warning vs a false alarm. Used to pick alert thresholds.
COST_MISS = 8.0
COST_FALSE_ALARM = 1.0

# Years
# The 2002 monsoon failed; the crisis peaks in spring/summer 2003.
# The demo replays Oct 2001 -> Sep 2003 (two failed monsoons). Never trained on.
REPLAY_START = "2001-10-01"
REPLAY_END = "2003-09-30"
HELDOUT_YEARS = [2001, 2002, 2003, 2004]  # replay window + buffer against leakage
# Alert thresholds are chosen from out-of-fold predictions (4 folds of whole years) on training years.
TEST_YEARS = [2019, 2020, 2021, 2022, 2023]  # strict forward test
