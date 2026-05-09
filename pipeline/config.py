"""
Configuration file for Forex Calendar Impact Analyzer
=====================================================
Centralized settings. Edit here instead of hunting through code.
"""

from datetime import datetime, timedelta

# ============================================================
# DATE RANGE - 3 years of history
# ============================================================
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

# ============================================================
# THE 8 MAJORS - Yahoo Finance ticker format
# ============================================================
FOREX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
}

# Pip value per pair (how many decimal places = 1 pip)
# JPY pairs: 0.01 = 1 pip | Others: 0.0001 = 1 pip
PIP_MULTIPLIER = {
    "EURUSD": 10000,
    "GBPUSD": 10000,
    "USDJPY": 100,
    "USDCHF": 10000,
    "AUDUSD": 10000,
    "USDCAD": 10000,
    "NZDUSD": 10000,
    "EURJPY": 100,
}

# ============================================================
# CURRENCIES WE CARE ABOUT (filter calendar noise)
# ============================================================
MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

# ============================================================
# OUTPUT PATHS
# ============================================================
DATA_DIR = "data"
CALENDAR_FILE = f"{DATA_DIR}/economic_calendar.csv"
OHLC_FILE = f"{DATA_DIR}/ohlc_daily.csv"
MASTER_FILE = f"{DATA_DIR}/master_dataset.csv"
STATS_FILE = f"{DATA_DIR}/event_statistics.csv"

# ============================================================
# EVENT IMPACT FILTER
# ============================================================
# "High" only for now. Later we can include Medium for richer data.
IMPACT_LEVELS = ["High"]

# ============================================================
# ATR PERIOD (for context-aware analysis)
# ============================================================
ATR_PERIOD = 20
