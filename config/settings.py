"""
System Configuration Parameters for AI/ML Stock Screening System.
Defines screening thresholds, indicator periods, ETQ/Avg price windows,
and ML model parameters.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# 1. Stock Screening Criteria
SCREENING_RULES = {
    "MIN_LTP": 30.0,
    "MAX_LTP": 500.0,
    "MIN_BID_QTY": 1_000_000,
    "MIN_ASK_QTY": 1_000_000,
}

# 2. Technical Indicator Parameters
INDICATOR_PARAMS = {
    "SMMA_FAST": 20,
    "SMMA_SLOW": 120,
}

# 3. Rolling Time Window Parameters (in minutes)
WINDOW_PARAMS = {
    "ETQ_WINDOWS_MIN": [5, 20, 60],
    "AVG_LTP_WINDOWS_MIN": [20, 60],
    "LTQ_FAST_MIN": 2,
    "LTQ_SLOW_MIN": 5,
}

# 4. Machine Learning Parameters
ML_PARAMS = {
    "MODEL_PATH": MODELS_DIR / "smma_crossover_model.pkl",
    "FEATURE_IMPORTANCE_PATH": MODELS_DIR / "feature_importance.json",
    "CONFIDENCE_THRESHOLD": 0.65,
    "RANDOM_SEED": 42,
    "FEATURE_NAMES": [
        "ltq_ratio_2m_5m",
        "etq_5m",
        "etq_20m",
        "etq_60m",
        "price_momentum_20m",
        "smma_spread_pct",
        "bid_ask_qty_ratio",
        "volatility_20m",
        "volume_20m",
    ],
}

# 5. Broker API Configuration Placeholders (SmartAPI & Fyers)
BROKER_CONFIG = {
    "ANGEL_ONE": {
        "API_KEY": os.getenv("ANGEL_API_KEY", "Gul267iQ"),
        "CLIENT_CODE": os.getenv("ANGEL_CLIENT_CODE", "AAAC413396"),
        "PASSWORD": os.getenv("ANGEL_PASSWORD", "2580"),
        "TOTP_KEY": os.getenv("ANGEL_TOTP_KEY", "QPLLOZTSXDWJF53CFJXHLEHMWU"),
    },
    "FYERS": {
        "CLIENT_ID": os.getenv("FYERS_CLIENT_ID", ""),
        "SECRET_KEY": os.getenv("FYERS_SECRET_KEY", ""),
        "REDIRECT_URI": os.getenv("FYERS_REDIRECT_URI", "http://127.0.0.1:5000/"),
        "ACCESS_TOKEN": os.getenv("FYERS_ACCESS_TOKEN", ""),
    }
}
