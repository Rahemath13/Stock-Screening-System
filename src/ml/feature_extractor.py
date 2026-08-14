"""
Quantitative Feature Extraction Engine for Machine Learning.
Engineers predictive features from price, SMMA indicators, LTQ, and ETQ statistics.
"""

from datetime import timedelta
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import ML_PARAMS


def extract_features_from_df(df: pd.DataFrame, indicator_dict: dict) -> dict:
    """
    Extracts a feature dictionary for a stock given its tick history and indicator output.
    """
    if df.empty or len(df) < 10:
        # Default fallback feature vector
        return {feat: 0.0 for feat in ML_PARAMS["FEATURE_NAMES"]}
        
    df = df.sort_values("timestamp").reset_index(drop=True)
    latest_tick = df.iloc[-1]
    ltp = float(latest_tick["ltp"])
    
    # 1. LTQ Ratio (Avg LTQ last 2 min vs Avg LTQ last 5 min)
    ltq_ratio_2m_5m = indicator_dict.get("ltq_ratio_2m_5m", 1.0)
    
    # 2. ETQ Metrics
    etq_5m = indicator_dict.get("etq_5m", 0.0)
    etq_20m = indicator_dict.get("etq_20m", 0.0)
    etq_60m = indicator_dict.get("etq_60m", 0.0)
    
    # 3. Price Momentum over 20 min
    avg_ltp_20m = indicator_dict.get("avg_ltp_20m", ltp)
    price_momentum_20m = ((ltp - avg_ltp_20m) / avg_ltp_20m) * 100.0 if avg_ltp_20m > 0 else 0.0
    
    # 4. SMMA Spread %
    smma_20 = indicator_dict.get("smma_20", ltp)
    smma_120 = indicator_dict.get("smma_120", ltp)
    smma_spread_pct = ((smma_20 - smma_120) / smma_120) * 100.0 if smma_120 > 0 else 0.0
    
    # 5. Bid/Ask Quantity Ratio
    bid_qty = float(latest_tick.get("bid_qty", 1.0))
    ask_qty = float(latest_tick.get("ask_qty", 1.0))
    bid_ask_qty_ratio = (bid_qty / ask_qty) if ask_qty > 0 else 1.0
    
    # 6. Volatility over 20 min
    latest_time = latest_tick["timestamp"]
    cutoff_20m = latest_time - timedelta(minutes=20)
    recent_prices = df[df["timestamp"] >= cutoff_20m]["ltp"]
    volatility_20m = float(recent_prices.std()) if len(recent_prices) > 2 else 0.0
    
    # 7. Total Volume over 20 min
    volume_20m = float(df[df["timestamp"] >= cutoff_20m]["ltq"].sum()) if "ltq" in df.columns else 0.0

    return {
        "ltq_ratio_2m_5m": float(ltq_ratio_2m_5m),
        "etq_5m": float(etq_5m),
        "etq_20m": float(etq_20m),
        "etq_60m": float(etq_60m),
        "price_momentum_20m": float(price_momentum_20m),
        "smma_spread_pct": float(smma_spread_pct),
        "bid_ask_qty_ratio": bid_ask_qty_ratio,
        "volatility_20m": volatility_20m,
        "volume_20m": volume_20m,
    }
