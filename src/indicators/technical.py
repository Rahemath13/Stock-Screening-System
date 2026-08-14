"""
Technical Indicators & Volume Aggregation Engine.
Implements SMMA (20), SMMA (120), ETQ (5m/20m/60m), and Avg LTP (20m/60m).
"""

from datetime import timedelta
import numpy as np
import pandas as pd


def calculate_smma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculates Smoothed Moving Average (SMMA / Wilder's Smoothing).
    Formula:
      - First SMMA value = SMA(period)
      - Subsequent SMMA = (prev_smma * (period - 1) + current_price) / period
    """
    if len(prices) < period:
        # Fallback if insufficient historical data
        return pd.Series(index=prices.index, data=prices.mean() if len(prices) > 0 else np.nan)
        
    smma_vals = np.empty(len(prices))
    smma_vals[:] = np.nan
    
    # First SMMA is the simple average of the first 'period' values
    first_sma = prices.iloc[:period].mean()
    smma_vals[period - 1] = first_sma
    
    prices_arr = prices.values
    for i in range(period, len(prices)):
        smma_vals[i] = (smma_vals[i - 1] * (period - 1) + prices_arr[i]) / period
        
    return pd.Series(smma_vals, index=prices.index)


def compute_etq(df: pd.Series, timestamp_col: pd.Series, minutes: int) -> float:
    """
    Computes total Exchange Traded Quantity (ETQ) executed in the last 'minutes'.
    """
    if df.empty or len(timestamp_col) == 0:
        return 0.0
        
    latest_time = timestamp_col.iloc[-1]
    cutoff = latest_time - timedelta(minutes=minutes)
    
    mask = timestamp_col >= cutoff
    return float(df[mask].sum())


def compute_avg_ltp(ltp_series: pd.Series, timestamp_col: pd.Series, minutes: int) -> float:
    """
    Computes average Last Traded Price (LTP) over the last 'minutes'.
    """
    if ltp_series.empty or len(timestamp_col) == 0:
        return 0.0
        
    latest_time = timestamp_col.iloc[-1]
    cutoff = latest_time - timedelta(minutes=minutes)
    
    mask = timestamp_col >= cutoff
    filtered_ltp = ltp_series[mask]
    
    if filtered_ltp.empty:
        return float(ltp_series.iloc[-1])
        
    return float(filtered_ltp.mean())


def compute_all_indicators_for_df(df: pd.DataFrame) -> dict:
    """
    Computes all required assignment metrics for a single stock's DataFrame:
    - SMMA(20), SMMA(120)
    - ETQ(5m), ETQ(20m), ETQ(60m)
    - Avg LTP(20m), Avg LTP(60m)
    - SMMA Crossover Signals
    """
    if df.empty or len(df) < 5:
        return {}
        
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    prices = df["ltp"]
    timestamps = df["timestamp"]
    ltqs = df["ltq"] if "ltq" in df.columns else pd.Series(0, index=df.index)
    
    # 1. SMMA Calculations
    smma_20_series = calculate_smma(prices, period=20)
    smma_120_series = calculate_smma(prices, period=120)
    
    smma_20_curr = float(smma_20_series.dropna().iloc[-1]) if not smma_20_series.dropna().empty else float(prices.iloc[-1])
    smma_120_curr = float(smma_120_series.dropna().iloc[-1]) if not smma_120_series.dropna().empty else float(prices.iloc[-1])
    
    # Check for crossover on recent ticks
    signal = "NEUTRAL"
    if len(smma_20_series.dropna()) >= 2 and len(smma_120_series.dropna()) >= 2:
        prev_smma_20 = float(smma_20_series.dropna().iloc[-2])
        prev_smma_120 = float(smma_120_series.dropna().iloc[-2])
        
        if prev_smma_20 <= prev_smma_120 and smma_20_curr > smma_120_curr:
            signal = "BUY"
        elif prev_smma_20 >= prev_smma_120 and smma_20_curr < smma_120_curr:
            signal = "SELL"
        elif smma_20_curr > smma_120_curr:
            signal = "BUY"
        else:
            signal = "SELL"

    # 2. ETQ Calculations
    etq_5m = compute_etq(ltqs, timestamps, minutes=5)
    etq_20m = compute_etq(ltqs, timestamps, minutes=20)
    etq_60m = compute_etq(ltqs, timestamps, minutes=60)
    
    # 3. Avg LTP Calculations
    avg_ltp_20m = compute_avg_ltp(prices, timestamps, minutes=20)
    avg_ltp_60m = compute_avg_ltp(prices, timestamps, minutes=60)
    
    # 4. LTQ 2m vs 5m calculations for AI feature
    avg_ltq_2m = compute_etq(ltqs, timestamps, minutes=2) / max(1, len(ltqs[timestamps >= (timestamps.iloc[-1] - timedelta(minutes=2))]))
    avg_ltq_5m = compute_etq(ltqs, timestamps, minutes=5) / max(1, len(ltqs[timestamps >= (timestamps.iloc[-1] - timedelta(minutes=5))]))
    ltq_ratio_2m_5m = (avg_ltq_2m / avg_ltq_5m) if avg_ltq_5m > 0 else 1.0

    return {
        "smma_20": round(smma_20_curr, 2),
        "smma_120": round(smma_120_curr, 2),
        "signal": signal,
        "etq_5m": etq_5m,
        "etq_20m": etq_20m,
        "etq_60m": etq_60m,
        "avg_ltp_20m": round(avg_ltp_20m, 2),
        "avg_ltp_60m": round(avg_ltp_60m, 2),
        "ltq_ratio_2m_5m": round(ltq_ratio_2m_5m, 2),
        "smma_20_series": smma_20_series,
        "smma_120_series": smma_120_series,
    }
