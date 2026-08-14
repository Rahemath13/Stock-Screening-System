"""
Sample Data Generator & Realistic Tick Stream Simulator.
Generates realistic tick streams for testing screening rules, SMMA technicals,
ETQ/LTQ windows, and AI/ML signal evaluation.
"""

import random
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Sample Universe of Stocks with realistic price tiers
SAMPLE_STOCKS = [
    # Compliant candidates (LTP ₹30 - ₹500 with high liquidity >10L)
    {"symbol": "IDFCFIRSTB", "base_ltp": 72.50, "liquid": True, "sector": "Banking"},
    {"symbol": "SUZLON", "base_ltp": 54.30, "liquid": True, "sector": "Energy"},
    {"symbol": "GMRINFRA", "base_ltp": 88.60, "liquid": True, "sector": "Infra"},
    {"symbol": "SOUTHBANK", "base_ltp": 32.10, "liquid": True, "sector": "Banking"},
    {"symbol": "IRB", "base_ltp": 62.40, "liquid": True, "sector": "Infra"},
    {"symbol": "HFCL", "base_ltp": 115.80, "liquid": True, "sector": "Telecom"},
    {"symbol": "NMDC", "base_ltp": 218.40, "liquid": True, "sector": "Mining"},
    {"symbol": "TATAPOWER", "base_ltp": 412.50, "liquid": True, "sector": "Energy"},
    {"symbol": "BHEL", "base_ltp": 285.30, "liquid": True, "sector": "Capital Goods"},
    {"symbol": "HUDCO", "base_ltp": 194.20, "liquid": True, "sector": "Finance"},
    {"symbol": "PNB", "base_ltp": 112.75, "liquid": True, "sector": "Banking"},
    {"symbol": "NHPC", "base_ltp": 94.60, "liquid": True, "sector": "Energy"},
    
    # Illiquid or non-compliant candidates for screener testing
    {"symbol": "SMALLCAP1", "base_ltp": 45.00, "liquid": False, "sector": "Others"}, # Illiquid (<10L qty)
    {"symbol": "RELIANCE", "base_ltp": 2854.65, "liquid": True, "sector": "Energy"},  # Price > 500
    {"symbol": "TCS", "base_ltp": 3692.10, "liquid": True, "sector": "IT"},         # Price > 500
    {"symbol": "INFY", "base_ltp": 1458.35, "liquid": True, "sector": "IT"},        # Price > 500
    {"symbol": "PENNYSTOCK", "base_ltp": 14.50, "liquid": True, "sector": "Others"}, # Price < 30
]


def generate_market_depth(ltp: float, liquid: bool):
    """Generates 5-level bid/ask market depth."""
    spread = round(max(0.05, ltp * 0.0005), 2)
    bids = []
    asks = []
    
    base_qty_mult = 1.0 if liquid else 0.05
    
    for i in range(5):
        bid_p = round(ltp - (i + 1) * spread, 2)
        ask_p = round(ltp + (i + 1) * spread, 2)
        
        # High volume on top levels for liquid stocks
        if i == 0 and liquid:
            bid_q = random.randint(1_050_000, 2_500_000)
            ask_q = random.randint(1_050_000, 2_500_000)
        else:
            bid_q = int(random.randint(100_000, 500_000) * base_qty_mult)
            ask_q = int(random.randint(100_000, 500_000) * base_qty_mult)
            
        bids.append({"price": bid_p, "quantity": bid_q})
        asks.append({"price": ask_p, "quantity": ask_q})
        
    return bids, asks


def generate_historical_ticks(symbol_info: dict, minutes: int = 180, ticks_per_min: int = 4):
    """Generates a DataFrame of historical tick data for time-series calculations."""
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=minutes)
    total_ticks = minutes * ticks_per_min
    
    symbol = symbol_info["symbol"]
    base_price = symbol_info["base_ltp"]
    is_liquid = symbol_info["liquid"]
    
    # Generate random walk with drift
    returns = np.random.normal(0.0001, 0.0015, total_ticks)
    price_path = base_price * np.exp(np.cumsum(returns))
    
    ticks = []
    curr_time = start_time
    time_step = timedelta(seconds=60 / ticks_per_min)
    
    for p in price_path:
        curr_time += time_step
        ltp = round(float(p), 2)
        ltq = random.randint(500, 15000) if is_liquid else random.randint(50, 1000)
        
        # Sudden LTQ surge emulation for signal testing
        if random.random() < 0.05:
            ltq *= random.randint(5, 12)
            
        bids, asks = generate_market_depth(ltp, is_liquid)
        
        ticks.append({
            "timestamp": curr_time,
            "symbol": symbol,
            "ltp": ltp,
            "ltq": ltq,
            "bid_price": bids[0]["price"],
            "bid_qty": bids[0]["quantity"],
            "ask_price": asks[0]["price"],
            "ask_qty": asks[0]["quantity"],
            "bids": bids,
            "asks": asks,
        })
        
    return pd.DataFrame(ticks)


def generate_full_market_snapshot():
    """Generates initial history for all stocks in universe."""
    all_data = {}
    for item in SAMPLE_STOCKS:
        df = generate_historical_ticks(item, minutes=180, ticks_per_min=4)
        all_data[item["symbol"]] = df
    return all_data
