"""
Automated Test Suite for Stock Screening System.
Validates:
1. Stock Screener (LTP ₹30-500 & Bid/Ask Qty > 10L)
2. SMMA(20) and SMMA(120) Technical Indicator calculation
3. ETQ (5m/20m/60m) rolling aggregation
4. Avg LTP (20m/60m) computation
5. AI/ML Predictor inference
"""

import sys
from pathlib import Path
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# pyrefly: ignore [missing-import]
from src.indicators.screener import StockScreener
# pyrefly: ignore [missing-import]
from src.indicators.technical import calculate_smma, compute_etq, compute_avg_ltp, compute_all_indicators_for_df
# pyrefly: ignore [missing-import]
from src.ml.predictor import SignalPredictor


class TestStockScreeningSystem(unittest.TestCase):

    def setUp(self):
        self.screener = StockScreener(min_ltp=30.0, max_ltp=500.0, min_bid_qty=1_000_000, min_ask_qty=1_000_000)
        self.predictor = SignalPredictor()

    def test_screener_rules(self):
        """Test LTP price bounds and liquidity quantity threshold enforcement."""
        # 1. Compliant Stock
        pass_tick = {"symbol": "IDFCFIRSTB", "ltp": 75.0, "bid_qty": 1_200_000, "ask_qty": 1_500_000}
        eval_pass = self.screener.evaluate_stock(pass_tick)
        self.assertTrue(eval_pass["fully_screened"])

        # 2. LTP > ₹500 (Fail)
        high_price_tick = {"symbol": "RELIANCE", "ltp": 2854.0, "bid_qty": 1_500_000, "ask_qty": 1_500_000}
        eval_high = self.screener.evaluate_stock(high_price_tick)
        self.assertFalse(eval_high["fully_screened"])
        self.assertFalse(eval_high["price_passed"])

        # 3. LTP < ₹30 (Fail)
        low_price_tick = {"symbol": "PENNY", "ltp": 15.0, "bid_qty": 1_500_000, "ask_qty": 1_500_000}
        eval_low = self.screener.evaluate_stock(low_price_tick)
        self.assertFalse(eval_low["fully_screened"])

        # 4. Bid Qty < 10L (Fail)
        low_bid_tick = {"symbol": "ILLIQUID", "ltp": 80.0, "bid_qty": 800_000, "ask_qty": 1_200_000}
        eval_low_bid = self.screener.evaluate_stock(low_bid_tick)
        self.assertFalse(eval_low_bid["fully_screened"])
        self.assertFalse(eval_low_bid["liquidity_passed"])

    def test_smma_calculation(self):
        """Test SMMA formula output correctness."""
        prices = pd.Series([10.0] * 30 + [20.0] * 30)
        smma_20 = calculate_smma(prices, period=20)
        self.assertIsNotNone(smma_20)
        self.assertEqual(len(smma_20), 60)
        # Verify first SMMA value equals SMA(20)
        self.assertAlmostEqual(smma_20.iloc[19], 10.0, places=2)
        # SMMA smoothing should reflect gradual rise towards 20
        self.assertGreater(smma_20.iloc[59], 10.0)
        self.assertLess(smma_20.iloc[59], 20.0)

    def test_etq_and_avg_ltp_windows(self):
        """Test rolling window ETQ sum and average LTP computation."""
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in reversed(range(10))]
        ltqs = pd.Series([1000] * 10)
        prices = pd.Series([100.0] * 10)

        etq_5m = compute_etq(ltqs, pd.Series(timestamps), minutes=5)
        # Should sum last 5 or 6 entries (~6000)
        self.assertGreaterEqual(etq_5m, 5000)

        avg_ltp_20m = compute_avg_ltp(prices, pd.Series(timestamps), minutes=20)
        self.assertAlmostEqual(avg_ltp_20m, 100.0, places=2)

    def test_ai_predictor(self):
        """Test AI Predictor confidence score and explainability output."""
        features = {
            "ltq_ratio_2m_5m": 1.45,
            "etq_5m": 2_500_000,
            "etq_20m": 8_500_000,
            "etq_60m": 22_000_000,
            "price_momentum_20m": 0.35,
            "smma_spread_pct": 0.42,
            "bid_ask_qty_ratio": 1.25,
            "volatility_20m": 1.2,
            "volume_20m": 15_000_000,
        }
        res = self.predictor.predict_signal(features, raw_signal="BUY")
        self.assertTrue("ACCEPT" in res["decision"] or "AVOID" in res["decision"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertTrue(len(res["reason"]) > 0)



if __name__ == "__main__":
    unittest.main()
