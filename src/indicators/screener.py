"""
Stock Screening and Liquidity Filter Module.
Filters stock universe based on strict prompt constraints:
1. LTP between ₹30 and ₹500
2. Bid Quantity > 1,000,000 (10 Lakhs)
3. Ask Quantity > 1,000,000 (10 Lakhs)
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import SCREENING_RULES


class StockScreener:
    """Evaluates individual stock quote snapshots against screening rules."""
    
    def __init__(self, min_ltp: float = None, max_ltp: float = None, min_bid_qty: int = None, min_ask_qty: int = None):
        self.min_ltp = min_ltp if min_ltp is not None else SCREENING_RULES["MIN_LTP"]
        self.max_ltp = max_ltp if max_ltp is not None else SCREENING_RULES["MAX_LTP"]
        self.min_bid_qty = min_bid_qty if min_bid_qty is not None else SCREENING_RULES["MIN_BID_QTY"]
        self.min_ask_qty = min_ask_qty if min_ask_qty is not None else SCREENING_RULES["MIN_ASK_QTY"]

    def is_price_compliant(self, ltp: float) -> bool:
        """Checks if LTP is within ₹30 to ₹500."""
        return self.min_ltp <= ltp <= self.max_ltp

    def is_liquid_compliant(self, bid_qty: int, ask_qty: int) -> bool:
        """Checks if both Bid Qty and Ask Qty exceed 10 Lakhs (1,000,000)."""
        return (bid_qty > self.min_bid_qty) and (ask_qty > self.min_ask_qty)

    def evaluate_stock(self, stock_tick: dict) -> dict:
        """
        Evaluates stock tick dictionary and returns screening breakdown.
        """
        ltp = float(stock_tick.get("ltp", 0.0))
        bid_qty = int(stock_tick.get("bid_qty", 0))
        ask_qty = int(stock_tick.get("ask_qty", 0))

        price_passed = self.is_price_compliant(ltp)
        liquidity_passed = self.is_liquid_compliant(bid_qty, ask_qty)

        fully_screened = price_passed and liquidity_passed

        return {
            "symbol": stock_tick.get("symbol", ""),
            "ltp": ltp,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "price_passed": price_passed,
            "liquidity_passed": liquidity_passed,
            "fully_screened": fully_screened,
        }
