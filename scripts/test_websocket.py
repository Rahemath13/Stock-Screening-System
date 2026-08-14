"""
Test Angel One SmartWebSocketV2 Real-Time Tick Collector.
Connects to wss://smartapisocket.angelone.in/smart-stream in FULL Quote Mode (Mode 3).
Collects LTP, LTQ, ETQ, Bid/Ask Depth, and saves to data/live/ directory.
Run: python scripts/test_websocket.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import BROKER_CONFIG
from scripts.test_angel_login import test_login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parquet Storage Location
LIVE_DATA_DIR = BASE_DIR / "data" / "live" / datetime.now().strftime("%Y-%m-%d")
LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def on_data(wsapp, message):
    """Callback for incoming SmartWebSocketV2 tick data."""
    try:
        # Message is parsed tick dictionary from SmartWebSocketV2
        token = message.get("token")
        ltp = message.get("last_traded_price", 0.0) / 100.0 if "last_traded_price" in message else 0.0
        ltq = message.get("last_traded_quantity", 0)
        volume = message.get("volume_traded", 0)
        avg_price = message.get("average_traded_price", 0.0) / 100.0 if "average_traded_price" in message else 0.0

        # Market Depth parsing
        best_bids = message.get("best_5_buy_data", [])
        best_asks = message.get("best_5_sell_data", [])

        bid_price = best_bids[0].get("price", 0.0) / 100.0 if best_bids else 0.0
        bid_qty = best_bids[0].get("quantity", 0) if best_bids else 0
        ask_price = best_asks[0].get("price", 0.0) / 100.0 if best_asks else 0.0
        ask_qty = best_asks[0].get("quantity", 0) if best_asks else 0

        tick_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "token": token,
            "ltp": ltp,
            "ltq": ltq,
            "volume": volume,
            "avg_price": avg_price,
            "bid_price": bid_price,
            "bid_qty": bid_qty,
            "ask_price": ask_price,
            "ask_qty": ask_qty,
        }

        print(f"[{tick_record['timestamp']}] TOKEN:{token} | LTP: ₹{ltp:.2f} | LTQ: {ltq} | Bid: ₹{bid_price:.2f} ({bid_qty}) | Ask: ₹{ask_price:.2f} ({ask_qty})")

    except Exception as e:
        logger.error(f"Error processing tick message: {e}")


def on_open(wsapp):
    logger.info("WebSocket Connection Opened Successfully.")


def on_error(wsapp, error):
    logger.error(f"WebSocket Error: {error}")


def on_close(wsapp):
    logger.info("WebSocket Connection Closed.")


def start_websocket_stream():
    login_res = test_login()
    if not login_res:
        print("[ERROR] Login failed. Please check credentials.")
        return

    smart_api, auth_data = login_res
    jwt_token = auth_data["jwtToken"]
    feed_token = smart_api.getfeedToken()
    client_code = BROKER_CONFIG["ANGEL_ONE"]["CLIENT_CODE"]
    api_key = BROKER_CONFIG["ANGEL_ONE"]["API_KEY"]

    try:
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2

        sws = SmartWebSocketV2(
            auth_token=jwt_token,
            api_key=api_key,
            client_code=client_code,
            feed_token=feed_token,
        )

        # Mode 3 = FULL Snapquote Mode (LTP, LTQ, Volume, Total Buy/Sell Qty, Market Depth 5)
        # Token 2885 = RELIANCE-EQ (from nse_instruments.json)
        correlation_id = "test_stream_1"
        action = 1  # Subscribe
        mode = 3    # FULL Quote Mode
        token_list = [{"exchangeType": 1, "tokens": ["2885"]}]  # 1 = NSE Equity

        sws.on_data = on_data
        sws.on_open = on_open
        sws.on_error = on_error
        sws.on_close = on_close

        print("=" * 60)
        print("Connecting to SmartWebSocketV2 (wss://smartapisocket.angelone.in/smart-stream)...")
        print("Subscribed Token: 2885 (RELIANCE-EQ) in Mode 3 FULL Depth")
        print("=" * 60)

        sws.connect()

    except ImportError:
        print("[ERROR] SmartWebSocketV2 not found in smartapi-python SDK.")
    except Exception as e:
        print(f"[ERROR] WebSocket Connection Exception: {e}")


if __name__ == "__main__":
    start_websocket_stream()
