"""
Test Angel One Historical Candle Data API.
Fetches 1-minute candles to pre-calculate SMMA(20) and SMMA(120) technical indicators.
Run: python scripts/test_historical.py
"""

import json
from datetime import datetime, timedelta
import sys
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.test_angel_login import test_login


def fetch_historical_candles(smart_api, symbol: str = "RELIANCE", token: str = "2885", days: int = 5):
    print("=" * 60)
    print(f"Fetching Historical 1-Minute Candles for {symbol} (Token: {token})...")
    print("=" * 60)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_MINUTE",
        "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
        "todate": end_time.strftime("%Y-%m-%d %H:%M"),
    }

    try:
        response = smart_api.getCandleData(params)
        if response and response.get("status"):
            candles = response["data"]
            print(f"[SUCCESS] Downloaded {len(candles):,} candles for {symbol}")

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Save historical parquet file
            hist_dir = BASE_DIR / "data" / "historical"
            hist_dir.mkdir(parents=True, exist_ok=True)
            out_file = hist_dir / f"{symbol}.parquet"
            df.to_parquet(out_file, index=False)

            print(f"Saved to: {out_file}")
            print(df.tail(5))
            return df
        else:
            print(f"[FAILED] Historical fetch failed: {response}")
            return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Exception during historical fetch: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    login_res = test_login()
    if login_res:
        smart_api, _ = login_res
        
        # Load token lookup from instruments JSON
        inst_file = BASE_DIR / "data" / "instruments" / "nse_instruments.json"
        lookup = {}
        if inst_file.exists():
            with open(inst_file, "r", encoding="utf-8") as f:
                lookup = json.load(f).get("token_lookup", {})

        target_stocks = [
            ("RELIANCE", lookup.get("RELIANCE", "2885")),
            ("TCS", lookup.get("TCS", "11536")),
            ("INFY", lookup.get("INFY", "1594")),
            ("TATAPOWER", lookup.get("TATAPOWER", "3426")),
            ("SUZLON", lookup.get("SUZLON", "12018")),
            ("IDFCFIRSTB", lookup.get("IDFCFIRSTB", "11184")),
            ("PNB", lookup.get("PNB", "10666")),
            ("GMRINFRA", lookup.get("GMRINFRA", "13528")),
        ]

        for sym, tok in target_stocks:
            if tok:
                fetch_historical_candles(smart_api, symbol=sym, token=str(tok), days=30)

