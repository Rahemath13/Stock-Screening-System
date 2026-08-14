"""
Download NSE Instrument Scrip Master from Angel One.
Fetches official instrument tokens for NSE Equity stocks and saves to data/instruments/nse_instruments.json
"""

import json
import os
import sys
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

INSTRUMENTS_DIR = BASE_DIR / "data" / "instruments"
INSTRUMENTS_DIR.mkdir(parents=True, exist_ok=True)

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
OUTPUT_FILE = INSTRUMENTS_DIR / "nse_instruments.json"


def download_nse_instruments():
    print("=" * 60)
    print("Downloading Angel One NSE Scrip Master...")
    print(f"URL: {SCRIP_MASTER_URL}")
    print("=" * 60)

    try:
        response = requests.get(SCRIP_MASTER_URL, timeout=30)
        response.raise_for_status()
        raw_data = response.json()

        print(f"Total Raw Instruments Downloaded: {len(raw_data):,}")

        # Filter NSE Equity Stocks (exch_seg == 'NSE' and symbol ending in '-EQ')
        nse_equity = []
        token_lookup = {}

        for item in raw_data:
            if item.get("exch_seg") == "NSE" and item.get("symbol", "").endswith("-EQ"):
                stock_record = {
                    "token": item.get("token"),
                    "symbol": item.get("symbol"),
                    "name": item.get("name"),
                    "expiry": item.get("expiry"),
                    "strike": item.get("strike"),
                    "lotsize": item.get("lotsize"),
                    "instrumenttype": item.get("instrumenttype"),
                    "exch_seg": item.get("exch_seg"),
                    "tick_size": item.get("tick_size"),
                }
                nse_equity.append(stock_record)
                clean_sym = item.get("symbol").replace("-EQ", "")
                token_lookup[clean_sym] = item.get("token")

        output_payload = {
            "total_stocks": len(nse_equity),
            "token_lookup": token_lookup,
            "instruments": nse_equity,
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        print(f"[SUCCESS] Filtered NSE Equity Stocks: {len(nse_equity):,}")
        print(f"Saved to: {OUTPUT_FILE}")
        print("Sample Token Mappings:")
        for sym in ["RELIANCE", "TCS", "INFY", "SUZLON", "IDFCFIRSTB"]:
            if sym in token_lookup:
                print(f"  - {sym}-EQ -> Token: {token_lookup[sym]}")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Downloading instruments failed: {e}")


if __name__ == "__main__":
    download_nse_instruments()
