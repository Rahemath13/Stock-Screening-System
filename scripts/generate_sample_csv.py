"""
Populate sample offline dataset CSV in data/sample/sample_nse_ticks.csv.
Provides a ready sample CSV dataset for offline testing and Streamlit CSV upload.
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.sample_generator import generate_full_market_snapshot

SAMPLE_DIR = BASE_DIR / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CSV = SAMPLE_DIR / "sample_nse_ticks.csv"


def create_sample_csv():
    snapshot = generate_full_market_snapshot()
    all_ticks = []
    for sym, df in snapshot.items():
        all_ticks.append(df)
        
    combined_df = pd.concat(all_ticks, ignore_index=True)
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
    
    # Save standard columns
    export_df = pd.DataFrame()
    export_df["timestamp"] = combined_df["timestamp"]
    export_df["symbol"] = combined_df["symbol"]
    export_df["ltp"] = combined_df["ltp"]
    export_df["ltq"] = combined_df["ltq"]
    export_df["bid_price"] = combined_df["bid_price"]
    export_df["bid_qty"] = combined_df["bid_qty"]
    export_df["ask_price"] = combined_df["ask_price"]
    export_df["ask_qty"] = combined_df["ask_qty"]

    export_df.to_csv(SAMPLE_CSV, index=False)
    print(f"[SUCCESS] Sample tick dataset created at: {SAMPLE_CSV}")
    print(f"Total Ticks: {len(export_df):,}")


if __name__ == "__main__":
    create_sample_csv()
