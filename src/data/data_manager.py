"""
Central Data Manager & Time-Series Memory Store.
Maintains rolling tick history per stock symbol and provides thread-safe access.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class DataManager:
    """Manages active stock universe ticks and historical windows."""
    
    def __init__(self, max_history_minutes: int = 180):
        self.max_history_minutes = max_history_minutes
        # Storage dictionary: symbol -> DataFrame of ticks
        self._buffers = {}
        
    def initialize_symbol_data(self, symbol: str, df: pd.DataFrame):
        """Initializes buffer with historical tick DataFrame."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        self._buffers[symbol] = df.copy()
        
    def add_tick(self, symbol: str, tick_data: dict):
        """Appends a new tick to the symbol buffer."""
        tick_data["timestamp"] = pd.to_datetime(tick_data.get("timestamp", datetime.now()))
        
        new_row = pd.DataFrame([tick_data])
        if symbol not in self._buffers or self._buffers[symbol].empty:
            self._buffers[symbol] = new_row
        else:
            self._buffers[symbol] = pd.concat([self._buffers[symbol], new_row], ignore_index=True)
            
        # Prune old records beyond max history
        cutoff = tick_data["timestamp"] - timedelta(minutes=self.max_history_minutes)
        self._buffers[symbol] = self._buffers[symbol][self._buffers[symbol]["timestamp"] >= cutoff]
        
    def get_symbol_df(self, symbol: str) -> pd.DataFrame:
        """Returns the full historical DataFrame for a symbol."""
        return self._buffers.get(symbol, pd.DataFrame())

    def get_all_symbols(self) -> list:
        """Returns list of all active symbols in buffer."""
        return list(self._buffers.keys())

    def get_latest_tick(self, symbol: str) -> dict:
        """Returns the most recent tick for a symbol."""
        df = self.get_symbol_df(symbol)
        if df.empty:
            return {}
        return df.iloc[-1].to_dict()
