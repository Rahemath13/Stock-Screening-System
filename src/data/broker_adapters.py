"""
Broker API Adapters for Real-Time Data Ingestion.
Supports Angel One SmartAPI, Fyers API v3, WebSocket streaming, and Parquet/CSV tick logs.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INSTRUMENTS_FILE = BASE_DIR / "data" / "instruments" / "nse_instruments.json"


class BaseBrokerAdapter(ABC):
    """Abstract Base Class for Broker Integrations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish connection."""
        pass
        
    @abstractmethod
    def fetch_quotes(self, symbols: list) -> dict:
        """Fetch current quotes for list of symbols."""
        pass
        
    @abstractmethod
    def start_websocket(self, symbols: list, callback_fn):
        """Start live WebSocket tick stream."""
        pass


class AngelOneAdapter(BaseBrokerAdapter):
    """Adapter for Angel One SmartAPI and SmartWebSocketV2."""
    
    def __init__(self, api_key: str, client_code: str, password: str, totp_key: str):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_key = totp_key
        self.smart_api = None
        self.token_lookup = self._load_token_lookup()
        
    def _load_token_lookup(self) -> dict:
        """Loads symbol to token lookup dictionary from data/instruments/nse_instruments.json."""
        if INSTRUMENTS_FILE.exists():
            try:
                with open(INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("token_lookup", {})
            except Exception as e:
                logger.error(f"Error loading instrument tokens: {e}")
        return {}

    def connect(self) -> bool:
        """Authenticate with Angel One SmartAPI using TOTP."""
        try:
            from SmartApi import SmartConnect  # pyright: ignore[reportMissingImports]
            import pyotp  # pyright: ignore[reportMissingImports]
            
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_key).now() if self.totp_key else ""
            data = self.smart_api.generateSession(self.client_code, self.password, totp)
            if isinstance(data, (bytes, str)):
                try:
                    data = json.loads(data)
                except Exception:
                    data = {}
            
            if isinstance(data, dict) and data.get("status"):
                logger.info("Angel One SmartAPI session established successfully.")
                return True
            else:
                logger.warning(f"Angel One auth failed: {data}")
                return False
        except ImportError:
            logger.info("smartapi-python SDK not installed.")
            return False
        except Exception as e:
            logger.error(f"Angel One Connection Error: {e}")
            return False

    def fetch_quotes(self, symbols: list) -> dict:
        """Fetch snapshot quote data from Angel One."""
        quotes = {}
        if not self.smart_api:
            return quotes
            
        for sym in symbols:
            clean_sym = sym.replace("-EQ", "")
            token = self.token_lookup.get(clean_sym, "")
            if not token:
                continue
                
            try:
                data = self.smart_api.ltpData("NSE", f"{clean_sym}-EQ", token)
                if isinstance(data, (bytes, str)):
                    try:
                        data = json.loads(data)
                    except Exception:
                        data = {}

                if isinstance(data, dict) and data.get("status"):
                    d = data.get("data", {})
                    quotes[clean_sym] = {
                        "timestamp": datetime.now(),
                        "symbol": clean_sym,
                        "token": token,
                        "ltp": float(d.get("ltp", 0)),
                        "ltq": int(d.get("tradeVolume", 0)),
                        "bid_price": float(d.get("bestBidPrice", 0)),
                        "bid_qty": int(d.get("bestBidSize", 0)),
                        "ask_price": float(d.get("bestAskPrice", 0)),
                        "ask_qty": int(d.get("bestAskSize", 0)),
                    }
            except Exception as e:
                logger.error(f"Error fetching quote for {sym}: {e}")
        return quotes

    def start_websocket(self, symbols: list, callback_fn):
        """Starts WebSocket V2 Live Stream in Mode 3 (FULL Quote Mode)."""
        logger.info("Initializing Angel One SmartWebSocketV2 Feed...")
        if not self.smart_api:
            logger.warning("SmartAPI not authenticated. Call connect() first.")
            return

        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # pyright: ignore[reportMissingImports]
            feed_token = self.smart_api.getfeedToken()
            
            sws = SmartWebSocketV2(
                auth_token=getattr(self.smart_api, "jwtToken", "") if self.smart_api else "",
                api_key=self.api_key,
                client_code=self.client_code,
                feed_token=feed_token,
            )

            # Map input symbols to numeric tokens
            tokens = []
            for s in symbols:
                clean_s = s.replace("-EQ", "")
                tok = self.token_lookup.get(clean_s, "")
                if tok:
                    tokens.append(str(tok))

            if not tokens:
                tokens = ["2885"] # Fallback to RELIANCE-EQ

            sws.on_data = callback_fn
            sws.connect()
        except Exception as e:
            logger.error(f"Error starting SmartWebSocketV2: {e}")


class FyersAdapter(BaseBrokerAdapter):
    """Adapter for Fyers API v3."""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.fyers = None
        
    def connect(self) -> bool:
        try:
            from fyers_apiv3 import fyersModel  # pyright: ignore[reportMissingImports]
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                is_async=False,
                token=self.access_token,
                log_path=""
            )
            profile = self.fyers.get_profile()
            if isinstance(profile, (bytes, str)):
                try:
                    profile = json.loads(profile)
                except Exception:
                    profile = {}

            if isinstance(profile, dict) and profile.get("code") == 200:
                logger.info("Fyers API connected successfully.")
                return True
            return False
        except ImportError:
            logger.info("Fyers API v3 SDK not installed.")
            return False
        except Exception as e:
            logger.error(f"Fyers Connection Error: {e}")
            return False

    def fetch_quotes(self, symbols: list) -> dict:
        return {}

    def start_websocket(self, symbols: list, callback_fn):
        pass


class ParquetDataIngestor:
    """Ingests tick data from Parquet/CSV files stored locally."""
    
    @staticmethod
    def load_from_parquet(file_path: str) -> pd.DataFrame:
        """Reads Parquet file and formats standardized column names."""
        df = pd.read_parquet(file_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
