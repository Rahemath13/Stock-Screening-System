"""
Machine Learning Model Training Pipeline.
Trains XGBoost/RandomForest models on quantitative features to predict crossover profitability.
Saves model checkpoint (.pkl) and feature importances (.json).
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import ML_PARAMS
from src.indicators.technical import calculate_smma

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles dataset generation, model fitting, evaluation, and serialization."""
    
    def __init__(self, random_state: int = ML_PARAMS["RANDOM_SEED"]):
        self.random_state = random_state
        self.model = None

    def generate_synthetic_training_dataset(self, n_samples: int = 1500) -> pd.DataFrame:
        """Generates realistic synthetic SMMA crossover training data for pre-training."""
        np.random.seed(self.random_state)
        
        ltq_ratio = np.random.normal(1.1, 0.35, n_samples)
        etq_5m = np.random.uniform(500_000, 5_000_000, n_samples)
        etq_20m = etq_5m * np.random.uniform(3.5, 4.5, n_samples)
        etq_60m = etq_20m * np.random.uniform(2.8, 3.2, n_samples)
        price_momentum = np.random.normal(0.05, 0.45, n_samples)
        smma_spread = np.random.normal(0.1, 0.5, n_samples)
        bid_ask_ratio = np.random.normal(1.05, 0.25, n_samples)
        volatility = np.random.uniform(0.5, 5.0, n_samples)
        volume_20m = np.random.uniform(1_000_000, 20_000_000, n_samples)
        
        df = pd.DataFrame({
            "ltq_ratio_2m_5m": ltq_ratio,
            "etq_5m": etq_5m,
            "etq_20m": etq_20m,
            "etq_60m": etq_60m,
            "price_momentum_20m": price_momentum,
            "smma_spread_pct": smma_spread,
            "bid_ask_qty_ratio": bid_ask_ratio,
            "volatility_20m": volatility,
            "volume_20m": volume_20m,
        })
        
        logit = (
            1.8 * (df["ltq_ratio_2m_5m"] - 1.0) +
            1.2 * (df["price_momentum_20m"]) +
            0.8 * (df["bid_ask_qty_ratio"] - 1.0) +
            0.05 * np.log1p(df["etq_20m"]) -
            0.2 * df["volatility_20m"]
        )
        prob = 1 / (1 + np.exp(-logit))
        df["is_profitable"] = (prob > 0.48).astype(int)
        
        return df

    def build_dataset_from_historical_candles(self) -> pd.DataFrame:

        """Reads historical parquet candle files, computes SMMA crossovers & features, and labels trade P&L."""
        hist_dir = BASE_DIR / "data" / "historical"
        if not hist_dir.exists():
            return self.generate_synthetic_training_dataset()

        parquet_files = list(hist_dir.glob("*.parquet"))
        if not parquet_files:
            return self.generate_synthetic_training_dataset()

        rows = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                if df.empty or len(df) < 130:
                    continue

                if "close" in df.columns and "ltp" not in df.columns:
                    df["ltp"] = df["close"]
                if "volume" in df.columns and "ltq" not in df.columns:
                    df["ltq"] = df["volume"]

                prices = df["ltp"]
                volumes = df["ltq"]
                smma_20 = calculate_smma(prices, 20)
                smma_120 = calculate_smma(prices, 120)

                # Find crossover events
                for i in range(121, len(df) - 10):
                    prev_20, curr_20 = smma_20.iloc[i - 1], smma_20.iloc[i]
                    prev_120, curr_120 = smma_120.iloc[i - 1], smma_120.iloc[i]

                    is_buy_cross = (prev_20 <= prev_120) and (curr_20 > curr_120)
                    is_sell_cross = (prev_20 >= prev_120) and (curr_20 < curr_120)

                    if not (is_buy_cross or is_sell_cross):
                        continue

                    entry_price = float(prices.iloc[i])
                    # Determine trade outcome over next 15 bars
                    future_prices = prices.iloc[i + 1: i + 16]
                    if future_prices.empty:
                        continue

                    if is_buy_cross:
                        pnl = (future_prices.max() - entry_price) / entry_price
                    else:
                        pnl = (entry_price - future_prices.min()) / entry_price

                    is_profitable = 1 if pnl > 0.002 else 0

                    # Extract quantitative features
                    vol_5m = float(volumes.iloc[max(0, i - 5):i + 1].sum())
                    vol_20m = float(volumes.iloc[max(0, i - 20):i + 1].sum())
                    vol_60m = float(volumes.iloc[max(0, i - 60):i + 1].sum())
                    vol_2m_avg = volumes.iloc[max(0, i - 2):i + 1].mean()
                    vol_5m_avg = volumes.iloc[max(0, i - 5):i + 1].mean()
                    ltq_ratio = (vol_2m_avg / vol_5m_avg) if vol_5m_avg > 0 else 1.0

                    price_momentum = (entry_price - prices.iloc[max(0, i - 20)]) / max(1.0, prices.iloc[max(0, i - 20)]) * 100.0
                    smma_spread = (curr_20 - curr_120) / max(1.0, curr_120) * 100.0
                    volatility = prices.iloc[max(0, i - 20):i + 1].std()

                    rows.append({
                        "ltq_ratio_2m_5m": ltq_ratio,
                        "etq_5m": vol_5m,
                        "etq_20m": vol_20m,
                        "etq_60m": vol_60m,
                        "price_momentum_20m": price_momentum,
                        "smma_spread_pct": smma_spread,
                        "bid_ask_qty_ratio": 1.05 if is_buy_cross else 0.95,
                        "volatility_20m": volatility,
                        "volume_20m": vol_20m,
                        "is_profitable": is_profitable,
                    })
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")

        if len(rows) < 50:
            return self.generate_synthetic_training_dataset()

        return pd.DataFrame(rows)

    def train_and_evaluate(self, df: pd.DataFrame | None = None) -> dict:
        """Trains ML classifier model and outputs performance metrics."""
        if df is None:
            df = self.build_dataset_from_historical_candles()
            
        X = df[ML_PARAMS["FEATURE_NAMES"]]
        y = df["is_profitable"]

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        
        # Train model (using GradientBoosting/RandomForest as universal sklearn fallback if XGBoost unavailable)
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state
            )
        except ImportError:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=self.random_state
            )
            
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
        }
        
        # Calculate feature importances
        feature_importances = {}
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            total_imp = float(np.sum(imp)) if np.sum(imp) > 0 else 1.0
            for name, score in zip(ML_PARAMS["FEATURE_NAMES"], imp):
                feature_importances[name] = round(float(score / total_imp) * 100.0, 2)
                
        # Save trained checkpoint and feature importances
        ML_PARAMS["MODEL_PATH"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, ML_PARAMS["MODEL_PATH"])
        
        fi_path = ML_PARAMS.get("FEATURE_IMPORTANCE_PATH", ML_PARAMS["MODEL_PATH"].parent / "feature_importance.json")
        with open(fi_path, "w", encoding="utf-8") as f:
            json.dump(feature_importances, f, indent=2)
            
        logger.info(f"Model saved to {ML_PARAMS['MODEL_PATH']} with accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
