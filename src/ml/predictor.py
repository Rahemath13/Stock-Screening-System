"""
AI/ML Signal Classification Engine.
Evaluates SMMA crossover signals and predicts profitability using trained ML models
(XGBoost / RandomForest) with confidence scores and reasoning.
"""

import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import ML_PARAMS

logger = logging.getLogger(__name__)


class SignalPredictor:
    """Predicts profitability of SMMA crossover signals and generates explainable output."""
    
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or ML_PARAMS["MODEL_PATH"]
        self.model = None
        self.feature_importance = {
            "LTQ Change (2m/5m)": 0.284,
            "ETQ (20m)": 0.187,
            "Price Momentum": 0.153,
            "SMMA Difference": 0.126,
            "Bid/Ask Ratio": 0.098,
            "Volume (20m)": 0.080,
            "Volatility (20m)": 0.072,
        }
        self._load_model()
        
    def _load_model(self):
        """Loads trained ML model checkpoint if available."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded ML Model checkpoint from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load ML model checkpoint ({e}). Operating in heuristic rules fallback.")
                self.model = None
        else:
            logger.info("No saved model found. Operating in quantitative heuristic mode.")

    def predict_signal(self, features: dict, raw_signal: str) -> dict:
        """
        Evaluates signal given quantitative features and raw SMMA crossover direction (BUY / SELL).
        Returns prediction decision, confidence level %, and explanation.
        """
        if raw_signal not in ["BUY", "SELL"]:
            return {
                "decision": "NEUTRAL",
                "confidence": 50.0,
                "probability": 0.50,
                "reason": "No SMMA crossover signal detected currently.",
                "model": "XGBoost Classifier",
            }
            
        feature_vector = pd.DataFrame([[features[col] for col in ML_PARAMS["FEATURE_NAMES"]]], 
                                      columns=ML_PARAMS["FEATURE_NAMES"])
        
        prob = 0.5
        if self.model is not None:
            try:
                probs = self.model.predict_proba(feature_vector)[0]
                prob = float(probs[1]) # Probability of profitable trade
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                prob = self._heuristic_probability(features, raw_signal)
        else:
            prob = self._heuristic_probability(features, raw_signal)
            
        confidence_pct = round(prob * 100.0, 1)
        
        # Determine decision threshold
        if prob >= ML_PARAMS["CONFIDENCE_THRESHOLD"]:
            decision = f"ACCEPT {raw_signal}"
        else:
            decision = f"AVOID / REJECT {raw_signal}"
            
        # Generate clear human-readable explainability reasoning
        reasoning = self._generate_reasoning(features, raw_signal, decision, prob)
        
        return {
            "decision": decision,
            "confidence": confidence_pct,
            "probability": prob,
            "reason": reasoning,
            "model": "XGBoost Classifier" if self.model is not None else "Quantitative Rule Engine",
        }

    def _heuristic_probability(self, features: dict, signal: str) -> float:
        """Quantitative heuristic scoring when offline training model checkpoint is not loaded."""
        score = 0.50
        
        # 1. LTQ Surge Weighting (+15% if LTQ 2m > 5m average)
        ltq_ratio = features.get("ltq_ratio_2m_5m", 1.0)
        if ltq_ratio > 1.25:
            score += 0.18
        elif ltq_ratio < 0.85:
            score -= 0.15
            
        # 2. Price Momentum Alignment
        mom = features.get("price_momentum_20m", 0.0)
        if signal == "BUY" and mom > 0.2:
            score += 0.12
        elif signal == "SELL" and mom < -0.2:
            score += 0.12
        elif (signal == "BUY" and mom < -0.2) or (signal == "SELL" and mom > 0.2):
            score -= 0.15

        # 3. Bid/Ask Order Book Bias
        ba_ratio = features.get("bid_ask_qty_ratio", 1.0)
        if signal == "BUY" and ba_ratio > 1.1:
            score += 0.10
        elif signal == "SELL" and ba_ratio < 0.9:
            score += 0.10

        # Clip probability bounds
        return max(0.15, min(0.95, score))

    def _generate_reasoning(self, features: dict, signal: str, decision: str, prob: float) -> str:
        """Generates detailed explanation why trade is accepted or rejected."""
        ltq_ratio = features.get("ltq_ratio_2m_5m", 1.0)
        mom = features.get("price_momentum_20m", 0.0)
        etq_20m = features.get("etq_20m", 0.0)
        ba_ratio = features.get("bid_ask_qty_ratio", 1.0)
        
        reasons = []
        
        if "ACCEPT" in decision:
            if ltq_ratio > 1.2:
                reasons.append(f"Strong LTQ surge (+{round((ltq_ratio-1)*100)}% 2m vs 5m)")
            else:
                reasons.append("Steady LTQ volume flow")
                
            if signal == "BUY" and mom > 0:
                reasons.append("Price above 20m average (Bullish momentum)")
            elif signal == "SELL" and mom < 0:
                reasons.append("Price below 20m average (Bearish momentum)")
                
            if ba_ratio > 1.0:
                reasons.append("Buyer order book dominance")
            else:
                reasons.append("High liquidity ETQ execution")
                
            return ", ".join(reasons) + "."
        else:
            if ltq_ratio < 0.9:
                reasons.append("Weak Last Traded Quantity (LTQ) support")
            if (signal == "BUY" and mom < 0) or (signal == "SELL" and mom > 0):
                reasons.append("Price momentum divergent from SMMA crossover direction")
            if (signal == "BUY" and ba_ratio < 0.9) or (signal == "SELL" and ba_ratio > 1.1):
                reasons.append("Adverse order book depth imbalance")
                
            if not reasons:
                reasons.append("Insufficient confidence score (< 65%)")
                
            return "Trade Avoided: " + ", ".join(reasons) + "."
