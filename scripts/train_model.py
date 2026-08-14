"""
CLI Script to Train or Retrain AI/ML Signal Classification Model.
Can be executed standalone: python scripts/train_model.py
"""

import sys
import warnings
from pathlib import Path

# Suppress serialization warnings
warnings.filterwarnings("ignore")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.trainer import ModelTrainer


def main():
    print("=" * 60)
    print("AI/ML Stock Screening System - Model Training Pipeline")
    print("=" * 60)
    
    try:
        trainer = ModelTrainer()
        print("Training model on quantitative SMMA & LTQ features...")
        metrics = trainer.train_and_evaluate()
        
        print("\n--- Model Evaluation Results ---")
        print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision : {metrics['precision'] * 100:.2f}%")
        print(f"Recall    : {metrics['recall'] * 100:.2f}%")
        print(f"AUC-ROC   : {metrics['auc_roc']:.4f}")
        print("=" * 60)
        print("Model checkpoint saved to: models/smma_crossover_model.pkl")
        print("Feature importances saved to: models/feature_importance.json")
    except Exception as e:
        print(f"[ERROR] Exception during model training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
