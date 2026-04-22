"""
evaluator.py — Fase 7
Evaluation metrics for meta-learner: AUC, Brier, accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_auc(y_true_binary: np.ndarray, y_pred_continuous: np.ndarray) -> float:
    """ROC-AUC using continuous scores (no threshold needed)."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true_binary, y_pred_continuous))
    except ValueError:
        return 0.5


def compute_brier(y_true_binary: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Brier score. Lower = better. Clips predictions to [0, 1]."""
    p = np.clip(y_pred_proba, 0.0, 1.0)
    return float(np.mean((p - y_true_binary.astype(float)) ** 2))


def compute_accuracy(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
    return float(np.mean(y_true_binary == y_pred_binary))


def evaluate_model(
    model,                       # MetaLearner instance
    X: pd.DataFrame,
    y_continuous: pd.Series,
    y_binary: pd.Series,
) -> dict:
    """
    Full evaluation pass. Returns dict:
      n, auc, brier, accuracy, mean_pred, std_pred
    """
    raw_preds = model.predict_batch(X)
    p75       = max(model._pred_p75, 1e-6)
    proba     = 1.0 / (1.0 + np.exp(-raw_preds / p75))  # sigmoid scaling

    y_b       = y_binary.values.astype(int)
    y_pred_b  = (proba > 0.5).astype(int)

    return {
        "n":         int(len(y_b)),
        "auc":       compute_auc(y_b, raw_preds),
        "brier":     compute_brier(y_b, proba),
        "accuracy":  compute_accuracy(y_b, y_pred_b),
        "mean_pred": float(np.mean(raw_preds)),
        "std_pred":  float(np.std(raw_preds)),
    }
