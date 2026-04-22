"""
meta_learner.py — Fase 7
XGBoost-based meta-learner that estimates per-agent reliability by context.

predict_weight(agent, ticker, signal, confidence, context) -> float in [0.1, 2.0]
  1.0 = neutral (EWA unchanged)
  >1.0 = agent historically reliable in this context → amplify
  <1.0 = unreliable → dampen

Guardrail: if the (agent_id, regime) combo has < MIN_SAMPLES_GUARDRAIL training rows,
return 1.0 instead of potentially overfit predictions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.feature_extractor import CATEGORICAL_COLS, extract_features

logger = logging.getLogger(__name__)

WEIGHT_MIN = 0.1
WEIGHT_MAX = 2.0
MIN_SAMPLES_GUARDRAIL = 500  # (agent, regime) combos with fewer rows → return 1.0

_CURRENT_PATH = Path("models/meta_learner_current.joblib")
_cached_learner: Optional["MetaLearner"] = None


class MetaLearner:
    """Wrapper around XGBRegressor with context-aware weight prediction."""

    def __init__(self) -> None:
        self._model = None
        self._feature_columns: list[str] = []
        self._pred_p75: float = 1.0          # 75th pct of |train predictions| for tanh scaling
        self._train_context_counts: dict[tuple[str, str], int] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train without validation (for final refit on all data)."""
        from xgboost import XGBRegressor

        self._build_context_counts(X_train)
        X_enc = self._encode(X_train, fit=True)

        self._model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        self._model.fit(X_enc, y_train)
        self._compute_p75(X_enc)

    def fit_with_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict | None = None,
    ) -> None:
        """Train with early stopping on validation set."""
        from xgboost import XGBRegressor

        self._build_context_counts(X_train)
        X_enc_tr  = self._encode(X_train, fit=True)
        X_enc_val = self._encode(X_val,   fit=False)

        kw = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8}
        if params:
            kw.update(params)

        self._model = XGBRegressor(
            **kw, random_state=42, n_jobs=-1,
            early_stopping_rounds=30, eval_metric="rmse",
        )
        self._model.fit(
            X_enc_tr, y_train,
            eval_set=[(X_enc_val, y_val)],
            verbose=False,
        )
        self._compute_p75(X_enc_tr)
        logger.info(
            "MetaLearner fitted (early_stopping), best_n=%d",
            self._model.best_iteration,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_weight(
        self,
        agent_id: str,
        ticker: str,
        signal: str,
        confidence: float,
        context: dict,
        horizon: str = "5d",
    ) -> float:
        """
        Returns multiplicative weight in [0.1, 2.0].
        Falls back to 1.0 on: missing model, sparse context, any error.
        """
        if self._model is None:
            return 1.0

        regime   = str(context.get("regime") or "UNKNOWN")
        n_train  = self._train_context_counts.get((agent_id, regime), 0)
        if n_train < MIN_SAMPLES_GUARDRAIL:
            return 1.0

        try:
            feat  = extract_features(agent_id, ticker, signal, confidence, horizon, context)
            row   = pd.DataFrame([feat])
            X_enc = self._encode(row, fit=False)
            raw   = float(self._model.predict(X_enc)[0])
            # tanh transform: maps raw prediction into (-1, +1), then shift to weight
            weight = 1.0 + float(np.tanh(raw / max(self._pred_p75, 1e-6)))
            return float(np.clip(weight, WEIGHT_MIN, WEIGHT_MAX))
        except Exception as exc:
            logger.debug("predict_weight error (%s) — returning 1.0", exc)
            return 1.0

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Raw continuous predictions for a batch (used by evaluator/train script)."""
        if self._model is None:
            return np.ones(len(X))
        X_enc = self._encode(X, fit=False)
        return self._model.predict(X_enc)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model":                self._model,
                "feature_columns":      self._feature_columns,
                "pred_p75":             self._pred_p75,
                "train_context_counts": self._train_context_counts,
            },
            path,
        )
        logger.info("MetaLearner saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MetaLearner":
        import joblib
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        payload = joblib.load(path)
        ml = cls()
        ml._model               = payload["model"]
        ml._feature_columns     = payload["feature_columns"]
        ml._pred_p75            = payload.get("pred_p75", 1.0)
        ml._train_context_counts = payload.get("train_context_counts", {})
        logger.info("MetaLearner loaded ← %s", path)
        return ml

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """One-hot encode categoricals, align to stored feature_columns."""
        cats  = [c for c in CATEGORICAL_COLS if c in X.columns]
        X_enc = pd.get_dummies(X, columns=cats, dtype=float)

        if fit:
            self._feature_columns = list(X_enc.columns)
            return X_enc

        # Align: add missing columns as 0, drop unknown columns
        for col in self._feature_columns:
            if col not in X_enc.columns:
                X_enc[col] = 0.0
        return X_enc[[c for c in self._feature_columns if c in X_enc.columns]]

    def _build_context_counts(self, X: pd.DataFrame) -> None:
        if "agent_id" in X.columns and "regime" in X.columns:
            counts = X.groupby(["agent_id", "regime"]).size().to_dict()
            self._train_context_counts = {k: int(v) for k, v in counts.items()}

    def _compute_p75(self, X_enc: pd.DataFrame) -> None:
        raw = self._model.predict(X_enc)
        self._pred_p75 = float(np.percentile(np.abs(raw), 75)) or 0.01


# ── Singleton loader ──────────────────────────────────────────────────────────

def load_current_learner() -> Optional[MetaLearner]:
    """
    Load production model from models/meta_learner_current.joblib.
    Caches in memory for the process lifetime. Returns None on any failure.
    """
    global _cached_learner
    if _cached_learner is not None:
        return _cached_learner
    if not _CURRENT_PATH.exists():
        logger.debug("No meta_learner_current.joblib — EWA fallback active.")
        return None
    try:
        _cached_learner = MetaLearner.load(_CURRENT_PATH)
        return _cached_learner
    except Exception as exc:
        logger.warning("MetaLearner load failed (%s) — EWA fallback active.", exc)
        return None
