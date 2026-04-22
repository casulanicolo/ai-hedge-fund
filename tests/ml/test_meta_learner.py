"""
tests/ml/test_meta_learner.py — Fase 7

Verifies:
  - MetaLearner fits on synthetic data without error
  - predict_weight returns float in [0.1, 2.0]
  - predict_weight returns 1.0 when model is None (fallback)
  - predict_weight returns 1.0 for sparse (agent, regime) combos (guardrail)
  - save/load round-trip produces identical predictions
  - load_current_learner returns None when file missing (fallback)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost", reason="xgboost not installed")

from src.ml.meta_learner import (
    WEIGHT_MAX,
    WEIGHT_MIN,
    MetaLearner,
    load_current_learner,
)
from src.ml.feature_extractor import extract_features


# ── Synthetic data ────────────────────────────────────────────────────────────

AGENTS   = ["warren_buffett_agent", "technical_analyst_agent", "sentiment_agent"]
TICKERS  = ["AAPL", "NVDA", "TSLA"]
REGIMES  = ["RISK_ON", "CAUTION", "RISK_OFF"]
SIGNALS  = ["BUY", "SELL", "HOLD"]
HORIZONS = ["1d", "5d", "20d"]
N_ROWS   = 600  # above MIN_SAMPLES_GUARDRAIL=500 for (agent, regime) counts


def _make_synthetic_df(n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        agent   = rng.choice(AGENTS)
        ticker  = rng.choice(TICKERS)
        regime  = rng.choice(REGIMES)
        signal  = rng.choice(SIGNALS)
        horizon = rng.choice(HORIZONS)
        context = {
            "regime":            regime,
            "vix_at_prediction": float(rng.uniform(12, 40)),
            "realized_vol_20d":  float(rng.uniform(0.1, 0.5)),
            "month":             int(rng.integers(1, 13)),
            "day_of_week":       int(rng.integers(0, 5)),
            "ewa_weight":        float(rng.uniform(0.5, 1.5)),
        }
        feat = extract_features(agent, ticker, signal, 0.7, horizon, context)
        feat["y_continuous"] = float(rng.normal(0, 0.02))
        feat["y_binary"]     = int(feat["y_continuous"] > 0)
        rows.append(feat)
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "agent_id", "ticker", "signal", "confidence", "horizon",
    "regime", "vix_at_prediction", "realized_vol_20d",
    "sector", "month", "day_of_week", "ewa_weight",
]


def _get_Xy(df: pd.DataFrame):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols], df["y_continuous"], df["y_binary"]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_runs_without_error(self):
        df = _make_synthetic_df()
        X, y, _ = _get_Xy(df)
        ml = MetaLearner()
        ml.fit(X, y)
        assert ml._model is not None

    def test_predict_batch_shape(self):
        df = _make_synthetic_df()
        X, y, _ = _get_Xy(df)
        ml = MetaLearner()
        ml.fit(X, y)
        preds = ml.predict_batch(X)
        assert preds.shape == (len(X),)

    def test_fit_with_validation(self):
        df   = _make_synthetic_df(800)
        X, y, _ = _get_Xy(df)
        split   = int(len(df) * 0.8)
        ml = MetaLearner()
        ml.fit_with_validation(X.iloc[:split], y.iloc[:split],
                               X.iloc[split:], y.iloc[split:])
        assert ml._model is not None
        assert ml._model.best_iteration > 0


class TestPredictWeight:
    @pytest.fixture(scope="class")
    def trained_ml(self):
        df = _make_synthetic_df(N_ROWS)
        X, y, _ = _get_Xy(df)
        ml = MetaLearner()
        ml.fit(X, y)
        return ml

    def test_weight_in_valid_range(self, trained_ml):
        context = {
            "regime": "RISK_ON", "vix_at_prediction": 15.0,
            "realized_vol_20d": 0.2, "month": 6, "day_of_week": 2, "ewa_weight": 1.0,
        }
        w = trained_ml.predict_weight(
            "warren_buffett_agent", "AAPL", "BUY", 0.8, context,
        )
        assert WEIGHT_MIN <= w <= WEIGHT_MAX, f"Weight {w} out of range [{WEIGHT_MIN}, {WEIGHT_MAX}]"

    def test_weight_is_float(self, trained_ml):
        context = {"regime": "CAUTION"}
        w = trained_ml.predict_weight("sentiment_agent", "TSLA", "SELL", 0.6, context)
        assert isinstance(w, float)

    def test_guardrail_sparse_combo_returns_neutral(self, trained_ml):
        """(agent, regime) combo with 0 training rows → must return 1.0."""
        w = trained_ml.predict_weight(
            "nonexistent_agent_xyz", "AAPL", "BUY", 0.8,
            {"regime": "RISK_OFF"},
        )
        assert w == 1.0, f"Expected 1.0 (guardrail), got {w}"


class TestFallback:
    def test_predict_weight_no_model_returns_1(self):
        ml = MetaLearner()
        assert ml._model is None
        w = ml.predict_weight("warren_buffett_agent", "AAPL", "BUY", 0.8, {})
        assert w == 1.0

    def test_predict_batch_no_model_returns_ones(self):
        df  = _make_synthetic_df(10)
        X, _, _ = _get_Xy(df)
        ml  = MetaLearner()
        out = ml.predict_batch(X)
        assert np.all(out == 1.0)

    def test_load_current_learner_missing_file(self, monkeypatch):
        """load_current_learner must return None when file absent."""
        import src.ml.meta_learner as ml_mod
        ml_mod._cached_learner = None  # reset cache
        monkeypatch.setattr(ml_mod, "_CURRENT_PATH", Path("/nonexistent/path/model.joblib"))
        result = load_current_learner()
        assert result is None


class TestSaveLoad:
    def test_round_trip_identical_predictions(self):
        df = _make_synthetic_df()
        X, y, _ = _get_Xy(df)
        ml = MetaLearner()
        ml.fit(X, y)
        preds_before = ml.predict_batch(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name
        try:
            ml.save(tmp_path)
            ml2     = MetaLearner.load(tmp_path)
            preds_after = ml2.predict_batch(X)
            np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=6)
        finally:
            os.unlink(tmp_path)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            MetaLearner.load("/nonexistent/path/model.joblib")
