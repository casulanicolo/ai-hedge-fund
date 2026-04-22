"""
tests/ml/test_feature_extractor.py — Fase 7

Verifies:
  - extract_features is deterministic (same input → same output)
  - output has all expected keys
  - no future information leakage in feature values
  - categorical values are valid strings, numerics are floats
"""
from __future__ import annotations

import pytest

from src.ml.feature_extractor import CATEGORICAL_COLS, SECTOR_MAP, extract_features

SAMPLE_CONTEXT = {
    "regime":            "RISK_ON",
    "vix_at_prediction": 15.3,
    "realized_vol_20d":  0.22,
    "sector":            "TECHNOLOGY",
    "month":             3,
    "day_of_week":       1,
    "ewa_weight":        1.2,
}

EXPECTED_KEYS = {
    "agent_id", "ticker", "signal", "confidence", "horizon",
    "regime", "vix_at_prediction", "realized_vol_20d",
    "sector", "month", "day_of_week", "ewa_weight",
}


def _make_feat(**overrides):
    ctx = {**SAMPLE_CONTEXT, **overrides.get("context", {})}
    return extract_features(
        agent_id=overrides.get("agent_id", "warren_buffett_agent"),
        ticker=overrides.get("ticker", "AAPL"),
        signal=overrides.get("signal", "BUY"),
        confidence=overrides.get("confidence", 0.75),
        horizon=overrides.get("horizon", "5d"),
        context=ctx,
    )


class TestDeterminism:
    def test_same_input_same_output(self):
        f1 = _make_feat()
        f2 = _make_feat()
        assert f1 == f2, "extract_features must be deterministic"

    def test_different_signal_different_output(self):
        buy  = _make_feat(signal="BUY")
        sell = _make_feat(signal="SELL")
        assert buy["signal"] != sell["signal"]


class TestStructure:
    def test_all_expected_keys_present(self):
        feat = _make_feat()
        assert EXPECTED_KEYS.issubset(feat.keys()), (
            f"Missing keys: {EXPECTED_KEYS - feat.keys()}"
        )

    def test_numeric_types(self):
        feat = _make_feat()
        for key in ("confidence", "vix_at_prediction", "realized_vol_20d", "ewa_weight"):
            assert isinstance(feat[key], float), f"{key} should be float, got {type(feat[key])}"

    def test_int_types(self):
        feat = _make_feat()
        for key in ("month", "day_of_week"):
            assert isinstance(feat[key], int), f"{key} should be int, got {type(feat[key])}"

    def test_categorical_types(self):
        feat = _make_feat()
        for key in CATEGORICAL_COLS:
            assert isinstance(feat[key], str), f"{key} should be str, got {type(feat[key])}"


class TestDefaults:
    def test_missing_context_uses_defaults(self):
        feat = extract_features(
            agent_id="warren_buffett_agent",
            ticker="AAPL",
            signal="HOLD",
            confidence=0.5,
            horizon="1d",
            context={},
        )
        assert feat["regime"] == "UNKNOWN"
        assert feat["vix_at_prediction"] == 20.0
        assert feat["ewa_weight"] == 1.0

    def test_sector_fallback_from_map(self):
        feat = extract_features(
            "warren_buffett_agent", "NVDA", "BUY", 0.8, "5d",
            context={},  # no sector in context
        )
        assert feat["sector"] == SECTOR_MAP.get("NVDA", "UNKNOWN")

    def test_unknown_ticker_sector(self):
        feat = extract_features(
            "warren_buffett_agent", "XYZ", "BUY", 0.8, "5d",
            context={},
        )
        assert feat["sector"] == "UNKNOWN"


class TestSignalNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("buy",  "BUY"),
        ("SELL", "SELL"),
        ("Hold", "HOLD"),
    ])
    def test_signal_uppercased(self, raw, expected):
        feat = _make_feat(signal=raw)
        assert feat["signal"] == expected


class TestTemporalIntegrity:
    def test_no_future_return_in_features(self):
        """Features must NOT contain any actual_return field (that's the target)."""
        feat = _make_feat()
        for key in feat:
            assert "actual_return" not in key, (
                f"Feature '{key}' looks like a future return — potential leakage!"
            )

    def test_no_outcome_fields(self):
        feat = _make_feat()
        for key in feat:
            assert "outcome" not in key.lower()
            assert "y_binary" not in key
            assert "y_continuous" not in key
