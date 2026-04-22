"""
train_meta_learner.py — Fase 7 nightly training script.

Steps:
  1. Build dataset from DB (predictions + outcomes)
  2. Time-based split: train ≤ 2024-06-30, val 2024-07→2024-12, test 2025-01→today
  3. Grid search over 5 XGBoost configs (early stopping on val)
  4. SHAP top-20 feature report
  5. Register in ml_model_registry
  6. Promote to production if AUC_val > current + 0.01 AND AUC_val ≥ 0.55

Run:
    python -m src.ml.train_meta_learner
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.init_db import get_connection
from src.ml.dataset_builder import build_dataset
from src.ml.evaluator import compute_auc, compute_brier, evaluate_model
from src.ml.meta_learner import MetaLearner

logger = logging.getLogger(__name__)

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports/ml")
MIN_ROWS    = 500

TRAIN_CUTOFF = "2024-06-30"
VAL_CUTOFF   = "2024-12-31"

FEATURE_COLS = [
    "agent_id", "ticker", "signal", "confidence", "horizon",
    "regime", "vix_at_prediction", "realized_vol_20d",
    "sector", "month", "day_of_week", "ewa_weight",
]

GRID = [
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.10},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["prediction_date"] <= TRAIN_CUTOFF]
    val   = df[(df["prediction_date"] > TRAIN_CUTOFF) & (df["prediction_date"] <= VAL_CUTOFF)]
    test  = df[df["prediction_date"] > VAL_CUTOFF]
    return train, val, test


def _get_current_auc() -> float:
    """AUC of currently promoted model from ml_model_registry. 0.0 if none."""
    try:
        conn = get_connection()
        row  = conn.execute(
            "SELECT auc_val FROM ml_model_registry WHERE promoted=1 "
            "ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return float(row["auc_val"]) if row and row["auc_val"] else 0.0
    except Exception:
        return 0.0


def _register_model(
    model_path: Path,
    n_rows: int,
    auc_train: float,
    auc_val: float,
    auc_test: float,
    brier_test: float,
    promoted: bool,
    notes: str = "",
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO ml_model_registry
            (model_path, trained_at, dataset_rows,
             auc_train, auc_val, auc_test, brier_test, promoted, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(model_path),
            datetime.now(timezone.utc).isoformat(),
            n_rows,
            round(auc_train, 4), round(auc_val, 4),
            round(auc_test, 4),  round(brier_test, 4),
            int(promoted),
            notes,
        ),
    )
    conn.commit()
    conn.close()


def _shap_report(model: MetaLearner, X: pd.DataFrame) -> str:
    """SHAP top-20 mean absolute feature importance."""
    try:
        import shap
        sample  = X.head(min(500, len(X)))
        X_enc   = model._encode(sample, fit=False)
        exp     = shap.TreeExplainer(model._model)
        sv      = exp.shap_values(X_enc)
        means   = np.abs(sv).mean(axis=0)
        top     = sorted(zip(X_enc.columns, means), key=lambda x: -x[1])[:20]
        lines   = [f"  {name:55s} {val:.4f}" for name, val in top]
        return "\n".join(lines)
    except Exception as exc:
        return f"SHAP report failed: {exc}"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("=== Meta-Learner Nightly Training (Fase 7) ===")

    df = build_dataset()
    if df.empty or len(df) < MIN_ROWS:
        logger.warning(
            "Insufficient data (%d rows, minimum %d). "
            "Run more pipeline iterations to accumulate predictions+outcomes.",
            len(df), MIN_ROWS,
        )
        return

    train, val, test = _split(df)
    logger.info(
        "Split — train: %d (%s→%s), val: %d, test: %d",
        len(train), df["prediction_date"].min(), TRAIN_CUTOFF,
        len(val), len(test),
    )

    if len(train) < 100 or len(val) < 50:
        logger.warning("Train (%d) or val (%d) too small — aborting.", len(train), len(val))
        return

    X_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_tr, y_tr = train[X_cols], train["y_continuous"]
    X_va, y_va = val[X_cols],   val["y_continuous"]

    has_test = len(test) >= 20
    X_te = test[X_cols] if has_test else pd.DataFrame()

    # ── Grid search ───────────────────────────────────────────────────────────
    best_model:   MetaLearner | None = None
    best_auc_val: float = 0.0
    best_cfg:     dict  = {}

    for cfg in GRID:
        try:
            ml = MetaLearner()
            ml.fit_with_validation(X_tr, y_tr, X_va, y_va, params=cfg)
            val_preds = ml.predict_batch(X_va)
            auc_v     = compute_auc(val["y_binary"].values, val_preds)
            logger.info("  cfg=%-55s auc_val=%.4f", str(cfg), auc_v)
            if auc_v > best_auc_val:
                best_auc_val = auc_v
                best_model   = ml
                best_cfg     = cfg
        except Exception as exc:
            logger.warning("Grid cfg %s failed: %s", cfg, exc)

    if best_model is None:
        logger.error("All grid configs failed — aborting training.")
        return

    logger.info("Best config: %s → auc_val=%.4f", best_cfg, best_auc_val)

    # ── Metrics ───────────────────────────────────────────────────────────────
    tr_preds  = best_model.predict_batch(X_tr)
    auc_train = compute_auc(train["y_binary"].values, tr_preds)

    auc_test   = 0.0
    brier_test = 0.25  # random baseline
    if has_test:
        metrics   = evaluate_model(best_model, X_te, test["y_continuous"], test["y_binary"])
        auc_test  = metrics["auc"]
        brier_test = metrics["brier"]
        logger.info("Test — auc=%.4f, brier=%.4f, n=%d", auc_test, brier_test, metrics["n"])

    # ── SHAP report ───────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    shap_text   = _shap_report(best_model, X_te if has_test else X_va)
    report_path = REPORTS_DIR / f"shap_{ts}.txt"
    report_path.write_text(
        f"SHAP Top-20 Features — {ts}\n"
        f"AUC_val={best_auc_val:.4f}  AUC_test={auc_test:.4f}  Brier_test={brier_test:.4f}\n"
        f"Config: {best_cfg}\n\n{shap_text}\n"
    )
    logger.info("SHAP report → %s", report_path)

    # ── Save model ────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"meta_learner_{ts}.joblib"
    best_model.save(model_path)

    # ── Promotion decision ────────────────────────────────────────────────────
    current_auc = _get_current_auc()
    promote = (
        best_auc_val >= current_auc + 0.01
        and best_auc_val >= 0.55
        and brier_test < 0.30
    )

    current_link = MODELS_DIR / "meta_learner_current.joblib"
    if promote:
        shutil.copy2(model_path, current_link)
        # Invalidate in-memory singleton
        import src.ml.meta_learner as _ml_mod
        _ml_mod._cached_learner = None
        logger.info(
            "PROMOTED: %s → meta_learner_current.joblib "
            "(auc_val=%.4f > current=%.4f)",
            model_path.name, best_auc_val, current_auc,
        )
    else:
        reason = (
            f"auc_val={best_auc_val:.4f} vs current={current_auc:.4f}"
            + (" [below 0.55]" if best_auc_val < 0.55 else "")
            + (f" [brier={brier_test:.3f}≥0.30]" if brier_test >= 0.30 else "")
        )
        logger.info("NOT promoted — %s", reason)

    _register_model(
        model_path=model_path,
        n_rows=len(df),
        auc_train=auc_train,
        auc_val=best_auc_val,
        auc_test=auc_test,
        brier_test=brier_test,
        promoted=promote,
        notes=f"best_cfg={best_cfg}",
    )
    logger.info("=== Training complete ===")


if __name__ == "__main__":
    run_training()
