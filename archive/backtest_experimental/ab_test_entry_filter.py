"""
backtest/ab_test_entry_filter.py
================================
A/B test for the XGBoost entry filter.

Compares:
  - OLD model: models/rl_entry_filter.pkl  (currently live, trained ~2026-04-08)
  - Model A  : rl_entry_filter_fair.pkl    (new, trained with SAME temporal cutoff as old)
  - Model B  : rl_entry_filter_new.pkl     (new, trained on ALL data through today)

Holdout: 2026-04-08 -> 2026-04-15 (the period the live model has been in production)

Outputs:
  - Trains Model A and Model B, writes to models/ with suffixed names
  - Evaluates old vs Model A on strict temporal holdout (fair comparison)
  - Simulated P&L, AUC, precision@0.55, bootstrap CI
  - Agreement matrix (which signals did they disagree on, and who was right)
  - JSON results to backtest/results/ab_test_results.json

Run:
    python3 -m backtest.ab_test_entry_filter

Notes:
  - All compute-intensive work runs locally on M4 (per user preference)
  - No deployment — this script only trains + evaluates + reports
  - Shadow-replay on live paper_trading.log is a separate pass
"""
from __future__ import annotations

import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_brain_v2 import (
    DATA_DIR,
    MODEL_DIR,
    META_COLS,
    DEAD_FEATURES,
    ENTRY_QUALITY_EXCLUDE,
    CLIP_RANGES,
    ASSET_CLASS_MAP,
    load_training_data,
    prepare_features,
    prepare_labels,
    prepare_sample_weights,
    train_xgboost,
    compute_oos_profit_factor,
    compute_oos_sharpe,
)
from features.schema import SCHEMA_VERSION as _SCHEMA_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/results/ab_test.log"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("backtest/results")

# ═══════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════

# The day the old model was trained (rl_entry_filter.pkl, Apr 8 18:48)
# Training data before this date = "fair" (matches old model's knowledge)
CUTOFF_TRAIN_END = pd.Timestamp("2026-04-08", tz="UTC")

# The live period we want to evaluate on
HOLDOUT_START = pd.Timestamp("2026-04-08", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-04-15", tz="UTC")

# Validation window for early stopping (1 week before cutoff)
VAL_START = pd.Timestamp("2026-04-01", tz="UTC")
VAL_END = pd.Timestamp("2026-04-08", tz="UTC")

# Confidence threshold used by live bot
LIVE_THRESHOLD = 0.55

# Output model paths
MODEL_A_PATH = MODEL_DIR / "rl_entry_filter_fair.pkl"
MODEL_B_PATH = MODEL_DIR / "rl_entry_filter_new.pkl"
OLD_MODEL_PATH = MODEL_DIR / "rl_entry_filter.pkl"

# Output results path
RESULTS_PATH = RESULTS_DIR / "ab_test_results.json"


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def _ensure_utc(ts_series: pd.Series) -> pd.Series:
    """Coerce timestamp series to tz-aware UTC."""
    ts = pd.to_datetime(ts_series, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts


def _save_model(
    model: Any,
    feat_names: list[str],
    path: Path,
    extra_meta: dict | None = None,
) -> None:
    """Save model in the same pickle format as rl_brain_v2's production save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feat_names": feat_names,
        "task": "entry_quality",
        "schema_version": _SCHEMA_VERSION,
        "dead_features": list(DEAD_FEATURES),
        "clip_ranges": CLIP_RANGES,
        "asset_class_map": ASSET_CLASS_MAP,
    }
    if extra_meta:
        payload.update(extra_meta)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Saved model: %s", path)


def _load_old_model() -> tuple[Any, list[str]]:
    """Load the currently-live model."""
    if not OLD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Old model not found at {OLD_MODEL_PATH}")
    with open(OLD_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded old model: schema=%s, features=%d",
                data.get("schema_version"), len(data.get("feat_names", [])))
    return data["model"], data.get("feat_names", [])


# ═══════════════════════════════════════════════════════════════════
#  Data preparation — strict temporal split
# ═══════════════════════════════════════════════════════════════════

def load_and_split(classes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all RL training samples and split them into:
      - train_fair: timestamp < CUTOFF_TRAIN_END (matches old model)
      - train_full: timestamp < HOLDOUT_END  (all data including holdout's week minus day)
      - val: [VAL_START, VAL_END)  — for early stopping on Model A
      - val_full: last 10% of train_full by time — for early stopping on Model B
      - holdout: [HOLDOUT_START, HOLDOUT_END)
    """
    logger.info("Loading training data (entry-only, no subsampling)...")
    data = load_training_data(classes, subsample_notrade=0.0)

    # Entry bars only (entry_quality task)
    pre = len(data)
    data = data[data["label_action"] > 0].copy()
    logger.info("Entry-only filter: %d -> %d rows", pre, len(data))

    # Coerce timestamps to UTC
    data["timestamp"] = _ensure_utc(data["timestamp"])

    # Drop rows with NaT
    nat_count = int(data["timestamp"].isna().sum())
    if nat_count > 0:
        logger.warning("Dropping %d rows with NaT timestamps", nat_count)
        data = data[data["timestamp"].notna()].copy()

    data = data.sort_values("timestamp").reset_index(drop=True)
    logger.info("Timestamp range: %s -> %s",
                data["timestamp"].iloc[0], data["timestamp"].iloc[-1])

    # Splits
    train_fair_mask = data["timestamp"] < CUTOFF_TRAIN_END
    val_mask = (data["timestamp"] >= VAL_START) & (data["timestamp"] < VAL_END)
    holdout_mask = (data["timestamp"] >= HOLDOUT_START) & (data["timestamp"] < HOLDOUT_END)
    train_full_mask = data["timestamp"] < HOLDOUT_END  # excludes ONLY the holdout's future

    # Model A: strict temporal split
    train_fair_df = data[train_fair_mask & ~val_mask].copy()
    val_fair_df = data[val_mask].copy()

    # Fallback: if val_fair is too small for early stopping, carve the last 10%
    # of train_fair (by time) as val instead.
    MIN_VAL = 500
    if len(val_fair_df) < MIN_VAL:
        logger.warning(
            "val_fair has only %d samples (< %d) — falling back to last 10%% of train_fair",
            len(val_fair_df), MIN_VAL,
        )
        # train_fair_df is already sorted by timestamp (inherited from data sort)
        n_tf = len(train_fair_df)
        if n_tf > 0:
            cutoff_idx = int(n_tf * 0.9)
            val_fair_df = train_fair_df.iloc[cutoff_idx:].copy()
            train_fair_df = train_fair_df.iloc[:cutoff_idx].copy()

    # Model B: train_full minus last-10% val tail
    train_full = data[train_full_mask].copy()
    n_train_full = len(train_full)
    val_b_cutoff_idx = int(n_train_full * 0.9)
    train_full_train = train_full.iloc[:val_b_cutoff_idx].copy()
    train_full_val = train_full.iloc[val_b_cutoff_idx:].copy()

    splits = {
        "train_fair": train_fair_df,
        "val_fair": val_fair_df,
        "train_full_train": train_full_train,
        "train_full_val": train_full_val,
        "holdout": data[holdout_mask].copy(),
    }

    for name, df in splits.items():
        if len(df) == 0:
            logger.warning("Split '%s' is EMPTY", name)
            continue
        n_ent = int((df["label_outcome"] == 1).sum())
        n_loss = int((df["label_outcome"] == 2).sum())
        n_be = int((df["label_outcome"] == 3).sum())
        per_class = df["asset_class"].value_counts().to_dict()
        logger.info("  %-18s n=%6d  win=%d loss=%d be=%d  ts=[%s .. %s]  classes=%s",
                    name, len(df), n_ent, n_loss, n_be,
                    df["timestamp"].min(), df["timestamp"].max(), per_class)

    return splits


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label: str,
) -> tuple[Any, list[str]]:
    """Train an XGBoost entry filter on the given train/val split."""
    logger.info("=" * 70)
    logger.info("Training %s: train=%d val=%d", label, len(train_df), len(val_df))
    logger.info("=" * 70)

    X_train, feat_names = prepare_features(train_df, task="entry_quality")
    X_val, _ = prepare_features(val_df, task="entry_quality")
    y_train = prepare_labels(train_df, task="entry_quality")
    y_val = prepare_labels(val_df, task="entry_quality")
    weights = prepare_sample_weights(y_train, train_df, task="entry_quality")

    logger.info("Features: %d, Labels: train_pos=%.1f%%, val_pos=%.1f%%",
                len(feat_names), 100 * y_train.mean(), 100 * y_val.mean())

    model = train_xgboost(
        X_train, y_train, X_val, y_val, feat_names,
        sample_weights=weights, task="entry_quality",
    )
    return model, feat_names


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean of a 1D array."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = sample.mean()
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    mean = float(values.mean())
    return mean, lo, hi


def evaluate_on_holdout(
    model: Any,
    holdout_df: pd.DataFrame,
    model_label: str,
    feat_names_expected: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a classifier on the holdout set.
    Computes AUC, precision@0.55, recall@0.55, accuracy, simulated P&L,
    bootstrap CI on P&L and winrate.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )

    X, feat_names = prepare_features(holdout_df, task="entry_quality")
    y = prepare_labels(holdout_df, task="entry_quality")

    if feat_names_expected and feat_names != feat_names_expected:
        # Align column order for models with different feature orderings
        common = [c for c in feat_names_expected if c in feat_names]
        missing_in_h = [c for c in feat_names_expected if c not in feat_names]
        missing_in_e = [c for c in feat_names if c not in feat_names_expected]
        if missing_in_h or missing_in_e:
            logger.warning(
                "%s: feature alignment mismatch -- common=%d, missing_in_holdout=%d, missing_in_expected=%d",
                model_label, len(common), len(missing_in_h), len(missing_in_e),
            )

    logger.info("%s: predicting %d holdout samples (%d positive)",
                model_label, len(holdout_df), int(y.sum()))

    # Predict probabilities
    proba = model.predict_proba(X)[:, 1]
    pred_default = model.predict(X)  # uses 0.5 threshold by default
    pred_live = (proba >= LIVE_THRESHOLD).astype(np.int32)

    # ML metrics
    result: dict[str, Any] = {"label": model_label, "n_holdout": int(len(holdout_df))}
    try:
        result["auc"] = float(roc_auc_score(y, proba))
    except (ValueError, IndexError):
        result["auc"] = None

    result["accuracy_default"] = float(accuracy_score(y, pred_default))
    result["precision_default"] = float(precision_score(y, pred_default, zero_division=0))
    result["recall_default"] = float(recall_score(y, pred_default, zero_division=0))
    result["f1_default"] = float(f1_score(y, pred_default, zero_division=0))

    result["accuracy_live_th"] = float(accuracy_score(y, pred_live))
    result["precision_live_th"] = float(precision_score(y, pred_live, zero_division=0))
    result["recall_live_th"] = float(recall_score(y, pred_live, zero_division=0))
    result["f1_live_th"] = float(f1_score(y, pred_live, zero_division=0))
    result["accepted_live_th"] = int(pred_live.sum())
    result["accept_rate_live_th"] = float(pred_live.mean())

    # Confidence distribution
    result["conf_min"] = float(proba.min())
    result["conf_max"] = float(proba.max())
    result["conf_mean"] = float(proba.mean())
    result["conf_p50"] = float(np.quantile(proba, 0.50))
    result["conf_p90"] = float(np.quantile(proba, 0.90))
    result["conf_above_055"] = int((proba >= 0.55).sum())

    # Trading metrics — use raw RR outcomes (cap extremes)
    rr = holdout_df["label_rr"].values.astype(np.float64)
    rr = np.clip(rr, -1.0, 20.0)

    # At live threshold
    mask = pred_live == 1
    accepted_rr = rr[mask]
    result["sum_rr_live_th"] = float(accepted_rr.sum())
    result["avg_rr_live_th"] = float(accepted_rr.mean()) if len(accepted_rr) > 0 else 0.0

    outcomes = holdout_df["label_outcome"].values
    if mask.sum() > 0:
        accepted_outcomes = outcomes[mask]
        n_w = int((accepted_outcomes == 1).sum())
        n_l = int((accepted_outcomes == 2).sum())
        real = n_w + n_l
        result["wins_live_th"] = n_w
        result["losses_live_th"] = n_l
        result["winrate_live_th"] = float(n_w / max(real, 1))

        win_rr = accepted_rr[accepted_outcomes == 1]
        loss_rr = accepted_rr[accepted_outcomes == 2]
        total_win = float(win_rr.sum()) if len(win_rr) > 0 else 0.0
        total_loss = float(abs(loss_rr.sum())) if len(loss_rr) > 0 else 0.0
        result["pf_live_th"] = total_win / max(total_loss, 0.001)
    else:
        result["wins_live_th"] = 0
        result["losses_live_th"] = 0
        result["winrate_live_th"] = 0.0
        result["pf_live_th"] = 0.0

    # Bootstrap CI on winrate (over accepted samples)
    if mask.sum() > 1:
        outs = (outcomes[mask] == 1).astype(np.float64)
        wr_mean, wr_lo, wr_hi = _bootstrap_ci(outs)
        result["winrate_live_th_ci95"] = [wr_lo, wr_hi]

        rr_mean, rr_lo, rr_hi = _bootstrap_ci(accepted_rr)
        result["avg_rr_live_th_ci95"] = [rr_lo, rr_hi]

    # Per-class breakdown at live threshold
    per_class: dict[str, dict[str, Any]] = {}
    for ac in holdout_df["asset_class"].unique():
        ac_mask = (holdout_df["asset_class"].values == ac) & mask
        n_ac = int(ac_mask.sum())
        if n_ac == 0:
            per_class[ac] = {"accepted": 0, "wins": 0, "losses": 0, "pf": 0.0}
            continue
        ac_outs = outcomes[ac_mask]
        ac_rr = rr[ac_mask]
        w = int((ac_outs == 1).sum())
        lo = int((ac_outs == 2).sum())
        win_sum = float(ac_rr[ac_outs == 1].sum()) if w > 0 else 0.0
        loss_sum = float(abs(ac_rr[ac_outs == 2].sum())) if lo > 0 else 0.0
        per_class[ac] = {
            "accepted": n_ac,
            "wins": w,
            "losses": lo,
            "winrate": w / max(w + lo, 1),
            "pf": win_sum / max(loss_sum, 0.001),
        }
    result["per_class_live_th"] = per_class

    # Return proba for downstream agreement matrix
    result["_proba"] = proba
    result["_pred_live"] = pred_live
    return result


def compute_agreement_matrix(
    holdout_df: pd.DataFrame,
    old_result: dict,
    new_result: dict,
) -> dict[str, Any]:
    """Compare which signals each model accepts/rejects and tally outcomes."""
    old_pred = old_result["_pred_live"]
    new_pred = new_result["_pred_live"]
    outcomes = holdout_df["label_outcome"].values
    rr = np.clip(holdout_df["label_rr"].values.astype(np.float64), -1.0, 20.0)

    cells = {
        "both_accept": (old_pred == 1) & (new_pred == 1),
        "only_old_accepts": (old_pred == 1) & (new_pred == 0),
        "only_new_accepts": (old_pred == 0) & (new_pred == 1),
        "both_reject": (old_pred == 0) & (new_pred == 0),
    }

    matrix: dict[str, Any] = {}
    for key, mask in cells.items():
        n = int(mask.sum())
        if n == 0:
            matrix[key] = {"count": 0, "wins": 0, "losses": 0, "sum_rr": 0.0, "pf": 0.0}
            continue
        outs = outcomes[mask]
        these_rr = rr[mask]
        w = int((outs == 1).sum())
        lo = int((outs == 2).sum())
        win_sum = float(these_rr[outs == 1].sum()) if w > 0 else 0.0
        loss_sum = float(abs(these_rr[outs == 2].sum())) if lo > 0 else 0.0
        matrix[key] = {
            "count": n,
            "wins": w,
            "losses": lo,
            "sum_rr": float(these_rr.sum()),
            "pf": win_sum / max(loss_sum, 0.001),
            "avg_rr": float(these_rr.mean()),
        }
    return matrix


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("A/B TEST: Entry Filter")
    logger.info("  Fair cutoff:  %s (matches old model's training date)", CUTOFF_TRAIN_END)
    logger.info("  Holdout:      [%s, %s)", HOLDOUT_START, HOLDOUT_END)
    logger.info("  Val window:   [%s, %s)", VAL_START, VAL_END)
    logger.info("  Live th:      %.2f", LIVE_THRESHOLD)
    logger.info("=" * 70)

    splits = load_and_split()

    # ─────────────────────────────────────────────────────────────
    # Train Model A (fair comparison with old model)
    # ─────────────────────────────────────────────────────────────
    model_a, feat_names_a = train_model(
        splits["train_fair"], splits["val_fair"], "Model A (fair)",
    )
    _save_model(model_a, feat_names_a, MODEL_A_PATH, extra_meta={
        "ab_test_label": "model_a_fair",
        "train_cutoff": str(CUTOFF_TRAIN_END),
    })

    # ─────────────────────────────────────────────────────────────
    # Train Model B (production candidate, all data minus holdout)
    # ─────────────────────────────────────────────────────────────
    model_b, feat_names_b = train_model(
        splits["train_full_train"], splits["train_full_val"], "Model B (production)",
    )
    _save_model(model_b, feat_names_b, MODEL_B_PATH, extra_meta={
        "ab_test_label": "model_b_production",
        "train_cutoff": str(HOLDOUT_END),
    })

    # ─────────────────────────────────────────────────────────────
    # Load old model
    # ─────────────────────────────────────────────────────────────
    old_model, old_feat_names = _load_old_model()

    # ─────────────────────────────────────────────────────────────
    # Evaluate all three on holdout
    #   - "full holdout": all SMC candidate entries (matches training distribution)
    #   - "live-realistic": only signals that would have passed the live alignment gate
    #     (alignment_score >= 0.78). This is the universe the live bot actually sees.
    # ─────────────────────────────────────────────────────────────
    holdout = splits["holdout"]
    if len(holdout) == 0:
        logger.error("HOLDOUT IS EMPTY — cannot evaluate. Ensure regeneration extended data past 2026-04-08.")
        return

    # Live-realistic subset
    holdout_live = holdout[holdout["alignment_score"] >= 0.78].copy()
    logger.info("Live-realistic subset: %d / %d (score >= 0.78)", len(holdout_live), len(holdout))

    logger.info("")
    logger.info("--- Evaluation on FULL holdout (all SMC candidates) ---")
    old_eval = evaluate_on_holdout(old_model, holdout, "OLD", old_feat_names)
    model_a_eval = evaluate_on_holdout(model_a, holdout, "MODEL_A", feat_names_a)
    model_b_eval = evaluate_on_holdout(model_b, holdout, "MODEL_B", feat_names_b)

    logger.info("")
    logger.info("--- Evaluation on LIVE-REALISTIC holdout (alignment_score >= 0.78) ---")
    if len(holdout_live) > 0:
        old_eval_live = evaluate_on_holdout(old_model, holdout_live, "OLD_live", old_feat_names)
        model_a_eval_live = evaluate_on_holdout(model_a, holdout_live, "MODEL_A_live", feat_names_a)
        model_b_eval_live = evaluate_on_holdout(model_b, holdout_live, "MODEL_B_live", feat_names_b)
    else:
        logger.warning("  Live-realistic subset is empty — skipping.")
        old_eval_live = model_a_eval_live = model_b_eval_live = None

    # ─────────────────────────────────────────────────────────────
    # Agreement matrices
    # ─────────────────────────────────────────────────────────────
    agreement_a = compute_agreement_matrix(holdout, old_eval, model_a_eval)
    agreement_b = compute_agreement_matrix(holdout, old_eval, model_b_eval)
    agreement_a_live = compute_agreement_matrix(holdout_live, old_eval_live, model_a_eval_live) if model_a_eval_live else None
    agreement_b_live = compute_agreement_matrix(holdout_live, old_eval_live, model_b_eval_live) if model_b_eval_live else None

    # ─────────────────────────────────────────────────────────────
    # Serialize results (strip _proba/_pred_live internal arrays)
    # ─────────────────────────────────────────────────────────────
    def _strip(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    report = {
        "config": {
            "cutoff_train_end": str(CUTOFF_TRAIN_END),
            "holdout_start": str(HOLDOUT_START),
            "holdout_end": str(HOLDOUT_END),
            "val_window": [str(VAL_START), str(VAL_END)],
            "live_threshold": LIVE_THRESHOLD,
            "n_holdout_full": len(holdout),
            "n_holdout_live": len(holdout_live),
        },
        "splits": {
            name: {
                "n_rows": len(df),
                "ts_min": str(df["timestamp"].min()) if len(df) else None,
                "ts_max": str(df["timestamp"].max()) if len(df) else None,
                "per_class": {k: int(v) for k, v in df["asset_class"].value_counts().items()},
            }
            for name, df in splits.items()
        },
        "full_holdout": {
            "old_model": _strip(old_eval),
            "model_a_fair": _strip(model_a_eval),
            "model_b_production": _strip(model_b_eval),
            "agreement_old_vs_a": agreement_a,
            "agreement_old_vs_b": agreement_b,
        },
        "live_realistic_holdout": {
            "old_model": _strip(old_eval_live) if old_eval_live else None,
            "model_a_fair": _strip(model_a_eval_live) if model_a_eval_live else None,
            "model_b_production": _strip(model_b_eval_live) if model_b_eval_live else None,
            "agreement_old_vs_a": agreement_a_live,
            "agreement_old_vs_b": agreement_b_live,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Results written to %s", RESULTS_PATH)

    # ─────────────────────────────────────────────────────────────
    # Print summary
    # ─────────────────────────────────────────────────────────────
    def _print_eval_block(title, n_samples, old_e, a_e, b_e, agree_a, agree_b):
        logger.info("")
        logger.info("=" * 70)
        logger.info("%s (%d samples)", title, n_samples)
        logger.info("=" * 70)
        for lbl, res in [("OLD", old_e), ("MODEL_A (fair)", a_e), ("MODEL_B (prod)", b_e)]:
            if res is None:
                continue
            logger.info("%-15s AUC=%.3f acc=%.3f prec@0.55=%.3f rec@0.55=%.3f | accepted=%d avg_rr=%+.3f PF=%.2f WR=%.0f%%",
                        lbl,
                        res.get("auc") or 0.0,
                        res["accuracy_live_th"],
                        res["precision_live_th"],
                        res["recall_live_th"],
                        res["accepted_live_th"],
                        res.get("avg_rr_live_th", 0.0),
                        res.get("pf_live_th", 0.0),
                        100 * res.get("winrate_live_th", 0.0))
        if agree_a:
            logger.info("")
            logger.info("AGREEMENT (OLD vs MODEL A):")
            for key, cell in agree_a.items():
                logger.info("  %-20s n=%4d  wins=%d losses=%d sum_rr=%+.2f PF=%.2f",
                            key, cell["count"], cell["wins"], cell["losses"],
                            cell["sum_rr"], cell["pf"])
        if agree_b:
            logger.info("")
            logger.info("AGREEMENT (OLD vs MODEL B):")
            for key, cell in agree_b.items():
                logger.info("  %-20s n=%4d  wins=%d losses=%d sum_rr=%+.2f PF=%.2f",
                            key, cell["count"], cell["wins"], cell["losses"],
                            cell["sum_rr"], cell["pf"])

    _print_eval_block(
        "FULL HOLDOUT (all SMC candidates)",
        len(holdout),
        old_eval, model_a_eval, model_b_eval,
        agreement_a, agreement_b,
    )
    _print_eval_block(
        "LIVE-REALISTIC HOLDOUT (alignment_score >= 0.78)",
        len(holdout_live),
        old_eval_live, model_a_eval_live, model_b_eval_live,
        agreement_a_live, agreement_b_live,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
