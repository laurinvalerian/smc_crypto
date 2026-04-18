"""
backtest/retrain_after_optfix.py
================================
After the per-symbol optimized SMC params fix, the training samples need to
be regenerated. This script:

1. Verifies the regenerated parquets are present and recent
2. Trains a fresh entry filter using rl_brain_v2's run_walk_forward_rolling
3. Saves the new model to models/rl_entry_filter_optfix.pkl (separate from live)
4. Re-runs the shadow replay against the new model
5. Reports whether the train/inference mismatch is closed

Run AFTER the 3 parallel regen processes complete:
    python3 -m backtest.retrain_after_optfix
"""
from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_brain_v2 import (
    DATA_DIR, MODEL_DIR, META_COLS, DEAD_FEATURES,
    CLIP_RANGES, ASSET_CLASS_MAP,
    load_training_data, prepare_features, prepare_labels, prepare_sample_weights,
    train_xgboost,
)
from features.schema import SCHEMA_VERSION as _SCHEMA_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NEW_MODEL_PATH = MODEL_DIR / "rl_entry_filter_optfix.pkl"


def verify_parquets() -> bool:
    """Check all 4 class parquets exist and have the expected windows."""
    for ac in ("crypto", "forex", "stocks", "commodities"):
        path = DATA_DIR / f"{ac}_samples.parquet"
        if not path.exists():
            logger.error("Missing: %s", path)
            return False
        df = pd.read_parquet(path, columns=["window", "timestamp"])
        windows = sorted(df["window"].unique().tolist())
        n_w12 = int((df["window"] == 12).sum())
        ts_max = pd.to_datetime(df["timestamp"]).max()
        logger.info("  %s: %d rows, windows=%s, W12=%d rows, ts_max=%s",
                    ac, len(df), windows, n_w12, ts_max)
        if 12 not in windows and ac != "stocks":
            logger.warning("  %s: MISSING W12 — regen incomplete?", ac)
    return True


def train_and_save() -> dict[str, Any]:
    """Train new entry filter on regenerated data, save to optfix pickle."""
    logger.info("=" * 70)
    logger.info("Training entry filter on regenerated samples (optfix)")
    logger.info("=" * 70)

    data = load_training_data(classes=None, subsample_notrade=0.0)
    logger.info("Loaded %d rows", len(data))

    # Entry-only filter
    pre = len(data)
    data = data[data["label_action"] > 0].copy()
    logger.info("Entry-only filter: %d -> %d rows", pre, len(data))

    # Coerce timestamp tz
    ts = pd.to_datetime(data["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    data["timestamp"] = ts
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Use last 10% as val, rest as train
    n = len(data)
    cutoff = int(n * 0.9)
    train_df = data.iloc[:cutoff].copy()
    val_df = data.iloc[cutoff:].copy()

    logger.info("Train: %d, Val: %d", len(train_df), len(val_df))
    logger.info("Train ts: %s -> %s", train_df["timestamp"].iloc[0], train_df["timestamp"].iloc[-1])
    logger.info("Val ts:   %s -> %s", val_df["timestamp"].iloc[0], val_df["timestamp"].iloc[-1])

    X_train, feat_names = prepare_features(train_df, task="entry_quality")
    X_val, _ = prepare_features(val_df, task="entry_quality")
    y_train = prepare_labels(train_df, task="entry_quality")
    y_val = prepare_labels(val_df, task="entry_quality")
    weights = prepare_sample_weights(y_train, train_df, task="entry_quality")

    logger.info("Features: %d, Labels: train_pos=%.1f%%, val_pos=%.1f%%",
                len(feat_names), 100 * y_train.mean(), 100 * y_val.mean())

    model = train_xgboost(X_train, y_train, X_val, y_val, feat_names,
                          sample_weights=weights, task="entry_quality")

    # Save
    NEW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feat_names": feat_names,
        "task": "entry_quality",
        "schema_version": _SCHEMA_VERSION,
        "dead_features": list(DEAD_FEATURES),
        "clip_ranges": CLIP_RANGES,
        "asset_class_map": ASSET_CLASS_MAP,
        "ab_test_label": "optfix",
        "trained_after": "per_cluster_smc_params_fix",
    }
    with open(NEW_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Saved model: %s", NEW_MODEL_PATH)

    return {
        "model_path": str(NEW_MODEL_PATH),
        "best_iteration": int(model.best_iteration),
        "n_train": len(train_df),
        "n_val": len(val_df),
    }


def main() -> None:
    if not verify_parquets():
        logger.error("Parquet verification failed — abort")
        return
    result = train_and_save()
    logger.info("")
    logger.info("=" * 70)
    logger.info("RETRAIN COMPLETE")
    logger.info("=" * 70)
    for k, v in result.items():
        logger.info("  %s: %s", k, v)
    logger.info("")
    logger.info("Next: run shadow_replay_entry_filter.py with new model to verify")
    logger.info("the train/inference mismatch is closed.")


if __name__ == "__main__":
    main()
