"""
Train the 4 Student heads end-to-end from Teacher v2 labels.

Heads
-----
    entry  — XGBClassifier → P(trade was worth taking | features)
    sl     — XGBRegressor  → optimal_sl_rr
    tp     — XGBRegressor  → optimal_tp_rr
    size   — XGBRegressor  → optimal_size (Sharpe-like reward/risk)

Data
----
    ``data/rl_training/{crypto,forex,stocks,commodities}_samples.parquet``,
    backfilled with Teacher v2 labels via ``teacher.backfill_parquet``.
    Each parquet must contain the 41 ENTRY_QUALITY_FEATURES plus the 4
    ``optimal_*`` target columns.

Walk-forward
------------
    12 windows in chronological order. For each head, for each fold:
    Train on W[0..k+1], validate on W[k+2], test on W[k+3]. The final
    production model is trained on W[0..10] with W[11] held out for
    validation/early stopping.

Anti-overfitting
----------------
    * early_stopping_rounds = 30
    * L2 reg (reg_lambda) scaled to data size
    * max_depth capped at 5 (tighter than the old 6 — audit showed
      auc_gap = 0.118 overfit on entry model with depth 6)
    * min_child_weight raised to 20 (was 10)
    * subsample + colsample 0.8

Usage
-----
    python3 train_student.py --classes crypto forex commodities
    python3 train_student.py --heads entry sl tp size
    python3 train_student.py --skip-eval   # skip walk-forward, only prod
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, roc_auc_score

# Reuse rl_brain_v2's feature pipeline so train/inference feature ordering
# matches the canonical pipeline. prepare_features handles clipping +
# asset_class_id/style_id injection consistently. The feature/class
# constants actually live in rl_brain_v2 (CLIP_RANGES/ASSET_CLASS_MAP/
# DEAD_FEATURES/ALL_CLASSES) and features.schema (ENTRY_QUALITY_FEATURES,
# SCHEMA_VERSION).
from features.schema import ENTRY_QUALITY_FEATURES, SCHEMA_VERSION
from rl_brain_v2 import (
    ALL_CLASSES,
    ASSET_CLASS_MAP,
    CLIP_RANGES,
    DEAD_FEATURES,
    prepare_features,
)

logger = logging.getLogger("train_student")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATA_DIR = Path("data/rl_training")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("backtest/results/rl")

# Tuned vs old rl_brain_v2 defaults (see docstring for rationale)
_XGB_BASE = dict(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    gamma=0.1,
    reg_alpha=0.3,
    reg_lambda=2.0,
    n_jobs=4,
    random_state=42,
    tree_method="hist",
    early_stopping_rounds=30,
)

HEAD_CONFIG: dict[str, dict[str, Any]] = {
    "entry": {
        "target": "optimal_entry",
        "estimator": xgb.XGBClassifier,
        "eval_metric": "logloss",
        "extra_params": {},
    },
    "sl": {
        "target": "optimal_sl_rr",
        "estimator": xgb.XGBRegressor,
        "eval_metric": "rmse",
        "extra_params": {},
    },
    "tp": {
        "target": "optimal_tp_rr",
        "estimator": xgb.XGBRegressor,
        "eval_metric": "rmse",
        "extra_params": {},
    },
    "size": {
        "target": "optimal_size",
        "estimator": xgb.XGBRegressor,
        "eval_metric": "rmse",
        "extra_params": {},
    },
}


def load_training_data(classes: list[str] | None,
                        subsample_per_class: int = 500_000) -> pd.DataFrame:
    """Load + stratified subsample parquets for the requested classes.

    Returns a single DataFrame with all classes concatenated, sorted by
    ``window`` for walk-forward ordering. Missing target columns trigger
    a hard error — run ``teacher.backfill_parquet`` first.
    """
    if classes is None:
        classes = ALL_CLASSES

    required_targets = [h["target"] for h in HEAD_CONFIG.values()]
    frames: list[pd.DataFrame] = []

    for cls in classes:
        path = DATA_DIR / f"{cls}_samples.parquet"
        if not path.exists():
            logger.warning("Missing parquet: %s", path)
            continue

        df = pd.read_parquet(path)

        # Entry-only filter — the student never sees no-trade rows
        if "label_action" in df.columns:
            df = df[df["label_action"] > 0]

        missing_targets = [t for t in required_targets if t not in df.columns]
        if missing_targets:
            raise RuntimeError(
                f"{path} missing teacher-v2 targets {missing_targets}. "
                f"Run `python3 -m teacher.backfill_parquet` first."
            )

        missing_feats = [f for f in ENTRY_QUALITY_FEATURES if f not in df.columns]
        if len(missing_feats) > 5:
            logger.error("%s schema drift — %d features missing, skipping",
                         path, len(missing_feats))
            continue

        # Stratified subsample on optimal_entry to keep class balance
        if len(df) > subsample_per_class:
            pos = df[df["optimal_entry"] == 1]
            neg = df[df["optimal_entry"] != 1]
            ratio = len(pos) / max(len(df), 1)
            n_pos = max(1, int(subsample_per_class * ratio))
            n_neg = subsample_per_class - n_pos
            pos_s = pos.sample(n=min(n_pos, len(pos)), random_state=42)
            neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=42)
            df = pd.concat([pos_s, neg_s], ignore_index=True)

        df["asset_class"] = cls
        frames.append(df)
        logger.info("Loaded %s: %d rows (optimal_entry pos rate %.2f%%)",
                    cls, len(df), 100 * df["optimal_entry"].mean())

    if not frames:
        raise RuntimeError("No training data loaded — check DATA_DIR and --classes")

    full = pd.concat(frames, ignore_index=True)
    if "window" in full.columns:
        full = full.sort_values("window").reset_index(drop=True)
    logger.info("Total: %d rows across %d classes, %d windows",
                len(full), len(frames),
                full["window"].nunique() if "window" in full.columns else 0)
    return full


def _train_one_head(head_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    classes_loaded: list[str]) -> tuple[Any, list[str], dict]:
    """Train a single head on train_df with val_df for early stopping."""
    cfg = HEAD_CONFIG[head_name]
    target = cfg["target"]

    X_train, feat_names = prepare_features(train_df, task="entry_quality")
    X_val, _ = prepare_features(val_df, task="entry_quality")

    y_train = train_df[target].values.astype(np.float32)
    y_val = val_df[target].values.astype(np.float32)

    params = {**_XGB_BASE, **cfg["extra_params"], "eval_metric": cfg["eval_metric"]}
    if cfg["estimator"] is xgb.XGBClassifier:
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0 and n_neg > 0:
            params["scale_pos_weight"] = n_neg / n_pos

    model = cfg["estimator"](**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Metrics
    metrics: dict[str, float] = {"best_iteration": int(model.best_iteration or 0)}
    try:
        if cfg["estimator"] is xgb.XGBClassifier:
            proba = model.predict_proba(X_val)
            if proba.shape[1] > 1:
                metrics["val_auc"] = float(roc_auc_score(y_val, proba[:, 1]))
        else:
            pred = model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, pred)))
            metrics["val_mae_label"] = float(np.mean(np.abs(pred - y_val)))
    except Exception as exc:
        logger.debug("metric computation failed for %s: %s", head_name, exc)

    # Top-5 feature importance
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        order = np.argsort(importances)[::-1][:5]
        top5 = [(feat_names[i], float(importances[i])) for i in order]
        logger.info("  [%s] top5 features: %s", head_name,
                    ", ".join(f"{n}={w:.3f}" for n, w in top5))

    return model, feat_names, metrics


def _walk_forward(head_name: str, data: pd.DataFrame, classes: list[str]) -> list[dict]:
    """Standard 5-fold walk-forward for diagnostic purposes."""
    if "window" not in data.columns:
        logger.warning("No `window` column — skipping walk-forward for %s", head_name)
        return []

    windows = sorted(data["window"].unique())
    if len(windows) < 4:
        logger.warning("Only %d windows — skipping walk-forward", len(windows))
        return []

    results: list[dict] = []
    n_folds = min(5, len(windows) - 3)
    for fold in range(n_folds):
        train_w = windows[:fold + 6]
        val_w = windows[fold + 6] if fold + 6 < len(windows) else windows[-2]
        test_w = windows[fold + 7] if fold + 7 < len(windows) else windows[-1]

        train_df = data[data["window"].isin(train_w)]
        val_df = data[data["window"] == val_w]
        test_df = data[data["window"] == test_w]

        if len(train_df) < 1000 or len(val_df) < 100 or len(test_df) < 100:
            logger.info("  fold %d skipped — insufficient rows", fold)
            continue

        logger.info("[%s] FOLD %d: train=%d, val=%d, test=%d",
                    head_name, fold, len(train_df), len(val_df), len(test_df))
        model, feat_names, metrics = _train_one_head(
            head_name, train_df, val_df, classes,
        )

        # Test-set metrics
        target = HEAD_CONFIG[head_name]["target"]
        X_test, _ = prepare_features(test_df, task="entry_quality")
        y_test = test_df[target].values
        if HEAD_CONFIG[head_name]["estimator"] is xgb.XGBClassifier:
            proba = model.predict_proba(X_test)
            if proba.shape[1] > 1:
                metrics["test_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            pred = model.predict(X_test)
            metrics["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, pred)))
        logger.info("  [%s] fold %d metrics: %s", head_name, fold,
                    {k: round(v, 4) for k, v in metrics.items()})
        results.append({"fold": fold, "head": head_name, **metrics})

        # Free model memory between folds
        del model
        gc.collect()

    return results


def train_production(head_name: str, data: pd.DataFrame, classes: list[str]) -> dict:
    """Train the final production model on all-but-last window, validate on last."""
    if "window" in data.columns:
        last_w = max(data["window"].unique())
        train_df = data[data["window"] != last_w]
        val_df = data[data["window"] == last_w]
    else:
        # Fallback — simple 90/10 split
        cut = int(len(data) * 0.9)
        train_df = data.iloc[:cut]
        val_df = data.iloc[cut:]

    logger.info("[%s] PRODUCTION: train=%d, val=%d", head_name, len(train_df), len(val_df))
    model, feat_names, metrics = _train_one_head(
        head_name, train_df, val_df, classes,
    )

    # Pickle in the same format as rl_brain_v2 so live loading is uniform
    model_data = {
        "model": model,
        "feat_names": feat_names,
        "task": f"student_{head_name}",
        "schema_version": SCHEMA_VERSION,
        "dead_features": list(DEAD_FEATURES),
        "clip_ranges": CLIP_RANGES,
        "asset_class_map": ASSET_CLASS_MAP,
        "teacher_version": "v2",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"student_{head_name}.pkl"
    tmp = out.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(model_data, f)
    tmp.replace(out)
    logger.info("[%s] saved → %s (val metrics: %s)", head_name, out,
                {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", choices=ALL_CLASSES, default=None)
    parser.add_argument("--heads", nargs="+", choices=list(HEAD_CONFIG),
                        default=list(HEAD_CONFIG))
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip walk-forward, train production only")
    parser.add_argument("--subsample", type=int, default=500_000,
                        help="Max rows per class (stratified)")
    args = parser.parse_args()

    data = load_training_data(args.classes, subsample_per_class=args.subsample)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {"teacher_version": "v2", "folds": {}, "production": {}}

    for head in args.heads:
        logger.info("=" * 70)
        logger.info("Training head: %s (target=%s)", head, HEAD_CONFIG[head]["target"])
        logger.info("=" * 70)

        if not args.skip_eval:
            fold_results = _walk_forward(head, data, args.classes or ALL_CLASSES)
            all_results["folds"][head] = fold_results
            gc.collect()

        metrics = train_production(head, data, args.classes or ALL_CLASSES)
        all_results["production"][head] = metrics
        gc.collect()

    results_path = RESULTS_DIR / "student_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved → %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
