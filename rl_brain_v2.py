"""
===================================================================
 rl_brain_v2.py  —  XGBoost Walk-Forward Trade Decision Model
 -----------------------------------------------------------
 Teacher-Student approach: learns which SMC entry signals lead to
 profitable trades using walk-forward validation with anti-overfitting.

 Training: Offline on historical data (causal features, lookahead labels)
 Inference: Real-time prediction from causal features only

 Usage:
     python3 -m rl_brain_v2 --train --walk-forward
     python3 -m rl_brain_v2 --train --walk-forward --classes crypto
     python3 -m rl_brain_v2 --train --asset-holdout
     python3 -m rl_brain_v2 --evaluate
===================================================================
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/results/rl_training.log"),
    ],
)
logger = logging.getLogger(__name__)

# ===================================================================
#  Constants
# ===================================================================

DATA_DIR = Path("data/rl_training")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("backtest/results/rl")

ALL_CLASSES = ["crypto", "forex", "stocks", "commodities"]

# Features that should NOT be used as model input
META_COLS = {"timestamp", "symbol", "asset_class", "window",
             "label_action", "label_rr", "label_outcome", "label_profitable"}

# Features to exclude from entry_quality task (data leaks)
ENTRY_QUALITY_EXCLUDE = {"has_entry_zone", "alignment_score"}

# Dead features: constant or redundant — carry no signal
DEAD_FEATURES = {"bias_strong", "daily_bias"}

# Outlier clipping ranges for known problematic features
CLIP_RANGES = {
    "ema20_dist_5m": (-3.0, 3.0),
    "ema50_dist_5m": (-3.0, 3.0),
    "ema20_dist_1h": (-3.0, 3.0),
    "ema50_dist_1h": (-3.0, 3.0),
}

# Asset class encoding
ASSET_CLASS_MAP = {"crypto": 0, "forex": 1, "stocks": 2, "commodities": 3}

# Walk-forward: minimum training windows before first fold
MIN_TRAIN_WINDOWS = 6


# ===================================================================
#  Data Loading
# ===================================================================

def load_training_data(
    classes: list[str] | None = None,
    subsample_notrade: float = 2.0,
) -> pd.DataFrame:
    """Load all RL training parquets, subsampling no-trade bars to fit in RAM."""
    if classes is None:
        classes = ALL_CLASSES

    dfs = []
    total_raw = 0
    total_entries = 0
    for ac in classes:
        path = DATA_DIR / f"{ac}_samples.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            raw_len = len(df)
            n_ent = int((df["label_action"] > 0).sum())
            total_raw += raw_len
            total_entries += n_ent

            # Downcast floats to float32 to save ~50% RAM
            float_cols = df.select_dtypes(include=["float64"]).columns
            df[float_cols] = df[float_cols].astype(np.float32)

            # Subsample or drop no-trade bars PER CLASS during loading
            if subsample_notrade == 0:
                df = df[df["label_action"] > 0].copy()
            elif subsample_notrade > 0:
                entries = df[df["label_action"] > 0]
                no_trade = df[df["label_action"] == 0]
                max_nt = int(len(entries) * subsample_notrade)
                if len(no_trade) > max_nt:
                    no_trade = no_trade.sample(n=max_nt, random_state=42)
                df = pd.concat([entries, no_trade], ignore_index=True)
                del entries, no_trade

            logger.info("Loaded %s: %d->%d samples (%d entries)",
                        ac, raw_len, len(df), n_ent)
            dfs.append(df)
            gc.collect()
        else:
            logger.warning("No data for %s at %s", ac, path)

    if not dfs:
        raise FileNotFoundError(f"No training data found in {DATA_DIR}")

    combined = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()
    logger.info("Total: %d raw -> %d subsampled, %d entries (%.1f%%)",
                total_raw, len(combined), total_entries,
                100 * total_entries / max(total_raw, 1))
    return combined


def prepare_features(df: pd.DataFrame, task: str = "entry_quality") -> tuple[np.ndarray, list[str]]:
    """Extract feature matrix with dead feature removal, clipping, and asset_class_id."""
    exclude = META_COLS | DEAD_FEATURES | (ENTRY_QUALITY_EXCLUDE if task == "entry_quality" else set())
    feat_cols = [c for c in df.columns if c not in exclude]

    X = df[feat_cols].values.astype(np.float32)

    # Clip known outlier columns
    for col_name, (lo, hi) in CLIP_RANGES.items():
        if col_name in feat_cols:
            col_idx = feat_cols.index(col_name)
            X[:, col_idx] = np.clip(X[:, col_idx], lo, hi)

    # Log nan_to_num replacements instead of silent masking
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan > 0 or n_inf > 0:
        logger.warning("Data quality: %d NaN, %d inf values replaced in %d x %d matrix",
                       n_nan, n_inf, X.shape[0], X.shape[1])
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

    # Add asset_class_id as a feature
    if "asset_class" in df.columns:
        ac_ids = df["asset_class"].map(ASSET_CLASS_MAP).fillna(0).values.astype(np.float32).reshape(-1, 1)
        X = np.hstack([X, ac_ids])
        feat_cols = feat_cols + ["asset_class_id"]

    return X, feat_cols


def prepare_labels(df: pd.DataFrame, task: str = "entry_quality") -> np.ndarray:
    """
    Prepare labels based on task:
    - "entry_quality": win (1) vs not-win (0), only on entry bars
    - "binary": profitable (1) vs not (0)
    - "direction": no_trade (0), long (1), short (2)
    """
    if task == "entry_quality":
        return (df["label_outcome"].values == 1).astype(np.int32)
    elif task == "binary":
        return df["label_profitable"].values.astype(np.int32)
    elif task == "direction":
        return df["label_action"].values.astype(np.int32)
    else:
        raise ValueError(f"Unknown task: {task}")


def prepare_sample_weights(
    y_train: np.ndarray,
    df_train: pd.DataFrame,
    task: str = "entry_quality",
) -> np.ndarray:
    """RR-weighted sample weights: high-RR wins count more, losses uniform."""
    weights = np.ones(len(y_train), dtype=np.float32)
    if task == "entry_quality":
        rr_vals = np.abs(df_train["label_rr"].values).astype(np.float32)
        # Clip RR for weighting to avoid extreme outliers dominating
        rr_clipped = np.clip(rr_vals, 1.0, 5.0)
        weights[y_train == 1] = rr_clipped[y_train == 1]
    else:
        entry_mask = df_train["label_action"].values > 0
        rr_vals = np.abs(df_train["label_rr"].values).astype(np.float32)
        weights[entry_mask] = np.clip(rr_vals[entry_mask], 0.5, 5.0)
        weights[~entry_mask] = 0.1
    return weights


# ===================================================================
#  XGBoost Model
# ===================================================================

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feat_names: list[str],
    sample_weights: np.ndarray | None = None,
    task: str = "entry_quality",
) -> Any:
    """Train XGBoost classifier with early stopping and regularization."""
    import xgboost as xgb

    n_classes = len(np.unique(y_train))
    if task in ("binary", "entry_quality") or n_classes == 2:
        objective = "binary:logistic"
        eval_metric = "auc"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    # Handle class imbalance
    if task in ("binary", "entry_quality"):
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale = n_neg / max(n_pos, 1)
    else:
        scale = 1.0

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale if task in ("binary", "entry_quality") else 1.0,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric=eval_metric,
        early_stopping_rounds=30,
        n_jobs=4,
        random_state=42,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=False,
    )

    logger.info("XGBoost trained: %d trees, best iteration %d",
                model.n_estimators, model.best_iteration)

    return model


# ===================================================================
#  Metrics
# ===================================================================

def compute_oos_sharpe(y_pred: np.ndarray, df_test: pd.DataFrame) -> float:
    """Compute OOS Sharpe ratio from predicted trades' actual RR outcomes."""
    trade_mask = y_pred == 1
    if trade_mask.sum() == 0:
        return 0.0
    trade_rr = df_test.loc[trade_mask, "label_rr"].values.astype(np.float64)
    # Clip extreme RR for Sharpe calculation
    trade_rr = np.clip(trade_rr, -1.0, 20.0)
    if len(trade_rr) < 2:
        return 0.0
    return float(np.mean(trade_rr) / max(np.std(trade_rr), 1e-6))


def compute_oos_profit_factor(y_pred: np.ndarray, df_test: pd.DataFrame) -> float:
    """Compute OOS profit factor from predicted trades."""
    trade_mask = y_pred == 1
    if trade_mask.sum() == 0:
        return 0.0
    trade_df = df_test[trade_mask]
    outcomes = trade_df["label_outcome"].values
    rr = trade_df["label_rr"].values.astype(np.float64)
    rr = np.clip(rr, -1.0, 20.0)
    win_rr = rr[outcomes == 1]
    loss_rr = rr[outcomes == 2]
    total_win = float(win_rr.sum()) if len(win_rr) > 0 else 0.0
    total_loss = float(abs(loss_rr.sum())) if len(loss_rr) > 0 else 0.0
    return total_win / max(total_loss, 0.001)


def evaluate_fold(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    feat_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "entry_quality",
) -> dict[str, Any]:
    """Evaluate a single fold with all trading metrics + overfitting check."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics: dict[str, Any] = {}

    # ML metrics (test)
    metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred))
    if task in ("binary", "entry_quality"):
        metrics["test_precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["test_recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["test_f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            metrics["test_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        except (ValueError, IndexError):
            metrics["test_auc"] = 0.0

    # ML metrics (train) — for overfitting detection
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)
    metrics["train_accuracy"] = float(accuracy_score(y_train, train_pred))
    try:
        metrics["train_auc"] = float(roc_auc_score(y_train, train_proba[:, 1])) if task in ("binary", "entry_quality") else 0.0
    except (ValueError, IndexError):
        metrics["train_auc"] = 0.0

    metrics["auc_gap"] = metrics.get("train_auc", 0) - metrics.get("test_auc", 0)
    metrics["overfit_flag"] = metrics["auc_gap"] > 0.10

    # Trading metrics
    trade_mask = y_pred == 1
    n_predicted = int(trade_mask.sum())
    metrics["n_predicted_trades"] = n_predicted

    if n_predicted > 0:
        trade_df = df_test[trade_mask]
        outcomes = trade_df["label_outcome"].values
        rr = trade_df["label_rr"].values.astype(np.float64)
        rr = np.clip(rr, -1.0, 20.0)

        n_win = int((outcomes == 1).sum())
        n_loss = int((outcomes == 2).sum())
        n_be = int((outcomes == 3).sum())
        real_trades = n_win + n_loss

        metrics["wins"] = n_win
        metrics["losses"] = n_loss
        metrics["breakeven"] = n_be
        metrics["oos_winrate"] = n_win / max(real_trades, 1)
        metrics["avg_win_rr"] = float(rr[outcomes == 1].mean()) if n_win > 0 else 0.0
        metrics["avg_loss_rr"] = float(rr[outcomes == 2].mean()) if n_loss > 0 else 0.0

    # OOS Sharpe and PF
    metrics["oos_sharpe"] = compute_oos_sharpe(y_pred, df_test)
    metrics["oos_pf"] = compute_oos_profit_factor(y_pred, df_test)

    # Per-asset-class breakdown
    per_class: dict[str, dict] = {}
    for ac in df_test["asset_class"].unique():
        ac_mask = (df_test["asset_class"] == ac).values & trade_mask
        n_ac = int(ac_mask.sum())
        if n_ac == 0:
            per_class[ac] = {"trades": 0, "wins": 0, "losses": 0, "winrate": 0.0, "pf": 0.0}
            continue
        ac_df = df_test[ac_mask]
        ac_out = ac_df["label_outcome"].values
        ac_rr = np.clip(ac_df["label_rr"].values.astype(np.float64), -1.0, 20.0)
        w = int((ac_out == 1).sum())
        lo = int((ac_out == 2).sum())
        win_sum = float(ac_rr[ac_out == 1].sum()) if w > 0 else 0.0
        loss_sum = float(abs(ac_rr[ac_out == 2].sum())) if lo > 0 else 0.0
        per_class[ac] = {
            "trades": n_ac, "wins": w, "losses": lo,
            "winrate": w / max(w + lo, 1),
            "pf": win_sum / max(loss_sum, 0.001),
        }
    metrics["per_class"] = per_class

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = sorted(zip(feat_names, importances.tolist()), key=lambda x: x[1], reverse=True)
        metrics["feature_importance"] = fi

    # Clean up train arrays
    del train_pred, train_proba
    gc.collect()

    return metrics


def compute_feature_stability(fold_results: list[dict], top_n: int = 10) -> list[tuple[str, int, float]]:
    """
    Compute feature stability across folds.
    Returns list of (feature_name, n_folds_in_top_N, avg_importance) sorted by consistency.
    """
    from collections import defaultdict
    feature_counts: dict[str, int] = defaultdict(int)
    feature_importance_sum: dict[str, float] = defaultdict(float)
    n_folds = len(fold_results)

    for result in fold_results:
        fi = result.get("feature_importance", [])
        top_features = {f[0] for f in fi[:top_n]}
        for feat, imp in fi:
            feature_importance_sum[feat] += imp
            if feat in top_features:
                feature_counts[feat] += 1

    stability = []
    for feat in feature_importance_sum:
        stability.append((
            feat,
            feature_counts.get(feat, 0),
            feature_importance_sum[feat] / n_folds,
        ))

    # Sort by: folds in top-N (desc), then avg importance (desc)
    stability.sort(key=lambda x: (-x[1], -x[2]))
    return stability


# ===================================================================
#  Reporting
# ===================================================================

def print_fold_result(fold_idx: int, train_wins: list, test_win: int, metrics: dict) -> None:
    """Print a single fold's results."""
    logger.info("--- Fold %d: Train W%s, Test W%d ---",
                fold_idx, [int(w) for w in train_wins], int(test_win))
    logger.info("  Train AUC: %.3f | Test AUC: %.3f | Gap: %.3f %s",
                metrics.get("train_auc", 0), metrics.get("test_auc", 0),
                metrics.get("auc_gap", 0),
                "** OVERFIT **" if metrics.get("overfit_flag") else "")
    logger.info("  OOS Sharpe: %.3f | OOS PF: %.2f | OOS WR: %.1f%% | Trades: %d",
                metrics.get("oos_sharpe", 0), metrics.get("oos_pf", 0),
                100 * metrics.get("oos_winrate", 0), metrics.get("n_predicted_trades", 0))
    logger.info("  Precision: %.3f | Recall: %.3f | F1: %.3f",
                metrics.get("test_precision", 0), metrics.get("test_recall", 0),
                metrics.get("test_f1", 0))

    for ac, stats in metrics.get("per_class", {}).items():
        if stats.get("trades", 0) > 0:
            logger.info("    %s: %d trades, WR=%.0f%%, PF=%.2f",
                        ac, stats["trades"], 100 * stats.get("winrate", 0), stats.get("pf", 0))

    fi = metrics.get("feature_importance", [])
    if fi:
        logger.info("  Top 5 features: %s",
                    ", ".join(f"{f[0]}={f[1]:.3f}" for f in fi[:5]))


def print_aggregate_results(fold_results: list[dict], stability: list[tuple]) -> None:
    """Print cross-fold aggregate summary."""
    n = len(fold_results)
    sharpes = [r.get("oos_sharpe", 0) for r in fold_results]
    pfs = [r.get("oos_pf", 0) for r in fold_results]
    wrs = [r.get("oos_winrate", 0) for r in fold_results]
    trades = [r.get("n_predicted_trades", 0) for r in fold_results]
    gaps = [r.get("auc_gap", 0) for r in fold_results]
    overfit_count = sum(1 for r in fold_results if r.get("overfit_flag"))

    logger.info("=" * 70)
    logger.info("AGGREGATE RESULTS (%d folds)", n)
    logger.info("=" * 70)
    logger.info("  OOS Sharpe:  %.3f +/- %.3f  (min=%.3f, max=%.3f)",
                np.mean(sharpes), np.std(sharpes), min(sharpes), max(sharpes))
    logger.info("  OOS PF:      %.2f +/- %.2f  (min=%.2f, max=%.2f)",
                np.mean(pfs), np.std(pfs), min(pfs), max(pfs))
    logger.info("  OOS WR:      %.1f%% +/- %.1f%%",
                100 * np.mean(wrs), 100 * np.std(wrs))
    logger.info("  Trades/fold: %.0f +/- %.0f  (total=%d)",
                np.mean(trades), np.std(trades), sum(trades))
    logger.info("  AUC Gap:     %.3f +/- %.3f  (overfit folds: %d/%d)",
                np.mean(gaps), np.std(gaps), overfit_count, n)

    if overfit_count > 0:
        logger.warning("  !! %d/%d folds show overfit (AUC gap > 0.10) !!", overfit_count, n)

    logger.info("")
    logger.info("Feature Stability (top-10 in how many folds):")
    for feat, count, avg_imp in stability[:20]:
        bar = "*" * count
        logger.info("  %-25s %d/%d folds  avg=%.4f  %s", feat, count, n, avg_imp, bar)


# ===================================================================
#  Walk-Forward Training (Rolling 5-Fold)
# ===================================================================

def run_walk_forward_rolling(
    classes: list[str] | None = None,
    task: str = "entry_quality",
) -> list[dict[str, Any]]:
    """
    Rolling walk-forward with expanding training window.

    With 12 windows (W0-W11) and MIN_TRAIN_WINDOWS=6:
    - Fold 0: Train W0-W5, Val W6, Test W7
    - Fold 1: Train W0-W6, Val W7, Test W8
    - Fold 2: Train W0-W7, Val W8, Test W9
    - Fold 3: Train W0-W8, Val W9, Test W10
    - Fold 4: Train W0-W9, Val W10, Test W11
    """
    subsample_ratio = 0.0 if task == "entry_quality" else 2.0
    data = load_training_data(classes, subsample_notrade=subsample_ratio)

    all_windows = sorted(data["window"].unique())
    n_win = len(all_windows)
    logger.info("Found %d windows: %s", n_win, list(all_windows))

    if n_win < MIN_TRAIN_WINDOWS + 2:
        logger.error("Need at least %d windows for walk-forward, found %d",
                     MIN_TRAIN_WINDOWS + 2, n_win)
        return []

    # Print per-window stats
    for w_id in all_windows:
        w = data[data["window"] == w_id]
        n_ent = int((w["label_action"] > 0).sum())
        n_w = int((w["label_outcome"] == 1).sum())
        n_l = int((w["label_outcome"] == 2).sum())
        logger.info("  W%d: %d samples, %d entries, %d win, %d loss (WR=%.0f%%)",
                    w_id, len(w), n_ent, n_w, n_l,
                    100 * n_w / max(n_w + n_l, 1))

    # For entry_quality, filter to entry bars only
    if task == "entry_quality":
        pre_filter = len(data)
        data = data[data["label_action"] > 0].copy()
        logger.info("Entry-only filter: %d -> %d rows", pre_filter, len(data))

    # Rolling folds
    fold_results: list[dict[str, Any]] = []
    n_folds = n_win - MIN_TRAIN_WINDOWS - 1

    for fold_idx in range(n_folds):
        train_end = MIN_TRAIN_WINDOWS + fold_idx
        train_wins = all_windows[:train_end]
        val_win = all_windows[train_end]
        test_win = all_windows[train_end + 1]

        logger.info("=" * 70)
        logger.info("FOLD %d/%d: Train W%s, Val W%d, Test W%d",
                    fold_idx, n_folds - 1,
                    [int(w) for w in train_wins], int(val_win), int(test_win))
        logger.info("=" * 70)

        train_data = data[data["window"].isin(train_wins)].copy()
        val_data = data[data["window"] == val_win].copy()
        test_data = data[data["window"] == test_win].copy()

        logger.info("Sizes: Train=%d, Val=%d, Test=%d",
                    len(train_data), len(val_data), len(test_data))

        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            logger.warning("Skipping fold %d: empty split", fold_idx)
            continue

        # Prepare features
        X_train, feat_names = prepare_features(train_data, task)
        X_val, _ = prepare_features(val_data, task)
        X_test, _ = prepare_features(test_data, task)

        y_train = prepare_labels(train_data, task)
        y_val = prepare_labels(val_data, task)
        y_test = prepare_labels(test_data, task)

        weights = prepare_sample_weights(y_train, train_data, task)

        logger.info("Labels: Train pos=%.1f%%, Val pos=%.1f%%, Test pos=%.1f%%",
                    100 * y_train.mean(), 100 * y_val.mean(), 100 * y_test.mean())

        # Train
        model = train_xgboost(X_train, y_train, X_val, y_val,
                              feat_names, sample_weights=weights, task=task)

        # Evaluate
        metrics = evaluate_fold(model, X_test, y_test, test_data.reset_index(drop=True),
                                feat_names, X_train, y_train, task)
        metrics["fold"] = fold_idx
        metrics["train_windows"] = [int(w) for w in train_wins]
        metrics["val_window"] = int(val_win)
        metrics["test_window"] = int(test_win)
        metrics["best_iteration"] = int(model.best_iteration)

        print_fold_result(fold_idx, train_wins, test_win, metrics)
        fold_results.append(metrics)

        # Cleanup
        del X_train, X_val, X_test, y_train, y_val, y_test, weights, model
        del train_data, val_data, test_data
        gc.collect()

    if not fold_results:
        logger.error("No folds completed!")
        return []

    # Feature stability
    stability = compute_feature_stability(fold_results)

    # Aggregate reporting
    print_aggregate_results(fold_results, stability)

    # Train final production model on all available data (W0-W10, val W10)
    logger.info("=" * 70)
    logger.info("TRAINING FINAL PRODUCTION MODEL (W0-%d, val W%d)",
                int(all_windows[-2]), int(all_windows[-1]))
    logger.info("=" * 70)

    final_train = data[data["window"].isin(all_windows[:-1])].copy()
    final_val = data[data["window"] == all_windows[-1]].copy()

    X_ft, feat_names_final = prepare_features(final_train, task)
    X_fv, _ = prepare_features(final_val, task)
    y_ft = prepare_labels(final_train, task)
    y_fv = prepare_labels(final_val, task)
    w_ft = prepare_sample_weights(y_ft, final_train, task)

    final_model = train_xgboost(X_ft, y_ft, X_fv, y_fv,
                                feat_names_final, sample_weights=w_ft, task=task)

    # Save model + metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "rl_brain_v2_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feat_names": feat_names_final,
            "task": task,
            "dead_features": list(DEAD_FEATURES),
            "clip_ranges": CLIP_RANGES,
            "asset_class_map": ASSET_CLASS_MAP,
            "fold_results_summary": [{
                "fold": r["fold"],
                "test_window": r["test_window"],
                "oos_sharpe": r.get("oos_sharpe", 0),
                "oos_pf": r.get("oos_pf", 0),
                "oos_winrate": r.get("oos_winrate", 0),
                "test_auc": r.get("test_auc", 0),
                "auc_gap": r.get("auc_gap", 0),
            } for r in fold_results],
        }, f)
    logger.info("Production model saved: %s", model_path)

    # Save detailed results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = json.loads(json.dumps(fold_results, default=_convert))
    with open(RESULTS_DIR / "fold_results.json", "w") as f:
        json.dump(results_clean, f, indent=2)

    # Save feature stability
    stab_df = pd.DataFrame(stability, columns=["feature", "folds_in_top10", "avg_importance"])
    stab_df.to_csv(RESULTS_DIR / "feature_stability.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR)

    # Save feature importance (from last fold for reference)
    if fold_results and "feature_importance" in fold_results[-1]:
        fi_df = pd.DataFrame(fold_results[-1]["feature_importance"],
                             columns=["feature", "importance"])
        fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    del final_train, final_val, X_ft, X_fv, y_ft, y_fv, w_ft, final_model, data
    gc.collect()

    return fold_results


# ===================================================================
#  Per-Asset Holdout
# ===================================================================

def run_asset_holdout(
    holdout_class: str = "commodities",
    task: str = "entry_quality",
) -> dict[str, Any]:
    """
    Train on all classes except holdout, test generalization on holdout.
    Compares: all-class model vs holdout-excluded model on the holdout class.
    """
    logger.info("=" * 70)
    logger.info("ASSET HOLDOUT: training WITHOUT %s, testing ON %s", holdout_class, holdout_class)
    logger.info("=" * 70)

    train_classes = [c for c in ALL_CLASSES if c != holdout_class]

    # Load all data
    all_data = load_training_data(ALL_CLASSES, subsample_notrade=0.0 if task == "entry_quality" else 2.0)

    if task == "entry_quality":
        all_data = all_data[all_data["label_action"] > 0].copy()

    all_windows = sorted(all_data["window"].unique())
    if len(all_windows) < MIN_TRAIN_WINDOWS + 2:
        logger.error("Not enough windows")
        return {}

    # Use the last fold split (largest training set)
    train_wins = all_windows[:-2]
    val_win = all_windows[-2]
    test_win = all_windows[-1]

    # Split data
    holdout_test = all_data[
        (all_data["window"] == test_win) &
        (all_data["asset_class"] == holdout_class)
    ].copy()

    if len(holdout_test) == 0:
        logger.warning("No holdout test data for %s in W%d", holdout_class, test_win)
        return {}

    # --- Model A: trained on ALL classes ---
    train_all = all_data[all_data["window"].isin(train_wins)].copy()
    val_all = all_data[all_data["window"] == val_win].copy()

    X_train_a, fn_a = prepare_features(train_all, task)
    X_val_a, _ = prepare_features(val_all, task)
    y_train_a = prepare_labels(train_all, task)
    y_val_a = prepare_labels(val_all, task)
    w_a = prepare_sample_weights(y_train_a, train_all, task)

    logger.info("Model A (all classes): Train=%d, Val=%d", len(train_all), len(val_all))
    model_a = train_xgboost(X_train_a, y_train_a, X_val_a, y_val_a, fn_a, w_a, task)

    X_hold, _ = prepare_features(holdout_test, task)
    y_hold = prepare_labels(holdout_test, task)
    y_pred_a = model_a.predict(X_hold)

    del X_train_a, X_val_a, y_train_a, y_val_a, w_a, train_all, val_all, model_a
    gc.collect()

    # --- Model B: trained WITHOUT holdout class ---
    train_excl = all_data[
        (all_data["window"].isin(train_wins)) &
        (all_data["asset_class"] != holdout_class)
    ].copy()
    val_excl = all_data[
        (all_data["window"] == val_win) &
        (all_data["asset_class"] != holdout_class)
    ].copy()

    X_train_b, fn_b = prepare_features(train_excl, task)
    X_val_b, _ = prepare_features(val_excl, task)
    y_train_b = prepare_labels(train_excl, task)
    y_val_b = prepare_labels(val_excl, task)
    w_b = prepare_sample_weights(y_train_b, train_excl, task)

    logger.info("Model B (excl %s): Train=%d, Val=%d", holdout_class, len(train_excl), len(val_excl))
    model_b = train_xgboost(X_train_b, y_train_b, X_val_b, y_val_b, fn_b, w_b, task)

    y_pred_b = model_b.predict(X_hold)

    del X_train_b, X_val_b, y_train_b, y_val_b, w_b, train_excl, val_excl, model_b
    gc.collect()

    # --- Compare ---
    results = {
        "holdout_class": holdout_class,
        "test_window": int(test_win),
        "n_holdout_samples": len(holdout_test),
        "n_positive": int(y_hold.sum()),
    }

    for label, y_pred in [("all_class", y_pred_a), ("excl_holdout", y_pred_b)]:
        n_pred = int((y_pred == 1).sum())
        sharpe = compute_oos_sharpe(y_pred, holdout_test.reset_index(drop=True))
        pf = compute_oos_profit_factor(y_pred, holdout_test.reset_index(drop=True))

        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = float(accuracy_score(y_hold, y_pred))

        results[f"{label}_trades"] = n_pred
        results[f"{label}_sharpe"] = sharpe
        results[f"{label}_pf"] = pf
        results[f"{label}_accuracy"] = acc

        logger.info("  %s on %s: %d trades, Sharpe=%.3f, PF=%.2f, Acc=%.3f",
                    label, holdout_class, n_pred, sharpe, pf, acc)

    # Generalization gap
    sharpe_gap = results.get("all_class_sharpe", 0) - results.get("excl_holdout_sharpe", 0)
    results["sharpe_gap"] = sharpe_gap
    logger.info("  Sharpe gap (all_class - excl_holdout): %.3f", sharpe_gap)
    if abs(sharpe_gap) < 0.1:
        logger.info("  -> Model generalizes well to %s (small gap)", holdout_class)
    else:
        logger.warning("  -> Significant gap — model may be memorizing %s patterns", holdout_class)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "asset_holdout.json", "w") as f:
        json.dump(results, f, indent=2)

    del all_data, holdout_test, X_hold, y_hold
    gc.collect()

    return results


# ===================================================================
#  Prediction Interface (for live/paper trading)
# ===================================================================

class RLBrainV2:
    """Inference wrapper for the trained model."""

    def __init__(self, model_path: str | Path = "models/rl_brain_v2_xgb.pkl"):
        self.model = None
        self.feat_names: list[str] = []
        self.task = "entry_quality"
        self._clip_ranges: dict = {}
        self._dead_features: set = set()
        self._asset_class_map: dict = ASSET_CLASS_MAP

        path = Path(model_path)
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feat_names = data["feat_names"]
            self.task = data.get("task", "entry_quality")
            self._clip_ranges = data.get("clip_ranges", {})
            self._dead_features = set(data.get("dead_features", []))
            self._asset_class_map = data.get("asset_class_map", ASSET_CLASS_MAP)
            logger.info("RLBrainV2 loaded from %s (%d features, task=%s)",
                        path, len(self.feat_names), self.task)
        else:
            logger.warning("No model found at %s - predictions disabled", path)

    def predict(self, features: dict[str, float]) -> tuple[str, float, float]:
        """
        Predict trade decision from feature dict.

        Returns (action, confidence, predicted_rr):
            action: "no_trade" | "long" | "short"
            confidence: 0.0-1.0
            predicted_rr: estimated RR (0 if no_trade)
        """
        if self.model is None:
            return "no_trade", 0.0, 0.0

        # Build feature vector in correct order
        x = np.array([features.get(f, 0.0) for f in self.feat_names],
                      dtype=np.float32).reshape(1, -1)

        # Apply clipping
        for col_name, (lo, hi) in self._clip_ranges.items():
            if col_name in self.feat_names:
                col_idx = self.feat_names.index(col_name)
                x[0, col_idx] = np.clip(x[0, col_idx], lo, hi)

        x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)

        proba = self.model.predict_proba(x)[0]

        if self.task in ("binary", "entry_quality"):
            confidence = float(proba[1])  # P(win)
            if confidence >= 0.6:
                bias = features.get("struct_1d", features.get("daily_bias", 0))
                if bias > 0:
                    return "long", confidence, 0.0
                elif bias < 0:
                    return "short", confidence, 0.0
            return "no_trade", confidence, 0.0

        elif self.task == "direction":
            action_idx = int(np.argmax(proba))
            confidence = float(proba[action_idx])
            actions = ["no_trade", "long", "short"]
            return actions[action_idx], confidence, 0.0

        return "no_trade", 0.0, 0.0


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="RL Brain V2 - XGBoost Walk-Forward Training")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Use rolling walk-forward validation (5 folds)")
    parser.add_argument("--asset-holdout", action="store_true",
                        help="Run per-asset holdout test")
    parser.add_argument("--holdout-class", default="commodities",
                        choices=ALL_CLASSES,
                        help="Asset class to hold out (default: commodities)")
    parser.add_argument("--task", choices=["binary", "direction", "entry_quality"],
                        default="entry_quality",
                        help="Prediction task (default: entry_quality)")
    parser.add_argument("--classes", nargs="+",
                        choices=ALL_CLASSES,
                        help="Asset classes to include")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate existing model")
    args = parser.parse_args()

    Path("backtest/results").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.train:
        if args.walk_forward:
            run_walk_forward_rolling(classes=args.classes, task=args.task)
        elif args.asset_holdout:
            run_asset_holdout(holdout_class=args.holdout_class, task=args.task)
        else:
            logger.info("Use --walk-forward for rolling validation or --asset-holdout")
            run_walk_forward_rolling(classes=args.classes, task=args.task)
    elif args.asset_holdout:
        run_asset_holdout(holdout_class=args.holdout_class, task=args.task)
    elif args.evaluate:
        brain = RLBrainV2()
        if brain.model is not None:
            data = load_training_data(args.classes, subsample_notrade=0.0)
            data = data[data["label_action"] > 0].copy()
            last_win = data["window"].max()
            test = data[data["window"] == last_win].reset_index(drop=True)
            X, feat_names = prepare_features(test, brain.task)
            y = prepare_labels(test, brain.task)
            y_pred = brain.model.predict(X)
            sharpe = compute_oos_sharpe(y_pred, test)
            pf = compute_oos_profit_factor(y_pred, test)
            from sklearn.metrics import accuracy_score, roc_auc_score
            proba = brain.model.predict_proba(X)
            logger.info("EVALUATE on W%d:", int(last_win))
            logger.info("  Accuracy: %.3f", accuracy_score(y, y_pred))
            try:
                logger.info("  AUC: %.3f", roc_auc_score(y, proba[:, 1]))
            except Exception:
                pass
            logger.info("  Trades: %d, Sharpe: %.3f, PF: %.2f",
                        int((y_pred == 1).sum()), sharpe, pf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
