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

from features.schema import SCHEMA_VERSION as _SCHEMA_VERSION

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
             "label_action", "label_rr", "label_outcome", "label_profitable",
             "label_exit_mechanism", "label_exit_bar", "label_max_favorable_rr",
             "label_tp_rr", "label_cost_rr",
             # Exit classifier labels (bar-level episodes)
             "label_hold_better", "bar_unrealized_rr", "final_net_rr",
             "outcome", "exit_mechanism", "max_favorable_rr", "entry_price",
             "tp_rr", "direction"}

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
    exclude = set(META_COLS) | DEAD_FEATURES | (ENTRY_QUALITY_EXCLUDE if task == "entry_quality" else set())
    if task == "early_exit":
        # bar_unrealized_rr is the most important feature for exit decisions
        exclude.discard("bar_unrealized_rr")
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

    # Add asset_class_id as a feature (skip if already present, e.g. exit episodes)
    if "asset_class" in df.columns and "asset_class_id" not in feat_cols:
        ac_ids = df["asset_class"].map(ASSET_CLASS_MAP).fillna(0).values.astype(np.float32).reshape(-1, 1)
        X = np.hstack([X, ac_ids])
        feat_cols = feat_cols + ["asset_class_id"]

    # Add style_id derived from ATR (SL proxy): <0.5%=scalp(0), >2%=swing(1), else=day(0.5)
    if "style_id" not in feat_cols:
        if "atr_5m_norm" in feat_cols:
            atr_idx = feat_cols.index("atr_5m_norm")
            atr_vals = X[:, atr_idx]
        elif "atr_5m_norm" in df.columns:
            atr_vals = df["atr_5m_norm"].values.astype(np.float32)
        else:
            atr_vals = np.full(X.shape[0], 0.01, dtype=np.float32)  # default=day
        style_vals = np.where(atr_vals < 0.005, 0.0, np.where(atr_vals > 0.02, 1.0, 0.5)).astype(np.float32).reshape(-1, 1)
        X = np.hstack([X, style_vals])
        feat_cols = feat_cols + ["style_id"]

    return X, feat_cols


def prepare_labels(df: pd.DataFrame, task: str = "entry_quality") -> np.ndarray:
    """
    Prepare labels based on task:
    - "entry_quality": win (1) vs not-win (0), only on entry bars
    - "binary": profitable (1) vs not (0)
    - "direction": no_trade (0), long (1), short (2)
    - "early_exit": hold (1) vs exit now (0), for bar-by-bar exit classifier
    """
    if task == "entry_quality":
        return (df["label_outcome"].values == 1).astype(np.int32)
    elif task == "binary":
        return df["label_profitable"].values.astype(np.int32)
    elif task == "direction":
        return df["label_action"].values.astype(np.int32)
    elif task == "early_exit":
        return df["label_hold_better"].values.astype(np.int32)
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


def train_xgboost_regressor(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feat_names: list[str],
    sample_weights: np.ndarray | None = None,
) -> Any:
    """Train XGBoost regressor with early stopping and regularization."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="rmse",
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

    logger.info("XGBRegressor trained: %d trees, best iteration %d",
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
    model_path = MODEL_DIR / "rl_entry_filter.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feat_names": feat_names_final,
            "task": task,
            "schema_version": _SCHEMA_VERSION,
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

    # Export symbol ranks as JSON (ships to server alongside model pickle)
    try:
        from backtest.generate_rl_data import compute_symbol_ranks
        all_ranks: dict[str, dict] = {}
        for ac in ALL_CLASSES:
            ac_ranks = compute_symbol_ranks(ac)
            if ac_ranks:
                all_ranks[ac] = ac_ranks
        ranks_path = MODEL_DIR / "symbol_ranks.json"
        with open(ranks_path, "w") as f:
            json.dump(all_ranks, f, indent=2)
        logger.info("Symbol ranks saved: %s (%d classes, %d symbols)",
                     ranks_path, len(all_ranks),
                     sum(len(v) for v in all_ranks.values()))
    except Exception as exc:
        logger.warning("Could not export symbol_ranks.json: %s", exc)

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
    with open(RESULTS_DIR / "fold_results_v2.json", "w") as f:
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
#  Regression Walk-Forward: Shared Helpers
# ===================================================================

def _regression_walk_forward_skeleton(
    classes: list[str] | None,
    label_fn,
    eval_fn,
    task_name: str,
    model_filename: str,
    results_filename: str,
) -> list[dict[str, Any]]:
    """
    Shared walk-forward skeleton for regression tasks (sizing, TP, BE).

    Args:
        label_fn: (df_entries) -> y array
        eval_fn: (model, X_test, df_test, feat_names) -> metrics dict
    """
    data = load_training_data(classes, subsample_notrade=0.0)
    all_windows = sorted(data["window"].unique())
    n_win = len(all_windows)
    logger.info("[%s] Found %d windows", task_name, n_win)

    if n_win < MIN_TRAIN_WINDOWS + 2:
        logger.error("Need at least %d windows, found %d", MIN_TRAIN_WINDOWS + 2, n_win)
        return []

    # Entry bars only
    pre_filter = len(data)
    data = data[data["label_action"] > 0].copy()
    logger.info("[%s] Entry-only filter: %d -> %d rows", task_name, pre_filter, len(data))

    fold_results: list[dict[str, Any]] = []
    n_folds = n_win - MIN_TRAIN_WINDOWS - 1

    for fold_idx in range(n_folds):
        train_end = MIN_TRAIN_WINDOWS + fold_idx
        train_wins = all_windows[:train_end]
        val_win = all_windows[train_end]
        test_win = all_windows[train_end + 1]

        logger.info("=" * 70)
        logger.info("[%s] FOLD %d/%d: Train W%s, Val W%d, Test W%d",
                    task_name, fold_idx, n_folds - 1,
                    [int(w) for w in train_wins], int(val_win), int(test_win))

        train_data = data[data["window"].isin(train_wins)].copy()
        val_data = data[data["window"] == val_win].copy()
        test_data = data[data["window"] == test_win].copy()

        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            logger.warning("Skipping fold %d: empty split", fold_idx)
            continue

        X_train, feat_names = prepare_features(train_data, "entry_quality")
        X_val, _ = prepare_features(val_data, "entry_quality")
        X_test, _ = prepare_features(test_data, "entry_quality")

        y_train = label_fn(train_data)
        y_val = label_fn(val_data)

        logger.info("  Train=%d, Val=%d, Test=%d | y_train: mean=%.3f, std=%.3f",
                    len(train_data), len(val_data), len(test_data),
                    float(np.mean(y_train)), float(np.std(y_train)))

        model = train_xgboost_regressor(X_train, y_train, X_val, y_val, feat_names)

        metrics = eval_fn(model, X_test, test_data.reset_index(drop=True), feat_names)
        metrics["fold"] = fold_idx
        metrics["train_windows"] = [int(w) for w in train_wins]
        metrics["test_window"] = int(test_win)
        metrics["best_iteration"] = int(model.best_iteration)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = sorted(zip(feat_names, model.feature_importances_.tolist()),
                        key=lambda x: x[1], reverse=True)
            metrics["feature_importance"] = fi
            logger.info("  Top 5: %s", ", ".join(f"{f}={v:.3f}" for f, v in fi[:5]))

        fold_results.append(metrics)

        del X_train, X_val, X_test, y_train, y_val, model, train_data, val_data, test_data
        gc.collect()

    if not fold_results:
        logger.error("[%s] No folds completed!", task_name)
        return []

    # Train production model
    logger.info("=" * 70)
    logger.info("[%s] TRAINING PRODUCTION MODEL (W0-%d, val W%d)",
                task_name, int(all_windows[-2]), int(all_windows[-1]))

    final_train = data[data["window"].isin(all_windows[:-1])].copy()
    final_val = data[data["window"] == all_windows[-1]].copy()

    X_ft, fn = prepare_features(final_train, "entry_quality")
    X_fv, _ = prepare_features(final_val, "entry_quality")
    y_ft = label_fn(final_train)
    y_fv = label_fn(final_val)

    final_model = train_xgboost_regressor(X_ft, y_ft, X_fv, y_fv, fn)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / model_filename
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feat_names": fn,
            "task": task_name,
            "schema_version": _SCHEMA_VERSION,
            "clip_ranges": CLIP_RANGES,
            "asset_class_map": ASSET_CLASS_MAP,
        }, f)
    logger.info("[%s] Production model saved: %s", task_name, model_path)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _c(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(RESULTS_DIR / results_filename, "w") as f:
        json.dump(json.loads(json.dumps(fold_results, default=_c)), f, indent=2)
    logger.info("[%s] Results saved: %s", task_name, RESULTS_DIR / results_filename)

    del final_train, final_val, X_ft, X_fv, y_ft, y_fv, final_model, data
    gc.collect()

    return fold_results


# ===================================================================
#  Position Sizing Model (Phase 2)
# ===================================================================

def _sizing_labels(df: pd.DataFrame) -> np.ndarray:
    """Label = clipped realized RR (net of fees)."""
    return np.clip(df["label_rr"].values.astype(np.float32), -1.0, 10.0)


def _sizing_eval(model, X_test, df_test, feat_names) -> dict[str, Any]:
    """Compare model-sized vs uniform-sized portfolio Sharpe on OOS data."""
    predicted_rr = model.predict(X_test).astype(np.float64)
    actual_rr = np.clip(df_test["label_rr"].values.astype(np.float64), -1.0, 20.0)

    n_trades = len(actual_rr)
    if n_trades < 10:
        return {"n_trades": n_trades, "uniform_sharpe": 0.0, "model_sharpe": 0.0, "improvement": 0.0}

    # Uniform sizing baseline
    uniform_sharpe = float(np.mean(actual_rr) / max(np.std(actual_rr), 1e-6))

    # Model sizing: map predicted RR → risk multiplier [0.33, 1.5]
    # Higher predicted RR → more capital allocated
    target_rr = 3.0
    risk_mult = np.clip(predicted_rr / target_rr, 0.33, 1.5)

    # Weighted portfolio returns
    weighted_returns = actual_rr * risk_mult
    model_sharpe = float(np.mean(weighted_returns) / max(np.std(weighted_returns), 1e-6))

    # PF comparison
    uniform_wins = actual_rr[actual_rr > 0].sum()
    uniform_losses = abs(actual_rr[actual_rr < 0].sum())
    uniform_pf = float(uniform_wins / max(uniform_losses, 0.001))

    model_wins = weighted_returns[weighted_returns > 0].sum()
    model_losses = abs(weighted_returns[weighted_returns < 0].sum())
    model_pf = float(model_wins / max(model_losses, 0.001))

    # Prediction quality
    pred_corr = float(np.corrcoef(predicted_rr, actual_rr)[0, 1]) if n_trades > 2 else 0.0
    rmse = float(np.sqrt(np.mean((predicted_rr - actual_rr) ** 2)))

    improvement = model_sharpe - uniform_sharpe

    logger.info("  [sizing] Uniform Sharpe=%.3f PF=%.2f | Model Sharpe=%.3f PF=%.2f | "
                "Improvement=%.3f | Corr=%.3f | RMSE=%.3f | Trades=%d",
                uniform_sharpe, uniform_pf, model_sharpe, model_pf,
                improvement, pred_corr, rmse, n_trades)

    return {
        "n_trades": n_trades,
        "uniform_sharpe": uniform_sharpe, "uniform_pf": uniform_pf,
        "model_sharpe": model_sharpe, "model_pf": model_pf,
        "improvement": improvement,
        "pred_corr": pred_corr, "rmse": rmse,
    }


def run_walk_forward_sizing(classes: list[str] | None = None) -> list[dict]:
    """Walk-forward validation for position sizing model."""
    logger.info("=" * 70)
    logger.info("POSITION SIZING MODEL — Walk-Forward Validation")
    logger.info("=" * 70)

    results = _regression_walk_forward_skeleton(
        classes, _sizing_labels, _sizing_eval,
        task_name="sizing",
        model_filename="rl_position_sizer.pkl",
        results_filename="sizing_results.json",
    )

    if results:
        improvements = [r.get("improvement", 0) for r in results]
        model_sharpes = [r.get("model_sharpe", 0) for r in results]
        uniform_sharpes = [r.get("uniform_sharpe", 0) for r in results]
        logger.info("=" * 70)
        logger.info("[sizing] AGGREGATE: Uniform Sharpe=%.3f | Model Sharpe=%.3f | "
                    "Improvement=%.3f +/- %.3f",
                    np.mean(uniform_sharpes), np.mean(model_sharpes),
                    np.mean(improvements), np.std(improvements))
        gate = np.mean(model_sharpes) > np.mean(uniform_sharpes)
        logger.info("[sizing] GATE %s: model_sharpe > uniform_sharpe",
                    "PASSED" if gate else "FAILED")
    return results


# ===================================================================
#  TP Optimization Model (Phase 3)
# ===================================================================

def _tp_labels(df: pd.DataFrame) -> np.ndarray:
    """Label = MFE (max favorable excursion) — what TP was achievable."""
    return np.clip(df["label_max_favorable_rr"].values.astype(np.float32), 0.0, 20.0)


def _tp_eval(model, X_test, df_test, feat_names) -> dict[str, Any]:
    """Simulate TP adjustment on OOS data. Compare adjusted vs original outcomes."""
    predicted_mfe = model.predict(X_test).astype(np.float64)
    planned_tp = df_test["label_tp_rr"].values.astype(np.float64)
    actual_mfe = df_test["label_max_favorable_rr"].values.astype(np.float64)
    actual_rr = np.clip(df_test["label_rr"].values.astype(np.float64), -1.0, 20.0)
    cost_rr = df_test["label_cost_rr"].values.astype(np.float64)

    n_trades = len(actual_rr)
    if n_trades < 10:
        return {"n_trades": n_trades, "original_sharpe": 0.0, "adjusted_sharpe": 0.0}

    # Adjust TP based on predicted MFE
    adjusted_tp = np.copy(planned_tp)
    # Reduce: predicted MFE much lower than planned TP → take profit earlier
    reduce_mask = predicted_mfe < (planned_tp * 0.7)
    adjusted_tp[reduce_mask] = np.maximum(predicted_mfe[reduce_mask] * 0.85, 0.5)
    # Extend: predicted MFE much higher than planned TP → let it run
    extend_mask = predicted_mfe > (planned_tp * 1.3)
    adjusted_tp[extend_mask] = np.minimum(predicted_mfe[extend_mask] * 0.8, planned_tp[extend_mask] * 1.5)

    # Simulate outcomes under adjusted TP
    # If actual MFE >= adjusted TP: trade would have hit adjusted TP
    # Otherwise: original outcome
    hits_adjusted = actual_mfe >= adjusted_tp
    adjusted_rr = np.where(
        hits_adjusted,
        adjusted_tp - cost_rr,  # hit adjusted TP, net of costs
        actual_rr,              # original outcome
    )
    adjusted_rr = np.clip(adjusted_rr, -1.0, 20.0)

    # Original metrics
    orig_sharpe = float(np.mean(actual_rr) / max(np.std(actual_rr), 1e-6))
    orig_wins = actual_rr[actual_rr > 0].sum()
    orig_losses = abs(actual_rr[actual_rr < 0].sum())
    orig_pf = float(orig_wins / max(orig_losses, 0.001))
    orig_wr = float((actual_rr > 0).mean())

    # Adjusted metrics
    adj_sharpe = float(np.mean(adjusted_rr) / max(np.std(adjusted_rr), 1e-6))
    adj_wins = adjusted_rr[adjusted_rr > 0].sum()
    adj_losses = abs(adjusted_rr[adjusted_rr < 0].sum())
    adj_pf = float(adj_wins / max(adj_losses, 0.001))
    adj_wr = float((adjusted_rr > 0).mean())

    # TP adjustment stats
    n_reduced = int(reduce_mask.sum())
    n_extended = int(extend_mask.sum())
    n_kept = n_trades - n_reduced - n_extended

    # MFE prediction quality
    pred_corr = float(np.corrcoef(predicted_mfe, actual_mfe)[0, 1]) if n_trades > 2 else 0.0
    rmse = float(np.sqrt(np.mean((predicted_mfe - actual_mfe) ** 2)))

    logger.info("  [tp] Original: Sharpe=%.3f PF=%.2f WR=%.1f%% | "
                "Adjusted: Sharpe=%.3f PF=%.2f WR=%.1f%%",
                orig_sharpe, orig_pf, 100 * orig_wr,
                adj_sharpe, adj_pf, 100 * adj_wr)
    logger.info("  [tp] TP changes: %d reduced, %d extended, %d kept | "
                "MFE corr=%.3f RMSE=%.3f",
                n_reduced, n_extended, n_kept, pred_corr, rmse)

    return {
        "n_trades": n_trades,
        "original_sharpe": orig_sharpe, "original_pf": orig_pf, "original_wr": orig_wr,
        "adjusted_sharpe": adj_sharpe, "adjusted_pf": adj_pf, "adjusted_wr": adj_wr,
        "sharpe_improvement": adj_sharpe - orig_sharpe,
        "pf_improvement": adj_pf - orig_pf,
        "n_reduced": n_reduced, "n_extended": n_extended, "n_kept": n_kept,
        "mfe_pred_corr": pred_corr, "mfe_rmse": rmse,
    }


def run_walk_forward_tp(classes: list[str] | None = None) -> list[dict]:
    """Walk-forward validation for TP optimization model."""
    logger.info("=" * 70)
    logger.info("TP OPTIMIZATION MODEL — Walk-Forward Validation")
    logger.info("=" * 70)

    results = _regression_walk_forward_skeleton(
        classes, _tp_labels, _tp_eval,
        task_name="tp",
        model_filename="rl_tp_optimizer.pkl",
        results_filename="tp_results.json",
    )

    if results:
        orig_pfs = [r.get("original_pf", 0) for r in results]
        adj_pfs = [r.get("adjusted_pf", 0) for r in results]
        orig_sharpes = [r.get("original_sharpe", 0) for r in results]
        adj_sharpes = [r.get("adjusted_sharpe", 0) for r in results]
        logger.info("=" * 70)
        logger.info("[tp] AGGREGATE: Original PF=%.2f Sharpe=%.3f | Adjusted PF=%.2f Sharpe=%.3f",
                    np.mean(orig_pfs), np.mean(orig_sharpes),
                    np.mean(adj_pfs), np.mean(adj_sharpes))
        gate = np.mean(adj_pfs) > np.mean(orig_pfs)
        logger.info("[tp] GATE %s: adjusted_pf > original_pf",
                    "PASSED" if gate else "FAILED")
    return results


# ===================================================================
#  Breakeven Management Model (Phase 4)
# ===================================================================

def derive_optimal_be_label(df: pd.DataFrame) -> np.ndarray:
    """
    Derive optimal BE ratchet level from MFE and outcome data.

    Logic:
    - Loss with MFE >= 1.5R: should have set BE at 1.0R
    - Loss with MFE >= 1.0R: should have set BE at 0.7R
    - Loss with MFE < 1.0R: no BE could help → 0.0
    - Win: protect at 40% of MFE (capped at 2.5R)
    - Breakeven: current strategy worked → 1.5R
    """
    outcome = df["label_outcome"].values
    mfe = df["label_max_favorable_rr"].values.astype(np.float32)

    be_label = np.zeros(len(df), dtype=np.float32)

    loss_mask = outcome == 2
    be_label[loss_mask & (mfe >= 1.5)] = 1.0
    be_label[loss_mask & (mfe >= 1.0) & (mfe < 1.5)] = 0.7
    # loss with MFE < 1.0 stays 0.0

    win_mask = outcome == 1
    be_label[win_mask] = np.clip(mfe[win_mask] * 0.4, 0.0, 2.5)

    be_mask = outcome == 3
    be_label[be_mask] = 1.5

    return be_label


def _be_eval(model, X_test, df_test, feat_names) -> dict[str, Any]:
    """Simulate BE management on OOS data. Compare model-BE vs no-BE vs fixed-BE."""
    predicted_be = model.predict(X_test).astype(np.float64)
    predicted_be = np.clip(predicted_be, 0.0, 3.0)

    actual_rr = np.clip(df_test["label_rr"].values.astype(np.float64), -1.0, 20.0)
    actual_mfe = df_test["label_max_favorable_rr"].values.astype(np.float64)
    cost_rr = df_test["label_cost_rr"].values.astype(np.float64)
    outcome = df_test["label_outcome"].values

    n_trades = len(actual_rr)
    if n_trades < 10:
        return {"n_trades": n_trades, "no_be_sharpe": 0.0, "model_be_sharpe": 0.0, "fixed_be_sharpe": 0.0}

    # --- No-BE baseline (original outcomes) ---
    no_be_rr = actual_rr.copy()

    # --- Fixed BE at 1.5R ---
    fixed_be_level = 1.5
    fixed_be_rr = actual_rr.copy()
    # Trades where MFE >= 1.5R but outcome is loss → saved by BE
    fixed_saves = (outcome == 2) & (actual_mfe >= fixed_be_level)
    fixed_be_rr[fixed_saves] = -cost_rr[fixed_saves]  # breakeven = lose only fees

    # --- Model BE ---
    model_be_rr = actual_rr.copy()
    # For each trade: if MFE >= predicted_be AND predicted_be > 0 AND outcome is loss → saved
    model_saves = (outcome == 2) & (predicted_be > 0.1) & (actual_mfe >= predicted_be)
    model_be_rr[model_saves] = -cost_rr[model_saves]

    def _metrics(rr, label):
        sharpe = float(np.mean(rr) / max(np.std(rr), 1e-6))
        wins = rr[rr > 0].sum()
        losses = abs(rr[rr < 0].sum())
        pf = float(wins / max(losses, 0.001))
        # Max drawdown (cumulative)
        cumsum = np.cumsum(rr)
        peak = np.maximum.accumulate(cumsum)
        dd = cumsum - peak
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0
        return {"sharpe": sharpe, "pf": pf, "max_dd": max_dd, "saves": 0}

    no_be = _metrics(no_be_rr, "no_be")
    fixed_be = _metrics(fixed_be_rr, "fixed_be")
    model_be = _metrics(model_be_rr, "model_be")

    n_fixed_saves = int(fixed_saves.sum())
    n_model_saves = int(model_saves.sum())
    n_losses = int((outcome == 2).sum())

    logger.info("  [be] No-BE: Sharpe=%.3f PF=%.2f DD=%.2f",
                no_be["sharpe"], no_be["pf"], no_be["max_dd"])
    logger.info("  [be] Fixed-1.5R: Sharpe=%.3f PF=%.2f DD=%.2f | saves=%d/%d losses",
                fixed_be["sharpe"], fixed_be["pf"], fixed_be["max_dd"],
                n_fixed_saves, n_losses)
    logger.info("  [be] Model-BE: Sharpe=%.3f PF=%.2f DD=%.2f | saves=%d/%d losses",
                model_be["sharpe"], model_be["pf"], model_be["max_dd"],
                n_model_saves, n_losses)

    return {
        "n_trades": n_trades, "n_losses": n_losses,
        "no_be_sharpe": no_be["sharpe"], "no_be_pf": no_be["pf"], "no_be_dd": no_be["max_dd"],
        "fixed_be_sharpe": fixed_be["sharpe"], "fixed_be_pf": fixed_be["pf"],
        "fixed_be_dd": fixed_be["max_dd"], "fixed_be_saves": n_fixed_saves,
        "model_be_sharpe": model_be["sharpe"], "model_be_pf": model_be["pf"],
        "model_be_dd": model_be["max_dd"], "model_be_saves": n_model_saves,
        "dd_improvement_vs_no_be": model_be["max_dd"] - no_be["max_dd"],
    }


def run_walk_forward_be(classes: list[str] | None = None) -> list[dict]:
    """Walk-forward validation for breakeven management model."""
    logger.info("=" * 70)
    logger.info("BREAKEVEN MANAGEMENT MODEL — Walk-Forward Validation")
    logger.info("=" * 70)

    results = _regression_walk_forward_skeleton(
        classes, derive_optimal_be_label, _be_eval,
        task_name="be",
        model_filename="rl_be_manager.pkl",
        results_filename="be_results.json",
    )

    if results:
        no_dds = [r.get("no_be_dd", 0) for r in results]
        model_dds = [r.get("model_be_dd", 0) for r in results]
        model_pfs = [r.get("model_be_pf", 0) for r in results]
        no_pfs = [r.get("no_be_pf", 0) for r in results]
        logger.info("=" * 70)
        logger.info("[be] AGGREGATE: No-BE DD=%.2f PF=%.2f | Model-BE DD=%.2f PF=%.2f",
                    np.mean(no_dds), np.mean(no_pfs),
                    np.mean(model_dds), np.mean(model_pfs))
        dd_improved = np.mean(model_dds) > np.mean(no_dds)  # less negative = better
        pf_ok = np.mean(model_pfs) >= 0.9 * np.mean(no_pfs)
        logger.info("[be] GATE %s: model_dd better AND model_pf >= 90%% of no_be_pf",
                    "PASSED" if (dd_improved and pf_ok) else "FAILED")
    return results


# ===================================================================
#  Exit Classifier Walk-Forward
# ===================================================================

def run_walk_forward_exit(
    classes: list[str] | None = None,
    exit_episodes_dir: str = "data/rl_training",
    min_trades: int = 100,
    min_auc: float = 0.52,
) -> dict[str, Any]:
    """
    Walk-forward validation for the early-exit classifier (5th model slot).

    Reads bar-by-bar exit episodes from {ac}_exit_episodes.parquet.
    Trains XGBoost to predict label_hold_better at each bar.

    Key constraints:
      - Train/test split by implied trade (grouped by symbol+direction+entry_price)
        so all bars of one trade stay on the same side of the split.
      - Recency weight: exponential decay by window index, applied per-trade uniformly.
      - Gate: AUC > min_auc on holdout AND old/new model agreement > 80% on HOLD preds.

    Saves: models/rl_exit_classifier.pkl (same format as other rl_*.pkl models)
    """
    if classes is None:
        classes = ALL_CLASSES

    logger.info("=" * 70)
    logger.info("EARLY-EXIT CLASSIFIER — Walk-Forward Validation")
    logger.info("=" * 70)

    episodes_dir = Path(exit_episodes_dir)
    all_frames: list[pd.DataFrame] = []

    for ac in classes:
        ep_path = episodes_dir / f"{ac}_exit_episodes.parquet"
        if not ep_path.exists():
            logger.warning("[exit] No episode file for %s: %s", ac, ep_path)
            continue
        try:
            df = pd.read_parquet(ep_path)
            df["asset_class"] = ac
            all_frames.append(df)
            logger.info("[exit] Loaded %d bar rows for %s", len(df), ac)
        except Exception as e:
            logger.warning("[exit] Failed to load %s: %s", ep_path, e)

    if not all_frames:
        logger.error("[exit] No episode data found. Run --exit-episodes first.")
        return {}

    episodes = pd.concat(all_frames, ignore_index=True)

    # Require label_hold_better to be present and binary
    if "label_hold_better" not in episodes.columns:
        logger.error("[exit] label_hold_better column missing from episodes")
        return {}

    episodes = episodes.dropna(subset=["label_hold_better"])
    episodes["label_hold_better"] = episodes["label_hold_better"].astype(int)

    n_total = len(episodes)
    n_hold_label = int((episodes["label_hold_better"] == 1).sum())
    balance = n_hold_label / max(n_total, 1)
    logger.info("[exit] %d bar rows | HOLD=%.1f%% EXIT=%.1f%%",
                n_total, 100 * balance, 100 * (1 - balance))

    # Build trade-level grouping key for split integrity
    # All bars of the same trade must stay on same split side
    group_keys = ["symbol", "asset_class", "direction", "entry_price", "window"]
    available_keys = [k for k in group_keys if k in episodes.columns]
    episodes["_trade_group"] = (
        episodes[available_keys].astype(str).agg("_".join, axis=1)
    )
    trade_ids = episodes["_trade_group"].unique()
    n_unique_trades = len(trade_ids)
    logger.info("[exit] Unique trade groups: %d", n_unique_trades)

    if n_unique_trades < min_trades:
        logger.warning(
            "[exit] Only %d trade groups (minimum %d) — not enough for reliable training. "
            "Run more paper trades or generate more historical episodes.",
            n_unique_trades, min_trades,
        )
        return {"status": "insufficient_data", "n_trades": n_unique_trades}

    # Time-ordered split: 80% train, 20% holdout — by window if available, else by trade group order
    if "window" in episodes.columns:
        all_windows = sorted(episodes["window"].unique())
        split_idx = max(1, int(len(all_windows) * 0.8))
        train_windows = set(all_windows[:split_idx])
        test_windows = set(all_windows[split_idx:])
        train_df = episodes[episodes["window"].isin(train_windows)].copy()
        test_df = episodes[episodes["window"].isin(test_windows)].copy()
    else:
        # Fallback: split by trade group order
        split = int(len(trade_ids) * 0.8)
        train_groups = set(trade_ids[:split])
        train_df = episodes[episodes["_trade_group"].isin(train_groups)].copy()
        test_df = episodes[~episodes["_trade_group"].isin(train_groups)].copy()

    logger.info("[exit] Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    if len(test_df) == 0 or len(train_df) == 0:
        logger.error("[exit] Empty train or test split")
        return {}

    # Recency weights: exponential decay by window, uniform per trade
    if "window" in train_df.columns:
        max_win = int(train_df["window"].max())
        decay_half_life = 200  # trades equivalent → scale by window count
        trade_weights = np.exp(
            (train_df["window"].values - max_win) / max(max_win * 0.3, 1)
        ).astype(np.float32)
    else:
        trade_weights = np.ones(len(train_df), dtype=np.float32)

    # Prepare features and labels
    X_train, feat_cols = prepare_features(train_df, task="early_exit")
    y_train = prepare_labels(train_df, task="early_exit")
    X_test, _ = prepare_features(test_df, task="early_exit")
    y_test = prepare_labels(test_df, task="early_exit")

    # Validation split from training (last 15% of train by window)
    val_size = max(1, int(len(X_train) * 0.15))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_tr = X_train[:-val_size]
    y_tr = y_train[:-val_size]
    w_tr = trade_weights[:-val_size]

    # Train XGBoost classifier
    model = train_xgboost(X_tr, y_tr, X_val, y_val, feat_cols,
                          sample_weights=w_tr, task="early_exit")

    if model is None:
        logger.error("[exit] Model training failed")
        return {}

    # Evaluate on holdout
    from sklearn.metrics import roc_auc_score, f1_score
    test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_proba))
    test_pred = (test_proba >= 0.5).astype(int)
    test_f1 = float(f1_score(y_test, test_pred, zero_division=0))

    logger.info("[exit] Holdout AUC=%.4f F1=%.4f", test_auc, test_f1)

    # Label balance sanity check
    hold_rate = float(test_pred.mean())
    logger.info("[exit] Predicted HOLD rate on holdout: %.1f%%", hold_rate * 100)

    # Gate: AUC must exceed threshold
    gate_passed = test_auc >= min_auc
    logger.info("[exit] GATE %s: AUC %.4f %s %.4f",
                "PASSED" if gate_passed else "FAILED",
                test_auc, ">=" if gate_passed else "<", min_auc)

    if not gate_passed:
        logger.warning("[exit] Gate failed — model NOT saved. Collect more data.")
        return {
            "status": "gate_failed", "auc": test_auc, "f1": test_f1,
            "n_trades": n_unique_trades,
        }

    # Load existing model for agreement check (if exists)
    existing_path = MODEL_DIR / "rl_exit_classifier.pkl"
    if existing_path.exists():
        try:
            with open(existing_path, "rb") as f:
                old_data = pickle.load(f)
            old_model = old_data.get("model")
            if old_model is not None:
                old_feat = old_data.get("feat_names", feat_cols)
                X_agree = np.array([
                    [X_test[j, feat_cols.index(f)] if f in feat_cols else 0.0
                     for f in old_feat]
                    for j in range(len(X_test))
                ], dtype=np.float32)
                old_pred = (old_model.predict_proba(X_agree)[:, 1] >= 0.5).astype(int)
                agreement = float((old_pred == test_pred).mean())
                logger.info("[exit] Model agreement with previous: %.1f%%", agreement * 100)
                if agreement < 0.80:
                    logger.warning(
                        "[exit] Agreement %.1f%% < 80%% — model NOT deployed (too different)",
                        agreement * 100,
                    )
                    return {
                        "status": "agreement_failed", "auc": test_auc,
                        "agreement": agreement, "n_trades": n_unique_trades,
                    }
        except Exception as e:
            logger.warning("[exit] Could not load existing model for agreement check: %s", e)

    # Save model
    from features.feature_extractor import SCHEMA_VERSION as _EXIT_SCHEMA
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": model,
        "feat_names": feat_cols,
        "clip_ranges": CLIP_RANGES,
        "schema_version": _EXIT_SCHEMA,
        "auc": test_auc,
        "f1": test_f1,
        "n_train": len(X_tr),
        "n_test": len(X_test),
        "n_trades": n_unique_trades,
        "task": "early_exit",
    }
    with open(existing_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info("[exit] Model saved → %s (AUC=%.4f)", existing_path, test_auc)

    return {
        "status": "deployed", "auc": test_auc, "f1": test_f1,
        "n_trades": n_unique_trades, "hold_rate": hold_rate,
    }


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


class RLBrainSuite:
    """Unified inference for all RL models (entry filter + TP optimizer + BE manager)."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        rl_cfg = cfg.get("rl_brain", {})
        self.enabled = rl_cfg.get("enabled", True)

        # Entry filter
        ef_cfg = rl_cfg.get("entry_filter", {})
        self.entry_filter_enabled = ef_cfg.get("enabled", True)
        self.confidence_threshold = ef_cfg.get("confidence_threshold", 0.6)
        self._entry_filter = self._load_model(
            ef_cfg.get("model_path", "models/rl_entry_filter.pkl")
        ) if self.entry_filter_enabled else None

        # TP optimizer
        tp_cfg = rl_cfg.get("tp_optimizer", {})
        self.tp_enabled = tp_cfg.get("enabled", False)
        self.max_tp_reduction = tp_cfg.get("max_tp_reduction", 0.3)
        self.max_tp_extension = tp_cfg.get("max_tp_extension", 0.5)
        self.min_tp_rr = tp_cfg.get("min_tp_rr", 0.5)
        self._tp_model = self._load_model(
            tp_cfg.get("model_path", "models/rl_tp_optimizer.pkl")
        ) if self.tp_enabled else None

        # BE manager
        be_cfg = rl_cfg.get("be_manager", {})
        self.be_enabled = be_cfg.get("enabled", False)
        self.min_be_rr = be_cfg.get("min_be_rr", 0.5)
        self.be_fee_buffer = be_cfg.get("fee_buffer", True)
        self._be_model = self._load_model(
            be_cfg.get("model_path", "models/rl_be_manager.pkl")
        ) if self.be_enabled else None

        # Exit classifier (5th slot) — shadow mode until validated
        exit_cfg = rl_cfg.get("exit_classifier", {})
        self.exit_enabled = exit_cfg.get("enabled", False)
        self.exit_threshold = exit_cfg.get("confidence_threshold", 0.65)
        self._exit_model = self._load_model(
            exit_cfg.get("model_path", "models/rl_exit_classifier.pkl")
        ) if self.exit_enabled else None

        active = []
        if self._entry_filter: active.append("entry_filter")
        if self._tp_model: active.append("tp_optimizer")
        if self._be_model: active.append("be_manager")
        if self._exit_model: active.append("exit_classifier")
        logger.info("RLBrainSuite initialized: %s", active if active else "no models loaded")

        # Hot-swap: track model file mtimes for live reload
        self._model_paths: dict[str, str] = {
            "entry_filter": ef_cfg.get("model_path", "models/rl_entry_filter.pkl"),
            "tp_optimizer": tp_cfg.get("model_path", "models/rl_tp_optimizer.pkl"),
            "be_manager": be_cfg.get("model_path", "models/rl_be_manager.pkl"),
            "exit_classifier": exit_cfg.get("model_path", "models/rl_exit_classifier.pkl"),
        }
        self._model_mtimes: dict[str, float] = {}
        for name, mpath in self._model_paths.items():
            p = Path(mpath)
            if p.exists():
                self._model_mtimes[mpath] = p.stat().st_mtime

        # Continuous learner config for rollback
        cl_cfg = cfg.get("continuous_learner", {})
        self._rollback_min_trades = cl_cfg.get("rollback_min_trades", 20)
        self._rollback_pnl_threshold = cl_cfg.get("rollback_cumulative_pnl", -0.03)
        self._rollback_winrate_threshold = cl_cfg.get("rollback_min_winrate", 0.20)

    @staticmethod
    def _load_model(path_str: str) -> dict | None:
        path = Path(path_str)
        if not path.exists():
            logger.warning("Model not found: %s", path)
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            logger.error("Corrupt model file %s: %s -- skipping", path, exc)
            return None
        # Schema version gate: refuse to load if model was trained with different schema
        model_sv = data.get("schema_version")
        if model_sv is not None and model_sv != _SCHEMA_VERSION:
            raise RuntimeError(
                f"Model-schema version mismatch: model has schema_version={model_sv}, "
                f"code expects schema_version={_SCHEMA_VERSION}. "
                f"Retrain the model or revert the schema change."
            )
        logger.info("Loaded model: %s (%d features, schema_v%s)", path, len(data.get("feat_names", [])), model_sv or "?")
        return data

    def check_and_reload_models(self) -> None:
        """Check if model files have been updated on disk and reload if so.
        Called every 60 seconds from the live bot's main loop.
        Also checks the rollback marker and triggers rollback if conditions met.
        """
        import json as _json

        slot_to_attr = {
            "entry_filter": "_entry_filter",
            "tp_optimizer": "_tp_model",
            "be_manager": "_be_model",
            "exit_classifier": "_exit_model",
        }

        for slot_name, model_path_str in self._model_paths.items():
            p = Path(model_path_str)
            if not p.exists():
                continue

            current_mtime = p.stat().st_mtime
            stored_mtime = self._model_mtimes.get(model_path_str, 0.0)

            if current_mtime <= stored_mtime:
                continue

            # File has changed -- attempt reload
            attr_name = slot_to_attr.get(slot_name)
            if attr_name is None:
                continue

            old_data = getattr(self, attr_name, None)
            old_version = old_data.get("version", "?") if old_data else "?"

            try:
                new_data = self._load_model(model_path_str)
                if new_data is None:
                    logger.error("[ML] Failed to reload %s -- keeping old model", model_path_str)
                    continue

                # Verify the model can produce predictions (basic sanity check)
                feat_names = new_data.get("feat_names", [])
                if not feat_names:
                    logger.error("[ML] Reloaded model %s has no feat_names -- keeping old", model_path_str)
                    continue

                new_version = new_data.get("version", "?")
                setattr(self, attr_name, new_data)
                self._model_mtimes[model_path_str] = current_mtime
                logger.info("[ML] Model reloaded: %s v%s->v%s", p.name, old_version, new_version)

            except Exception as exc:
                logger.error("[ML] Error reloading %s: %s -- keeping old model", model_path_str, exc)

        # Check rollback marker
        rollback_marker = Path("models/.rollback_watch")
        if not rollback_marker.exists():
            return

        try:
            with open(rollback_marker) as f:
                marker = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return

        # Rollback evaluation is done by the live bot which tracks trades
        # under the new model.  This method is a hook point -- the live bot
        # should call _check_rollback_conditions() with its accumulated stats.

    def check_rollback_conditions(
        self,
        cumulative_pnl: float,
        win_rate: float,
        n_trades: int,
    ) -> bool:
        """Evaluate whether the current model should be rolled back.

        Called by the live bot after each trade close, with cumulative
        stats since the last model deployment.

        Returns True if rollback was triggered and executed.
        """
        import json as _json

        rollback_marker = Path("models/.rollback_watch")
        if not rollback_marker.exists():
            return False

        try:
            with open(rollback_marker) as f:
                marker = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return False

        min_trades = marker.get("min_trades", self._rollback_min_trades)
        if n_trades < min_trades:
            return False

        pnl_threshold = marker.get("cumulative_pnl_threshold", self._rollback_pnl_threshold)
        wr_threshold = marker.get("min_winrate_threshold", self._rollback_winrate_threshold)

        should_rollback = (cumulative_pnl < pnl_threshold) or (win_rate < wr_threshold)

        if not should_rollback:
            return False

        logger.warning(
            "[ML] ROLLBACK TRIGGERED: cumPnL=%.2f%% (threshold=%.2f%%), "
            "WR=%.1f%% (threshold=%.1f%%), trades=%d",
            cumulative_pnl * 100, pnl_threshold * 100,
            win_rate * 100, wr_threshold * 100, n_trades,
        )

        prev_version = marker.get("prev_version")
        if prev_version is None:
            logger.error("[ML] No prev_version in rollback marker -- cannot rollback")
            return False

        # Execute rollback for all model slots
        try:
            for slot_name in ("entry_filter", "exit_classifier"):
                model_path = self._model_paths.get(slot_name)
                if not model_path:
                    continue
                mp = Path(model_path)
                # Try _prev.pkl first (backed up by continuous_learner before each retrain)
                prev_path = mp.with_name(mp.stem + "_prev" + mp.suffix)
                # Fallback: _v1.pkl (original known-good baseline)
                v1_path = mp.with_name(mp.stem + "_v1" + mp.suffix)
                source = None
                if prev_path.exists():
                    source = prev_path
                elif v1_path.exists():
                    source = v1_path
                if source is not None:
                    import shutil
                    shutil.copy2(source, mp)
                    logger.info("[ML] Rolled back %s from %s", mp.name, source.name)
                else:
                    logger.warning("[ML] No rollback source for %s (tried %s, %s)",
                                   mp.name, prev_path.name, v1_path.name)

            # Remove rollback marker
            rollback_marker.unlink(missing_ok=True)

            # Force reload models
            self.check_and_reload_models()

            logger.info("[ML] Rollback complete -- models restored to previous versions")
            return True

        except Exception as exc:
            logger.error("[ML] Rollback execution failed: %s", exc)
            return False

    def _build_features(self, features: dict[str, float], model_data: dict) -> np.ndarray:
        """Build feature vector from dict, matching model's expected feature order."""
        feat_names = model_data["feat_names"]
        x = np.array([features.get(f, 0.0) for f in feat_names],
                      dtype=np.float32).reshape(1, -1)
        for col_name, (lo, hi) in model_data.get("clip_ranges", {}).items():
            if col_name in feat_names:
                idx = feat_names.index(col_name)
                x[0, idx] = np.clip(x[0, idx], lo, hi)
        return np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)

    def predict_entry(self, features: dict[str, float]) -> tuple[bool, float]:
        """Predict whether to take entry. Returns (take, confidence)."""
        if not self.enabled or self._entry_filter is None:
            return True, 1.0  # pass-through if disabled

        x = self._build_features(features, self._entry_filter)
        proba = self._entry_filter["model"].predict_proba(x)[0]
        confidence = float(proba[1])  # P(win)
        return confidence >= self.confidence_threshold, confidence

    def predict_tp_adjustment(
        self, features: dict[str, float], planned_tp_rr: float,
    ) -> float:
        """Predict adjusted TP level. Returns adjusted TP in R-multiples."""
        if not self.enabled or self._tp_model is None or planned_tp_rr <= 0:
            return planned_tp_rr  # pass-through

        x = self._build_features(features, self._tp_model)
        predicted_mfe = float(self._tp_model["model"].predict(x)[0])
        predicted_mfe = max(predicted_mfe, 0.1)

        adjusted = planned_tp_rr
        if predicted_mfe < planned_tp_rr * 0.7:
            # MFE much lower than planned → reduce TP
            adjusted = max(predicted_mfe * 0.85, self.min_tp_rr)
            # Clamp reduction
            min_tp = planned_tp_rr * (1.0 - self.max_tp_reduction)
            adjusted = max(adjusted, min_tp)
        elif predicted_mfe > planned_tp_rr * 1.3:
            # MFE much higher → extend TP
            adjusted = min(predicted_mfe * 0.8, planned_tp_rr * (1.0 + self.max_tp_extension))

        return adjusted

    def predict_be_level(
        self, features: dict[str, float], cost_rr: float = 0.0,
    ) -> float:
        """Predict optimal BE ratchet level. Returns RR at which to move SL to entry.
        0.0 means no BE (let trade play out). >0 means move to BE when unrealized RR reaches this."""
        if not self.enabled or self._be_model is None:
            return 0.0  # no BE

        x = self._build_features(features, self._be_model)
        predicted_be = float(self._be_model["model"].predict(x)[0])
        predicted_be = max(predicted_be, 0.0)

        if predicted_be < self.min_be_rr:
            return 0.0  # below threshold, don't use BE

        # Add fee buffer if configured
        if self.be_fee_buffer and cost_rr > 0:
            predicted_be = max(predicted_be, cost_rr * 2.0)

        return min(predicted_be, 3.0)  # cap at 3R

    def predict_early_exit(
        self, bar_features: dict[str, float],
    ) -> tuple[bool, float]:
        """
        Predict whether to exit the trade early at the current bar.

        Returns (should_exit_now: bool, confidence: float).

        Safe defaults when model is unavailable:
          - Returns (False, 0.0) → HOLD, no action taken.
          - This is the correct shadow-mode behavior: log but don't act.

        bar_features dict should include:
          bars_held, bar_unrealized_rr, sl_distance_pct, max_favorable_seen,
          be_triggered, asset_class_id (and any overlap with entry features)
        """
        if not self.enabled or self._exit_model is None:
            return False, 0.0  # shadow mode / model not yet trained

        try:
            # Schema version check: warn if model was trained on different feature schema
            model_schema = self._exit_model.get("schema_version")
            if model_schema is not None:
                from features.feature_extractor import FeatureExtractor
                FeatureExtractor.check_schema(model_schema)

            x = self._build_features(bar_features, self._exit_model)
            proba = self._exit_model["model"].predict_proba(x)[0]
            # proba[1] = P(label_hold_better=1) = P(HOLD is better)
            # We EXIT when P(HOLD) < threshold, i.e., P(EXIT) > (1 - threshold)
            p_hold = float(proba[1])
            should_exit = p_hold < (1.0 - self.exit_threshold)
            return should_exit, 1.0 - p_hold  # confidence = P(EXIT)
        except Exception as exc:
            logger.warning("predict_early_exit failed: %s", exc)
            return False, 0.0  # safe fallback


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
    parser.add_argument("--task", choices=["binary", "direction", "entry_quality",
                                          "sizing", "tp", "be", "all_regression",
                                          "exit"],
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
        if args.task == "sizing":
            run_walk_forward_sizing(classes=args.classes)
        elif args.task == "tp":
            run_walk_forward_tp(classes=args.classes)
        elif args.task == "be":
            run_walk_forward_be(classes=args.classes)
        elif args.task == "exit":
            run_walk_forward_exit(classes=args.classes)
        elif args.task == "all_regression":
            logger.info("Training ALL regression models sequentially...")
            run_walk_forward_sizing(classes=args.classes)
            gc.collect()
            run_walk_forward_tp(classes=args.classes)
            gc.collect()
            run_walk_forward_be(classes=args.classes)
        elif args.walk_forward:
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
