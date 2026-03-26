"""
═══════════════════════════════════════════════════════════════════
 rl_brain_v2.py  —  XGBoost + NN Trade Decision Model
 ─────────────────────────────────────────────────────
 Learns which SMC market states lead to profitable trades.

 Training: Offline on historical data (causal features, lookahead labels)
 Inference: Real-time prediction from causal features only

 Usage:
     python3 -m rl_brain_v2 --train --walk-forward
     python3 -m rl_brain_v2 --train --model xgboost
     python3 -m rl_brain_v2 --evaluate
═══════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = Path("data/rl_training")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("backtest/results/rl")

# Features that should NOT be used as model input
META_COLS = {"timestamp", "symbol", "asset_class", "window",
             "label_action", "label_rr", "label_outcome", "label_profitable"}

# Asset class encoding for model
ASSET_CLASS_MAP = {"crypto": 0, "forex": 1, "stocks": 2, "commodities": 3}


# ═══════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_training_data(
    classes: list[str] | None = None,
    subsample_notrade: float = 2.0,
) -> pd.DataFrame:
    """Load all RL training parquets, subsampling no-trade bars to fit in RAM."""
    if classes is None:
        classes = ["crypto", "forex", "stocks", "commodities"]

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

            # Subsample no-trade bars PER CLASS during loading
            if subsample_notrade > 0:
                entries = df[df["label_action"] > 0]
                no_trade = df[df["label_action"] == 0]
                max_nt = int(len(entries) * subsample_notrade)
                if len(no_trade) > max_nt:
                    no_trade = no_trade.sample(n=max_nt, random_state=42)
                df = pd.concat([entries, no_trade], ignore_index=True)

            logger.info("Loaded %s: %d→%d samples (%d entries)",
                        ac, raw_len, len(df), n_ent)
            dfs.append(df)
            del entries, no_trade
            gc.collect()
        else:
            logger.warning("No data for %s at %s", ac, path)

    if not dfs:
        raise FileNotFoundError(f"No training data found in {DATA_DIR}")

    combined = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()
    logger.info("Total: %d raw → %d subsampled, %d entries (%.1f%%)",
                total_raw, len(combined), total_entries,
                100 * total_entries / max(total_raw, 1))
    return combined


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract feature matrix from DataFrame."""
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values.astype(np.float32)
    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    return X, feat_cols


def prepare_labels(df: pd.DataFrame, task: str = "binary") -> np.ndarray:
    """
    Prepare labels based on task:
    - "binary": profitable (1) vs not (0)
    - "multiclass": no_trade (0), long_win (1), short_win (2), long_loss (3), short_loss (4)
    - "direction": no_trade (0), long (1), short (2)
    """
    if task == "binary":
        return df["label_profitable"].values.astype(np.int32)
    elif task == "direction":
        return df["label_action"].values.astype(np.int32)
    elif task == "multiclass":
        action = df["label_action"].values
        outcome = df["label_outcome"].values
        labels = np.zeros(len(df), dtype=np.int32)
        # 1 = profitable long, 2 = profitable short
        labels[(action == 1) & (outcome == 1)] = 1
        labels[(action == 2) & (outcome == 1)] = 2
        # 3 = losing long, 4 = losing short
        labels[(action == 1) & (outcome == 2)] = 3
        labels[(action == 2) & (outcome == 2)] = 4
        return labels
    else:
        raise ValueError(f"Unknown task: {task}")


# ═══════════════════════════════════════════════════════════════════
#  XGBoost Model
# ═══════════════════════════════════════════════════════════════════

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feat_names: list[str],
    sample_weights: np.ndarray | None = None,
    task: str = "binary",
) -> Any:
    """Train XGBoost classifier with early stopping."""
    import xgboost as xgb

    n_classes = len(np.unique(y_train))
    if task == "binary" or n_classes == 2:
        objective = "binary:logistic"
        eval_metric = "auc"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    # Handle class imbalance
    if task == "binary":
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale = n_neg / max(n_pos, 1)
    else:
        scale = 1.0

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale if task == "binary" else 1.0,
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


# ═══════════════════════════════════════════════════════════════════
#  Evaluation & Diagnostics
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    feat_names: list[str],
    task: str = "binary",
) -> dict[str, Any]:
    """Comprehensive model evaluation with trading-specific metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics: dict[str, Any] = {}

    # Standard ML metrics
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    if task == "binary":
        metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_test, y_pred, zero_division=0)
        try:
            metrics["auc"] = roc_auc_score(y_test, y_proba[:, 1])
        except (ValueError, IndexError):
            metrics["auc"] = 0.0

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    # Trading-specific metrics: simulate trades based on predictions
    trade_mask = y_pred == 1 if task == "binary" else y_pred > 0
    n_predicted_trades = int(trade_mask.sum())
    metrics["n_predicted_trades"] = n_predicted_trades

    if n_predicted_trades > 0:
        trade_df = df_test[trade_mask].copy()
        actual_outcomes = trade_df["label_outcome"].values
        actual_rr = trade_df["label_rr"].values

        n_actual_win = int((actual_outcomes == 1).sum())
        n_actual_loss = int((actual_outcomes == 2).sum())
        n_actual_be = int((actual_outcomes == 3).sum())
        n_no_trade = int((actual_outcomes == 0).sum())

        metrics["predicted_trades"] = {
            "win": n_actual_win,
            "loss": n_actual_loss,
            "be": n_actual_be,
            "no_trade_bars": n_no_trade,
            "total": n_predicted_trades,
        }

        real_trades = n_actual_win + n_actual_loss
        if real_trades > 0:
            metrics["trading_winrate"] = n_actual_win / real_trades
            win_rr = actual_rr[actual_outcomes == 1]
            loss_rr = actual_rr[actual_outcomes == 2]
            metrics["avg_win_rr"] = float(win_rr.mean()) if len(win_rr) > 0 else 0
            metrics["avg_loss_rr"] = float(loss_rr.mean()) if len(loss_rr) > 0 else 0

            total_win = win_rr.sum() if len(win_rr) > 0 else 0
            total_loss = abs(loss_rr.sum()) if len(loss_rr) > 0 else 0
            metrics["profit_factor"] = float(total_win / max(total_loss, 0.001))
            metrics["expectancy"] = float(actual_rr[actual_outcomes != 0].mean())

    # Per-asset-class breakdown
    per_class: dict[str, dict] = {}
    for ac in df_test["asset_class"].unique():
        ac_mask = (df_test["asset_class"] == ac).values & trade_mask
        n_ac = int(ac_mask.sum())
        if n_ac == 0:
            per_class[ac] = {"trades": 0}
            continue
        ac_df = df_test[ac_mask]
        ac_out = ac_df["label_outcome"].values
        ac_rr = ac_df["label_rr"].values
        w = int((ac_out == 1).sum())
        l = int((ac_out == 2).sum())
        per_class[ac] = {
            "trades": n_ac,
            "wins": w, "losses": l,
            "winrate": w / max(w + l, 1),
            "avg_rr": float(ac_rr[ac_out != 0].mean()) if (ac_out != 0).any() else 0,
            "pf": float(ac_rr[ac_out == 1].sum() / max(abs(ac_rr[ac_out == 2].sum()), 0.001))
                if (ac_out == 1).any() else 0,
        }
    metrics["per_class"] = per_class

    # Per-symbol top performers
    if n_predicted_trades > 0:
        sym_stats = trade_df.groupby("symbol").agg(
            trades=("label_action", "count"),
            wins=("label_profitable", "sum"),
            avg_rr=("label_rr", "mean"),
        )
        sym_stats["winrate"] = sym_stats["wins"] / sym_stats["trades"]
        metrics["top_symbols"] = sym_stats.sort_values("wins", ascending=False).head(10).to_dict()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        metrics["feature_importance_top20"] = fi[:20]

    return metrics


def print_diagnostics(metrics: dict[str, Any], label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    logger.info("═══ %s EVALUATION ═══", label.upper() if label else "MODEL")

    logger.info("ML Metrics:")
    logger.info("  Accuracy:  %.3f", metrics.get("accuracy", 0))
    logger.info("  Precision: %.3f", metrics.get("precision", 0))
    logger.info("  Recall:    %.3f", metrics.get("recall", 0))
    logger.info("  F1:        %.3f", metrics.get("f1", 0))
    logger.info("  AUC:       %.3f", metrics.get("auc", 0))
    logger.info("  Confusion: %s", metrics.get("confusion_matrix", []))

    logger.info("Trading Metrics:")
    logger.info("  Predicted trades: %d", metrics.get("n_predicted_trades", 0))
    pt = metrics.get("predicted_trades", {})
    if pt:
        logger.info("    Win: %d, Loss: %d, BE: %d, NoTrade bars: %d",
                     pt.get("win", 0), pt.get("loss", 0),
                     pt.get("be", 0), pt.get("no_trade_bars", 0))
    logger.info("  Win Rate:   %.1f%%", 100 * metrics.get("trading_winrate", 0))
    logger.info("  Avg Win RR: %.2f", metrics.get("avg_win_rr", 0))
    logger.info("  Avg Loss RR: %.2f", metrics.get("avg_loss_rr", 0))
    logger.info("  PF:          %.2f", metrics.get("profit_factor", 0))
    logger.info("  Expectancy:  %.3f", metrics.get("expectancy", 0))

    logger.info("Per Asset Class:")
    for ac, stats in metrics.get("per_class", {}).items():
        if stats.get("trades", 0) > 0:
            logger.info("  %s: %d trades, WR=%.0f%%, PF=%.2f, avgRR=%.2f",
                         ac, stats["trades"], 100 * stats.get("winrate", 0),
                         stats.get("pf", 0), stats.get("avg_rr", 0))

    logger.info("Feature Importance (top 10):")
    for feat, imp in metrics.get("feature_importance_top20", [])[:10]:
        logger.info("  %-25s %.4f", feat, imp)


# ═══════════════════════════════════════════════════════════════════
#  Walk-Forward Training
# ═══════════════════════════════════════════════════════════════════

def _subsample_notrade(df: pd.DataFrame, ratio: float = 3.0) -> pd.DataFrame:
    """Subsample no-trade bars to ratio:1 vs entry bars. Saves RAM."""
    entries = df[df["label_action"] > 0]
    no_trade = df[df["label_action"] == 0]
    max_notrade = int(len(entries) * ratio)
    if len(no_trade) > max_notrade:
        no_trade = no_trade.sample(n=max_notrade, random_state=42)
        logger.info("  Subsampled no-trade: %d → %d (%.0f:1 ratio)",
                     int((df["label_action"] == 0).sum()), len(no_trade), ratio)
    return pd.concat([entries, no_trade], ignore_index=True)


def run_walk_forward(
    classes: list[str] | None = None,
    task: str = "binary",
) -> dict[str, Any]:
    """
    Walk-forward training with expanding window:
    - Train on W0..W(n-2), validate on W(n-1), test on W(n)
    - With 12 windows: Train W0-W9, Val W10, Test W11
    - Subsamples no-trade bars to fit in 8GB RAM
    """
    data = load_training_data(classes)

    # Discover all windows present in data
    all_windows = sorted(data["window"].unique())
    n_win = len(all_windows)
    logger.info("Found %d windows: %s", n_win, list(all_windows))

    if n_win < 3:
        logger.error("Need at least 3 windows for walk-forward (train/val/test)!")
        return {}

    # Print per-window stats
    for w_id in all_windows:
        w = data[data["window"] == w_id]
        n_ent = int((w["label_action"] > 0).sum())
        n_win_t = int((w["label_outcome"] == 1).sum())
        n_loss = int((w["label_outcome"] == 2).sum())
        logger.info("  W%d: %d samples, %d entries, %d win, %d loss (WR=%.0f%%)",
                     w_id, len(w), n_ent, n_win_t, n_loss,
                     100 * n_win_t / max(n_win_t + n_loss, 1))

    # Expanding window: train on all but last 2, val on second-to-last, test on last
    train_windows = all_windows[:-2]
    val_window = all_windows[-2]
    test_window = all_windows[-1]

    logger.info("Split: Train W%s, Val W%d, Test W%d",
                [int(w) for w in train_windows], int(val_window), int(test_window))

    # Split into train/val/test
    train_data = data[data["window"].isin(train_windows)].copy()
    val_data = data[data["window"] == val_window].copy()
    test_data = data[data["window"] == test_window].copy()
    del data; gc.collect()

    logger.info("Sizes: Train=%d, Val=%d, Test=%d",
                len(train_data), len(val_data), len(test_data))

    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        logger.error("Not enough data for walk-forward!")
        return {}

    X_train, feat_names = prepare_features(train_data)
    y_train = prepare_labels(train_data, task)
    X_val, _ = prepare_features(val_data)
    y_val = prepare_labels(val_data, task)
    X_test, _ = prepare_features(test_data)
    y_test = prepare_labels(test_data, task)

    # Sample weights: RR-weighted for entries, low weight for no-trade
    weights = np.ones(len(train_data), dtype=np.float32)
    entry_mask = train_data["label_action"].values > 0
    rr_vals = np.abs(train_data["label_rr"].values)
    weights[entry_mask] = np.clip(rr_vals[entry_mask], 0.5, 5.0)
    weights[~entry_mask] = 0.1

    # Free DataFrames after extracting arrays
    train_meta = train_data[list(META_COLS & set(train_data.columns))].copy()
    val_meta = val_data[list(META_COLS & set(val_data.columns))].copy()
    test_meta = test_data  # Keep test_data for evaluation
    del train_data, val_data; gc.collect()

    logger.info("Training XGBoost (task=%s)...", task)
    model = train_xgboost(X_train, y_train, X_val, y_val,
                           feat_names, sample_weights=weights, task=task)

    # Evaluate on test window (unseen)
    logger.info("Evaluating on W%d (out-of-sample)...", int(test_window))
    metrics = evaluate_model(model, X_test, y_test, test_meta, feat_names, task)
    print_diagnostics(metrics, f"W{int(test_window)} OOS")

    # Also evaluate on training data for overfitting check
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)
    from sklearn.metrics import accuracy_score, roc_auc_score
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = metrics.get("accuracy", 0)
    try:
        train_auc = roc_auc_score(y_train, train_proba[:, 1]) if task == "binary" else 0
    except Exception:
        train_auc = 0
    test_auc = metrics.get("auc", 0)
    logger.info("Overfitting check: Train ACC=%.3f AUC=%.3f, Test ACC=%.3f AUC=%.3f",
                train_acc, train_auc, test_acc, test_auc)
    del X_train, y_train, X_val, y_val, train_pred, train_proba, weights; gc.collect()

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "rl_brain_v2_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feat_names": feat_names, "task": task}, f)
    logger.info("Model saved: %s", model_path)

    # Save metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_clean = json.loads(json.dumps(metrics, default=_convert))
    with open(RESULTS_DIR / "evaluation_w2.json", "w") as f:
        json.dump(metrics_clean, f, indent=2)

    # Save feature importance
    fi = metrics.get("feature_importance_top20", [])
    if fi:
        fi_df = pd.DataFrame(fi, columns=["feature", "importance"])
        fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
        logger.info("Feature importance saved: %s",
                     RESULTS_DIR / "feature_importance.csv")

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  Prediction Interface (for live/paper trading)
# ═══════════════════════════════════════════════════════════════════

class RLBrainV2:
    """Inference wrapper for the trained model."""

    def __init__(self, model_path: str | Path = "models/rl_brain_v2_xgb.pkl"):
        self.model = None
        self.feat_names: list[str] = []
        self.task = "binary"

        path = Path(model_path)
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.feat_names = data["feat_names"]
            self.task = data.get("task", "binary")
            logger.info("RLBrainV2 loaded from %s (%d features, task=%s)",
                        path, len(self.feat_names), self.task)
        else:
            logger.warning("No model found at %s — predictions disabled", path)

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
        x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)

        proba = self.model.predict_proba(x)[0]

        if self.task == "binary":
            confidence = float(proba[1])  # P(profitable)
            if confidence >= 0.6:
                # Use daily bias from features to determine direction
                bias = features.get("daily_bias", 0)
                if bias > 0:
                    return "long", confidence, 0.0
                elif bias < 0:
                    return "short", confidence, 0.0
            return "no_trade", confidence, 0.0

        elif self.task == "direction":
            # proba: [no_trade, long, short]
            action_idx = int(np.argmax(proba))
            confidence = float(proba[action_idx])
            actions = ["no_trade", "long", "short"]
            return actions[action_idx], confidence, 0.0

        return "no_trade", 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Brain V2 — Train & Evaluate")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Use walk-forward validation")
    parser.add_argument("--model", choices=["xgboost", "nn"], default="xgboost",
                        help="Model type (default: xgboost)")
    parser.add_argument("--task", choices=["binary", "direction"], default="binary",
                        help="Prediction task (default: binary)")
    parser.add_argument("--classes", nargs="+",
                        choices=["crypto", "forex", "stocks", "commodities"],
                        help="Asset classes to include")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate existing model")
    args = parser.parse_args()

    Path("backtest/results").mkdir(parents=True, exist_ok=True)

    if args.train:
        if args.walk_forward:
            run_walk_forward(classes=args.classes, task=args.task)
        else:
            logger.info("Use --walk-forward for proper validation")
            run_walk_forward(classes=args.classes, task=args.task)
    elif args.evaluate:
        brain = RLBrainV2()
        if brain.model is not None:
            data = load_training_data(args.classes)
            w2 = data[data["window"] == 2]
            X, feat_names = prepare_features(w2)
            y = prepare_labels(w2, brain.task)
            metrics = evaluate_model(brain.model, X, y, w2, feat_names, brain.task)
            print_diagnostics(metrics, "EVALUATE")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
