"""
continuous_learner.py — Continuous XGBoost retraining from live trades.

Reads closed-trade data from the trade journal + teacher feedback,
combines with subsampled backtest data, and retrains the entry_quality
XGBoost model from scratch. The new model is written atomically for
hot-reload by rl_brain_v2.py.

Usage:
    Launched as an asyncio task from live_multi_bot.py:
        asyncio.create_task(run_continuous_learner(config, shutdown_event))
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pickle
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.schema import ENTRY_QUALITY_FEATURES, SCHEMA_VERSION

logger = logging.getLogger(__name__)

# ===================================================================
#  Constants (mirrored from rl_brain_v2.py)
# ===================================================================

MODEL_DIR = Path("models")
DATA_DIR = Path("data/rl_training")
META_PATH = MODEL_DIR / "rl_entry_filter_meta.json"

DEAD_FEATURES: set[str] = {"bias_strong", "daily_bias"}

CLIP_RANGES: dict[str, tuple[float, float]] = {
    "ema20_dist_5m": (-3.0, 3.0),
    "ema50_dist_5m": (-3.0, 3.0),
    "ema20_dist_1h": (-3.0, 3.0),
    "ema50_dist_1h": (-3.0, 3.0),
}

ASSET_CLASS_MAP: dict[str, int] = {
    "crypto": 0, "forex": 1, "stocks": 2, "commodities": 3,
}

ALL_CLASSES = ["crypto", "forex", "stocks", "commodities"]

FEEDBACK_PATH = Path("live_results/teacher_feedback.jsonl")

DEFAULT_GRADE_WEIGHTS: dict[str, float] = {
    "A+": 1.5, "A": 1.5, "B+": 1.0, "B": 1.0, "C": 1.0, "D": 1.5,
}


# ===================================================================
#  1. Collect live trade data from journal + teacher feedback
# ===================================================================

def collect_live_data(
    db_path: str,
    feedback_path: str | Path = FEEDBACK_PATH,
    grade_weights: dict[str, float] | None = None,
    timeout_weight: float = 0.5,
) -> pd.DataFrame:
    """Load closed trades from the journal DB with teacher-grade weighting.

    Opens a SEPARATE read-only SQLite connection. Never writes to the DB.
    Returns a DataFrame with the 41 ENTRY_QUALITY_FEATURES + label + sample_weight,
    or an empty DataFrame if no usable data.
    """
    if grade_weights is None:
        grade_weights = DEFAULT_GRADE_WEIGHTS

    feedback_path = Path(feedback_path)

    # ── Load teacher feedback grades keyed by trade_id ────────────
    grade_lookup: dict[str, str] = {}
    if feedback_path.exists():
        try:
            with open(feedback_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        tid = rec.get("trade_id", "")
                        grade = rec.get("analysis", {}).get("grade", "")
                        if tid and grade:
                            grade_lookup[tid] = grade
                    except json.JSONDecodeError:
                        continue
            logger.info("Loaded %d teacher grades from %s", len(grade_lookup), feedback_path)
        except OSError as exc:
            logger.warning("Cannot read feedback file %s: %s", feedback_path, exc)

    # ── Query journal DB (read-only) ─────────────────────────────
    if not Path(db_path).exists():
        logger.warning("Journal DB not found: %s", db_path)
        return pd.DataFrame()

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT trade_id, entry_features, outcome, pnl_pct, exit_reason, "
            "score, style, asset_class, direction "
            "FROM trades "
            "WHERE exit_time IS NOT NULL "
            "AND entry_features IS NOT NULL "
            "AND entry_features != '{}'"
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error("Journal DB query failed: %s", exc)
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()

    if not rows:
        logger.info("No closed trades with entry_features in journal")
        return pd.DataFrame()

    # ── Parse entry_features JSON into feature rows ──────────────
    feat_names = list(ENTRY_QUALITY_FEATURES)
    records: list[dict[str, Any]] = []

    for row in rows:
        try:
            feats = json.loads(row["entry_features"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Check all 41 features present
        missing = [f for f in feat_names if f not in feats]
        if missing:
            continue

        record: dict[str, Any] = {f: feats[f] for f in feat_names}
        record["trade_id"] = row["trade_id"]
        record["outcome"] = row["outcome"]
        record["pnl_pct"] = row["pnl_pct"] or 0.0
        record["exit_reason"] = row["exit_reason"] or ""
        records.append(record)

    dropped = len(rows) - len(records)
    if dropped > 0:
        logger.info("Dropped %d trades with incomplete features (of %d total)", dropped, len(rows))

    if not records:
        logger.info("No trades with complete 41-feature set")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── Label: win=1, else=0 ─────────────────────────────────────
    df["label"] = (df["outcome"] == "win").astype(np.int32)

    # ── Sample weights ───────────────────────────────────────────
    weights = np.full(len(df), 2.0, dtype=np.float64)  # live 2x base

    for i, tid in enumerate(df["trade_id"]):
        grade = grade_lookup.get(tid, "")
        weights[i] *= grade_weights.get(grade, 1.0)

    # Timeout penalty
    timeout_mask = df["exit_reason"] == "timeout"
    weights[timeout_mask.values] *= timeout_weight

    df["sample_weight"] = weights

    # Drop helper columns
    df.drop(columns=["trade_id", "outcome", "pnl_pct", "exit_reason"], inplace=True)

    logger.info("Collected %d live trades (%d wins, %d losses)",
                len(df), int(df["label"].sum()), int((df["label"] == 0).sum()))
    return df


# ===================================================================
#  2. Load subsampled backtest data
# ===================================================================

def load_backtest_data(
    data_dir: str | Path = DATA_DIR,
    subsample_per_class: int = 50_000,
    classes: list[str] | None = None,
) -> pd.DataFrame:
    """Load backtest parquets, subsample per class, return with 41 features."""
    data_dir = Path(data_dir)
    if classes is None:
        classes = ALL_CLASSES

    feat_names = list(ENTRY_QUALITY_FEATURES)
    frames: list[pd.DataFrame] = []

    for cls in classes:
        parquet_path = data_dir / f"{cls}_samples.parquet"
        if not parquet_path.exists():
            logger.warning("Backtest parquet not found: %s", parquet_path)
            continue

        try:
            raw = pd.read_parquet(parquet_path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", parquet_path, exc)
            continue

        if "label_outcome" not in raw.columns:
            logger.warning("No label_outcome in %s, skipping", parquet_path)
            continue

        # Stratified subsample
        if len(raw) > subsample_per_class:
            pos = raw[raw["label_outcome"] == 1]
            neg = raw[raw["label_outcome"] != 1]
            ratio = len(pos) / max(len(raw), 1)
            n_pos = max(1, int(subsample_per_class * ratio))
            n_neg = subsample_per_class - n_pos
            pos_sample = pos.sample(n=min(n_pos, len(pos)), random_state=42)
            neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=42)
            raw = pd.concat([pos_sample, neg_sample], ignore_index=True)
            logger.info("Subsampled %s to %d rows (stratified)", cls, len(raw))

        # Extract the 41 features
        available = [f for f in feat_names if f in raw.columns]
        missing = [f for f in feat_names if f not in raw.columns]

        if missing:
            # Fill missing features with 0
            for m in missing:
                raw[m] = 0.0
            logger.warning("%s missing features filled with 0: %s", cls, missing)

        subset = raw[feat_names].copy()
        subset["label"] = (raw["label_outcome"] == 1).astype(np.int32)
        subset["sample_weight"] = 1.0

        frames.append(subset)
        logger.info("Loaded %d backtest samples from %s", len(subset), cls)

    if not frames:
        logger.warning("No backtest data loaded")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ===================================================================
#  3. Retrain XGBoost from scratch
# ===================================================================

def retrain_if_ready(config: dict) -> bool:
    """Full XGBoost retrain on backtest + live data. Returns True on success."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    cl_cfg = config.get("continuous_learner", {})
    journal_cfg = config.get("journal", {})

    min_trades = cl_cfg.get("min_trades_for_retrain", 50)
    min_val_trades = cl_cfg.get("min_validation_trades", 15)
    val_split = cl_cfg.get("validation_split", 0.30)
    auc_gate = cl_cfg.get("auc_gate", 0.55)
    subsample_per_class = cl_cfg.get("backtest_subsample_per_class", 50_000)
    halflife = cl_cfg.get("recency_halflife_trades", 200)
    grade_weights = cl_cfg.get("grade_weights", DEFAULT_GRADE_WEIGHTS)
    timeout_weight = cl_cfg.get("timeout_weight", 0.5)
    db_path = journal_cfg.get("db_path", "trade_journal/journal.db")
    feedback_path = cl_cfg.get("feedback_path", str(FEEDBACK_PATH))

    # ── Memory guard ─────────────────────────────────────────────
    try:
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            logger.warning("Memory usage %.1f%% > 80%%, skipping retrain", mem.percent)
            return False
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        if rss_gb > 5.0:
            logger.warning("RSS %.2f GB > 5 GB, skipping retrain", rss_gb)
            return False
    except ImportError:
        pass  # psutil optional

    # ── Collect live data ────────────────────────────────────────
    live_df = collect_live_data(db_path, feedback_path, grade_weights, timeout_weight)
    n_live = len(live_df)
    if n_live < min_trades:
        logger.info("Only %d live trades (need %d), skipping retrain", n_live, min_trades)
        return False

    n_val = int(n_live * val_split)
    if n_val < min_val_trades:
        logger.info("Validation set would be %d trades (need %d), skipping", n_val, min_val_trades)
        return False

    # ── Load backtest data ───────────────────────────────────────
    backtest_df = load_backtest_data(subsample_per_class=subsample_per_class)

    # ── Walk-forward split: last 30% of live = validation ────────
    feat_names = list(ENTRY_QUALITY_FEATURES)
    split_idx = n_live - n_val

    live_train = live_df.iloc[:split_idx]
    live_val = live_df.iloc[split_idx:]

    # ── Combine backtest + live train ────────────────────────────
    if len(backtest_df) > 0:
        train_df = pd.concat([backtest_df, live_train], ignore_index=True)
    else:
        train_df = live_train.copy()

    # ── Apply recency halflife to live training weights ──────────
    # Position from newest: live_train rows are at the end of train_df
    n_bt = len(train_df) - len(live_train)
    weights = train_df["sample_weight"].values.copy()
    if len(live_train) > 0 and halflife > 0:
        n_lt = len(live_train)
        ages = np.arange(n_lt - 1, -1, -1, dtype=np.float64)  # newest=0
        decay = np.exp(-math.log(2) * ages / halflife)
        weights[n_bt:] *= decay

    # ── Prepare arrays ───────────────────────────────────────────
    X_train = train_df[feat_names].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int32)
    w_train = weights.astype(np.float64)

    X_val = live_val[feat_names].values.astype(np.float32)
    y_val = live_val["label"].values.astype(np.int32)

    # Clip known outlier features
    for col_name, (lo, hi) in CLIP_RANGES.items():
        if col_name in feat_names:
            col_idx = feat_names.index(col_name)
            X_train[:, col_idx] = np.clip(X_train[:, col_idx], lo, hi)
            X_val[:, col_idx] = np.clip(X_val[:, col_idx], lo, hi)

    # NaN/inf cleanup
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=5.0, neginf=-5.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=5.0, neginf=-5.0)

    # ── Class imbalance ──────────────────────────────────────────
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale = n_neg / max(n_pos, 1)

    # ── Train XGBoost from scratch ───────────────────────────────
    logger.info("Training XGBoost: %d train (%d backtest + %d live), %d val",
                len(X_train), n_bt, len(live_train), len(X_val))

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=30,
        n_jobs=2,
        random_state=42,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=w_train,
        verbose=False,
    )

    logger.info("XGBoost trained: best iteration %d / %d trees",
                model.best_iteration, model.n_estimators)

    # ── Validate ─────────────────────────────────────────────────
    y_prob = model.predict_proba(X_val)[:, 1]
    try:
        val_auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        logger.warning("AUC computation failed (single class?), skipping")
        return False

    # Approximate profit factor: predicted-win actual profits vs losses
    preds = (y_prob >= 0.5).astype(int)
    pred_win_mask = preds == 1
    wins_correct = int(((preds == 1) & (y_val == 1)).sum())
    losses_wrong = int(((preds == 1) & (y_val == 0)).sum())
    val_pf = wins_correct / max(losses_wrong, 1)
    val_wr = wins_correct / max(int(pred_win_mask.sum()), 1)

    logger.info("Validation: AUC=%.4f, PF=%.2f, WR=%.1f%%", val_auc, val_pf, val_wr * 100)

    if val_auc < auc_gate:
        logger.warning("AUC %.4f < gate %.4f, rejecting model", val_auc, auc_gate)
        return False
    if val_pf < 1.0:
        logger.warning("PF %.2f < 1.0, rejecting model", val_pf)
        return False

    # ── Performance tracking: compare with previous ──────────────
    prev_meta: dict[str, Any] = {}
    if META_PATH.exists():
        try:
            with open(META_PATH, "r") as fh:
                prev_meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    if prev_meta:
        prev_auc = prev_meta.get("validation_auc", 0)
        prev_pf = prev_meta.get("validation_pf", 0)
        logger.info("Delta vs previous: AUC %+.4f, PF %+.2f",
                     val_auc - prev_auc, val_pf - prev_pf)

    # ── Backup existing model ────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "rl_entry_filter.pkl"
    backup_path = MODEL_DIR / "rl_entry_filter_prev.pkl"

    if model_path.exists():
        try:
            shutil.copy2(str(model_path), str(backup_path))
            logger.info("Backed up existing model to %s", backup_path)
        except OSError as exc:
            logger.error("Backup failed: %s", exc)
            return False

    # ── Atomic write: .tmp then rename ───────────────────────────
    model_data = {
        "model": model,
        "feat_names": list(ENTRY_QUALITY_FEATURES),
        "task": "entry_quality",
        "schema_version": SCHEMA_VERSION,
        "dead_features": list(DEAD_FEATURES),
        "clip_ranges": CLIP_RANGES,
        "asset_class_map": ASSET_CLASS_MAP,
    }

    tmp_path = model_path.with_suffix(".pkl.tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(model_data, f)
        os.rename(str(tmp_path), str(model_path))
    except OSError as exc:
        logger.error("Atomic model write failed: %s", exc)
        if tmp_path.exists():
            tmp_path.unlink()
        return False

    logger.info("New model written: %s", model_path)

    # ── Write meta.json ──────────────────────────────────────────
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_live_trades": n_live,
        "n_backtest_samples": n_bt,
        "validation_auc": round(val_auc, 6),
        "validation_pf": round(val_pf, 4),
        "validation_wr": round(val_wr, 4),
        "n_trees": int(model.best_iteration + 1),
    }
    try:
        tmp_meta = META_PATH.with_suffix(".json.tmp")
        with open(tmp_meta, "w") as fh:
            json.dump(meta, fh, indent=2)
        os.rename(str(tmp_meta), str(META_PATH))
    except OSError as exc:
        logger.warning("Meta write failed: %s", exc)

    logger.info("Retrain complete: %d trees, AUC=%.4f, PF=%.2f",
                meta["n_trees"], val_auc, val_pf)
    return True


# ===================================================================
#  4. Auto-rollback check
# ===================================================================

def check_auto_rollback(config: dict, db_path: str) -> bool:
    """Check post-deploy trade performance and rollback if degraded.

    Returns True if a rollback was triggered.
    """
    cl_cfg = config.get("continuous_learner", {})
    min_trades = cl_cfg.get("rollback_min_trades", 20)
    pnl_threshold = cl_cfg.get("rollback_cumulative_pnl", -0.03)
    wr_threshold = cl_cfg.get("rollback_min_winrate", 0.20)

    # Read last deploy timestamp from meta
    if not META_PATH.exists():
        return False

    try:
        with open(META_PATH, "r") as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False

    deploy_ts = meta.get("timestamp")
    if not deploy_ts:
        return False

    # Query trades closed since deploy
    if not Path(db_path).exists():
        return False

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT outcome, pnl_pct FROM trades "
            "WHERE exit_time IS NOT NULL AND exit_time > ?",
            (deploy_ts,),
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error("Rollback check DB query failed: %s", exc)
        return False
    finally:
        if conn is not None:
            conn.close()

    if len(rows) < min_trades:
        return False

    # Evaluate
    pnl_total = sum(r[1] or 0.0 for r in rows)
    wins = sum(1 for r in rows if r[0] == "win")
    wr = wins / len(rows)

    if pnl_total < pnl_threshold or wr < wr_threshold:
        logger.warning(
            "Auto-rollback triggered: %d trades, cumPnL=%.2f%%, WR=%.1f%% "
            "(thresholds: pnl<%.2f%%, wr<%.1f%%)",
            len(rows), pnl_total * 100, wr * 100,
            pnl_threshold * 100, wr_threshold * 100,
        )
        rolled = manual_rollback("entry_filter")
        if rolled:
            # Write marker so rl_brain_v2.py picks it up
            marker = MODEL_DIR / ".rollback_watch"
            try:
                marker.write_text(
                    json.dumps({
                        "reason": "auto_rollback",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trades_evaluated": len(rows),
                        "cum_pnl": pnl_total,
                        "win_rate": wr,
                    })
                )
            except OSError:
                pass
        return rolled

    return False


# ===================================================================
#  5. Manual rollback
# ===================================================================

def manual_rollback(slot_name: str = "entry_filter") -> bool:
    """Roll back a model slot to its previous version. Idempotent.

    Slot mapping: entry_filter -> models/rl_entry_filter.pkl
    """
    slot_map = {
        "entry_filter": "rl_entry_filter",
        "tp_optimizer": "rl_tp_optimizer",
        "be_manager": "rl_be_manager",
        "exit_classifier": "rl_exit_classifier",
    }
    base_name = slot_map.get(slot_name)
    if not base_name:
        logger.error("Unknown model slot: %s", slot_name)
        return False

    model_path = MODEL_DIR / f"{base_name}.pkl"
    prev_path = MODEL_DIR / f"{base_name}_prev.pkl"
    v1_path = MODEL_DIR / f"{base_name}_v1.pkl"

    source: Path | None = None
    if prev_path.exists():
        source = prev_path
    elif v1_path.exists():
        source = v1_path
    else:
        logger.error("No backup found for %s (tried %s, %s)", slot_name, prev_path, v1_path)
        return False

    try:
        shutil.copy2(str(source), str(model_path))
        logger.info("Rolled back %s from %s", model_path, source)
        return True
    except OSError as exc:
        logger.error("Rollback copy failed: %s", exc)
        return False


# ===================================================================
#  6. Async main loop
# ===================================================================

async def run_continuous_learner(
    config: dict,
    shutdown_event: asyncio.Event,
) -> None:
    """Main async loop — runs in the live bot's event loop."""
    cl_cfg = config.get("continuous_learner", {})

    if not cl_cfg.get("enabled", False):
        logger.info("Continuous learner disabled in config")
        return

    interval_hours = cl_cfg.get("retrain_interval_hours", 1)
    interval_sec = interval_hours * 3600
    journal_cfg = config.get("journal", {})
    db_path = journal_cfg.get("db_path", "trade_journal/journal.db")

    # ── Health check ─────────────────────────────────────────────
    if not Path(db_path).exists():
        logger.warning("Continuous learner: journal DB not found at %s, waiting", db_path)
    else:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL"
            )
            closed = cursor.fetchone()[0]
            conn.close()
            logger.info("Continuous learner started: %d closed trades in journal", closed)
        except sqlite3.Error as exc:
            logger.warning("Continuous learner: journal DB health check failed: %s", exc)

    logger.info("Continuous learner loop: interval=%dh, min_trades=%d",
                interval_hours, cl_cfg.get("min_trades_for_retrain", 50))

    # ── Main loop ────────────────────────────────────────────────
    loop = asyncio.get_running_loop()

    while True:
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_sec)
            # shutdown_event was set — exit cleanly
            logger.info("Continuous learner: shutdown signal received")
            break
        except asyncio.TimeoutError:
            pass  # interval elapsed, proceed with retrain check

        try:
            retrained = await loop.run_in_executor(None, retrain_if_ready, config)
            if retrained:
                logger.info("Continuous learner: retrain succeeded")
        except Exception:
            logger.exception("Continuous learner: retrain failed")

        try:
            rolled_back = await loop.run_in_executor(
                None, check_auto_rollback, config, db_path,
            )
            if rolled_back:
                logger.warning("Continuous learner: auto-rollback executed")
        except Exception:
            logger.exception("Continuous learner: rollback check failed")
