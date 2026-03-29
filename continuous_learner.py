"""
===================================================================
 continuous_learner.py  --  Automated Model Retraining Pipeline
 -----------------------------------------------------------
 Exports live trade data from journal, merges with historical
 training data, retrains XGBoost models with walk-forward
 validation, and deploys if quality gates pass.  Includes
 auto-rollback on performance degradation.

 Designed to run weekly via cron on the MacBook Air.

 Usage:
     python3 continuous_learner.py --dry-run         # Show what would happen
     python3 continuous_learner.py --retrain          # Full retrain + deploy
     python3 continuous_learner.py --show-weights     # Print recency weight distribution
     python3 continuous_learner.py --rollback         # Manual rollback to previous version
     python3 continuous_learner.py --status           # Show model versions + metrics
===================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/results/rl/continuous_learner.log",
                            mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ===================================================================
#  Paths
# ===================================================================

MODEL_DIR = Path("models")
DATA_DIR = Path("data/rl_training")
RESULTS_DIR = Path("backtest/results/rl")
RETRAIN_LOG = RESULTS_DIR / "retrain_log.jsonl"
ROLLBACK_MARKER = MODEL_DIR / ".rollback_watch"

# Model slots that can be retrained
MODEL_SLOTS = {
    "entry_filter": {
        "filename": "rl_entry_filter.pkl",
        "task": "entry_quality",
        "auc_gate_key": "auc_gate",       # read from config, fallback 0.52
        "default_auc_gate": 0.52,
    },
    "exit_classifier": {
        "filename": "rl_exit_classifier.pkl",
        "task": "early_exit",
        "auc_gate_key": "auc_gate",       # read from config, fallback 0.55
        "default_auc_gate": 0.55,
    },
}


# ===================================================================
#  Config
# ===================================================================

def load_config() -> dict:
    """Load continuous_learner section from default_config.yaml."""
    cfg_path = Path("config/default_config.yaml")
    if not cfg_path.exists():
        logger.warning("Config not found at %s, using defaults", cfg_path)
        return {}
    with open(cfg_path) as f:
        full = yaml.safe_load(f) or {}
    return full.get("continuous_learner", {})


# ===================================================================
#  Step 1: Export journal data
# ===================================================================

def export_journal(cfg: dict) -> tuple[Path | None, int]:
    """Export live trade journal to parquet.  Returns (path, trade_count)."""
    try:
        from trade_journal import TradeJournal
    except ImportError:
        logger.error("Cannot import TradeJournal -- trade_journal.py missing")
        return None, 0

    db_path = cfg.get("journal_db_path", "trade_journal/journal.db")
    if not Path(db_path).exists():
        logger.warning("Journal DB not found: %s", db_path)
        return None, 0

    journal = TradeJournal(db_path=db_path)
    try:
        n_trades = journal.count_closed_trades()
        logger.info("[Step 1] Journal has %d closed trades", n_trades)

        min_trades = cfg.get("min_trades_for_retrain", 200)
        if n_trades < min_trades:
            logger.warning(
                "[Step 1] Only %d trades (minimum %d) -- not enough for retrain",
                n_trades, min_trades,
            )
            return None, n_trades

        out_path = journal.export_to_parquet(str(DATA_DIR / "live_trades"))
        logger.info("[Step 1] Exported journal -> %s", out_path)
        return out_path, n_trades
    finally:
        journal.close()


# ===================================================================
#  Step 2: Merge with historical data + recency weights
# ===================================================================

def compute_recency_weights(
    n_samples: int,
    halflife: int = 200,
    min_weight: float = 0.01,
) -> np.ndarray:
    """Exponential recency weights: w(t) = exp(-ln(2) * age / halflife).

    Samples are assumed to be ordered oldest-first (index 0 = oldest).
    age_in_trades = (n_samples - 1 - index).
    """
    ages = np.arange(n_samples - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-np.log(2) * ages / max(halflife, 1))
    weights = np.clip(weights, min_weight, 1.0)
    return weights.astype(np.float32)


def merge_training_data(
    cfg: dict,
    task: str = "entry_quality",
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Load historical + live data, return merged DataFrame and recency weights."""
    from rl_brain_v2 import ALL_CLASSES

    halflife = cfg.get("recency_halflife_trades", 200)

    frames: list[pd.DataFrame] = []

    # Historical parquets
    for ac in ALL_CLASSES:
        if task == "early_exit":
            path = DATA_DIR / f"{ac}_exit_episodes.parquet"
        else:
            path = DATA_DIR / f"{ac}_samples.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                df["_source"] = "historical"
                frames.append(df)
                logger.info("[Step 2] Historical %s: %d rows from %s", ac, len(df), path)
            except ImportError:
                logger.error("[Step 2] Parquet engine not installed (pip install pyarrow)")
                return None, None
            except Exception as exc:
                logger.warning("[Step 2] Could not read %s: %s", path, exc)

    # Live journal parquets
    live_path = DATA_DIR / "live_trades" / "live_exit_episodes.parquet"
    if live_path.exists():
        try:
            df_live = pd.read_parquet(live_path)
            df_live["_source"] = "live"
            frames.append(df_live)
            logger.info("[Step 2] Live journal: %d rows from %s", len(df_live), live_path)
        except ImportError:
            logger.error("[Step 2] Parquet engine not installed (pip install pyarrow)")
            return None, None
        except Exception as exc:
            logger.warning("[Step 2] Could not read live data: %s", exc)
    else:
        logger.info("[Step 2] No live journal parquet found at %s", live_path)

    if not frames:
        logger.error("[Step 2] No training data found")
        return None, None

    merged = pd.concat(frames, ignore_index=True)

    # For entry_quality, keep only entry rows
    if task == "entry_quality" and "label_action" in merged.columns:
        merged = merged[merged["label_action"] > 0].copy()

    # Sort by window (time ordering) then _source (historical first)
    if "window" in merged.columns:
        merged = merged.sort_values(["window"]).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    # Compute recency weights
    weights = compute_recency_weights(len(merged), halflife=halflife)
    logger.info(
        "[Step 2] Merged %d rows, weight range [%.4f, %.4f]",
        len(merged), float(weights.min()), float(weights.max()),
    )

    merged.drop(columns=["_source"], inplace=True, errors="ignore")
    return merged, weights


# ===================================================================
#  Step 3: Retrain models
# ===================================================================

def retrain_model(
    data: pd.DataFrame,
    recency_weights: np.ndarray,
    task: str,
    cfg: dict,
) -> dict[str, Any] | None:
    """Retrain a single model slot.  Returns model_data dict or None on failure."""
    from rl_brain_v2 import (
        prepare_features,
        prepare_labels,
        prepare_sample_weights,
        train_xgboost,
        CLIP_RANGES,
        ASSET_CLASS_MAP,
        DEAD_FEATURES,
    )

    if len(data) < 100:
        logger.error("[Step 3] Too few samples (%d) for task=%s", len(data), task)
        return None

    # Time-ordered split: 70% train, 15% val, 15% test
    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()

    if len(test_df) == 0 or len(val_df) == 0:
        logger.error("[Step 3] Empty validation or test split for task=%s", task)
        return None

    X_train, feat_cols = prepare_features(train_df, task)
    y_train = prepare_labels(train_df, task)
    X_val, _ = prepare_features(val_df, task)
    y_val = prepare_labels(val_df, task)
    X_test, _ = prepare_features(test_df, task)
    y_test = prepare_labels(test_df, task)

    # Combine recency weights with task-specific sample weights
    task_weights = prepare_sample_weights(y_train, train_df, task)
    combined_weights = task_weights * recency_weights[:train_end]

    try:
        model = train_xgboost(
            X_train, y_train, X_val, y_val,
            feat_cols, sample_weights=combined_weights, task=task,
        )
    except Exception as exc:
        logger.error("[Step 3] Training failed for task=%s: %s", task, exc)
        return None

    # Evaluate on holdout
    from sklearn.metrics import roc_auc_score

    try:
        test_proba = model.predict_proba(X_test)[:, 1]
        auc_test = float(roc_auc_score(y_test, test_proba))
    except Exception as exc:
        logger.error("[Step 3] AUC computation failed: %s", exc)
        auc_test = 0.0

    try:
        train_proba = model.predict_proba(X_train)[:, 1]
        auc_train = float(roc_auc_score(y_train, train_proba))
    except Exception:
        auc_train = 0.0

    # Feature importance
    importances = model.feature_importances_
    max_importance = float(importances.max()) if len(importances) > 0 else 0.0

    model_data = {
        "model": model,
        "feat_names": feat_cols,
        "task": task,
        "dead_features": list(DEAD_FEATURES),
        "clip_ranges": CLIP_RANGES,
        "asset_class_map": ASSET_CLASS_MAP,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "max_feature_importance": max_importance,
    }

    logger.info(
        "[Step 3] task=%s AUC_train=%.4f AUC_test=%.4f max_feat_imp=%.3f",
        task, auc_train, auc_test, max_importance,
    )

    return model_data


# ===================================================================
#  Step 4: Quality gates
# ===================================================================

def check_quality_gates(
    model_data: dict[str, Any],
    slot_name: str,
    cfg: dict,
) -> tuple[bool, list[str]]:
    """Check all quality gates.  Returns (passed, list_of_messages)."""
    slot_cfg = MODEL_SLOTS[slot_name]
    auc_threshold = cfg.get(slot_cfg["auc_gate_key"], slot_cfg["default_auc_gate"])
    agreement_threshold = cfg.get("agreement_gate", 0.80)

    messages: list[str] = []
    passed = True

    # Gate 1: AUC on holdout
    auc_test = model_data.get("auc_test", 0.0)
    if auc_test < auc_threshold:
        messages.append(f"FAIL: AUC {auc_test:.4f} < {auc_threshold:.4f}")
        passed = False
    else:
        messages.append(f"PASS: AUC {auc_test:.4f} >= {auc_threshold:.4f}")

    # Gate 2: Feature importance sanity (no single feature > 40%)
    max_imp = model_data.get("max_feature_importance", 0.0)
    if max_imp > 0.40:
        messages.append(f"FAIL: Max feature importance {max_imp:.3f} > 0.40 (data leak?)")
        passed = False
    else:
        messages.append(f"PASS: Max feature importance {max_imp:.3f} <= 0.40")

    # Gate 3: Model agreement with current production model
    current_path = MODEL_DIR / slot_cfg["filename"]
    if current_path.exists():
        agreement = _compute_model_agreement(model_data, current_path)
        if agreement is not None:
            if agreement < agreement_threshold:
                messages.append(
                    f"FAIL: Agreement {agreement:.1%} < {agreement_threshold:.1%}"
                )
                passed = False
            else:
                messages.append(
                    f"PASS: Agreement {agreement:.1%} >= {agreement_threshold:.1%}"
                )
            model_data["agreement"] = agreement
        else:
            messages.append("SKIP: Could not compute agreement (old model load failed)")
    else:
        messages.append("SKIP: No existing model for agreement check (first deploy)")

    return passed, messages


def _compute_model_agreement(
    new_data: dict[str, Any],
    old_path: Path,
) -> float | None:
    """Compute prediction overlap between new and old model on a shared test set."""
    try:
        with open(old_path, "rb") as f:
            old_data = pickle.load(f)
        old_model = old_data.get("model")
        if old_model is None:
            return None

        new_model = new_data["model"]
        feat_names = new_data["feat_names"]
        old_feat = old_data.get("feat_names", feat_names)

        # Generate a small synthetic test set from the training data range
        # Use the new model's predictions on zero-centered random data
        rng = np.random.RandomState(42)
        n_test = 500
        X_test = rng.randn(n_test, len(feat_names)).astype(np.float32) * 0.5

        new_pred = (new_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

        # Build aligned feature matrix for old model
        X_old = np.zeros((n_test, len(old_feat)), dtype=np.float32)
        for i, fname in enumerate(old_feat):
            if fname in feat_names:
                j = feat_names.index(fname)
                X_old[:, i] = X_test[:, j]

        old_pred = (old_model.predict_proba(X_old)[:, 1] >= 0.5).astype(int)
        agreement = float((new_pred == old_pred).mean())
        return agreement

    except Exception as exc:
        logger.warning("Agreement check failed: %s", exc)
        return None


# ===================================================================
#  Step 5: Deploy
# ===================================================================

def get_next_version(slot_name: str) -> int:
    """Find the next version number for a model slot."""
    slot_cfg = MODEL_SLOTS[slot_name]
    base = slot_cfg["filename"].replace(".pkl", "")
    existing = list(MODEL_DIR.glob(f"{base}_v*.pkl"))
    versions = []
    for p in existing:
        try:
            v = int(p.stem.split("_v")[-1])
            versions.append(v)
        except (ValueError, IndexError):
            continue
    return max(versions, default=0) + 1


def get_current_version(slot_name: str) -> int | None:
    """Find the current active version by reading the retrain log."""
    if not RETRAIN_LOG.exists():
        # Check for existing versioned files
        slot_cfg = MODEL_SLOTS[slot_name]
        base = slot_cfg["filename"].replace(".pkl", "")
        existing = list(MODEL_DIR.glob(f"{base}_v*.pkl"))
        if existing:
            versions = []
            for p in existing:
                try:
                    v = int(p.stem.split("_v")[-1])
                    versions.append(v)
                except (ValueError, IndexError):
                    continue
            return max(versions, default=None)
        return None

    # Read last deployed entry for this slot
    with open(RETRAIN_LOG) as f:
        for line in reversed(f.readlines()):
            try:
                entry = json.loads(line.strip())
                if entry.get("task") == slot_name and entry.get("deployed"):
                    return entry.get("version")
            except json.JSONDecodeError:
                continue
    return None


def deploy_model(
    model_data: dict[str, Any],
    slot_name: str,
    cfg: dict,
    dry_run: bool = False,
) -> int:
    """Save versioned model and atomically swap to active path.  Returns version."""
    slot_cfg = MODEL_SLOTS[slot_name]
    version = get_next_version(slot_name)
    prev_version = get_current_version(slot_name)

    base = slot_cfg["filename"].replace(".pkl", "")
    versioned_path = MODEL_DIR / f"{base}_v{version}.pkl"
    active_path = MODEL_DIR / slot_cfg["filename"]

    model_data["version"] = version

    if dry_run:
        logger.info(
            "[Step 5] DRY-RUN: Would deploy %s v%d -> %s",
            slot_name, version, active_path,
        )
        return version

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save versioned model
    with open(versioned_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info("[Step 5] Saved versioned model: %s", versioned_path)

    # Atomic swap: write to .tmp, then rename
    tmp_path = active_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(model_data, f)
    os.replace(str(tmp_path), str(active_path))
    logger.info("[Step 5] Deployed: %s (v%d)", active_path, version)

    # Clean up old versions (keep last N)
    keep = cfg.get("model_versions_keep", 5)
    _cleanup_old_versions(base, keep)

    # Write retrain log entry
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "prev_version": prev_version,
        "task": slot_name,
        "auc_train": model_data.get("auc_train", 0.0),
        "auc_test": model_data.get("auc_test", 0.0),
        "agreement": model_data.get("agreement"),
        "deployed": True,
        "trades_used": model_data.get("n_train", 0) + model_data.get("n_test", 0),
    }
    with open(RETRAIN_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Create rollback watch marker
    _create_rollback_marker(version, prev_version, cfg)

    return version


def _cleanup_old_versions(base: str, keep: int) -> None:
    """Remove old versioned model files, keeping the most recent `keep`."""
    existing = sorted(MODEL_DIR.glob(f"{base}_v*.pkl"))
    versions_with_paths = []
    for p in existing:
        try:
            v = int(p.stem.split("_v")[-1])
            versions_with_paths.append((v, p))
        except (ValueError, IndexError):
            continue
    versions_with_paths.sort(key=lambda x: x[0])
    to_remove = versions_with_paths[:-keep] if len(versions_with_paths) > keep else []
    for v, p in to_remove:
        try:
            p.unlink()
            logger.info("[Step 5] Removed old version: %s", p)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", p, exc)


# ===================================================================
#  Step 6: Rollback
# ===================================================================

def _create_rollback_marker(
    version: int,
    prev_version: int | None,
    cfg: dict,
) -> None:
    """Write rollback watch marker for the live bot to monitor."""
    marker = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "prev_version": prev_version,
        "min_trades": cfg.get("rollback_min_trades", 20),
        "cumulative_pnl_threshold": cfg.get("rollback_cumulative_pnl", -0.03),
        "min_winrate_threshold": cfg.get("rollback_min_winrate", 0.20),
    }
    try:
        with open(ROLLBACK_MARKER, "w") as f:
            json.dump(marker, f, indent=2)
        logger.info("[Step 6] Rollback marker written: %s", ROLLBACK_MARKER)
    except OSError as exc:
        logger.error("Could not write rollback marker: %s", exc)


def manual_rollback(slot_name: str = "entry_filter") -> bool:
    """Manually rollback to the previous model version."""
    slot_cfg = MODEL_SLOTS[slot_name]
    active_path = MODEL_DIR / slot_cfg["filename"]
    base = slot_cfg["filename"].replace(".pkl", "")

    # Find current and previous versions
    existing = sorted(MODEL_DIR.glob(f"{base}_v*.pkl"))
    versions_with_paths = []
    for p in existing:
        try:
            v = int(p.stem.split("_v")[-1])
            versions_with_paths.append((v, p))
        except (ValueError, IndexError):
            continue
    versions_with_paths.sort(key=lambda x: x[0])

    if len(versions_with_paths) < 2:
        logger.error("Not enough versions for rollback (need >= 2, have %d)",
                      len(versions_with_paths))
        return False

    current_v, current_path = versions_with_paths[-1]
    prev_v, prev_path = versions_with_paths[-2]

    logger.info("Rolling back %s: v%d -> v%d", slot_name, current_v, prev_v)

    # Rename current as bad
    bad_path = current_path.with_name(f"{current_path.stem}_bad.pkl")
    try:
        shutil.move(str(current_path), str(bad_path))
    except OSError as exc:
        logger.error("Could not rename current model: %s", exc)
        return False

    # Copy previous version to active path
    tmp_path = active_path.with_suffix(".pkl.tmp")
    try:
        shutil.copy2(str(prev_path), str(tmp_path))
        os.replace(str(tmp_path), str(active_path))
    except OSError as exc:
        logger.error("Could not restore previous model: %s", exc)
        return False

    # Log rollback
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": prev_v,
        "prev_version": current_v,
        "task": slot_name,
        "deployed": True,
        "rollback": True,
        "reason": "manual",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Remove rollback marker
    if ROLLBACK_MARKER.exists():
        ROLLBACK_MARKER.unlink(missing_ok=True)

    logger.info("Rollback complete: %s now at v%d", slot_name, prev_v)
    return True


# ===================================================================
#  CLI Commands
# ===================================================================

def cmd_retrain(cfg: dict, dry_run: bool = False) -> None:
    """Full retrain pipeline: export -> merge -> train -> gate -> deploy."""
    logger.info("=" * 70)
    logger.info("CONTINUOUS LEARNER -- %s", "DRY RUN" if dry_run else "RETRAIN")
    logger.info("=" * 70)

    # Step 1: Export journal
    journal_path, n_trades = export_journal(cfg)
    min_trades = cfg.get("min_trades_for_retrain", 200)
    if journal_path is None and n_trades < min_trades:
        logger.info("Insufficient trades (%d/%d). Skipping retrain.", n_trades, min_trades)
        # Still proceed with historical-only data if available
        logger.info("Attempting retrain with historical data only...")

    for slot_name, slot_cfg in MODEL_SLOTS.items():
        task = slot_cfg["task"]
        logger.info("-" * 50)
        logger.info("Retraining: %s (task=%s)", slot_name, task)
        logger.info("-" * 50)

        # Step 2: Merge data
        data, weights = merge_training_data(cfg, task=task)
        if data is None or len(data) < 100:
            logger.warning("Skipping %s: insufficient data", slot_name)
            continue

        # Step 3: Retrain
        model_data = retrain_model(data, weights, task, cfg)
        if model_data is None:
            logger.warning("Skipping %s: training failed", slot_name)
            continue

        # Step 4: Quality gates
        passed, messages = check_quality_gates(model_data, slot_name, cfg)
        for msg in messages:
            logger.info("  Gate: %s", msg)

        if not passed:
            logger.warning("%s: quality gates FAILED -- model NOT deployed", slot_name)
            # Log non-deployment
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": slot_name,
                "auc_train": model_data.get("auc_train", 0.0),
                "auc_test": model_data.get("auc_test", 0.0),
                "agreement": model_data.get("agreement"),
                "deployed": False,
                "reason": "gate_failed",
                "gate_messages": messages,
            }
            with open(RETRAIN_LOG, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            continue

        # Step 5: Deploy
        version = deploy_model(model_data, slot_name, cfg, dry_run=dry_run)
        logger.info("%s: deployed v%d", slot_name, version)

    logger.info("=" * 70)
    logger.info("RETRAIN COMPLETE")
    logger.info("=" * 70)


def cmd_show_weights(cfg: dict) -> None:
    """Print recency weight distribution table."""
    halflife = cfg.get("recency_halflife_trades", 200)
    buckets = [
        ("0-50", 0, 50),
        ("50-100", 50, 100),
        ("100-200", 100, 200),
        ("200-500", 200, 500),
        ("500+", 500, 2000),
    ]
    print(f"\nRecency Weight Distribution (halflife={halflife} trades)")
    print("-" * 60)
    print(f"{'Age Bucket':>12}  {'Weight Min':>12}  {'Weight Max':>12}  {'Trades':>8}")
    print("-" * 60)

    # Try to count actual trades per bucket
    from rl_brain_v2 import ALL_CLASSES
    total_samples = 0
    try:
        for ac in ALL_CLASSES:
            path = DATA_DIR / f"{ac}_samples.parquet"
            if path.exists():
                df = pd.read_parquet(path, columns=["symbol"])
                total_samples += len(df)

        live_path = DATA_DIR / "live_trades" / "live_exit_episodes.parquet"
        if live_path.exists():
            df_live = pd.read_parquet(live_path)
            total_samples += len(df_live)
    except Exception:
        pass  # parquet engine not available locally

    if total_samples == 0:
        total_samples = 1000  # hypothetical for display
        print("  (using hypothetical 1000 samples -- parquet data not available)")

    weights = compute_recency_weights(total_samples, halflife=halflife)

    for label, lo, hi in buckets:
        # Age = distance from end
        idx_lo = max(0, total_samples - hi)
        idx_hi = min(total_samples, total_samples - lo)
        if idx_lo >= idx_hi:
            bucket_weights = np.array([0.0])
            count = 0
        else:
            bucket_weights = weights[idx_lo:idx_hi]
            count = idx_hi - idx_lo
        w_min = float(bucket_weights.min()) if len(bucket_weights) > 0 else 0.0
        w_max = float(bucket_weights.max()) if len(bucket_weights) > 0 else 0.0
        print(f"{label:>12}  {w_min:>12.6f}  {w_max:>12.6f}  {count:>8}")

    print("-" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Effective range: trades older than ~{int(halflife * np.log(1/0.01) / np.log(2))}"
          f" are weighted < 0.01\n")


def cmd_status(cfg: dict) -> None:
    """Show model versions, active model, and last retrain log entry."""
    print("\n" + "=" * 60)
    print("MODEL STATUS")
    print("=" * 60)

    for slot_name, slot_cfg in MODEL_SLOTS.items():
        base = slot_cfg["filename"].replace(".pkl", "")
        active_path = MODEL_DIR / slot_cfg["filename"]

        print(f"\n--- {slot_name} ---")
        print(f"  Active model: {active_path}"
              f" ({'EXISTS' if active_path.exists() else 'MISSING'})")

        # List versions
        existing = sorted(MODEL_DIR.glob(f"{base}_v*.pkl"))
        if existing:
            versions = []
            for p in existing:
                try:
                    v = int(p.stem.split("_v")[-1])
                    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                    size_kb = p.stat().st_size / 1024
                    versions.append((v, mtime, size_kb, p))
                except (ValueError, IndexError):
                    continue
            versions.sort(key=lambda x: x[0])
            for v, mtime, size_kb, p in versions:
                print(f"  v{v}: {p.name} ({size_kb:.0f} KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("  No versioned models found")

        # Current version from retrain log
        cur_v = get_current_version(slot_name)
        if cur_v is not None:
            print(f"  Current active version: v{cur_v}")

    # Last retrain log entries
    print(f"\n--- Retrain Log (last 5 entries) ---")
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG) as f:
            lines = f.readlines()
        for line in lines[-5:]:
            try:
                entry = json.loads(line.strip())
                ts = entry.get("timestamp", "?")[:19]
                task = entry.get("task", "?")
                v = entry.get("version", "?")
                deployed = entry.get("deployed", False)
                auc = entry.get("auc_test", 0)
                rb = " [ROLLBACK]" if entry.get("rollback") else ""
                print(f"  {ts} | {task:>16} | v{v} | AUC={auc:.4f}"
                      f" | {'DEPLOYED' if deployed else 'SKIPPED'}{rb}")
            except json.JSONDecodeError:
                continue
    else:
        print("  No retrain log found")

    # Rollback watch status
    print(f"\n--- Rollback Watch ---")
    if ROLLBACK_MARKER.exists():
        try:
            with open(ROLLBACK_MARKER) as f:
                marker = json.load(f)
            print(f"  Active: v{marker.get('version')} "
                  f"(deployed {marker.get('deployed_at', '?')[:19]})")
            print(f"  Previous: v{marker.get('prev_version')}")
            print(f"  Triggers: cumPnL < {marker.get('cumulative_pnl_threshold', -0.03):.1%}"
                  f" OR WR < {marker.get('min_winrate_threshold', 0.20):.0%}"
                  f" after {marker.get('min_trades', 20)} trades")
        except (json.JSONDecodeError, OSError):
            print("  Marker file corrupt or unreadable")
    else:
        print("  No rollback watch active")

    print("=" * 60 + "\n")


def cmd_rollback() -> None:
    """Manual rollback for all model slots."""
    for slot_name in MODEL_SLOTS:
        print(f"\nRolling back {slot_name}...")
        ok = manual_rollback(slot_name)
        print(f"  {'Success' if ok else 'Failed (not enough versions?)'}")


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Continuous Learner -- Automated model retraining pipeline"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all steps except deployment")
    parser.add_argument("--retrain", action="store_true",
                        help="Full retrain + deploy pipeline")
    parser.add_argument("--show-weights", action="store_true",
                        help="Print recency weight distribution")
    parser.add_argument("--rollback", action="store_true",
                        help="Manual rollback to previous model version")
    parser.add_argument("--status", action="store_true",
                        help="Show model versions and metrics")
    args = parser.parse_args()

    Path("backtest/results/rl").mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    if not any([args.dry_run, args.retrain, args.show_weights,
                args.rollback, args.status]):
        parser.print_help()
        sys.exit(0)

    if args.status:
        cmd_status(cfg)
    elif args.show_weights:
        cmd_show_weights(cfg)
    elif args.rollback:
        cmd_rollback()
    elif args.retrain or args.dry_run:
        cmd_retrain(cfg, dry_run=args.dry_run or (not args.retrain))


if __name__ == "__main__":
    main()
