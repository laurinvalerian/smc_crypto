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
import gc
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
    # Base grade trust level. Combined with outcome multipliers + D-loss boost
    # downstream to produce the final sample weight.
    # Design rationale (2026-04-17 tuning):
    #   Old design used D=1.5 (same as A+) to "learn from bad trades harder".
    #   Problem: that boosted D+WIN samples too — lucky wins on poor setups
    #   got the same weight as confirmed A+ wins. The model then learned to
    #   chase bad patterns.
    #   New design: D base is LOW (0.5), but D+LOSS gets a post-hoc boost
    #   in the weight loop so "reject bad-grade mistakes" intent is preserved
    #   without "chase lucky bad-grade wins."
    "A+": 1.5, "A": 1.5, "B+": 1.0, "B": 1.0, "C": 0.8, "D": 0.5,
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
                        # live_teacher writes under "teacher" key; legacy records
                        # may still use "analysis" — check both.
                        grade = (
                            rec.get("teacher", {}).get("grade")
                            or rec.get("analysis", {}).get("grade")
                            or ""
                        )
                        if tid and grade and grade != "N/A":
                            grade_lookup[str(tid)] = grade
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
            "score, style, asset_class, direction, rr_actual, max_favorable_pct, rr_target "
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
        record["rr_actual"] = row["rr_actual"] or 0.0
        record["max_favorable_pct"] = row["max_favorable_pct"] or 0.0
        record["rr_target"] = row["rr_target"] or 3.0
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

    # Track per-row grade (needed for outcome-conditional boost below)
    trade_grades = [grade_lookup.get(tid, "") for tid in df["trade_id"]]

    for i, grade in enumerate(trade_grades):
        weights[i] *= grade_weights.get(grade, 1.0)

    # tp_achievement-based weighting (MFE / TP_distance) + grade×outcome conditional
    if "rr_actual" in df.columns and "max_favorable_pct" in df.columns:
        rr_actual = df["rr_actual"].abs().values
        mfe = df["max_favorable_pct"].abs().values
        tp_target = df["rr_target"].values if "rr_target" in df.columns else np.full(len(df), 3.0)
        tp_target = np.maximum(tp_target, 0.1)
        tp_achievement = np.clip(mfe / tp_target, 0.0, 1.5)

        exit_reason = df["exit_reason"].values
        outcome = df["label"].values  # 1=win, 0=loss

        for i in range(len(df)):
            er = exit_reason[i] if i < len(exit_reason) else ""
            tpa = tp_achievement[i]
            rr = rr_actual[i]
            grade = trade_grades[i]

            # Sanity: tp_hit must be win, sl_hit must be loss — fix mislabeled journal rows
            if er == "tp_hit" and outcome[i] == 0:
                er = "manual"
            elif er == "sl_hit" and outcome[i] == 1:
                er = "manual"

            if er == "tp_hit":
                weights[i] *= max(rr, 1.0)  # full win weight
            elif er == "sl_hit":
                w = min(rr, 1.5)
                if tpa > 0.60:
                    w *= 0.3
                elif tpa > 0.20:
                    w *= 0.7
                weights[i] *= max(w, 0.1)
            elif er == "timeout":
                if tpa >= 0.60:
                    weights[i] *= min(tpa * max(rr, 1.0), 4.0)
                elif tpa >= 0.20:
                    weights[i] *= 0.3
                else:
                    weights[i] *= 0.5
            elif "be" in str(er):
                weights[i] *= 0.3 if tpa >= 0.30 else 0.1

            # D-grade loss boost: recover the "learn from bad-grade mistakes" intent
            # without boosting D-grade lucky wins. Multiplier 3.0 ≈ restoring the old
            # D=1.5 base weight for this specific slice (new D=0.5 × 3.0 = 1.5).
            if grade == "D" and outcome[i] == 0:
                weights[i] *= 3.0

    # Cap combined weights — raised from 6.0 to 10.0 so large RR winners keep their
    # signal strength. With grade=1.5 × base=2.0 × rr=3.3 already hits old cap.
    weights = np.clip(weights, 0.1, 10.0)

    df["sample_weight"] = weights

    # Drop helper columns
    df.drop(columns=["trade_id", "outcome", "pnl_pct", "exit_reason", "rr_actual", "max_favorable_pct", "rr_target"], inplace=True)

    logger.info("Collected %d live trades (%d wins, %d losses)",
                len(df), int(df["label"].sum()), int((df["label"] == 0).sum()))
    return df


# ===================================================================
#  1b. Collect counterfactual data from rejected signals
# ===================================================================

def collect_counterfactual_data(
    db_path: str,
    horizon: str = "scalp",
) -> pd.DataFrame:
    """Load rejected signals with filled outcomes for counterfactual learning.

    Deduplicates by (symbol, direction, 30-min window) to avoid overweighting
    repeated rejections of the same setup.

    Labels: outcome=="win" → 1 (model should have accepted),
            outcome=="loss" → 0 (model correctly rejected),
            outcome=="timeout" → based on MFE (>60% of TP distance = 1, else 0).
    Weight: 0.3× base (lower than actual trades, counterfactual uncertainty).
    """
    if not Path(db_path).exists():
        return pd.DataFrame()

    outcome_col = f"outcome_{horizon}"
    mfe_col = f"mfe_{horizon}"

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        if horizon not in ("scalp", "day", "swing"):
            return pd.DataFrame()
        cursor = conn.execute(
            f"SELECT id, timestamp, symbol, direction, entry_price, sl_price, tp_price, "
            f"entry_features, xgb_confidence, {outcome_col}, {mfe_col} "
            f"FROM rejected_signals "
            f"WHERE {outcome_col} IS NOT NULL "
            f"AND entry_features IS NOT NULL "
            f"AND entry_features != '{{}}' "
            # Include both XGB-rejected (conf>=0) and capacity-rejected (conf=-1)
            f"ORDER BY timestamp ASC"
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error("Counterfactual query failed: %s", exc)
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()

    if not rows:
        logger.info("No counterfactual data with filled %s outcomes", horizon)
        return pd.DataFrame()

    # ── Deduplicate: one sample per (symbol, direction, 30-min window) ──
    deduped: list[sqlite3.Row] = []
    seen: dict[tuple[str, str], str] = {}  # (sym, dir) -> last_ts
    for row in rows:
        key = (row["symbol"], row["direction"])
        ts_str = row["timestamp"]
        last_ts = seen.get(key)
        if last_ts:
            try:
                t_cur = datetime.fromisoformat(ts_str)
                t_last = datetime.fromisoformat(last_ts)
                if abs((t_cur - t_last).total_seconds()) < 1800:
                    continue  # duplicate within 30-min window
            except (ValueError, TypeError):
                pass
        seen[key] = ts_str
        deduped.append(row)

    logger.info("Counterfactual dedup: %d → %d samples", len(rows), len(deduped))

    # ── Parse features and build training rows ──────────────────────
    feat_names = list(ENTRY_QUALITY_FEATURES)
    records: list[dict[str, Any]] = []

    for row in deduped:
        try:
            feats = json.loads(row["entry_features"])
        except (json.JSONDecodeError, TypeError):
            continue

        missing = [f for f in feat_names if f not in feats]
        if missing:
            continue

        record: dict[str, Any] = {f: feats[f] for f in feat_names}

        outcome = row[outcome_col]
        mfe = row[mfe_col] or 0.0

        # Label assignment
        if outcome == "win":
            record["label"] = 1  # model was wrong to reject
        elif outcome == "loss":
            record["label"] = 0  # model correctly rejected
        else:  # timeout
            # Use MFE heuristic: if price moved >60% toward TP, it was a good trade
            entry = row["entry_price"] or 0.0
            sl = row["sl_price"] or 0.0
            tp = row["tp_price"] or 0.0
            if entry > 0 and tp > 0 and sl > 0:
                tp_dist = abs(tp - entry) / entry
                mfe_ratio = mfe / tp_dist if tp_dist > 0 else 0.0
                record["label"] = 1 if mfe_ratio > 0.6 else 0
            else:
                record["label"] = 0

        # Lower weight: counterfactual data has inherent uncertainty
        record["sample_weight"] = 0.3
        records.append(record)

    if not records:
        logger.info("No counterfactual samples with complete features")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    n_pos = int(df["label"].sum())
    logger.info("Counterfactual data: %d samples (%d positive, %d negative)",
                len(df), n_pos, len(df) - n_pos)
    return df


# ───────────────────────────────────────────────────────────────────
#  Focal counterfactual collector (missed-wins only, style-matched)
# ───────────────────────────────────────────────────────────────────

# style_id mapping in entry_features (see live_multi_bot.py:2428)
_STYLE_ID_TO_HORIZON = {
    0.0: ("scalp", "outcome_scalp"),
    0.5: ("day",   "outcome_day"),
    1.0: ("swing", "outcome_swing"),
}


def collect_counterfactual_focal_wins(db_path: str) -> pd.DataFrame:
    """Focal-loss counterfactual collector: ONLY missed wins, style-matched.

    Philosophy: the entry model only learns from its mistakes. Rejected
    signals that turned into real TP hits ARE mistakes — the model should
    have accepted them. Losses and timeouts are confirmations or noise and
    don't teach the model anything new.

    Design choices:
      1. Query the journal ONCE for all rejected signals with populated features
      2. For each row, parse style_id from entry_features (0.0=scalp, 0.5=day, 1.0=swing)
      3. Look up the style-matching outcome column (avoids multi-horizon duplication)
      4. Keep only rows where outcome == "win" (real TP hit, no MFE heuristic)
      5. Label = 1, weight = 1.0 (same as live trades — missed wins are as important)
      6. Dedup by (symbol, direction, 30-min window) to prevent repeat-signal bias

    Why not also include losses/timeouts?
      - Losses: model was right to reject, confirmation has low info value
      - Timeouts: neither TP nor SL hit, ambiguous — MFE heuristic was noisy
      - Backtest + live trades already cover these directions
      - Focal loss says: weight mistakes, ignore confirmations

    Returns a DataFrame with label=1 on all rows, weight=1.0, matching
    ENTRY_QUALITY_FEATURES schema.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, timestamp, symbol, direction, entry_price, sl_price, tp_price, "
            "entry_features, xgb_confidence, "
            "outcome_scalp, outcome_day, outcome_swing "
            "FROM rejected_signals "
            "WHERE entry_features IS NOT NULL "
            "  AND entry_features != '{}' "
            "ORDER BY timestamp ASC"
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        logger.error("Focal counterfactual query failed: %s", exc)
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()

    if not rows:
        logger.info("focal_cf: no rejected_signals with features")
        return pd.DataFrame()

    # ── Per-row: parse style, look up matching outcome, keep only wins ──
    feat_names = list(ENTRY_QUALITY_FEATURES)
    candidates: list[dict[str, Any]] = []
    n_scanned = 0
    n_wins_by_style = {"scalp": 0, "day": 0, "swing": 0, "unknown": 0}
    for row in rows:
        n_scanned += 1
        try:
            feats = json.loads(row["entry_features"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Determine style from features (fallback: day if missing/unknown)
        style_id_raw = feats.get("style_id", 0.5)
        try:
            style_id = float(style_id_raw)
        except (TypeError, ValueError):
            style_id = 0.5
        # Round to nearest known style_id to survive float comparisons
        style_id_nearest = min(
            _STYLE_ID_TO_HORIZON.keys(),
            key=lambda k: abs(k - style_id),
        )
        style_name, outcome_col = _STYLE_ID_TO_HORIZON[style_id_nearest]

        # Look up the style-matching outcome (only one per signal — no duplication)
        outcome = row[outcome_col]
        if outcome != "win":
            continue  # skip: loss, timeout, or not-yet-filled

        # Verify all features present
        missing = [f for f in feat_names if f not in feats]
        if missing:
            continue

        record: dict[str, Any] = {f: feats[f] for f in feat_names}
        record["label"] = 1  # missed win → model should have accepted
        record["sample_weight"] = 1.0  # equal to live trades — critical learning signal
        record["_style"] = style_name
        record["_signal_ts"] = row["timestamp"]
        record["_symbol"] = row["symbol"]
        record["_direction"] = row["direction"]
        candidates.append(record)
        n_wins_by_style[style_name] = n_wins_by_style.get(style_name, 0) + 1

    if not candidates:
        logger.info("focal_cf: scanned %d rows, 0 missed wins found (model has no acceptance mistakes yet)", n_scanned)
        return pd.DataFrame()

    # ── Dedup: one sample per (symbol, direction, 30-min window) ──
    # Prevents the same rejected setup from being counted multiple times
    # when it keeps triggering over consecutive bars.
    deduped: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    for rec in candidates:
        key = (rec["_symbol"], rec["_direction"])
        ts_str = rec["_signal_ts"]
        last_ts = seen.get(key)
        if last_ts:
            try:
                t_cur = datetime.fromisoformat(ts_str)
                t_last = datetime.fromisoformat(last_ts)
                if abs((t_cur - t_last).total_seconds()) < 1800:
                    continue
            except (ValueError, TypeError):
                pass
        seen[key] = ts_str
        # Strip metadata columns
        clean = {k: v for k, v in rec.items() if not k.startswith("_")}
        deduped.append(clean)

    df = pd.DataFrame(deduped)
    logger.info(
        "focal_cf: scanned %d rows → %d missed wins by style %s → %d after dedup",
        n_scanned,
        len(candidates),
        dict((k, v) for k, v in n_wins_by_style.items() if v > 0),
        len(df),
    )
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

        # bug_005 fix (2026-04-17): schema version / column-presence check.
        # Primary gate: the parquet must contain all ENTRY_QUALITY_FEATURES
        # (41 features). Silent zero-fill is still allowed for a SMALL number
        # of missing features (graceful handling of minor schema evolution),
        # but >5 missing columns means the parquet is schema-incompatible and
        # we refuse to train on it. Metadata SCHEMA_VERSION marker is optional
        # and only meaningful for parquets regenerated after this version.
        missing = [f for f in feat_names if f not in raw.columns]
        if len(missing) > 5:
            logger.error("%s SCHEMA DRIFT: %d/%d features missing (>%d threshold) — "
                         "skipping. Regenerate via `python3 -m backtest.generate_rl_data --class %s`. "
                         "Missing: %s",
                         parquet_path, len(missing), len(feat_names), 5, cls,
                         missing[:10] + (["..."] if len(missing) > 10 else []))
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

        if missing:
            # Minor drift (<=5 missing) — fill with 0 + warning
            for m in missing:
                raw[m] = 0.0
            logger.warning("%s minor schema drift — %d feature(s) filled with 0: %s",
                           cls, len(missing), missing)

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

    # ── Load counterfactual data from rejected signals ───────────
    # Mode selection:
    #   "focal" (default): only missed wins, style-matched, weight=1.0
    #     → model learns "take these, you missed them" — pure correction signal
    #     → no multi-horizon duplication, no MFE heuristic noise
    #   "all_horizons": legacy — all 3 horizons concatenated, weight=0.3
    #     → same signal may appear up to 3×, losses/timeouts included
    cf_mode = cl_cfg.get("counterfactual_mode", "focal")
    if cf_mode == "focal":
        counterfactual_df = collect_counterfactual_focal_wins(db_path)
    else:
        cf_frames = []
        for hz in ("scalp", "day", "swing"):
            cf = collect_counterfactual_data(db_path, horizon=hz)
            if len(cf) > 0:
                cf_frames.append(cf)
        counterfactual_df = pd.concat(cf_frames, ignore_index=True) if cf_frames else pd.DataFrame()

    # ── Walk-forward split: last 30% of live = validation ────────
    feat_names = list(ENTRY_QUALITY_FEATURES)
    split_idx = n_live - n_val

    live_train = live_df.iloc[:split_idx]
    live_val = live_df.iloc[split_idx:]

    # ── Combine backtest + live train + counterfactual ───────────
    frames = [f for f in [backtest_df, live_train, counterfactual_df] if len(f) > 0]
    train_df = pd.concat(frames, ignore_index=True) if frames else live_train.copy()

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
#  3b. Retrain exit models (TP, SL, BE) from live trade outcomes
# ===================================================================

def retrain_exit_models_if_ready(config: dict) -> dict[str, bool]:
    """Retrain TP/SL/BE models from live trade data + backtest data.

    Returns dict of {model_name: retrained_bool}.
    These models use ENTRY features to predict EXIT outcomes:
    - TP: predict max favorable excursion (MFE) → optimize TP distance
    - SL: predict optimal SL category (tighten/keep/widen) from MAE
    - BE: predict optimal BE trigger level from outcome + MFE
    """
    import xgboost as xgb

    cl_cfg = config.get("continuous_learner", {})
    journal_cfg = config.get("journal", {})
    db_path = journal_cfg.get("db_path", "trade_journal/journal.db")
    min_trades = cl_cfg.get("min_trades_for_retrain", 50)
    subsample_per_class = cl_cfg.get("backtest_subsample_per_class", 50_000)

    results: dict[str, bool] = {"tp": False, "sl": False, "be": False}

    # ── Memory guard ─────────────────────────────────────────────
    try:
        import psutil
        if psutil.virtual_memory().percent > 80:
            logger.warning("Memory > 80%%, skipping exit model retrain")
            return results
    except ImportError:
        pass

    # ── Collect live trade outcomes from journal ─────────────────
    if not Path(db_path).exists():
        return results

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT entry_features, outcome, exit_reason, pnl_pct, rr_actual, rr_target, "
            "max_favorable_pct, max_adverse_pct, asset_class, entry_price, sl_original "
            "FROM trades WHERE exit_time IS NOT NULL AND entry_features IS NOT NULL"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.error("Exit model retrain: DB query failed: %s", exc)
        return results
    finally:
        if conn:
            conn.close()

    if len(rows) < min_trades:
        logger.info("Only %d closed trades with features (need %d), skipping exit retrain",
                    len(rows), min_trades)
        return results

    # ── Build DataFrame from live trades ─────────────────────────
    feat_names = list(ENTRY_QUALITY_FEATURES)
    live_rows = []

    for row in rows:
        features_json, outcome, exit_reason, pnl_pct, rr_actual, rr_target, mfe_pct, mae_pct, ac, entry_price, sl_original = row
        try:
            feats = json.loads(features_json) if isinstance(features_json, str) else {}
        except json.JSONDecodeError:
            continue

        # Map outcome to numeric: 1=win, 2=loss, 3=breakeven
        outcome_num = {"win": 1, "loss": 2, "breakeven": 3}.get(outcome, 0)
        if outcome_num == 0:
            continue

        # Map exit_reason to mechanism: 1=tp, 2=sl, 3=be, 4=timeout
        exit_mech = {"tp_hit": 1, "sl_hit": 2, "be_hit": 3, "timeout": 4}.get(exit_reason, 4)

        planned_tp_rr = rr_target or 3.0
        actual_rr = rr_actual or 0.0

        # Convert MFE/MAE from pct to R-multiples using exact SL distance from journal
        sl_pct = abs(sl_original - entry_price) / entry_price if entry_price and sl_original else 0.0
        if sl_pct > 0.00005:  # minimum viable sl_pct (0.005%)
            mfe_rr = min((mfe_pct or 0.0) / sl_pct, 20.0)
            mae_rr = min((mae_pct or 0.0) / sl_pct, 5.0)
        else:
            mfe_rr = 0.0
            mae_rr = 0.0

        entry = {fn: feats.get(fn, 0.0) for fn in feat_names}
        entry["label_outcome"] = outcome_num
        entry["label_exit_mechanism"] = exit_mech
        entry["label_rr"] = actual_rr
        entry["label_tp_rr"] = planned_tp_rr
        entry["label_max_favorable_rr"] = max(mfe_rr, 0.0)
        entry["label_mae_rr"] = max(mae_rr, 0.0)
        entry["label_cost_rr"] = 0.01  # approximate
        # bug_004 fix (2026-04-17): live trades have no post-TP tracking yet
        # (Exit-Model-Suite roadmap). Previously we emitted NaN here which, via
        # _tp_labels fallback, propagated as MFE-capped values anyway. We now
        # write the MFE directly so the retrain pipeline doesn't silently swallow
        # NaN rows.
        # KNOWN LIMITATION: for live TP-hits, MFE is capped at the TP level, so
        # the resulting post_tp_max_rr reads as "MFE = TP exactly" while backtest
        # TP-hits capture the actual post-exit excursion. This biases the TP
        # model toward treating TP-level as a ceiling for live-like setups. The
        # full fix requires Post-TP live tracking (see .omc/plans/exit-model-suite.md).
        entry["label_post_tp_max_rr"] = max(mfe_rr, 0.0)
        entry["label_post_tp_reversal_rr"] = 0.0  # no reversal observed ≠ NaN
        entry["sample_weight"] = 2.0  # live trades weighted 2x vs backtest (matches entry filter)
        entry["asset_class"] = ac or "unknown"
        live_rows.append(entry)

    if not live_rows:
        return results

    live_df = pd.DataFrame(live_rows)
    logger.info("Exit model retrain: %d live trades with features", len(live_df))

    # ── Load backtest data for blending ──────────────────────────
    # bug_005 fix (2026-04-17): schema-drift check before concat. If a parquet
    # is missing required label columns (rather than just features), the
    # downstream label_fn will produce NaN labels and poison training.
    _required_exit_cols = {
        "label_outcome", "label_action", "label_mae_rr", "label_max_favorable_rr",
        "label_rr", "label_tp_rr", "label_cost_rr", "label_exit_mechanism",
    }
    backtest_frames = []
    for ac in ALL_CLASSES:
        path = DATA_DIR / f"{ac}_samples.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error("Exit retrain: failed to read %s: %s — skipping", path, exc)
            continue
        missing_cols = _required_exit_cols - set(df.columns)
        if missing_cols:
            logger.error("Exit retrain: %s SCHEMA DRIFT — missing label columns %s, "
                         "skipping (regenerate with generate_rl_data)",
                         path, sorted(missing_cols))
            continue
        missing_feats = [f for f in ENTRY_QUALITY_FEATURES if f not in df.columns]
        if len(missing_feats) > 5:
            logger.error("Exit retrain: %s SCHEMA DRIFT — %d/%d features missing, skipping",
                         path, len(missing_feats), len(ENTRY_QUALITY_FEATURES))
            continue
        entries = df[df["label_action"] > 0]
        if len(entries) > subsample_per_class:
            entries = entries.sample(n=subsample_per_class, random_state=42)
        entries = entries.copy()
        entries["sample_weight"] = 1.0
        entries["asset_class"] = ac
        backtest_frames.append(entries)

    if not backtest_frames:
        logger.warning("No backtest data for exit model blending")
        return results

    backtest_df = pd.concat(backtest_frames, ignore_index=True)
    del backtest_frames
    gc.collect()

    # ── Stratified 80/20 split BEFORE concat (bug_002 fix) ──────
    # Old code concatenated then split temporally → live trades (always
    # "newest") landed entirely in test set, so their sample_weight=2.0
    # boost never influenced training. Here we split each source
    # independently so both train + test contain live signal.
    logger.info("Exit model retrain: backtest=%d, live=%d samples", len(backtest_df), len(live_df))

    # Sort each by its own temporal key
    if "timestamp" in backtest_df.columns:
        backtest_df = backtest_df.sort_values("timestamp").reset_index(drop=True)
    elif "window" in backtest_df.columns:
        backtest_df = backtest_df.sort_values("window").reset_index(drop=True)
    # live_df is already appended in journal insertion order

    bt_split = int(len(backtest_df) * 0.8)
    live_split = int(len(live_df) * 0.8)

    train_df = pd.concat(
        [backtest_df.iloc[:bt_split], live_df.iloc[:live_split]],
        ignore_index=True,
    )
    test_df = pd.concat(
        [backtest_df.iloc[bt_split:], live_df.iloc[live_split:]],
        ignore_index=True,
    )
    del backtest_df, live_df
    gc.collect()

    logger.info("Exit model retrain: train=%d, test=%d samples (live mixed into both)",
                len(train_df), len(test_df))

    # ── Retrain each exit model ──────────────────────────────────
    # Phase 2.5 cleanup (2026-04-18): tp_optimizer + sl_adjuster removed from
    # the retrain loop. Both were disabled in default_config.yaml after the
    # 2026-04-17 audit (TP: PF drops -0.03..-0.19 in 4/5 folds; SL: 0/5.3M
    # WIDEN predictions = Tighten-only classifier). Continuing to retrain
    # them wasted CPU on a 4-core/8GB server AND silently overwrote pickles
    # that were never reviewed against the disable rationale (Ultrareview
    # Bug_019). Only the BE-Manager remains here — and only because BE is
    # explicitly enabled in config (rl_brain.be_manager.enabled: true).
    # Phase 2.4 will decide Student-Brain vs Legacy entry-filter; until then,
    # entry filter retrains stay in retrain_if_ready() above (separate path).
    from rl_brain_v2 import (
        derive_optimal_be_label, _be_eval,
        prepare_features, META_COLS,
    )

    for task_name, label_fn, eval_fn, model_filename in [
        ("be", derive_optimal_be_label, _be_eval, "rl_be_manager.pkl"),
    ]:
        try:
            logger.info("[exit_retrain] Training %s model...", task_name)

            # Prepare features using canonical pipeline (clipping, asset_class_id, style_id).
            # task="entry_quality" excludes `has_entry_zone` (schema v3) — exit models are
            # trained on the same feature set as the entry filter for consistency (bug_007).
            X_train, feat_cols = prepare_features(train_df, task="entry_quality")
            X_test, _ = prepare_features(test_df, task="entry_quality")

            y_train = label_fn(train_df)
            y_test = label_fn(test_df)
            w_train = train_df["sample_weight"].values.astype(np.float64)

            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=10, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="rmse", early_stopping_rounds=30,
                n_jobs=2, random_state=42, tree_method="hist",
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                      sample_weight=w_train, verbose=False)

            logger.info("[exit_retrain] %s trained: %d trees", task_name, model.best_iteration + 1)

            # Evaluate
            eval_result = eval_fn(model, X_test, test_df, feat_cols)
            logger.info("[exit_retrain] %s eval: %s",
                        task_name, {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in eval_result.items() if not isinstance(v, (dict, list))})

            # Gate check
            # NOTE: DD values are always <= 0 (sum of negative drawdowns). "greater" means
            # "less negative" = shallower drawdown = better. Same semantic as BE-Gate below.
            # bug_001 (2026-04-17): SL gate was PF-only — a model with +0.1 PF but 2× worse
            # DD could pass. Now SL requires BOTH pf AND dd to improve, analog to BE.
            # TP gate stays PF-only intentionally: _tp_eval returns no DD metric.
            gate_passed = False
            gate_reason = ""
            if task_name == "tp":
                pf_ok = eval_result.get("adjusted_pf", 0) > eval_result.get("original_pf", 0)
                gate_passed = pf_ok
                gate_reason = f"pf {eval_result.get('adjusted_pf',0):.3f} vs {eval_result.get('original_pf',0):.3f} (PF-only: _tp_eval has no DD)"
            elif task_name == "be":
                dd_ok = eval_result.get("model_be_dd", 0) > eval_result.get("no_be_dd", 0)
                pf_ok = eval_result.get("model_be_pf", 0) >= 0.9 * eval_result.get("no_be_pf", 0)
                gate_passed = dd_ok and pf_ok
                gate_reason = f"dd_ok={dd_ok} pf_ok={pf_ok}"
            elif task_name == "sl":
                pf_ok = eval_result.get("adjusted_pf", 0) > eval_result.get("original_pf", 0)
                dd_ok = eval_result.get("adjusted_dd", 0) > eval_result.get("original_dd", 0)
                gate_passed = pf_ok and dd_ok
                gate_reason = (
                    f"pf_ok={pf_ok} (adj={eval_result.get('adjusted_pf',0):.3f} vs "
                    f"orig={eval_result.get('original_pf',0):.3f}) "
                    f"dd_ok={dd_ok} (adj={eval_result.get('adjusted_dd',0):.1f} vs "
                    f"orig={eval_result.get('original_dd',0):.1f})"
                )

            if not gate_passed:
                logger.info("[exit_retrain] %s gate FAILED, keeping existing model (%s)",
                            task_name, gate_reason)
                continue
            logger.info("[exit_retrain] %s gate passed: %s", task_name, gate_reason)

            # Save with backup
            model_path = MODEL_DIR / model_filename
            backup_path = MODEL_DIR / model_filename.replace(".pkl", "_prev.pkl")
            if model_path.exists():
                shutil.copy2(str(model_path), str(backup_path))

            model_data = {
                "model": model,
                "feat_names": feat_cols,
                "task": task_name,
                "schema_version": SCHEMA_VERSION,
                "dead_features": list(DEAD_FEATURES),
                "clip_ranges": CLIP_RANGES,
                "asset_class_map": ASSET_CLASS_MAP,
            }
            tmp = model_path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(model_data, f)
            os.rename(str(tmp), str(model_path))

            results[task_name] = True
            logger.info("[exit_retrain] %s SAVED: %s (gate passed)", task_name, model_path)

        except Exception:
            logger.exception("[exit_retrain] %s training failed", task_name)

    return results


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
        "sl_adjuster": "rl_sl_adjuster",
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
                logger.info("Continuous learner: entry filter retrain succeeded")
        except Exception:
            logger.exception("Continuous learner: entry filter retrain failed")

        # Retrain exit models (TP, SL, BE) from live trade outcomes
        try:
            exit_results = await loop.run_in_executor(
                None, retrain_exit_models_if_ready, config,
            )
            retrained_models = [k for k, v in exit_results.items() if v]
            if retrained_models:
                logger.info("Continuous learner: exit models retrained: %s", retrained_models)
        except Exception:
            logger.exception("Continuous learner: exit model retrain failed")

        try:
            rolled_back = await loop.run_in_executor(
                None, check_auto_rollback, config, db_path,
            )
            if rolled_back:
                logger.warning("Continuous learner: auto-rollback executed")
        except Exception:
            logger.exception("Continuous learner: rollback check failed")
