"""Drift monitor — detects train/inference feature distribution drift.

Runs as an asyncio task inside live_multi_bot.py (spawned next to
continuous_learner). Every `interval_hours` (default 1h):
  1. Load last N signals' features from rejected_signals.entry_features
  2. Load reference training distribution from data/rl_training/*_samples.parquet
  3. Compute KS statistic + PSI per key feature
  4. Log MAJOR drift alerts (KS > 0.25 or PSI > 0.25)
  5. Persist state to data/drift_state.json for dashboard
  6. (Future) Pause continuous_learner if drift detected on critical features

Design goals:
  - Read-only relative to journal DB (WAL mode, safe concurrent)
  - No network calls (uses local parquets only)
  - Fast (~1-2s per run), runs in executor to not block event loop
  - Rolls up alerts by severity (MAJOR vs MINOR) and by feature criticality
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════

CRITICAL_FEATURES = {
    # Features the OPTFIX bug hit directly — any drift here is a P0 alert
    "alignment_score",
    "adx_1h",
}
IMPORTANT_FEATURES = {
    # Features that drive signal generation — MAJOR drift is concerning
    "struct_1d", "struct_4h", "struct_1h",
    "h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
    "precision_trigger", "has_entry_zone",
    "premium_discount", "atr_1h_norm", "rsi_1h",
}
MINOR_DRIFT_KS = 0.10
MAJOR_DRIFT_KS = 0.25
MINOR_DRIFT_PSI = 0.10
MAJOR_DRIFT_PSI = 0.25
MAX_LIVE_SIGNALS = 200  # cap for performance
PARQUET_SAMPLE_SIZE = 50_000  # subsample training parquet for stability


# ════════════════════════════════════════════════════════════════════
# Stats helpers
# ════════════════════════════════════════════════════════════════════

def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.concatenate([a_sorted, b_sorted])
    all_vals.sort()
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return 0.0
    try:
        bins = np.unique(np.quantile(e, np.linspace(0, 1, buckets + 1)))
    except Exception:
        return 0.0
    if len(bins) < 2:
        return 0.0
    e_counts, _ = np.histogram(e, bins=bins)
    a_counts, _ = np.histogram(a, bins=bins)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    e_pct = np.where(e_pct == 0, 1e-6, e_pct)
    a_pct = np.where(a_pct == 0, 1e-6, a_pct)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


# ════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════

def _load_live_features(db_path: str, max_n: int = MAX_LIVE_SIGNALS) -> pd.DataFrame:
    """Load live feature vectors from rejected_signals.entry_features.

    Returns empty DataFrame when the DB exists but the ``rejected_signals``
    table has not been created yet (fresh install, pre-bot-boot, or paper
    session that never emitted a near-miss). This is a common case during
    the pre-funded paper phase start-up; an empty return keeps the cron
    wrapper from surfacing a spurious error.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rejected_signals'"
        ).fetchone() is not None
        if not has_table:
            return pd.DataFrame()
        rows = conn.execute(f"""
            SELECT symbol, asset_class, timestamp, entry_features
            FROM rejected_signals
            WHERE entry_features IS NOT NULL
              AND entry_features != ''
              AND entry_features != '{{}}'
            ORDER BY timestamp DESC
            LIMIT {int(max_n)}
        """).fetchall()
    finally:
        conn.close()
    out = []
    for sym, ac, ts, feats_json in rows:
        try:
            feats = json.loads(feats_json)
        except (json.JSONDecodeError, TypeError):
            continue
        feats["symbol"] = sym
        feats["asset_class"] = ac
        feats["_ts"] = ts
        out.append(feats)
    return pd.DataFrame(out)


def _load_training_reference(
    asset_classes: list[str],
    symbols_hint: list[str] | None = None,
    sample_size: int = PARQUET_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Load reference features from training parquets.

    Matches target asset classes. Optionally filters to symbols seen in
    the live sample for a fair comparison (avoids cross-symbol drift noise).
    """
    data_dir = Path("data/rl_training")
    if not data_dir.exists():
        return pd.DataFrame()
    frames = []
    for ac in asset_classes:
        p = data_dir / f"{ac}_samples.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            logger.warning("drift_monitor: failed to load %s: %s", p, exc)
            continue
        if "asset_class" in df.columns:
            df = df[df["asset_class"] == ac]
        if symbols_hint and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols_hint)]
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ════════════════════════════════════════════════════════════════════
# Core analysis
# ════════════════════════════════════════════════════════════════════

def compute_drift_report(
    live_df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_list: list[str] | None = None,
) -> dict[str, Any]:
    """Compute per-feature KS + PSI + severity classification."""
    if live_df.empty or train_df.empty:
        return {
            "features": [],
            "major_count": 0,
            "minor_count": 0,
            "critical_major": [],
            "important_major": [],
            "other_major": [],
            "n_live": len(live_df),
            "n_train": len(train_df),
        }

    if feature_list is None:
        feature_list = sorted(set(live_df.columns) & set(train_df.columns) - {"symbol", "asset_class", "_ts", "timestamp", "window"})

    results = []
    critical_major, important_major, other_major = [], [], []
    major_count, minor_count = 0, 0

    for feat in feature_list:
        if feat not in live_df.columns or feat not in train_df.columns:
            continue
        live_vals = pd.to_numeric(live_df[feat], errors="coerce").dropna().values
        train_vals = pd.to_numeric(train_df[feat], errors="coerce").dropna().values
        if len(live_vals) == 0 or len(train_vals) == 0:
            continue
        ks = _ks_statistic(live_vals, train_vals)
        psi = _psi(train_vals, live_vals)

        if ks > MAJOR_DRIFT_KS or psi > MAJOR_DRIFT_PSI:
            severity = "MAJOR"
            major_count += 1
            if feat in CRITICAL_FEATURES:
                critical_major.append(feat)
            elif feat in IMPORTANT_FEATURES:
                important_major.append(feat)
            else:
                other_major.append(feat)
        elif ks > MINOR_DRIFT_KS or psi > MINOR_DRIFT_PSI:
            severity = "MINOR"
            minor_count += 1
        else:
            severity = "OK"

        results.append({
            "feature": feat,
            "ks": round(ks, 4),
            "psi": round(psi, 4),
            "live_mean": round(float(live_vals.mean()), 5),
            "train_mean": round(float(train_vals.mean()), 5),
            "live_std": round(float(live_vals.std()), 5),
            "train_std": round(float(train_vals.std()), 5),
            "severity": severity,
            "is_critical": feat in CRITICAL_FEATURES,
        })

    return {
        "features": results,
        "major_count": major_count,
        "minor_count": minor_count,
        "critical_major": critical_major,
        "important_major": important_major,
        "other_major": other_major,
        "n_live": len(live_df),
        "n_train": len(train_df),
    }


def _should_alert(report: dict[str, Any]) -> tuple[bool, str]:
    """Decide if the report merits an alert, and at what level."""
    if report.get("critical_major"):
        return True, "CRITICAL"
    if report.get("important_major"):
        return True, "WARNING"
    return False, "INFO"


# ════════════════════════════════════════════════════════════════════
# Runtime entrypoint
# ════════════════════════════════════════════════════════════════════

def run_drift_check_once(
    db_path: str,
    asset_classes: list[str] | None = None,
    state_file: str = "data/drift_state.json",
) -> dict[str, Any]:
    """Execute one drift check cycle (blocking, for executor use)."""
    live_df = _load_live_features(db_path)
    if live_df.empty:
        logger.info("drift_monitor: no live signals in journal yet")
        return {"ok": False, "reason": "no_live_data"}

    # Determine asset classes + symbols to compare
    if asset_classes is None:
        asset_classes = sorted(live_df["asset_class"].dropna().unique().tolist())
    symbols_hint = sorted(live_df["symbol"].dropna().unique().tolist())

    train_df = _load_training_reference(asset_classes, symbols_hint=symbols_hint)
    if train_df.empty:
        logger.warning("drift_monitor: no training reference data for %s", asset_classes)
        return {"ok": False, "reason": "no_train_data"}

    report = compute_drift_report(live_df, train_df)
    should_alert, level = _should_alert(report)

    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report["asset_classes"] = asset_classes
    report["should_alert"] = should_alert
    report["alert_level"] = level

    # Log
    if level == "CRITICAL":
        logger.error(
            "[DRIFT CRITICAL] %d major, %d minor. Critical features: %s. Important: %s",
            report["major_count"], report["minor_count"],
            report["critical_major"], report["important_major"],
        )
    elif level == "WARNING":
        logger.warning(
            "[DRIFT WARNING] %d major, %d minor. Important features: %s",
            report["major_count"], report["minor_count"],
            report["important_major"],
        )
    else:
        logger.info(
            "drift_monitor: OK — %d major, %d minor (n_live=%d, n_train=%d)",
            report["major_count"], report["minor_count"],
            report["n_live"], report["n_train"],
        )

    # Persist state
    try:
        state_path = Path(state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    except OSError as exc:
        logger.warning("drift_monitor: state persist failed: %s", exc)

    return report


async def run_drift_monitor(
    config: dict,
    shutdown_event: asyncio.Event,
) -> None:
    """Main async loop — runs in the live bot's event loop."""
    dm_cfg = config.get("drift_monitor", {})
    if not dm_cfg.get("enabled", True):  # default ON — monitoring is safe
        logger.info("drift_monitor: disabled in config")
        return

    interval_hours = dm_cfg.get("interval_hours", 1)
    interval_sec = max(60, interval_hours * 3600)  # min 1 min for testing
    journal_cfg = config.get("journal", {})
    db_path = journal_cfg.get("db_path", "trade_journal/journal.db")
    state_file = dm_cfg.get("state_file", "data/drift_state.json")

    logger.info(
        "drift_monitor started: interval=%.1fh, critical features=%s",
        interval_hours, list(CRITICAL_FEATURES),
    )

    loop = asyncio.get_running_loop()
    # First check after 60s (give bot time to boot and collect initial signals)
    initial_delay = dm_cfg.get("initial_delay_sec", 60)
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=initial_delay)
        logger.info("drift_monitor: shutdown before first check")
        return
    except asyncio.TimeoutError:
        pass

    while True:
        try:
            await loop.run_in_executor(None, run_drift_check_once, db_path, None, state_file)
        except Exception:
            logger.exception("drift_monitor: check failed")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_sec)
            logger.info("drift_monitor: shutdown signal received")
            break
        except asyncio.TimeoutError:
            pass  # normal interval elapsed


# ════════════════════════════════════════════════════════════════════
# CLI entrypoint (for manual runs / testing)
# ════════════════════════════════════════════════════════════════════

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="trade_journal/journal.db")
    parser.add_argument("--state-file", default="data/drift_state.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report = run_drift_check_once(args.db, state_file=args.state_file)
    print(json.dumps({
        "ok": report.get("ok", True),
        "alert_level": report.get("alert_level"),
        "major_count": report.get("major_count"),
        "critical_major": report.get("critical_major"),
        "important_major": report.get("important_major"),
        "n_live": report.get("n_live"),
        "n_train": report.get("n_train"),
    }, indent=2, default=str))
    if args.verbose and "features" in report:
        print("\n=== Per-feature table ===")
        print(pd.DataFrame(report["features"]).to_string(index=False))


if __name__ == "__main__":
    main()
