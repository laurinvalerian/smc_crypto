"""
backtest/shadow_replay_entry_filter.py
=======================================
Replays the live XGB REJECT / XGB ACCEPT decisions from paper_trading.log
against all three candidate models:
  - OLD model (currently live)
  - Model A  (fair retrain, cutoff 2026-04-08)
  - Model B  (production retrain, all data)

This is the "most realistic" evaluation: instead of abstract backtest metrics,
it looks at exactly the signals the live bot rejected and asks:
  (a) Which of these would the new models have accepted?
  (b) For each decision, what does the generated sample say the outcome would have been?

Data flow:
  paper_trading.log  ─────┐
                          ├─→ match by (symbol, timestamp) ─→  features
  rl_training/*.parquet ──┘

For each rejected signal we get:
  - conf_old    (from log, already known)
  - conf_new_A  (re-score with Model A's feature vector)
  - conf_new_B  (re-score with Model B's feature vector)
  - label       (from samples parquet: outcome/RR)
  - accept@0.55 decision per model

Run:
    python3 -m backtest.shadow_replay_entry_filter
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_brain_v2 import (
    DATA_DIR,
    MODEL_DIR,
    load_training_data,
    prepare_features,
    prepare_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LOG_PATH = Path("/tmp/ab_test/live_logs/paper_trading.log")
OLD_MODEL_PATH = MODEL_DIR / "rl_entry_filter.pkl"
MODEL_A_PATH = MODEL_DIR / "rl_entry_filter_fair.pkl"
MODEL_B_PATH = MODEL_DIR / "rl_entry_filter_new.pkl"
# Post train/inference mismatch fix (per-cluster optimized SMC params).
# Loaded if present; ignored otherwise.
MODEL_OPTFIX_PATH = MODEL_DIR / "rl_entry_filter_optfix.pkl"
RESULTS_PATH = Path("backtest/results/shadow_replay_results.json")

START_TS = pd.Timestamp("2026-04-08", tz="UTC")
END_TS = pd.Timestamp("2026-04-15", tz="UTC")
LIVE_THRESHOLD = 0.55

# Live log symbol conventions use CCXT-style for crypto ("BTC/USDT:USDT")
# but samples parquet uses plain "BTCUSDT". Normalize with a map.
def _normalize_symbol(sym: str) -> str:
    s = sym.strip()
    # Crypto: "BTC/USDT:USDT" -> "BTCUSDT"
    if "/" in s:
        base, rest = s.split("/", 1)
        quote = rest.split(":", 1)[0]
        return f"{base}{quote}"
    return s


# ═══════════════════════════════════════════════════════════════════
#  Parse paper_trading.log
# ═══════════════════════════════════════════════════════════════════

XGB_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] "
    r"XGB (?P<decision>REJECT|ACCEPT) (?P<symbol>\S+) conf=(?P<conf>[0-9.]+) "
    r"score=(?P<score>[0-9.]+)"
)


def parse_log(path: Path) -> pd.DataFrame:
    """Extract all XGB decisions from the paper log."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = XGB_LINE.match(line)
            if not m:
                continue
            ts = pd.Timestamp(m.group("ts"), tz="UTC")
            if ts < START_TS or ts >= END_TS:
                continue
            rows.append({
                "log_ts": ts,
                "symbol_raw": m.group("symbol"),
                "symbol": _normalize_symbol(m.group("symbol")),
                "decision": m.group("decision"),
                "conf_live": float(m.group("conf")),
                "score": float(m.group("score")),
            })
    df = pd.DataFrame(rows)
    logger.info("Parsed %d XGB decisions from %s in [%s, %s)",
                len(df), path, START_TS, END_TS)
    if len(df):
        logger.info("  REJECT=%d  ACCEPT=%d",
                    int((df["decision"] == "REJECT").sum()),
                    int((df["decision"] == "ACCEPT").sum()))
    return df


# ═══════════════════════════════════════════════════════════════════
#  Match live decisions to generated samples
# ═══════════════════════════════════════════════════════════════════

def load_samples_for_period() -> pd.DataFrame:
    """
    Load the freshly-generated samples parquet, keeping only entry bars in the
    shadow-replay window. We keep timestamp, symbol, asset_class, label_* and
    all feature columns.
    """
    all_dfs = []
    for ac in ["crypto", "forex", "stocks", "commodities"]:
        path = DATA_DIR / f"{ac}_samples.parquet"
        if not path.exists():
            logger.warning("Samples missing for %s: %s", ac, path)
            continue
        df = pd.read_parquet(path)
        # Coerce timestamp to tz-aware UTC
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df["timestamp"] = ts
        # Entry bars only + within replay window
        df = df[
            (df["label_action"] > 0)
            & (df["timestamp"] >= START_TS)
            & (df["timestamp"] < END_TS)
        ].copy()
        logger.info("  %s: %d entry bars in replay window", ac, len(df))
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    out = pd.concat(all_dfs, ignore_index=True)
    return out


def match_decisions_to_samples(
    decisions: pd.DataFrame, samples: pd.DataFrame
) -> pd.DataFrame:
    """
    Match each live XGB decision to the nearest sample by (symbol, timestamp).
    Tolerance: +/- 5 minutes (the bot logs after feature computation).
    """
    if len(decisions) == 0 or len(samples) == 0:
        return pd.DataFrame()

    samples = samples.copy()
    # symbols in samples parquet match log-normalized names for forex/stocks/commodities
    # For crypto they match too (BTCUSDT matches after _normalize_symbol).
    samples["symbol_norm"] = samples["symbol"].astype(str)

    # Exact timestamp match attempt first (live logs typically lag sample timestamps
    # by a few seconds due to feature computation)
    decisions = decisions.copy()
    decisions["sample_ts"] = decisions["log_ts"].dt.floor("5min")

    # Merge: (symbol, sample_ts) -> sample row
    merged = decisions.merge(
        samples.rename(columns={"symbol": "symbol_sample"}),
        left_on=["symbol", "sample_ts"],
        right_on=["symbol_norm", "timestamp"],
        how="left",
        suffixes=("", "_sample"),
    )

    n_matched = int(merged["label_outcome"].notna().sum())
    logger.info("Exact-floor match: %d/%d decisions matched", n_matched, len(decisions))

    # For unmatched, widen by +/- 5 min (try the bar just before)
    if n_matched < len(decisions):
        unmatched_mask = merged["label_outcome"].isna()
        n_unmatched = int(unmatched_mask.sum())
        logger.info("Attempting wider match for %d decisions...", n_unmatched)
        # Fallback: for each unmatched decision, find closest sample by (symbol, ts)
        for idx in merged.index[unmatched_mask]:
            sym = merged.at[idx, "symbol"]
            ts = merged.at[idx, "log_ts"]
            cand = samples[samples["symbol_norm"] == sym]
            if len(cand) == 0:
                continue
            dt = (cand["timestamp"] - ts).abs()
            if dt.min() > pd.Timedelta(minutes=15):
                continue
            j = dt.idxmin()
            # Copy features/labels from the closest sample
            for col in samples.columns:
                if col in ("symbol", "symbol_norm"):
                    continue
                merged.at[idx, col] = samples.at[j, col]

    matched = merged[merged["label_outcome"].notna()].copy()
    logger.info("Final matched: %d/%d decisions", len(matched), len(decisions))
    return matched


# ═══════════════════════════════════════════════════════════════════
#  Re-score with each model
# ═══════════════════════════════════════════════════════════════════

def _load_model(path: Path) -> tuple[Any, list[str]] | None:
    if not path.exists():
        logger.error("Model not found: %s", path)
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("feat_names", [])


def _sample_only_df(matched: pd.DataFrame) -> pd.DataFrame:
    """Drop the decision-origin helper columns so prepare_features only sees sample columns."""
    drop_cols = [
        "log_ts", "symbol_raw", "decision", "conf_live", "score",
        "sample_ts", "symbol_norm", "symbol_sample",
    ]
    keep = [c for c in matched.columns if c not in drop_cols]
    return matched[keep]


def score_with_model(
    model_tuple: tuple[Any, list[str]] | None,
    matched: pd.DataFrame,
    label: str,
) -> np.ndarray | None:
    if model_tuple is None or len(matched) == 0:
        return None
    model, feat_names = model_tuple
    feature_df = _sample_only_df(matched)
    X, feat_used = prepare_features(feature_df, task="entry_quality")
    # Sanity check
    if feat_used != feat_names:
        logger.warning(
            "%s: feature order mismatch (used=%d, expected=%d)",
            label, len(feat_used), len(feat_names),
        )
    proba = model.predict_proba(X)[:, 1]
    logger.info("  %s: %d scored, mean conf=%.3f, accept@0.55=%d",
                label, len(proba), float(proba.mean()), int((proba >= LIVE_THRESHOLD).sum()))
    return proba


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SHADOW REPLAY: Entry Filter")
    logger.info("  Log:     %s", LOG_PATH)
    logger.info("  Window:  [%s, %s)", START_TS, END_TS)
    logger.info("  Models:  old=%s  A=%s  B=%s",
                OLD_MODEL_PATH.name, MODEL_A_PATH.name, MODEL_B_PATH.name)
    logger.info("=" * 70)

    decisions = parse_log(LOG_PATH)
    if len(decisions) == 0:
        logger.error("No decisions found in log — aborting")
        return

    samples = load_samples_for_period()
    if len(samples) == 0:
        logger.error("No samples loaded — aborting")
        return

    matched = match_decisions_to_samples(decisions, samples)
    if len(matched) == 0:
        logger.error("No matches found — aborting")
        return

    # Load all models (optfix is the post-mismatch-fix model, loaded if present)
    old_mt = _load_model(OLD_MODEL_PATH)
    a_mt = _load_model(MODEL_A_PATH)
    b_mt = _load_model(MODEL_B_PATH)
    optfix_mt = _load_model(MODEL_OPTFIX_PATH) if MODEL_OPTFIX_PATH.exists() else None

    logger.info("Scoring with models...")
    proba_old = score_with_model(old_mt, matched, "OLD")
    proba_a = score_with_model(a_mt, matched, "MODEL_A")
    proba_b = score_with_model(b_mt, matched, "MODEL_B")
    proba_optfix = score_with_model(optfix_mt, matched, "MODEL_OPTFIX") if optfix_mt else None

    # Build comparison table
    table_data = {
        "log_ts": matched["log_ts"].astype(str),
        "symbol": matched["symbol"],
        "asset_class": matched["asset_class"],
        "decision_live": matched["decision"],
        "conf_live": matched["conf_live"],
        "conf_old_rescored": proba_old if proba_old is not None else np.nan,
        "conf_a": proba_a if proba_a is not None else np.nan,
        "conf_b": proba_b if proba_b is not None else np.nan,
        "label_outcome": matched["label_outcome"].astype(int),
        "label_rr": np.clip(matched["label_rr"].astype(float), -1.0, 20.0),
    }
    if proba_optfix is not None:
        table_data["conf_optfix"] = proba_optfix
    table = pd.DataFrame(table_data)

    # Sanity: conf_live vs conf_old_rescored should be near-identical (same model,
    # same features). Any large gap indicates feature mismatch.
    if proba_old is not None:
        diff = (table["conf_old_rescored"] - table["conf_live"]).abs()
        logger.info("OLD re-score vs live conf: mean_abs_diff=%.4f max=%.4f",
                    float(diff.mean()), float(diff.max()))

    # Aggregate per-model decisions and outcomes at 0.55 threshold
    summary: dict[str, Any] = {"n_matched": len(matched)}
    model_probas = [
        ("old_live", matched["conf_live"].values),
        ("model_a", proba_a),
        ("model_b", proba_b),
    ]
    if proba_optfix is not None:
        model_probas.append(("model_optfix", proba_optfix))
    for name, proba in model_probas:
        if proba is None:
            continue
        accepted = proba >= LIVE_THRESHOLD
        n_acc = int(accepted.sum())
        outcomes = matched["label_outcome"].values[accepted] if n_acc > 0 else np.array([])
        rr = np.clip(matched["label_rr"].values[accepted].astype(np.float64), -1.0, 20.0) if n_acc > 0 else np.array([])
        n_win = int((outcomes == 1).sum()) if n_acc > 0 else 0
        n_loss = int((outcomes == 2).sum()) if n_acc > 0 else 0
        win_rr = float(rr[outcomes == 1].sum()) if n_win > 0 else 0.0
        loss_rr = float(abs(rr[outcomes == 2].sum())) if n_loss > 0 else 0.0
        summary[name] = {
            "conf_mean": float(np.mean(proba)),
            "conf_p90": float(np.quantile(proba, 0.90)),
            "accepted": n_acc,
            "accept_rate": float(n_acc / len(matched)),
            "wins": n_win,
            "losses": n_loss,
            "sum_rr": float(rr.sum()) if n_acc > 0 else 0.0,
            "pf": win_rr / max(loss_rr, 0.001),
            "winrate": n_win / max(n_win + n_loss, 1),
        }

    # Agreement tables (live vs each new model)
    def _agreement(live: np.ndarray, new: np.ndarray, label: str) -> dict:
        live_acc = live >= LIVE_THRESHOLD
        new_acc = new >= LIVE_THRESHOLD
        cells = {
            "both_reject": (~live_acc) & (~new_acc),
            "only_new_accept": (~live_acc) & new_acc,
            "only_old_accept": live_acc & (~new_acc),
            "both_accept": live_acc & new_acc,
        }
        out = {}
        for k, mask in cells.items():
            n = int(mask.sum())
            if n == 0:
                out[k] = {"count": 0, "wins": 0, "losses": 0, "sum_rr": 0.0}
                continue
            outs = matched["label_outcome"].values[mask]
            rr = np.clip(matched["label_rr"].values[mask].astype(np.float64), -1.0, 20.0)
            n_w = int((outs == 1).sum())
            n_l = int((outs == 2).sum())
            out[k] = {
                "count": n,
                "wins": n_w,
                "losses": n_l,
                "sum_rr": float(rr.sum()),
                "winrate": n_w / max(n_w + n_l, 1),
                "avg_rr": float(rr.mean()),
            }
        return out

    agreement_a = _agreement(matched["conf_live"].values, proba_a, "A") if proba_a is not None else None
    agreement_b = _agreement(matched["conf_live"].values, proba_b, "B") if proba_b is not None else None

    # Per-symbol breakdown
    per_symbol = {}
    for sym, sub in table.groupby("symbol"):
        per_symbol[sym] = {
            "n": int(len(sub)),
            "conf_live_mean": float(sub["conf_live"].mean()),
            "conf_a_mean": float(sub["conf_a"].mean()) if "conf_a" in sub else None,
            "conf_b_mean": float(sub["conf_b"].mean()) if "conf_b" in sub else None,
            "winrate": float((sub["label_outcome"] == 1).mean()),
            "sum_rr": float(sub["label_rr"].sum()),
        }

    report = {
        "config": {
            "log_path": str(LOG_PATH),
            "window": [str(START_TS), str(END_TS)],
            "live_threshold": LIVE_THRESHOLD,
        },
        "summary": summary,
        "agreement_old_vs_a": agreement_a,
        "agreement_old_vs_b": agreement_b,
        "per_symbol": per_symbol,
    }

    # Also write the per-decision table as CSV for inspection
    csv_path = RESULTS_PATH.with_suffix(".csv")
    table.to_csv(csv_path, index=False)
    logger.info("Per-decision table: %s", csv_path)

    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Results: %s", RESULTS_PATH)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SHADOW REPLAY SUMMARY (%d live decisions matched)", len(matched))
    logger.info("=" * 70)
    for name in ("old_live", "model_a", "model_b", "model_optfix"):
        if name not in summary:
            continue
        s = summary[name]
        logger.info("%-13s accepted=%2d/%d (%.0f%%)  wins=%d losses=%d sum_rr=%+.2f PF=%.2f WR=%.0f%%",
                    name, s["accepted"], len(matched), 100 * s["accept_rate"],
                    s["wins"], s["losses"], s["sum_rr"], s["pf"], 100 * s["winrate"])

    # Verify train/inference parity: conf_old_rescored vs conf_live
    if proba_old is not None:
        live_conf = matched["conf_live"].values
        delta = np.abs(proba_old - live_conf)
        mean_delta = float(delta.mean())
        max_delta = float(delta.max())
        logger.info("")
        logger.info("TRAIN/INFERENCE PARITY (OLD model rescored vs live conf):")
        logger.info("  mean_abs_diff = %.4f  max = %.4f", mean_delta, max_delta)
        if mean_delta < 0.05:
            logger.info("  ✓ FIX VERIFIED — mean delta < 0.05 (was 0.339 before fix)")
        elif mean_delta < 0.15:
            logger.info("  ~ PARTIAL FIX — delta improved but not closed")
        else:
            logger.info("  ✗ STILL MISMATCHED — regen/retrain may not have worked")


if __name__ == "__main__":
    main()
