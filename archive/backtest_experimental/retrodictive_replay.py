"""Retrodictive replay: what would the OPTFIX model have EARNED on the 49
rejected signals vs the OLD live model?

For each rejected signal:
  1. Fetch 5m OHLCV from OANDA from signal ts → +24h
  2. Simulate bracket order (first-touch SL/TP, pessimistic on same-bar tie)
  3. Compute R (reward-to-risk multiple) for "what if taken"
  4. Re-score with OLD model (from _prev backup) and NEW model (optfix)
  5. For each threshold (0.50, 0.55, 0.60), compute:
       - how many signals each model would accept
       - sum of R across accepted
       - win rate
       - profit factor

Runs on server (has OANDA creds + v20 + models + journal).

Usage (on server):
    /root/bot/.venv/bin/python3 /root/bot/backtest/retrodictive_replay.py \
        > /root/bot/retro_results.json

Then pull results via scp.
"""
import json
import os
import pickle
import sqlite3
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BOT_ROOT = Path("/root/bot")
JOURNAL = BOT_ROOT / "trade_journal" / "journal.db"
OLD_MODEL = BOT_ROOT / "models" / "rl_entry_filter.pkl_prev"
NEW_MODEL = BOT_ROOT / "models" / "rl_entry_filter.pkl"
WINDOW_HOURS = 24
OANDA_GRANULARITY = "M5"

ENTRY_QUALITY_EXCLUDE = {"has_entry_zone"}  # features dropped at predict time


def load_model(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def prep_features(feature_dict: dict, model: dict) -> np.ndarray:
    feature_names = (
        model.get("feat_names")
        or model.get("feature_names")
        or []
    )
    if not feature_names:
        m = model.get("model") or model
        feature_names = list(getattr(m, "feature_names_in_", []) or [])
    if not feature_names:
        raise RuntimeError("Model has no feature_names")
    clip_ranges = model.get("clip_ranges", {}) or {}
    row = []
    for name in feature_names:
        v = feature_dict.get(name)
        if v is None:
            v = 0.0
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        if name in clip_ranges:
            lo, hi = clip_ranges[name]
            if v < lo:
                v = float(lo)
            elif v > hi:
                v = float(hi)
        row.append(v)
    return np.asarray(row, dtype=np.float32).reshape(1, -1)


def predict_conf(model: dict, features: dict) -> float:
    X = prep_features(features, model)
    m = model.get("model") or model
    try:
        proba = m.predict_proba(X)
        return float(proba[0][1])
    except Exception:
        pred = m.predict(X)
        return float(pred[0])


def fetch_oanda_candles(symbol: str, start: datetime, end: datetime):
    import requests
    token = os.getenv("OANDA_ACCESS_TOKEN")
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{symbol}/candles"
    params = {
        "granularity": OANDA_GRANULARITY,
        "from": start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "price": "M",
    }
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        params=params,
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OANDA fetch failed {symbol} {r.status_code}: {r.text[:200]}")
    candles = r.json().get("candles", [])
    rows = []
    for c in candles:
        if not c.get("complete", True):
            continue
        mid = c.get("mid", {})
        rows.append({
            "timestamp": pd.to_datetime(c["time"], utc=True),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
        })
    return pd.DataFrame(rows)


def simulate_bracket(df: pd.DataFrame, direction: str, entry: float, sl: float, tp: float) -> dict:
    if df.empty:
        return {"outcome": "no_data", "r": None, "close_ts": None}
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return {"outcome": "invalid_risk", "r": None, "close_ts": None}
    rr_target = reward / risk
    first_touch_entry = False
    for _, bar in df.iterrows():
        hi, lo = bar["high"], bar["low"]
        if direction == "long":
            if not first_touch_entry:
                if lo <= entry <= hi:
                    first_touch_entry = True
                else:
                    continue
            hit_sl = lo <= sl
            hit_tp = hi >= tp
        else:
            if not first_touch_entry:
                if lo <= entry <= hi:
                    first_touch_entry = True
                else:
                    continue
            hit_sl = hi >= sl
            hit_tp = lo <= tp
        if hit_sl and hit_tp:
            return {"outcome": "ambiguous_sl", "r": -1.0, "close_ts": str(bar["timestamp"])}
        if hit_sl:
            return {"outcome": "sl", "r": -1.0, "close_ts": str(bar["timestamp"])}
        if hit_tp:
            return {"outcome": "tp", "r": float(rr_target), "close_ts": str(bar["timestamp"])}
    if not first_touch_entry:
        return {"outcome": "no_entry", "r": None, "close_ts": None}
    last_close = float(df.iloc[-1]["close"])
    if direction == "long":
        pnl_r = (last_close - entry) / risk
    else:
        pnl_r = (entry - last_close) / risk
    return {"outcome": "timeout", "r": float(pnl_r), "close_ts": str(df.iloc[-1]["timestamp"])}


def main():
    old_model = load_model(OLD_MODEL)
    new_model = load_model(NEW_MODEL)
    print(f"[info] OLD model features: {len(old_model.get('feature_names') or [])}", file=sys.stderr)
    print(f"[info] NEW model features: {len(new_model.get('feature_names') or [])}", file=sys.stderr)

    conn = sqlite3.connect(str(JOURNAL))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, symbol, asset_class, direction, entry_price,
               sl_price, tp_price, xgb_confidence, alignment_score, entry_features
        FROM rejected_signals
        ORDER BY timestamp
    """)
    rej = [dict(r) for r in cur.fetchall()]
    print(f"[info] Loaded {len(rej)} rejected signals", file=sys.stderr)

    cur.execute("""
        SELECT trade_id, entry_time, symbol, asset_class, direction, entry_price,
               sl_original, tp, exit_price, exit_time, exit_reason, outcome,
               rr_actual, pnl_pct, xgb_confidence
        FROM trades
        WHERE entry_time >= '2026-04-08'
          AND asset_class = 'forex'
        ORDER BY entry_time
    """)
    trades = [dict(r) for r in cur.fetchall()]
    print(f"[info] Loaded {len(trades)} forex trades since Apr 8", file=sys.stderr)
    conn.close()

    results = []
    for i, row in enumerate(rej):
        ts = pd.to_datetime(row["timestamp"], utc=True)
        end = ts + timedelta(hours=WINDOW_HOURS)
        if end > pd.Timestamp.utcnow():
            end = pd.Timestamp.utcnow()
        try:
            df = fetch_oanda_candles(row["symbol"], ts.to_pydatetime(), end.to_pydatetime())
        except Exception as e:
            print(f"[warn] fetch {row['symbol']} {ts}: {e}", file=sys.stderr)
            df = pd.DataFrame()
        sim = simulate_bracket(df, row["direction"], row["entry_price"], row["sl_price"], row["tp_price"])

        old_score = None
        new_score = None
        entry_feats = {}
        try:
            entry_feats = json.loads(row["entry_features"]) if row["entry_features"] else {}
            old_score = predict_conf(old_model, entry_feats)
            new_score = predict_conf(new_model, entry_feats)
        except Exception as e:
            print(f"[warn] rescore {row['symbol']} {ts}: {e}", file=sys.stderr)

        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "symbol": row["symbol"],
            "direction": row["direction"],
            "entry": row["entry_price"],
            "sl": row["sl_price"],
            "tp": row["tp_price"],
            "old_live_conf": row["xgb_confidence"],
            "alignment_score": row["alignment_score"],
            "old_rescore": old_score,
            "new_rescore": new_score,
            "sim": sim,
        })
        print(f"[{i+1}/{len(rej)}] {row['symbol']} {row['direction']} old_live={row['xgb_confidence']:.3f} old_re={old_score} new={new_score} sim={sim['outcome']} r={sim.get('r')}", file=sys.stderr)

    def stats_for_mask(mask, R):
        taken = sum(1 for m in mask if m)
        r_taken = [r for m, r in zip(mask, R) if m and r is not None]
        if not r_taken:
            return {"taken": taken, "evaluable": 0, "sum_r": 0.0, "wins": 0, "losses": 0, "wr": 0.0, "pf": None, "avg_r": 0.0}
        wins = sum(1 for r in r_taken if r > 0)
        losses = sum(1 for r in r_taken if r <= 0)
        sum_r = sum(r_taken)
        gross_win = sum(r for r in r_taken if r > 0)
        gross_loss = abs(sum(r for r in r_taken if r <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else None
        return {
            "taken": taken,
            "evaluable": len(r_taken),
            "sum_r": round(sum_r, 3),
            "wins": wins,
            "losses": losses,
            "wr": round(wins / len(r_taken), 3) if r_taken else 0.0,
            "pf": round(pf, 3) if pf is not None else None,
            "avg_r": round(sum_r / len(r_taken), 3) if r_taken else 0.0,
        }

    R_sim = [r["sim"].get("r") for r in results]
    old_rescores = [r["old_rescore"] for r in results]
    new_rescores = [r["new_rescore"] for r in results]
    old_live = [r["old_live_conf"] for r in results]

    summary = {"rejected_total": len(rej), "with_r": sum(1 for r in R_sim if r is not None)}
    for thr in (0.50, 0.55, 0.60, 0.65):
        summary[f"old_live_thr_{thr:.2f}"] = stats_for_mask(
            [(c is not None and c >= thr) for c in old_live], R_sim
        )
        summary[f"old_rescore_thr_{thr:.2f}"] = stats_for_mask(
            [(c is not None and c >= thr) for c in old_rescores], R_sim
        )
        summary[f"new_rescore_thr_{thr:.2f}"] = stats_for_mask(
            [(c is not None and c >= thr) for c in new_rescores], R_sim
        )

    # Include the 3 accepted trades for full "what actually happened" picture
    actual_trades_stats = {
        "count": len(trades),
        "closed": sum(1 for t in trades if t.get("exit_time")),
        "sum_r": round(sum(t.get("rr_actual") or 0 for t in trades), 3),
        "sum_pnl_pct": round(sum(t.get("pnl_pct") or 0 for t in trades), 3),
    }

    out = {
        "results": results,
        "summary": summary,
        "actual_executed_trades": trades,
        "actual_trades_stats": actual_trades_stats,
    }
    json.dump(out, sys.stdout, default=str, indent=2)


if __name__ == "__main__":
    main()
