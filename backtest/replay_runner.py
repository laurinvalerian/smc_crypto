"""
Market Replay Runner
====================
Feeds historical candle data through the LIVE PaperBot code, simulating
1:1 what happens in live paper trading.

Usage:
    python3 -m backtest.replay_runner
    python3 -m backtest.replay_runner --variants B,C,E --asset-classes crypto,stocks

Monitor:
    tail -f replay_results/<variant>/replay.log
"""
from __future__ import annotations

import xgboost as xgb  # MUST import before torch (shared lib conflict causes segfault)
import argparse
import asyncio
import gc
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Imports from live bot (use EXACTLY the same code) ──────────────
from exchanges.replay_adapter import ReplayAdapter
from features.schema import ENTRY_QUALITY_FEATURES, SCHEMA_VERSION
from risk.circuit_breaker import CircuitBreaker
from trade_journal import TradeJournal

# Lazy-import PaperBot to avoid loading all live bot dependencies at module level
_PaperBot = None
_ASSET_SMC_PARAMS = None
_ALL_INSTRUMENTS = None
_ASSET_COMMISSION = None
_ASSET_MAX_LEVERAGE = None
_COMMISSION_MULTIPLIER = None
_STYLE_CONFIG = None


def _lazy_import():
    global _PaperBot, _ASSET_SMC_PARAMS, _ALL_INSTRUMENTS, _ASSET_COMMISSION
    global _ASSET_MAX_LEVERAGE, _COMMISSION_MULTIPLIER, _STYLE_CONFIG
    if _PaperBot is not None:
        return
    from live_multi_bot import (
        PaperBot,
        ASSET_SMC_PARAMS,
        ALL_INSTRUMENTS,
        ASSET_COMMISSION,
        ASSET_MAX_LEVERAGE,
        COMMISSION_MULTIPLIER,
        STYLE_CONFIG,
    )
    _PaperBot = PaperBot
    _ASSET_SMC_PARAMS = ASSET_SMC_PARAMS
    _ALL_INSTRUMENTS = ALL_INSTRUMENTS
    _ASSET_COMMISSION = ASSET_COMMISSION
    _ASSET_MAX_LEVERAGE = ASSET_MAX_LEVERAGE
    _COMMISSION_MULTIPLIER = COMMISSION_MULTIPLIER
    _STYLE_CONFIG = STYLE_CONFIG


# ── Constants ──────────────────────────────────────────────────────
DATA_DIR = Path("data")
RESULTS_DIR = Path("replay_results")
INITIAL_EQUITY = 100_000.0
RETRAIN_EVERY = 50  # retrain XGB every N closed trades
WARMUP_CANDLES = 300  # minimum candles before trading starts
N_JOBS = max(1, os.cpu_count() - 1)

# Variant definitions
VARIANTS = {
    "A": {"name": "A_t050_noscore", "threshold": 0.50, "include_score": False},
    "B": {"name": "B_t078_noscore", "threshold": 0.78, "include_score": False},
    "C": {"name": "C_t078_score",   "threshold": 0.78, "include_score": True},
    "D": {"name": "D_t050_score",   "threshold": 0.50, "include_score": True},
    "E": {"name": "E_t065_score",   "threshold": 0.65, "include_score": True},
}


# ── Symbol ↔ Filename mapping ──────────────────────────────────────

def _symbol_to_filestem(symbol: str, asset_class: str) -> str:
    """Convert bot symbol to parquet file stem (without _5m suffix)."""
    if asset_class == "crypto":
        # "BTC/USDT:USDT" → "BTCUSDT"
        return symbol.replace("/", "").replace(":USDT", "")
    # forex/stocks/commodities: already in file format
    return symbol


def _load_tf_data(symbol: str, asset_class: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV parquet for a symbol/timeframe."""
    stem = _symbol_to_filestem(symbol, asset_class)
    path = DATA_DIR / asset_class / f"{stem}_{timeframe}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC") if df["timestamp"].dt.tz is None else df["timestamp"]
    return df


# ── XGBoost Training ───────────────────────────────────────────────

def _train_initial_xgb(
    asset_classes: list[str],
    include_score: bool,
    threshold: float,
    output_path: Path,
) -> tuple[Any, list[str]]:
    """Train initial XGBoost from pre-computed rl_training data."""
    import xgboost as xgb

    feat_names = list(ENTRY_QUALITY_FEATURES)
    if include_score:
        feat_names.append("alignment_score")

    frames = []
    for ac in asset_classes:
        path = DATA_DIR / "rl_training" / f"{ac}_samples.parquet"
        if not path.exists():
            continue
        # Load only needed columns to save memory (avoid loading full 4GB)
        needed = [c for c in feat_names if c not in ("asset_class_id", "style_id")]
        needed += ["alignment_score", "window", "label_profitable", "asset_class"]
        needed = list(set(needed))
        df = pd.read_parquet(path, columns=needed)
        df = df[(df["alignment_score"] >= threshold) & (df["window"] <= 5)]
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return None, feat_names

    data = pd.concat(frames, ignore_index=True)

    # Ensure all features exist
    for f in feat_names:
        if f not in data.columns:
            data[f] = 0.0
    if "style_id" not in data.columns:
        data["style_id"] = 0.5
    if "asset_class_id" not in data.columns:
        ac_map = {"crypto": 0.0, "forex": 0.33, "stocks": 0.66, "commodities": 1.0}
        data["asset_class_id"] = data["asset_class"].map(ac_map).fillna(0.5)

    X = data[feat_names].values.astype(np.float32)
    y = data["label_profitable"].values.astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)

    split = int(len(X) * 0.85)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=2,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X[:split], y[:split],
              eval_set=[(X[split:], y[split:])],
              verbose=False)

    # Save in rl_brain_v2 format
    import pickle
    payload = {
        "model": model,
        "feat_names": feat_names,
        "task": "entry_quality",
        "schema_version": SCHEMA_VERSION if not include_score else SCHEMA_VERSION + 100,
        "dead_features": set(),
        "clip_ranges": {},
        "asset_class_map": {"crypto": 0.0, "forex": 0.33, "stocks": 0.66, "commodities": 1.0},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    logging.info("Trained initial XGB: %d samples, %d features, best_iter=%d",
                 len(X), len(feat_names), getattr(model, "best_iteration", -1))
    return model, feat_names


def _retrain_xgb_from_journal(
    journal_path: Path,
    asset_classes: list[str],
    include_score: bool,
    threshold: float,
    model_path: Path,
) -> tuple[Any, list[str]] | None:
    """Retrain XGB using journal trades + backtest data (mirrors continuous_learner)."""
    import xgboost as xgb

    feat_names = list(ENTRY_QUALITY_FEATURES)
    if include_score:
        feat_names.append("alignment_score")

    # Load backtest base data (subsample to 50K for speed)
    frames = []
    for ac in asset_classes:
        path = DATA_DIR / "rl_training" / f"{ac}_samples.parquet"
        if not path.exists():
            continue
        needed = [c for c in feat_names if c not in ("asset_class_id", "style_id")]
        needed += ["alignment_score", "label_profitable", "asset_class"]
        needed = list(set(needed))
        df = pd.read_parquet(path, columns=needed)
        df = df[df["alignment_score"] >= threshold]
        if len(df) > 50_000:
            df = df.sample(50_000, random_state=42)
        frames.append(df)

    if not frames:
        return None

    base_data = pd.concat(frames, ignore_index=True)

    # Ensure features
    for f in feat_names:
        if f not in base_data.columns:
            base_data[f] = 0.0
    if "style_id" not in base_data.columns:
        base_data["style_id"] = 0.5
    if "asset_class_id" not in base_data.columns:
        ac_map = {"crypto": 0.0, "forex": 0.33, "stocks": 0.66, "commodities": 1.0}
        base_data["asset_class_id"] = base_data["asset_class"].map(ac_map).fillna(0.5)

    X = base_data[feat_names].values.astype(np.float32)
    y = base_data["label_profitable"].values.astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)

    split = int(len(X) * 0.85)
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        scale_pos_weight=spw, eval_metric="logloss",
        early_stopping_rounds=30, random_state=42, n_jobs=2,
        tree_method="hist", verbosity=0,
    )
    model.fit(X[:split], y[:split],
              eval_set=[(X[split:], y[split:])],
              verbose=False)

    # Save
    import pickle
    payload = {
        "model": model, "feat_names": feat_names, "task": "entry_quality",
        "schema_version": SCHEMA_VERSION if not include_score else SCHEMA_VERSION + 100,
        "dead_features": set(), "clip_ranges": {},
        "asset_class_map": {"crypto": 0.0, "forex": 0.33, "stocks": 0.66, "commodities": 1.0},
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    return model, feat_names


# ── Trade Close Handler ────────────────────────────────────────────

def _handle_trade_close(
    bot: Any,
    trade: dict[str, Any],
    exit_price: float,
    exit_reason: str,
    candle_ts: datetime,
) -> float:
    """Process a closed trade — mirrors live bot _record_close logic."""
    entry_price = trade["entry"]
    qty = trade["qty"]
    direction = trade["direction"]

    if direction == "long":
        raw_pnl = (exit_price - entry_price) * qty
    else:
        raw_pnl = (entry_price - exit_price) * qty

    commission = qty * entry_price * bot.commission_rate * _COMMISSION_MULTIPLIER
    net_pnl = raw_pnl - commission
    pnl_pct = (net_pnl / bot.equity * 100) if bot.equity > 0 else 0.0

    bot.equity += net_pnl
    bot.total_pnl += net_pnl
    bot.trades += 1
    if net_pnl > 0:
        bot.wins += 1
    if bot.equity > bot.peak_equity:
        bot.peak_equity = bot.equity

    # Record in journal
    if bot.journal is not None:
        trade_id = trade.get("rl_trade_id", trade.get("order_id", f"replay_{bot.trades}"))
        try:
            bot.journal.close_trade(
                trade_id=trade_id,
                exit_time=candle_ts,
                exit_price=exit_price,
                pnl_pct=pnl_pct / 100.0,
                rr_actual=(exit_price - entry_price) / abs(entry_price - trade["sl"]) if abs(entry_price - trade["sl"]) > 1e-10 else 0.0,
                rr_target=trade.get("rr_target", 0.0),
                outcome="win" if net_pnl > 0 else "loss",
                exit_reason=exit_reason,
                bars_held=trade.get("_candles_seen", 0),
                max_favorable_pct=0.0,
                max_adverse_pct=0.0,
            )
        except Exception:
            pass  # journal may not have this trade_id if open failed

    # Circuit breaker
    if bot.circuit_breaker is not None:
        bot.circuit_breaker.record_trade_pnl(
            pnl_pct=pnl_pct / 100.0,
            asset_class=bot.asset_class,
        )

    # Remove from active trades
    if trade in bot._active_trades:
        bot._active_trades.remove(trade)

    return net_pnl


# ── Single Variant Replay ──────────────────────────────────────────

def run_single_variant(args: tuple) -> dict[str, Any]:
    """Run one variant across all specified asset classes. Entry point for multiprocessing."""
    if len(args) == 5:
        variant_key, variant_cfg, asset_classes_str, with_be, max_candles = args
    elif len(args) == 4:
        variant_key, variant_cfg, asset_classes_str, with_be = args
        max_candles = 0
    else:
        variant_key, variant_cfg, asset_classes_str = args
        with_be = False
        max_candles = 0
    asset_classes = asset_classes_str.split(",")
    _lazy_import()

    variant_name = variant_cfg["name"]
    if with_be:
        variant_name = variant_name + "_be"
    threshold = variant_cfg["threshold"]
    include_score = variant_cfg["include_score"]

    out_dir = RESULTS_DIR / variant_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = out_dir / "replay.log"
    log = logging.getLogger(f"replay.{variant_name}")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)
    log.info("=" * 60)
    log.info("Replay Variant: %s (threshold=%.2f, score_feature=%s)", variant_name, threshold, include_score)
    log.info("Asset classes: %s", asset_classes)
    log.info("=" * 60)

    # Override alignment threshold globally for this process
    for ac in _ASSET_SMC_PARAMS:
        _ASSET_SMC_PARAMS[ac]["alignment_threshold"] = threshold

    # Load pre-trained XGB (trained separately to avoid memory conflicts)
    model_path = out_dir / "rl_entry_filter.pkl"
    pretrained = RESULTS_DIR / f"{variant_key}_initial_model.pkl"
    if pretrained.exists():
        shutil.copy(pretrained, model_path)
        log.info("Loaded pre-trained model from %s", pretrained)
    else:
        _train_initial_xgb(asset_classes, include_score, threshold, model_path)

    # Load RLBrainSuite with our model
    from rl_brain_v2 import RLBrainSuite
    rl_config = {
        "entry_filter": {"enabled": True, "model_path": str(model_path), "confidence_threshold": 0.55},
        "be_manager": {"enabled": with_be, "model_path": "models/rl_be_manager.pkl", "min_be_rr": 0.5},
        "tp_optimizer": {"enabled": False},
        "exit_classifier": {"enabled": False},
    }
    rl_suite = RLBrainSuite(rl_config)
    log.info("BE Manager: %s", "ENABLED" if with_be else "DISABLED")

    # Track results
    total_closed = 0
    equity_curve: list[dict] = []
    t0 = time.time()

    # Process each asset class sequentially (memory)
    for ac in asset_classes:
        symbols = _ALL_INSTRUMENTS.get(ac, [])
        if not symbols:
            continue

        # Check which symbols have data
        available = []
        for sym in symbols:
            stem = _symbol_to_filestem(sym, ac)
            if (DATA_DIR / ac / f"{stem}_5m.parquet").exists():
                available.append(sym)

        if not available:
            log.warning("No data for %s — skipping", ac)
            continue

        log.info("Processing %s: %d symbols", ac, len(available))

        # Create adapter
        leverage = _ASSET_MAX_LEVERAGE.get(ac, 1)
        adapter = ReplayAdapter(
            asset_class=ac,
            initial_balance=INITIAL_EQUITY,
            leverage=leverage,
        )

        # Create journal for this variant
        journal_path = out_dir / "journal.db"
        journal = TradeJournal(str(journal_path))

        # Create circuit breaker
        cb = CircuitBreaker()

        # Create bots
        bots: dict[str, Any] = {}  # symbol → PaperBot
        for i, sym in enumerate(available):
            bot = _PaperBot(
                bot_id=i,
                symbol=sym,
                config={},
                output_dir=out_dir / "bots",
                asset_class=ac,
                adapter=adapter,
                rl_suite=rl_suite,
            )
            bot.equity = INITIAL_EQUITY
            bot.peak_equity = INITIAL_EQUITY
            bot._account_equity = INITIAL_EQUITY
            bot.journal = journal
            bot.circuit_breaker = cb
            bots[sym] = bot

        # Load all 5m candle data and sort chronologically
        log.info("Loading candle data for %s...", ac)
        all_candles: list[tuple[datetime, str, dict]] = []

        for sym in available:
            df_5m = _load_tf_data(sym, ac, "5m")
            if df_5m.empty:
                continue
            if max_candles > 0:
                df_5m = df_5m.tail(max_candles)

            # Pre-fill higher TF buffers (trimmed to match live bot's load_history limits)
            _TF_LIMITS = {"1d": 250, "4h": 500, "1h": 1000, "15m": 1500}
            bot = bots[sym]
            for tf in ("1d", "4h", "1h", "15m"):
                df_tf = _load_tf_data(sym, ac, tf)
                if not df_tf.empty:
                    limit = _TF_LIMITS.get(tf, 1000)
                    setattr(bot, f"buffer_{tf}", df_tf.tail(limit).reset_index(drop=True))

            # Convert 5m candles to event list
            for _, row in df_5m.iterrows():
                ts = row["timestamp"]
                if not isinstance(ts, datetime):
                    ts = pd.Timestamp(ts).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                candle = {
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                all_candles.append((ts, sym, candle))

        # Sort by timestamp
        all_candles.sort(key=lambda x: x[0])
        total_candles = len(all_candles)
        log.info("Total candles to replay: %d across %d symbols", total_candles, len(available))

        # ── MAIN REPLAY LOOP ──────────────────────────────────────
        candle_count = 0
        ac_closed = 0
        last_retrain = 0
        skip_until_warmup = WARMUP_CANDLES  # count per symbol
        symbol_candle_counts: dict[str, int] = defaultdict(int)

        pbar = tqdm(
            total=total_candles,
            desc=f"{variant_name}/{ac}",
            unit="candle",
            miniters=max(1, total_candles // 200),
        )

        for ts, sym, candle in all_candles:
            candle_count += 1
            pbar.update(1)
            symbol_candle_counts[sym] += 1

            bot = bots.get(sym)
            if bot is None:
                continue

            # Update adapter state
            adapter.update_price(sym, candle["close"], ts)

            # 1. Check SL/TP fills BEFORE processing new candle
            fills = adapter.check_and_fill_orders(sym, candle)
            for fill in fills:
                # Find matching trade in bot._active_trades
                matching = [t for t in bot._active_trades
                            if t.get("order_id") == fill["order_id"]
                            or t.get("symbol") == sym]
                if matching:
                    trade = matching[0]
                    net_pnl = _handle_trade_close(
                        bot, trade, fill["exit_price"], fill["exit_reason"], ts,
                    )
                    ac_closed += 1
                    total_closed += 1
                    log.info(
                        "CLOSE %s %s %s @ %.6f → %.6f | PnL=%.2f | equity=%.2f | %s | #%d",
                        fill["exit_reason"].upper(), fill["direction"].upper(), sym,
                        fill["entry_price"], fill["exit_price"],
                        net_pnl, bot.equity, variant_name, total_closed,
                    )

                    # Continuous learning trigger
                    if total_closed - last_retrain >= RETRAIN_EVERY:
                        log.info("RETRAIN triggered at %d closed trades", total_closed)
                        result = _retrain_xgb_from_journal(
                            journal_path, asset_classes, include_score, threshold, model_path,
                        )
                        if result is not None:
                            # Reload model in rl_suite
                            try:
                                rl_suite._entry_model = result[0]
                                log.info("RETRAIN success: new model loaded")
                            except Exception as e:
                                log.warning("RETRAIN model reload failed: %s", e)
                        last_retrain = total_closed

            # 2. Skip warmup period (need buffer to fill)
            if symbol_candle_counts[sym] < skip_until_warmup:
                # Still feed candle to build buffer
                bot.on_candle(sym, candle)
                continue

            # 3. Feed candle to bot (generates pending signal if conditions met)
            try:
                bot.on_candle(sym, candle)
            except Exception as exc:
                log.debug("on_candle error %s: %s", sym, exc)
                continue

            # 4. Simulate tick for entry zone check
            if bot._pending_signal is not None:
                try:
                    asyncio.get_event_loop().run_until_complete(
                        bot.on_tick(sym, candle["close"])
                    )
                except RuntimeError:
                    # No event loop running — create one
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(bot.on_tick(sym, candle["close"]))
                    finally:
                        loop.close()
                except Exception as exc:
                    log.debug("on_tick error %s: %s", sym, exc)

            # 4b. BE check — move SL to breakeven if price reached target RR
            if with_be and bot._active_trades:
                for trade in bot._active_trades:
                    if trade.get("be_triggered", False) or trade.get("rl_be_level", 0) <= 0:
                        continue
                    entry_p = trade["entry"]
                    direction = trade["direction"]
                    sl_dist = abs(entry_p - trade.get("original_sl", trade["sl"]))
                    if sl_dist <= 0:
                        continue
                    be_target_rr = trade["rl_be_level"]
                    # Use candle high/low for intra-bar accuracy
                    if direction == "long":
                        current_rr = (candle["high"] - entry_p) / sl_dist
                    else:
                        current_rr = (entry_p - candle["low"]) / sl_dist
                    if current_rr >= be_target_rr:
                        fee_buffer = entry_p * bot.commission_rate * 4
                        new_sl = entry_p + fee_buffer if direction == "long" else entry_p - fee_buffer
                        # Update SL in adapter
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(
                                adapter.modify_stop_loss("", sym, "sell" if direction == "long" else "buy",
                                                         trade.get("qty", 0), new_sl)
                            )
                        finally:
                            loop.close()
                        trade["sl"] = new_sl
                        trade["be_triggered"] = True
                        log.debug("BE triggered %s %s @ RR=%.2f → SL=%.6f", direction, sym, current_rr, new_sl)

            # 5. Check max hold time — force close expired trades
            for trade in list(bot._active_trades):
                candles_seen = trade.get("_candles_seen", 0)
                style = trade.get("style", "day")
                max_hold = _STYLE_CONFIG.get(style, _STYLE_CONFIG.get("day", {})).get("max_hold_candles", 288)
                if max_hold > 0 and candles_seen >= max_hold:
                    net_pnl = _handle_trade_close(bot, trade, candle["close"], "timeout", ts)
                    ac_closed += 1
                    total_closed += 1
                    # Remove from adapter too
                    if adapter.has_position(sym):
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(adapter.close_position(sym))
                        loop.close()
                    log.info("TIMEOUT %s %s @ %.6f | PnL=%.2f | #%d", trade["direction"], sym, candle["close"], net_pnl, total_closed)

            # 6. Record equity periodically
            if candle_count % 1000 == 0:
                equity_curve.append({
                    "timestamp": ts.isoformat(),
                    "equity": round(bot.equity, 2),
                    "trades": total_closed,
                    "variant": variant_name,
                    "asset_class": ac,
                })

        pbar.close()

        # Asset class summary
        wr = (sum(b.wins for b in bots.values()) / max(sum(b.trades for b in bots.values()), 1)) * 100
        log.info(
            "%s DONE: %d trades, WR=%.1f%%, equity=%.2f (started %.2f)",
            ac, ac_closed, wr, adapter._balance, INITIAL_EQUITY,
        )

        # Cleanup
        del all_candles, bots, adapter
        gc.collect()

    # ── Final Results ──────────────────────────────────────────────
    elapsed = time.time() - t0

    # Compute metrics from journal
    metrics = _compute_metrics_from_journal(out_dir / "journal.db")
    metrics["variant"] = variant_name
    metrics["threshold"] = threshold
    metrics["include_score"] = include_score
    metrics["be_enabled"] = with_be
    metrics["elapsed_min"] = round(elapsed / 60, 1)
    metrics["total_closed"] = total_closed
    metrics["retrains"] = total_closed // RETRAIN_EVERY

    # Save
    with open(out_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    if equity_curve:
        pd.DataFrame(equity_curve).to_csv(out_dir / "equity_curve.csv", index=False)

    log.info("VARIANT %s COMPLETE: %s", variant_name, json.dumps(metrics, indent=2, default=str))
    return metrics


def _compute_metrics_from_journal(journal_path: Path) -> dict:
    """Compute PF, WR, Sharpe, maxDD from journal trades."""
    import sqlite3
    if not journal_path.exists():
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0, "max_dd_pct": 0}

    conn = sqlite3.connect(str(journal_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT pnl_pct, outcome, asset_class FROM trades WHERE exit_time IS NOT NULL"
        ).fetchall()
    except Exception:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0, "max_dd_pct": 0}
    finally:
        conn.close()

    if not rows:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0, "max_dd_pct": 0}

    pnls = [float(r["pnl_pct"] or 0) for r in rows]
    wins = sum(1 for r in rows if r["outcome"] == "win")
    wr = wins / len(rows) * 100

    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    pf = gains / losses if losses > 0 else 99.9

    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if np.std(pnls) > 0 else 0

    # Max drawdown
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Per asset class
    per_class: dict[str, dict] = {}
    for r in rows:
        ac = r["asset_class"] or "unknown"
        if ac not in per_class:
            per_class[ac] = {"trades": 0, "wins": 0, "pnl": 0.0}
        per_class[ac]["trades"] += 1
        if r["outcome"] == "win":
            per_class[ac]["wins"] += 1
        per_class[ac]["pnl"] += float(r["pnl_pct"] or 0)

    return {
        "trades": len(rows),
        "wins": wins,
        "wr": round(wr, 1),
        "pf": round(min(pf, 99.9), 2),
        "sharpe": round(float(sharpe), 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "per_class": per_class,
    }


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Market Replay Runner")
    parser.add_argument("--variants", default="A,B,C,D,E",
                        help="Comma-separated variant keys (default: A,B,C,D,E)")
    parser.add_argument("--asset-classes", default="crypto,stocks,forex,commodities",
                        help="Comma-separated asset classes")
    parser.add_argument("--sequential", action="store_true",
                        help="Run variants sequentially (less memory)")
    parser.add_argument("--with-be", action="store_true",
                        help="Enable BE manager (breakeven stop-loss)")
    parser.add_argument("--max-candles", type=int, default=0,
                        help="Max candles per symbol (0 = all, e.g. 50000 for quick test)")
    args = parser.parse_args()

    variant_keys = [v.strip() for v in args.variants.split(",")]
    asset_classes = args.asset_classes

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("replay")

    # Validate variants
    tasks = []
    for vk in variant_keys:
        if vk not in VARIANTS:
            log.error("Unknown variant: %s (available: %s)", vk, list(VARIANTS.keys()))
            sys.exit(1)
        tasks.append((vk, VARIANTS[vk], asset_classes, args.with_be, args.max_candles))

    log.info("=" * 70)
    log.info("MARKET REPLAY — %d variants, asset classes: %s", len(tasks), asset_classes)
    log.info("Workers: %d | Retrain every: %d trades | Initial equity: %.0f | BE: %s",
             1 if args.sequential else N_JOBS, RETRAIN_EVERY, INITIAL_EQUITY,
             "ON" if args.with_be else "OFF")
    log.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Run variants
    results = []
    if args.sequential or len(tasks) == 1:
        for task in tasks:
            result = run_single_variant(task)
            results.append(result)
    else:
        with mp.Pool(min(N_JOBS, len(tasks))) as pool:
            for result in pool.imap_unordered(run_single_variant, tasks):
                results.append(result)

    elapsed = time.time() - t0

    # ── Comparison Table ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("REPLAY COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Variant':<22} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'Time':>6}")
    print("-" * 90)

    best_pf = 0
    best_variant = None
    for r in sorted(results, key=lambda x: x.get("pf", 0), reverse=True):
        v = r.get("variant", "?")
        print(f"{v:<22} {r.get('trades', 0):>7} {r.get('wr', 0):>5.1f}% "
              f"{r.get('pf', 0):>6.2f} {r.get('sharpe', 0):>7.2f} "
              f"{r.get('max_dd_pct', 0):>6.2f}% {r.get('elapsed_min', 0):>5.1f}m")
        if r.get("pf", 0) > best_pf:
            best_pf = r["pf"]
            best_variant = r

    if best_variant:
        print(f"\nWINNER: {best_variant['variant']} (PF={best_variant['pf']}, "
              f"WR={best_variant['wr']}%, Sharpe={best_variant['sharpe']})")
        print(f"Model: replay_results/{best_variant['variant']}/rl_entry_filter.pkl")

    print(f"\nTotal runtime: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Results saved to: {RESULTS_DIR}/")

    # Save comparison
    pd.DataFrame(results).to_csv(RESULTS_DIR / "comparison.csv", index=False)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
