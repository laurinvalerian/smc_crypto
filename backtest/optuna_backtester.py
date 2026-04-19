"""
═══════════════════════════════════════════════════════════════════
 backtest/optuna_backtester.py
 ─────────────────────────────
 Walk-Forward Optimization with Optuna (Scalp-Day Hybrid, no tiers).

 Features:
   • Rolling walk-forward windows (configurable train/test months)
   • Optuna Bayesian optimisation (configurable trials per window)
   • Single alignment-threshold gate (core.constants.ALIGNMENT_THRESHOLD)
   • Confidence-based risk sizing (core.sizing.compute_risk_fraction)
   • Circuit breaker simulation (daily -3%, weekly -5%, class -2%,
     all-time -8%)
   • REAL price-path simulation: walks candles forward to check
     whether SL or TP is hit first (no synthetic win probability)
   • Monte Carlo with compound equity (not simple PnL sum)
   • 6-gate OOS validation: PF, trades, Sharpe, Monte Carlo,
     max DD (-10%), parameter stability
   • Extracts top 20% best parameter sets automatically
   • Generates parameter-importance ranking (plot + CSV)
   • All results stored in /backtest/results

 Usage:
     python -m backtest.optuna_backtester                 # default config
     python -m backtest.optuna_backtester --config path   # custom config
     python -m backtest.optuna_backtester --monte-carlo   # with MC check
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from strategies.smc_multi_style import SMCMultiStyleStrategy, TradeSignal
from filters.trend_strength import compute_adx, check_momentum_confluence
from filters.volume_liquidity import compute_volume_score
from filters.session_filter import compute_session_score
from filters.zone_quality import compute_zone_quality
from risk.circuit_breaker import CircuitBreaker
from core.constants import ALIGNMENT_THRESHOLD, SCALP_MAX_HOLD_BARS
from core.sizing import compute_risk_amount
from core.metrics import (
    sharpe_daily as _sharpe_daily,
    return_moments as _return_moments,
    deflated_sharpe_ratio as _deflated_sharpe,
    trial_sharpe_variance as _trial_sharpe_variance,
)

# Parallelism budget (override via env: BACKTEST_SIGNAL_WORKERS, BACKTEST_OPTUNA_WORKERS).
# Signal precompute is numpy-heavy → scales with physical cores.
# Optuna trials share one SQLite store → cap conservatively to avoid write contention.
_CPU_COUNT = os.cpu_count() or 4
_SIGNAL_WORKERS = int(os.environ.get("BACKTEST_SIGNAL_WORKERS", max(2, _CPU_COUNT - 2)))
_OPTUNA_WORKERS = int(os.environ.get("BACKTEST_OPTUNA_WORKERS", max(1, _CPU_COUNT // 2)))

# ── Logging ───────────────────────────────────────────────────────
_results_dir = Path("backtest/results")
_results_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_results_dir / "backtest.log", mode="w")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════
#  Config helpers
# ═══════════════════════════════════════════════════════════════════

def load_config(path: str = "config/default_config.yaml") -> dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ═══════════════════════════════════════════════════════════════════
#  Walk-forward window generation
# ═══════════════════════════════════════════════════════════════════

def generate_wf_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
) -> list[dict[str, pd.Timestamp]]:
    """
    Generate rolling walk-forward windows.
    Each window: { train_start, train_end, test_start, test_end }
    """
    windows: list[dict[str, pd.Timestamp]] = []
    cursor = start

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = test_start

    return windows


# Signal gate + risk sizing moved to core.constants.ALIGNMENT_THRESHOLD and
# core.sizing.compute_risk_amount (Scalp-Day Hybrid refocus, 2026-04-19).


# ═══════════════════════════════════════════════════════════════════
#  Price-path data cache (for realistic trade simulation)
# ═══════════════════════════════════════════════════════════════════

# Module-level cache: symbol → sorted DataFrame with columns [open, high, low, close]
# Indexed by tz-aware UTC DatetimeIndex at 5m (or 1m) resolution.
_price_cache: dict[str, pd.DataFrame] = {}


def load_price_data_for_symbols(
    symbols: list[str],
    config: dict[str, Any],
) -> None:
    """
    Pre-load 5m (or 1m) OHLCV data for all symbols into _price_cache.
    Called once before simulate_trades to avoid repeated disk I/O.
    """
    data_dirs: list[Path] = [Path(config["data"]["data_dir"])]
    for key in ("crypto_dir", "forex_dir", "stocks_dir", "commodities_dir"):
        d = config["data"].get(key)
        if d:
            p = Path(d)
            if p.exists() and p not in data_dirs:
                data_dirs.append(p)

    for sym in symbols:
        if sym in _price_cache:
            continue
        safe = sym.replace("/", "_").replace(":", "_")
        # Prefer 5m, fall back to 1m
        loaded = False
        for tf in ("5m", "1m"):
            for d in data_dirs:
                path = d / f"{safe}_{tf}.parquet"
                if path.exists():
                    try:
                        df = pd.read_parquet(path)
                        # Ensure timestamp is the index
                        if "timestamp" in df.columns:
                            df = df.set_index("timestamp")
                        df = df[["open", "high", "low", "close"]]
                        if df.index.tz is None:
                            df.index = df.index.tz_localize("UTC")
                        _price_cache[sym] = df.sort_index()
                        loaded = True
                    except Exception as e:
                        logger.debug("Failed loading %s: %s", path, e)
                    break
            if loaded:
                break


# Maximum holding period — Scalp-Day Hybrid: 4h on 5m (SSOT: core.constants).
_MAX_BARS_DEFAULT = SCALP_MAX_HOLD_BARS


def _resolve_trade_outcome(
    sig: TradeSignal,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0001,
    be_ratchet_r: float = 1.5,
    timeout_rr_threshold: float = 0.5,
    max_hold_bars: int = SCALP_MAX_HOLD_BARS,
) -> tuple[str, float, pd.Timestamp | None]:
    """
    Walk forward through real price bars after entry to determine outcome.

    Breakeven-only stop logic (like a patient SMC trader):
    - Before +1R: original SL stays
    - At +1R: SL moves to net-breakeven (entry + actual round-trip costs)
    - After +1R: SL stays at breakeven — NO further trailing
    - Price has room to breathe and reach structure-based TP
    - TP hit → full win at target RR
    - BE-SL hit → ~zero PnL (covers fees exactly)
    - Timeout → close at last bar close

    Returns:
        (outcome, exit_price, exit_timestamp) where outcome is "win", "loss",
        "breakeven", or "skip". exit_timestamp is None for "skip".
    """
    df = _price_cache.get(sig.symbol)
    if df is None or df.empty:
        return "skip", sig.entry_price, None

    entry_ts = sig.timestamp
    if not hasattr(entry_ts, 'tzinfo') or entry_ts.tzinfo is None:
        entry_ts = pd.Timestamp(entry_ts, tz="UTC")

    mask = df.index > entry_ts
    future_bars = df.loc[mask]

    if future_bars.empty:
        return "skip", sig.entry_price, None

    future_bars = future_bars.iloc[:max_hold_bars]

    is_long = sig.direction == "long"
    entry = sig.entry_price
    original_sl = sig.stop_loss
    tp = sig.take_profit
    sl_dist = abs(entry - original_sl)  # 1R in price terms

    # Breakeven stop state
    current_sl = original_sl
    reached_1r = False
    # Net-breakeven buffer: actual round-trip costs (entry + exit)
    fee_buffer = entry * (commission_pct * 2 + slippage_pct * 2)

    for bar_ts, bar in future_bars.iterrows():
        high = bar["high"]
        low = bar["low"]

        if is_long:
            # Check if we've reached +1R (move to net-breakeven once)
            if not reached_1r and sl_dist > 0:
                bar_best_r = (high - entry) / sl_dist
                if bar_best_r >= be_ratchet_r:
                    # +1.5R reached — but did original SL also get hit on same bar?
                    if low <= original_sl:
                        # AMBIGUOUS: both +1.5R and SL on same bar → conservative = LOSS
                        return "loss", original_sl, bar_ts
                    reached_1r = True
                    current_sl = max(current_sl, entry + fee_buffer)

            # Check SL first (conservative: worst case within bar)
            if low <= current_sl:
                if reached_1r:
                    return "breakeven", current_sl, bar_ts
                return "loss", current_sl, bar_ts
            # Check TP
            if high >= tp:
                return "win", tp, bar_ts

        else:  # short
            if not reached_1r and sl_dist > 0:
                bar_best_r = (entry - low) / sl_dist
                if bar_best_r >= be_ratchet_r:
                    # +1.5R reached — but did original SL also get hit on same bar?
                    if high >= original_sl:
                        # AMBIGUOUS: both +1.5R and SL on same bar → conservative = LOSS
                        return "loss", original_sl, bar_ts
                    reached_1r = True
                    current_sl = min(current_sl, entry - fee_buffer)

            if high >= current_sl:
                if reached_1r:
                    return "breakeven", current_sl, bar_ts
                return "loss", current_sl, bar_ts
            if low <= tp:
                return "win", tp, bar_ts

    # Timeout: close at last bar's close — classify by actual RR achieved
    last_close = float(future_bars.iloc[-1]["close"])
    last_ts = future_bars.index[-1]
    if is_long:
        timeout_pnl = last_close - entry
    else:
        timeout_pnl = entry - last_close
    timeout_rr = timeout_pnl / sl_dist if sl_dist > 0 else 0.0

    if timeout_rr >= timeout_rr_threshold:
        return "win", last_close, last_ts
    elif timeout_rr <= -timeout_rr_threshold:
        return "loss", last_close, last_ts
    else:
        return "breakeven", last_close, last_ts


# ═══════════════════════════════════════════════════════════════════
#  Trade simulation with REAL price paths + Circuit Breaker
# ═══════════════════════════════════════════════════════════════════

def simulate_trades(
    signals: list[TradeSignal],
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0001,
    account_size: float = 100_000,
    use_circuit_breaker: bool = True,
    alignment_threshold: float = ALIGNMENT_THRESHOLD,
    asset_class: str = "crypto",
    risk_per_trade_override: float | None = None,
    max_equity_for_sizing: float | None = None,
    be_ratchet_r: float = 1.5,
    timeout_rr_threshold: float = 0.5,
    max_hold_bars: int = SCALP_MAX_HOLD_BARS,
) -> pd.DataFrame:
    """
    Simulate trades using REAL price-path outcomes.

    For each signal, walks forward through actual 5m candle data
    to check whether SL or TP is hit first. No synthetic win probability.

    Features:
    - Real price-path simulation (SL/TP hit detection on actual candles)
    - Alignment-threshold gate (rejects sig.alignment_score < threshold)
    - Confidence-based risk sizing via core.sizing.compute_risk_amount
    - Compound position sizing (risk % of current equity)
    - Circuit breaker (daily -3%, weekly -5%, class -2%, all-time -8%)
    - Scalp-Day Hybrid max hold: 4h (48 bars @ 5m)
    - Timeout trades closed at market price
    """
    if not signals:
        return pd.DataFrame()

    # Sort signals chronologically — required for concurrent position tracking
    signals = sorted(signals, key=lambda s: s.timestamp)

    cb = CircuitBreaker() if use_circuit_breaker else None
    # Suppress CB logging during simulation — it toggles pause/resume rapidly
    # which produces thousands of log lines. CB logic still works, just silent.
    _cb_logger = logging.getLogger("risk.circuit_breaker")
    _cb_orig_level = _cb_logger.level
    _cb_logger.setLevel(logging.CRITICAL)
    equity = account_size

    rows: list[dict[str, Any]] = []
    rejected_count = 0
    skipped_no_data = 0

    # Track open positions per symbol — live bot only allows ONE position per symbol
    open_positions: dict[str, pd.Timestamp] = {}  # symbol → exit_timestamp

    for sig in signals:
        # ── Concurrent position check — only 1 position per symbol ──
        if sig.symbol in open_positions:
            if sig.timestamp < open_positions[sig.symbol]:
                continue  # Skip — already holding this symbol

        sl_dist = abs(sig.entry_price - sig.stop_loss)
        tp_dist = abs(sig.take_profit - sig.entry_price)
        if sl_dist <= 0 or tp_dist <= 0:
            continue

        rr = tp_dist / sl_dist
        alignment = sig.alignment_score

        # ── Alignment-threshold gate (replaces AAA++/AAA+ tier dispatch) ──
        if alignment < alignment_threshold:
            rejected_count += 1
            continue

        # ── Circuit breaker check ────────────────────────────────
        if cb is not None:
            trade_time = sig.timestamp.to_pydatetime()
            if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo is None:
                trade_time = trade_time.replace(tzinfo=timezone.utc)
            can_trade, reason = cb.can_trade(asset_class, trade_time)
            if not can_trade:
                rejected_count += 1
                continue

        # ── Dynamic position sizing (%-based risk on CURRENT equity) ─
        sl_pct = sl_dist / sig.entry_price
        if sl_pct <= 0:
            continue

        # Equity used for sizing — capped if max_equity_for_sizing is set
        sizing_equity = equity
        if max_equity_for_sizing is not None:
            sizing_equity = min(equity, max_equity_for_sizing)

        # Confidence-based sizing via core.sizing (Scalp-Day Hybrid, no tiers).
        risk_amount = compute_risk_amount(alignment, sizing_equity)
        # Optuna-tuned max risk cap (overrides hard 3% default)
        max_risk_pct = risk_per_trade_override if risk_per_trade_override is not None else 0.03
        risk_amount = min(risk_amount, sizing_equity * max_risk_pct)
        if risk_amount <= 0 or equity <= 0:
            continue

        position_notional = risk_amount / sl_pct  # $ notional value
        position_size = position_notional / sig.entry_price  # qty in base units

        # Size reduction from circuit breaker
        if cb is not None:
            size_factor = cb.get_size_factor()
            position_size *= size_factor
            position_notional *= size_factor
            risk_amount *= size_factor

        # Slippage + commission cost (as % of notional, entry + exit)
        cost_pct = commission_pct * 2 + slippage_pct * 2
        cost = position_notional * cost_pct

        # ── REAL price-path outcome ──────────────────────────────
        outcome, exit_price, exit_ts = _resolve_trade_outcome(
            sig, commission_pct, slippage_pct,
            be_ratchet_r=be_ratchet_r,
            timeout_rr_threshold=timeout_rr_threshold,
            max_hold_bars=max_hold_bars,
        )

        if outcome == "skip":
            skipped_no_data += 1
            continue

        # Track position close time for concurrent position prevention
        if exit_ts is not None:
            open_positions[sig.symbol] = exit_ts

        # Calculate actual PnL from price movement
        if sig.direction == "long":
            price_pnl_pct = (exit_price - sig.entry_price) / sig.entry_price
        else:
            price_pnl_pct = (sig.entry_price - exit_price) / sig.entry_price

        pnl = position_notional * price_pnl_pct - cost

        # Actual realized RR (for logging)
        actual_rr = price_pnl_pct / sl_pct if sl_pct > 0 else 0.0

        # Record risk_pct BEFORE equity update (for accurate MC calculations)
        risk_pct_at_entry = risk_amount / equity if equity > 0 else 0

        # Update equity (never go below 0)
        equity = max(0, equity + pnl)

        # Bankrupt check — stop trading if equity drops below 10% of initial
        if equity < account_size * 0.10:
            logger.warning("BANKRUPT: equity %.0f < 10%% of initial %.0f", equity, account_size)
            break

        # Record in circuit breaker
        if cb is not None:
            pnl_pct_cb = pnl / account_size  # Always relative to INITIAL account
            trade_time = sig.timestamp.to_pydatetime()
            if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo is None:
                trade_time = trade_time.replace(tzinfo=timezone.utc)
            cb.record_trade_pnl(pnl_pct_cb, asset_class, sig.symbol, trade_time)

        rows.append(
            {
                "timestamp": sig.timestamp,
                "symbol": sig.symbol,
                "direction": sig.direction,
                "style": sig.style,
                "entry": sig.entry_price,
                "sl": sig.stop_loss,
                "tp": sig.take_profit,
                "exit_price": exit_price,
                "rr": rr,
                "actual_rr": actual_rr,
                "qty": position_size,
                "leverage": sig.leverage,
                "alignment": alignment,
                "outcome": outcome,
                "pnl": pnl,
                "cost": cost,
                "equity": equity,
                "risk_pct": risk_pct_at_entry,
            }
        )

    # Restore CB logging
    _cb_logger.setLevel(_cb_orig_level)

    if rejected_count > 0 or skipped_no_data > 0:
        logger.info(
            "Trade simulation: %d executed, %d rejected (gate/CB), %d skipped (no price data)",
            len(rows), rejected_count, skipped_no_data,
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Performance metrics (enhanced)
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(trades_df: pd.DataFrame, account_size: float = 100_000) -> dict[str, float]:
    """Compute comprehensive performance metrics.

    `sharpe` is the daily-equity-return Sharpe (crypto-annualized sqrt(365)),
    implemented in core.metrics per Phase-B quality-upgrade. `sharpe_naive`
    is the pre-Phase-B per-trade-PnL Sharpe kept only for debug comparison —
    do not use it downstream. `skew`, `kurt_nonexcess`, `n_obs_daily` are
    inputs needed by the Deflated Sharpe Ratio gate.
    """
    if trades_df.empty:
        return {
            "total_pnl": 0.0, "profit_factor": 0.0, "pf_real": 0.0,
            "max_drawdown": 0.0, "sharpe": 0.0, "sharpe_naive": 0.0,
            "skew": 0.0, "kurt_nonexcess": 3.0, "n_obs_daily": 0,
            "winrate": 0.0, "winrate_real": 0.0, "be_rate": 0.0,
            "total_trades": 0, "n_wins": 0, "n_losses": 0, "n_breakeven": 0,
            "recovery_factor": 0.0, "avg_rr": 0.0,
            "trades_aaa_pp": 0, "trades_aaa_p": 0,
            "pnl_per_trade": 0.0, "expectancy": 0.0,
        }

    pnl = trades_df["pnl"]
    outcomes = trades_df["outcome"] if "outcome" in trades_df.columns else pd.Series()

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    total_pnl = float(pnl.sum())
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 1e-9

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    winrate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0

    # ── Real metrics (excluding breakeven exits) ──────────────────
    n_be = int((outcomes == "breakeven").sum()) if len(outcomes) > 0 else 0
    n_real_wins = int((outcomes == "win").sum()) if len(outcomes) > 0 else 0
    n_real_losses = int((outcomes == "loss").sum()) if len(outcomes) > 0 else 0
    be_rate = n_be / len(pnl) if len(pnl) > 0 else 0.0

    # Real WR: only TP-hits vs real losses (exclude BE)
    real_decisions = n_real_wins + n_real_losses
    winrate_real = n_real_wins / real_decisions if real_decisions > 0 else 0.0

    # Real PF: profit from wins only vs losses only (BE excluded from both)
    real_win_pnl = float(pnl[outcomes == "win"].sum()) if n_real_wins > 0 else 0.0
    real_loss_pnl = abs(float(pnl[outcomes == "loss"].sum())) if n_real_losses > 0 else 0.0
    if real_loss_pnl > 0:
        pf_real = real_win_pnl / real_loss_pnl
    elif real_win_pnl > 0:
        pf_real = 100.0  # Cap: no losses but has wins — don't let PF go to infinity
    else:
        pf_real = 0.0

    # Cumulative equity & drawdown
    equity = account_size + pnl.cumsum()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Sharpe (Phase B quality-upgrade, 2026-04-19):
    #   - `sharpe` = daily-equity-return Sharpe, crypto-annualized sqrt(365)
    #   - `sharpe_naive` = old per-trade-PnL sqrt(252) — kept only for debug
    sharpe = _sharpe_daily(trades_df, account_size=account_size)
    if len(pnl) > 1 and pnl.std() > 0:
        sharpe_naive = float((pnl.mean() / pnl.std()) * math.sqrt(252))
    else:
        sharpe_naive = 0.0
    n_obs_daily, skew, kurt_nonexcess = _return_moments(
        trades_df, account_size=account_size,
    )

    # Recovery factor
    max_dd_usd = abs(max_drawdown * account_size) if max_drawdown != 0 else 1e-9
    recovery_factor = total_pnl / max_dd_usd if max_dd_usd > 0 else 0.0

    # Average RR (use actual realized RR if available, else target RR)
    if "actual_rr" in trades_df.columns:
        avg_rr = float(trades_df["actual_rr"].mean())
    elif "rr" in trades_df.columns:
        avg_rr = float(trades_df["rr"].mean())
    else:
        avg_rr = 0.0

    # Tier system killed 2026-04-19. Keys retained for dashboard/legacy
    # callers but always 0 — downstream code should prefer total_trades.
    trades_aaa_pp = 0
    trades_aaa_p = 0

    # Expectancy using REAL wins/losses only (excluding BE trades)
    real_wins_pnl = pnl[outcomes == "win"] if len(outcomes) > 0 else pnl[pnl > 0]
    real_losses_pnl = pnl[outcomes == "loss"] if len(outcomes) > 0 else pnl[pnl < 0]
    avg_win = float(real_wins_pnl.mean()) if len(real_wins_pnl) > 0 else 0.0
    avg_loss = abs(float(real_losses_pnl.mean())) if len(real_losses_pnl) > 0 else 0.0
    expectancy = (winrate_real * avg_win) - ((1 - winrate_real) * avg_loss)

    return {
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "pf_real": pf_real,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sharpe_naive": sharpe_naive,
        "skew": skew,
        "kurt_nonexcess": kurt_nonexcess,
        "n_obs_daily": n_obs_daily,
        "winrate": winrate,
        "winrate_real": winrate_real,
        "be_rate": be_rate,
        "total_trades": int(len(pnl)),
        "n_wins": n_real_wins,
        "n_losses": n_real_losses,
        "n_breakeven": n_be,
        "recovery_factor": recovery_factor,
        "avg_rr": avg_rr,
        "trades_aaa_pp": trades_aaa_pp,
        "trades_aaa_p": trades_aaa_p,
        "pnl_per_trade": total_pnl / len(pnl) if len(pnl) > 0 else 0.0,
        "expectancy": expectancy,
    }


# ═══════════════════════════════════════════════════════════════════
#  Monte Carlo Robustness Check
# ═══════════════════════════════════════════════════════════════════

def monte_carlo_check(
    trades_df: pd.DataFrame,
    account_size: float = 100_000,
    n_simulations: int = 1000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Shuffle trades 1000x using R-multiple compounding.

    Each trade is expressed as its realized R-multiple (actual_rr):
    how many risk-units it actually returned based on real price action.
    Examples: -1.0R = SL hit, +4.0R = TP hit at RR 4, +1.7R = timeout
    partial win, -0.3R = timeout partial loss. NOT fixed values.

    Each simulation:
    1. Shuffles the R-multiples
    2. For each trade: risks a fixed % of current equity (median
       risk_pct from actual trades), PnL = equity × risk_pct × R
    3. Compounds equity after each trade

    This produces genuinely different equity curves because early
    winners compound differently than late winners.
    """
    if trades_df.empty or len(trades_df) < 10:
        return {
            "robust": False,
            "reason": "Not enough trades for Monte Carlo",
            "median_pnl": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "pct_profitable": 0.0,
            "worst_dd_95pct": 0.0,
            "n_simulations": 0,
        }

    # Extract R-multiples from actual trades
    # actual_rr = realized R-multiple (positive for wins, negative for losses)
    if "actual_rr" in trades_df.columns:
        r_multiples = trades_df["actual_rr"].values.copy()
    else:
        # Fallback: compute from PnL and risk_pct
        risk_pct = trades_df["risk_pct"].values
        pnl = trades_df["pnl"].values
        equity_after = trades_df["equity"].values
        r_multiples = np.zeros(len(pnl))
        for i in range(len(pnl)):
            eq_before = equity_after[i] - pnl[i]
            risk_amt = eq_before * risk_pct[i] if eq_before > 0 else 1.0
            r_multiples[i] = pnl[i] / risk_amt if risk_amt > 0 else 0.0

    # Median risk % per trade (used for compound sizing in simulation)
    if "risk_pct" in trades_df.columns:
        median_risk_pct = float(trades_df["risk_pct"].median())
    else:
        median_risk_pct = 0.01  # 1% default

    # Cap extreme R-multiples to prevent single-trade distortion
    r_multiples = np.clip(r_multiples, -2.0, 15.0)

    rng = np.random.RandomState(42)

    final_equities: list[float] = []
    max_dds: list[float] = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(r_multiples)

        # Compound equity through shuffled R-multiples
        equity = account_size
        equity_curve = np.empty(len(shuffled))
        for i, r in enumerate(shuffled):
            risk_amount = equity * median_risk_pct
            trade_pnl = risk_amount * r
            equity = max(equity + trade_pnl, 0.0)
            equity_curve[i] = equity

        final_equities.append(equity)

        # Max drawdown of compounded equity curve
        running_max = np.maximum.accumulate(equity_curve)
        dd = np.where(running_max > 0, (equity_curve - running_max) / running_max, 0.0)
        max_dds.append(float(np.min(dd)))

    final_arr = np.array(final_equities)
    final_pnls = final_arr - account_size
    max_dds_arr = np.array(max_dds)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(final_pnls, alpha / 2 * 100))
    ci_upper = float(np.percentile(final_pnls, (1 - alpha / 2) * 100))
    median_pnl = float(np.median(final_pnls))
    pct_profitable = float(np.mean(final_pnls > 0))

    # Robust if 95% CI lower bound is profitable
    robust = ci_lower > 0

    worst_dd_95 = float(np.percentile(max_dds_arr, 5))  # 5th percentile = worst 5%

    return {
        "robust": robust,
        "reason": "95% CI profitable" if robust else "95% CI includes losses",
        "median_pnl": median_pnl,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pct_profitable": pct_profitable,
        "worst_dd_95pct": worst_dd_95,
        "n_simulations": n_simulations,
    }


# ═══════════════════════════════════════════════════════════════════
#  Anti-Overfitting Validation Gates
# ═══════════════════════════════════════════════════════════════════

def validate_oos_results(
    oos_metrics: dict[str, float],
    mc_result: dict[str, Any] | None = None,
    stability_result: dict[str, Any] | None = None,
    dsr_result: dict[str, Any] | None = None,
    max_dd_limit: float = -0.10,
    dsr_threshold: float = 0.70,
    asset_class: str = "crypto",
) -> dict[str, Any]:
    """
    Check if out-of-sample results pass anti-overfitting gates.

    8 Gates (ALL must pass):
    1. Profit Factor >= 1.5
    2. Minimum trades (per-class: 20 for crypto/stocks, 5 for forex/commodities)
    3. Sharpe >= 0.5
    4. Monte Carlo robust (95% CI profitable)
    5. Max Drawdown > -10% (funded account compliance)
    6. Parameter stability (±10% change < 50% PF shift)
    7. Win Rate > 20% AND positive expectancy (quality check)
    8. Deflated Sharpe Ratio >= threshold (Bailey & Lopez de Prado 2014) —
       probability the Sharpe is genuine after accounting for selection bias
       over `n_trials`. Phase-B quality-upgrade (2026-04-19).

    Deploy to funded account requires `dsr_threshold >= 0.95`. During the
    research phase we use 0.70 to keep a signal from the existing flow.
    """
    gates: dict[str, bool] = {}
    reasons: list[str] = []

    # Gate 1: Profit Factor (use real PF excluding breakeven exits)
    pf = oos_metrics.get("pf_real", oos_metrics.get("profit_factor", 0))
    gates["profit_factor_ok"] = pf >= 1.5
    if not gates["profit_factor_ok"]:
        reasons.append(f"Profit Factor(real) {pf:.2f} < 1.5")

    # Gate 2: Minimum trades — per-class (forex/commodities have fewer instruments)
    min_trades_required = 5 if asset_class in ("forex", "commodities") else 20
    n_trades = oos_metrics.get("total_trades", 0)
    gates["min_trades_ok"] = n_trades >= min_trades_required
    if not gates["min_trades_ok"]:
        reasons.append(f"Only {n_trades} trades (need >= {min_trades_required})")

    # Gate 3: Sharpe
    sharpe = oos_metrics.get("sharpe", 0)
    gates["sharpe_ok"] = sharpe >= 0.5
    if not gates["sharpe_ok"]:
        reasons.append(f"Sharpe {sharpe:.2f} < 0.5")

    # Gate 4: Monte Carlo
    if mc_result is not None:
        gates["monte_carlo_ok"] = mc_result.get("robust", False)
        if not gates["monte_carlo_ok"]:
            reasons.append(f"Monte Carlo: {mc_result.get('reason', 'failed')}")
        # Also check Monte Carlo worst-case DD
        mc_worst_dd = mc_result.get("worst_dd_95pct", 0.0)
        if mc_worst_dd < max_dd_limit:
            gates["monte_carlo_ok"] = False
            reasons.append(f"Monte Carlo worst DD {mc_worst_dd*100:.1f}% exceeds {max_dd_limit*100:.0f}% limit")
    else:
        gates["monte_carlo_ok"] = True  # Skip if not run

    # Gate 5: Max Drawdown (funded account compliance)
    max_dd = oos_metrics.get("max_drawdown", 0)
    gates["max_dd_ok"] = max_dd >= max_dd_limit
    if not gates["max_dd_ok"]:
        reasons.append(f"Max DD {max_dd*100:.1f}% exceeds {max_dd_limit*100:.0f}% limit (funded account)")

    # Gate 6: Parameter stability (relaxed for small samples)
    if stability_result is not None:
        n_real = oos_metrics.get("n_real_wins", 0) + oos_metrics.get("n_real_losses", 0)
        if n_real < 30:
            # Small sample: ±10% param change adds/removes 1-2 trades → huge PF swing
            # This is statistics, not overfitting. Skip gate but warn.
            gates["stability_ok"] = True
            if not stability_result.get("stable", False):
                reasons.append(
                    f"Stability skipped (only {n_real} real trades < 30): "
                    f"±10% → {stability_result.get('max_pf_change_pct', 0):.0f}% shift"
                )
        else:
            gates["stability_ok"] = stability_result.get("stable", False)
            if not gates["stability_ok"]:
                reasons.append(
                    f"Parameter unstable: ±10% change causes "
                    f"{stability_result.get('max_pf_change_pct', 0):.0f}% PF shift (max 50%)"
                )
    else:
        gates["stability_ok"] = True  # Skip if not run

    # Gate 7: Trade quality — Win Rate > 20% AND positive expectancy
    # Uses real WR (excluding breakeven exits) for honest assessment
    wr = oos_metrics.get("winrate_real", oos_metrics.get("winrate", 0))
    expectancy = oos_metrics.get("expectancy", 0)
    wr_ok = wr > 0.20
    exp_ok = expectancy > 0
    gates["quality_ok"] = wr_ok and exp_ok
    if not wr_ok:
        reasons.append(f"Win Rate {wr*100:.1f}% <= 20%")
    if not exp_ok:
        reasons.append(f"Negative expectancy: {expectancy:.2f}")

    # Gate 8: Deflated Sharpe Ratio — probability observed Sharpe is not
    # a selection-bias artefact of running `n_trials` random strategies.
    if dsr_result is not None:
        dsr_value = dsr_result.get("dsr", 0.0)
        gates["dsr_ok"] = dsr_value >= dsr_threshold
        if not gates["dsr_ok"]:
            reasons.append(
                f"DSR {dsr_value:.3f} < {dsr_threshold:.2f} "
                f"(selection bias over {dsr_result.get('n_trials', '?')} trials)"
            )
    else:
        gates["dsr_ok"] = True  # Skip if not computed

    all_passed = all(gates.values())

    # ── Sanity warnings (don't auto-reject, but flag suspicious metrics) ──
    warnings: list[str] = []
    if pf > 10:
        warnings.append(f"Suspicious PF={pf:.1f} (>10)")
    if wr > 0.80:
        warnings.append(f"Suspicious WR={wr*100:.0f}% (>80%)")
    be_rate = oos_metrics.get("be_rate", 0)
    if be_rate > 0.50:
        warnings.append(f"High BE rate={be_rate*100:.0f}% (>50%)")

    verdict = "PASS - Ready for paper trading" if all_passed else "FAIL - " + "; ".join(reasons)
    if warnings:
        verdict += " | WARNINGS: " + "; ".join(warnings)

    return {
        "passed": all_passed,
        "gates": gates,
        "reasons": reasons,
        "warnings": warnings,
        "verdict": verdict,
    }


def compute_dsr_for_oos(
    study: optuna.Study,
    oos_metrics: dict[str, float],
) -> dict[str, Any] | None:
    """Post-Optuna DSR for the selected OOS run.

    Collects the Sharpe of every completed trial to estimate the null
    distribution's variance, then applies the Bailey & Lopez de Prado
    (2014) haircut to the OOS-Sharpe using the return moments computed
    in `compute_metrics`.

    Returns `None` when the study has <2 completed trials with positive
    Sharpe, or when variance collapses to 0 — both signals that DSR
    cannot be meaningfully computed.
    """
    trial_sharpes: list[float] = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        sr = t.user_attrs.get("sharpe")
        if sr is not None and not np.isnan(sr):
            trial_sharpes.append(float(sr))

    if len([s for s in trial_sharpes if s > 0]) < 2:
        return None

    trial_var = _trial_sharpe_variance(trial_sharpes)
    if trial_var <= 0:
        return None

    n_obs = int(oos_metrics.get("n_obs_daily", 0))
    if n_obs < 2:
        return None

    observed = float(oos_metrics.get("sharpe", 0.0))
    skew = float(oos_metrics.get("skew", 0.0))
    kurt = float(oos_metrics.get("kurt_nonexcess", 3.0))

    dsr = _deflated_sharpe(
        observed_sharpe=observed,
        sharpe_variance=trial_var,
        n_trials=len(trial_sharpes),
        observation_count=n_obs,
        skewness=skew,
        kurtosis=kurt,
    )

    return {
        "dsr": dsr,
        "observed_sharpe": observed,
        "trial_sharpe_variance": trial_var,
        "n_trials": len(trial_sharpes),
        "n_obs_daily": n_obs,
        "skew": skew,
        "kurt_nonexcess": kurt,
    }


# ═══════════════════════════════════════════════════════════════════
#  Optuna objective
# ═══════════════════════════════════════════════════════════════════

# Bump this version whenever signal generation logic changes (TP calc, filters, etc.)
# Old caches with different versions are automatically ignored.
SIGNAL_CACHE_VERSION = "v16"  # v16: causal SMC indicators (no lookahead)


def _signal_cache_key(
    smc_cfg: dict[str, Any],
    symbols: list[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> str:
    """Build a deterministic hash for the signal cache."""
    key_data = json.dumps({
        "version": SIGNAL_CACHE_VERSION,
        "smc": {k: smc_cfg.get(k) for k in sorted(smc_cfg)},
        "symbols": sorted(symbols),
        "start": str(window_start),
        "end": str(window_end),
    }, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _precompute_window_signals(
    config: dict[str, Any],
    symbols: list[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    symbol_to_asset: dict[str, str] | None = None,
) -> list[TradeSignal]:
    """
    Generate signals ONCE per window with fixed SMC params (from config defaults).
    Signals include full alignment scores + meta flags.
    Optuna trials then only filter/trade these signals with different thresholds.

    Caches signals to disk (pickle) — if the same window + params + symbols
    are requested again, loads from cache instead of regenerating (~10-15min saved).
    """
    # Use config defaults for SMC params — these are not tuned per trial
    # In per-class mode: overlay asset-specific smc_profiles on top of global smc
    smc_cfg = config.get("smc", {})
    if symbol_to_asset:
        classes = set(symbol_to_asset.values())
        if len(classes) == 1:
            asset_class = classes.pop()
            profile = config.get("smc_profiles", {}).get(asset_class)
            if profile:
                smc_cfg = {**smc_cfg, **profile}
                logger.info("Using smc_profiles[%s]: swing=%d fvg=%.4f ob=%d liq=%.4f",
                            asset_class,
                            profile.get("swing_length", "?"),
                            profile.get("fvg_threshold", 0),
                            profile.get("order_block_lookback", 0),
                            profile.get("liquidity_range_percent", 0))
    # Detect asset class for per-class mode
    _asset_class = None
    if symbol_to_asset:
        classes = set(symbol_to_asset.values())
        if len(classes) == 1:
            _asset_class = classes.pop()
    params = {
        "swing_length": smc_cfg.get("swing_length", 10),
        "fvg_threshold": smc_cfg.get("fvg_threshold", 0.0004),
        "order_block_lookback": smc_cfg.get("order_block_lookback", 20),
        "liquidity_range_percent": smc_cfg.get("liquidity_range_percent", 0.005),
        "risk_reward": 3.0,  # Fallback RR (only used when no structure TP found)
        "alignment_threshold": 0.0,  # No filtering — let all signals through
        "style_weights": {"day": 1.0},
        "asset_class": _asset_class,  # Passed to strategy for forex-specific lookbacks/scoring
    }

    # ── Check disk cache ──────────────────────────────────────────
    cache_dir = Path("backtest/results/signal_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hash = _signal_cache_key(smc_cfg, symbols, window_start, window_end)
    cache_file = cache_dir / f"signals_{cache_hash}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_signals = pickle.load(f)
            logger.info(
                "CACHE HIT: Loaded %d signals from %s (window %s → %s)",
                len(cached_signals), cache_file.name,
                window_start.date(), window_end.date(),
            )
            return cached_signals
        except Exception as exc:
            logger.warning("Cache load failed (%s), regenerating: %s", cache_file.name, exc)

    # ── Generate signals (slow path) ──────────────────────────────
    strategy = SMCMultiStyleStrategy(config, params)

    def _gen_signals(sym):
        try:
            return strategy.generate_signals(sym, start=window_start, end=window_end)
        except Exception as exc:
            logger.debug("Signal gen failed for %s: %s", sym, exc)
            return []

    # Process in batches of 30 to limit peak memory (worker-count tunable).
    all_signals: list[TradeSignal] = []
    batch_size = 30
    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        batch_results = Parallel(n_jobs=_SIGNAL_WORKERS)(
            delayed(_gen_signals)(sym) for sym in batch
        )
        for sublist in batch_results:
            all_signals.extend(sublist)
        del batch_results
        gc.collect()

    # ── Save to disk cache ────────────────────────────────────────
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(all_signals, f)
        logger.info(
            "CACHE SAVE: %d signals → %s (window %s → %s)",
            len(all_signals), cache_file.name,
            window_start.date(), window_end.date(),
        )
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)

    logger.info(
        "Precomputed %d signals for %d instruments (window %s → %s)",
        len(all_signals), len(symbols),
        window_start.date(), window_end.date(),
    )
    return all_signals


def _build_objective(
    config: dict[str, Any],
    precomputed_signals: list[TradeSignal],
    window_index: int = 0,
    results_dir: Path | None = None,
    symbol_to_asset: dict[str, str] | None = None,
):
    """Return an Optuna objective function that filters precomputed signals."""

    # Pre-group signals by asset class (once, not per trial)
    signals_by_class: dict[str, list[TradeSignal]] = {}
    for sig in precomputed_signals:
        ac = (symbol_to_asset or {}).get(sig.symbol, "crypto")
        signals_by_class.setdefault(ac, []).append(sig)

    def objective(trial: optuna.Trial) -> float:
        # Only tune filtering + trading params (signals are precomputed)
        tuning = config.get("tuning", {})
        alignment_threshold = trial.suggest_float(
            "alignment_threshold",
            tuning.get("alignment_threshold_min", 0.60),
            tuning.get("alignment_threshold_max", 0.90),
            step=0.05,
        )
        min_rr = trial.suggest_categorical(
            "risk_reward", config["risk_reward"]["options"]
        )
        leverage = trial.suggest_int(
            "leverage", config["leverage"]["min"], config["leverage"]["max"]
        )
        risk_per_trade = trial.suggest_float(
            "risk_per_trade",
            config["risk_per_trade"]["min"],
            config["risk_per_trade"]["max"],
            step=0.001,
        )

        # Filter precomputed signals by alignment threshold and min RR
        # TP is structure-based (from signal generation) — NOT overridden
        filtered: list[TradeSignal] = []
        for sig in precomputed_signals:
            if sig.alignment_score < alignment_threshold:
                continue
            if sig.risk_reward < min_rr:
                continue
            # Keep original structure-based TP, only override leverage
            filtered.append(TradeSignal(
                timestamp=sig.timestamp,
                symbol=sig.symbol,
                direction=sig.direction,
                style=sig.style,
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                risk_reward=sig.risk_reward,
                position_size=sig.position_size,
                leverage=leverage,
                alignment_score=sig.alignment_score,
                meta=sig.meta,
            ))

        if not filtered:
            return 0.0

        # Re-group filtered signals
        filt_by_class: dict[str, list[TradeSignal]] = {}
        for sig in filtered:
            ac = (symbol_to_asset or {}).get(sig.symbol, "crypto")
            filt_by_class.setdefault(ac, []).append(sig)

        # Equity cap: 2× initial account prevents unrealistic compound growth
        max_eq = config["account"]["size"] * 2

        all_trades: list[pd.DataFrame] = []
        for ac, sigs in filt_by_class.items():
            t = simulate_trades(
                sigs,
                commission_pct=ASSET_COMMISSION.get(ac, config["backtest"]["commission_pct"]),
                slippage_pct=ASSET_SLIPPAGE.get(ac, config["backtest"]["slippage_pct"]),
                account_size=config["account"]["size"],
                use_circuit_breaker=True,
                asset_class=ac,
                risk_per_trade_override=risk_per_trade,
                max_equity_for_sizing=max_eq,
            )
            if not t.empty:
                all_trades.append(t)

        trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        if not trades.empty:
            trades = trades.sort_values("timestamp").reset_index(drop=True)
        metrics = compute_metrics(trades, account_size=config["account"]["size"])

        # Save trades for this trial (only first 10 trials to save disk)
        if results_dir is not None and not trades.empty and trial.number < 10:
            trades_sorted = trades.sort_values("timestamp")
            trades_sorted.to_csv(
                results_dir / f"trades_window{window_index}_trial{trial.number}.csv",
                index=False,
            )

        # Phase-C objective (2026-04-19): daily Sharpe with hard quality guards.
        # Replaces the pre-Phase-B composite (pf × sharpe × dd × trades × wr),
        # which was a data-mining invitation — no theoretical grounding.
        #
        # DSR is computed post-hoc on best_trial (gates deploy, not optimization).
        # Here we only need a noise-robust score surface for Optuna to navigate.
        trades_count = metrics["total_trades"]
        pf_real = metrics["pf_real"]
        max_dd = metrics["max_drawdown"]
        sharpe = metrics["sharpe"]

        # Hard guards — insufficient evidence fails the trial outright.
        # Thresholds calibrated to Lopez de Prado "minimum track record" (Ch. 14):
        #   - 20 trades below any robust Sharpe estimate
        #   - PF<1.3 is indistinguishable from noise at this sample size
        #   - DD>15% violates funded-account compliance envelope (-8% all-time).
        if trades_count < 20:
            return 0.0
        if pf_real < 1.3:
            return 0.0
        if max_dd < -0.15:
            return 0.0

        # Soft winrate penalty — real-WR > 80 % on this regime is almost
        # always a BE-ratchet/TP-hit artefact, not genuine edge.
        wr = metrics.get("winrate_real", 0.5)
        wr_factor = 1.0 if wr <= 0.80 else max(0.3, 1.0 - (wr - 0.80) * 5)

        score = max(0.0, sharpe) * wr_factor

        for k, v in metrics.items():
            trial.set_user_attr(k, v)

        return score

    return objective


# ═══════════════════════════════════════════════════════════════════
#  Top-20 % extraction
# ═══════════════════════════════════════════════════════════════════

def extract_top_params(
    study: optuna.Study,
    top_pct: float = 0.20,
) -> pd.DataFrame:
    """Extract the top fraction of trials sorted by objective value."""
    trials_data: list[dict[str, Any]] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "value": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        trials_data.append(row)

    if not trials_data:
        return pd.DataFrame()

    df = pd.DataFrame(trials_data).sort_values("value", ascending=False).reset_index(drop=True)
    n_top = max(1, int(len(df) * top_pct))
    return df.head(n_top)


# ═══════════════════════════════════════════════════════════════════
#  Parameter importance
# ═══════════════════════════════════════════════════════════════════

def compute_param_importance(study: optuna.Study, results_dir: Path) -> pd.DataFrame:
    """Use Optuna's fANOVA importance evaluator."""
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as exc:
        logger.warning("Could not compute parameter importance: %s", exc)
        return pd.DataFrame()

    imp_df = (
        pd.DataFrame(
            list(importance.items()), columns=["parameter", "importance"]
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = results_dir / "param_importance.csv"
    imp_df.to_csv(csv_path, index=False)
    logger.info("Parameter importance saved → %s", csv_path)

    try:
        import plotly.express as px
        fig = px.bar(
            imp_df, x="importance", y="parameter", orientation="h",
            title="Parameter Importance (fANOVA)",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        plot_path = results_dir / "param_importance.html"
        fig.write_html(str(plot_path))
        logger.info("Importance plot saved → %s", plot_path)
    except ImportError:
        logger.warning("plotly not installed – skipping importance plot")

    return imp_df


# ═══════════════════════════════════════════════════════════════════
#  Available symbols (all Parquet files in data/)
# ═══════════════════════════════════════════════════════════════════

def get_available_symbols(data_dir: Path) -> list[str]:
    """Return all symbols with a 5m Parquet file in data_dir (legacy single-dir)."""
    parquets = sorted(data_dir.glob("*_5m.parquet"))
    symbols = [
        p.stem.replace("_5m", "").replace("_", "/").replace("/USDT/USDT", "/USDT:USDT")
        for p in parquets
    ]
    return symbols


def get_multi_asset_symbols(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """
    Return symbols grouped by asset class from all data directories.

    Returns dict like:
      {"crypto": ["BTC/USDT:USDT", ...], "forex": ["EUR_USD", ...],
       "stocks": ["AAPL", ...], "commodities": ["XAU_USD", ...]}
    """
    data_cfg = cfg["data"]
    result: dict[str, list[str]] = {}

    # Crypto: data/crypto/ → Top 30 by 5m file size (proxy for volume/liquidity).
    # Post-Phase-1: 1m parquets are no longer downloaded; 5m is the entry TF.
    crypto_dir = Path(data_cfg.get("crypto_dir", "data/crypto"))
    max_crypto = cfg.get("volume_filter", {}).get("max_crypto_symbols", 30)
    if crypto_dir.exists():
        parquets = list(crypto_dir.glob("*_5m.parquet"))
        # Rank by file size (more data = more liquid/actively traded)
        sized = []
        for p in parquets:
            raw = p.stem.replace("_5m", "")
            if "USDT" in raw and raw != "volume":
                sized.append((raw, p.stat().st_size))
        sized.sort(key=lambda x: x[1], reverse=True)
        crypto_syms = [sym for sym, _ in sized[:max_crypto]]
        if crypto_syms:
            result["crypto"] = crypto_syms

    # Phase 1 (2026-04-18): Crypto-Only refocus. Forex/Stocks/Commodities
    # loaders removed. The data/ subdirectories may still exist on disk with
    # historical parquets but are no longer scanned — the config has no
    # forex_dir/stocks_dir/commodities_dir keys after the Phase 1 strip.
    return result


# Phase 2.1 SSOT (2026-04-18): values imported from core.constants.
# Crypto-only after Phase 1 strip — dict form retained for caller compatibility.
from core.constants import COMMISSION, SLIPPAGE
ASSET_COMMISSION: dict[str, float] = {"crypto": COMMISSION}
ASSET_SLIPPAGE: dict[str, float] = {"crypto": SLIPPAGE}


# ═══════════════════════════════════════════════════════════════════
#  Parameter Stability Check
# ═══════════════════════════════════════════════════════════════════

def check_parameter_stability(
    study: optuna.Study,
    best_params: dict[str, Any],
    config: dict[str, Any],
    precomputed_signals: list[TradeSignal],
    perturbation_pct: float = 0.10,
    n_perturbations: int = 5,
    asset_class: str = "crypto",
) -> dict[str, Any]:
    """
    Check if small parameter changes (±10%) drastically change performance.
    Uses precomputed signals — only perturbs filtering/trading params.
    """

    def _run_with_params(params):
        alignment_th = params.get("alignment_threshold", 0.65)
        min_rr = params.get("risk_reward", 2.0)
        rpt = params.get("risk_per_trade", None)
        max_eq = config["account"]["size"] * 2
        filtered = []
        for sig in precomputed_signals:
            if sig.alignment_score < alignment_th:
                continue
            if sig.risk_reward < min_rr:
                continue
            filtered.append(TradeSignal(
                timestamp=sig.timestamp, symbol=sig.symbol,
                direction=sig.direction, style=sig.style,
                entry_price=sig.entry_price, stop_loss=sig.stop_loss,
                take_profit=sig.take_profit, risk_reward=sig.risk_reward,
                position_size=sig.position_size,
                leverage=params.get("leverage", 10),
                alignment_score=sig.alignment_score, meta=sig.meta,
            ))
        trades = simulate_trades(
            filtered,
            commission_pct=ASSET_COMMISSION.get(asset_class, config["backtest"]["commission_pct"]),
            slippage_pct=ASSET_SLIPPAGE.get(asset_class, config["backtest"]["slippage_pct"]),
            account_size=config["account"]["size"],
            risk_per_trade_override=rpt,
            max_equity_for_sizing=max_eq,
        )
        return compute_metrics(trades, account_size=config["account"]["size"])

    base_metrics = _run_with_params(best_params)
    base_pf = base_metrics.get("pf_real", base_metrics["profit_factor"])

    perturbed_pfs: list[float] = []
    rng = np.random.RandomState(123)

    for _ in range(n_perturbations):
        perturbed = dict(best_params)
        for key, val in perturbed.items():
            if isinstance(val, dict):
                continue  # skip nested dicts (style_weights)
            if isinstance(val, (int, float)):
                factor = 1.0 + rng.uniform(-perturbation_pct, perturbation_pct)
                if isinstance(val, int):
                    # Use ceil/floor to ensure integer params actually change
                    new_val = val * factor
                    perturbed[key] = max(1, int(round(new_val)))
                else:
                    perturbed[key] = val * factor

        metrics = _run_with_params(perturbed)
        perturbed_pfs.append(metrics.get("pf_real", metrics["profit_factor"]))

    # Check stability: if any perturbation drops PF by >50%, it's unstable
    pf_changes = [abs(pf - base_pf) / max(base_pf, 0.01) for pf in perturbed_pfs]
    max_change = max(pf_changes) if pf_changes else 0
    is_stable = max_change < 0.50  # Less than 50% change

    return {
        "stable": is_stable,
        "base_pf": base_pf,
        "perturbed_pfs": perturbed_pfs,
        "max_pf_change_pct": max_change * 100,
        "warning": "" if is_stable else f"Parameter sensitivity: ±10% change causes {max_change*100:.0f}% PF shift",
    }


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run(
    config_path: str = "config/default_config.yaml",
) -> None:
    """
    Full walk-forward Optuna backtest pipeline (alignment-threshold gate).

    All validation gates (Monte Carlo, stability, max DD) always run.
    """
    cfg = load_config(config_path)
    data_dir = Path(cfg["data"]["data_dir"])
    results_dir = Path(cfg["backtest"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_months = cfg["backtest"]["train_months"]
    test_months = cfg["backtest"]["test_months"]
    n_trials = cfg["backtest"]["n_trials"]
    top_pct = cfg["backtest"]["top_percent"]
    study_name = cfg["backtest"]["study_name"]
    storage = cfg["backtest"]["storage"]
    account_size = cfg["account"]["size"]

    start = pd.Timestamp(cfg["data"]["start_date"], tz="UTC")
    end = pd.Timestamp(datetime.now(timezone.utc))

    # Multi-asset: discover symbols from all asset-class directories
    multi_assets = get_multi_asset_symbols(cfg)
    all_symbols_flat: list[str] = []
    symbol_to_asset: dict[str, str] = {}
    for ac, syms in multi_assets.items():
        for s in syms:
            all_symbols_flat.append(s)
            symbol_to_asset[s] = ac

    total_instruments = sum(len(v) for v in multi_assets.values())
    logger.info(
        "Multi-asset universe: %d instruments (crypto=%d, forex=%d, stocks=%d, commodities=%d)",
        total_instruments,
        len(multi_assets.get("crypto", [])),
        len(multi_assets.get("forex", [])),
        len(multi_assets.get("stocks", [])),
        len(multi_assets.get("commodities", [])),
    )

    # Fallback: legacy single data_dir if no multi-asset dirs
    if not all_symbols_flat:
        all_symbols_flat = get_available_symbols(data_dir)
        for s in all_symbols_flat:
            symbol_to_asset[s] = "crypto"

    windows = generate_wf_windows(start, end, train_months, test_months)
    logger.info("Walk-forward windows: %d", len(windows))

    # Pre-load 5m/1m price data for all symbols (real price-path simulation)
    logger.info("Loading price data for %d instruments...", len(all_symbols_flat))
    load_price_data_for_symbols(all_symbols_flat, cfg)
    logger.info("Price data loaded for %d instruments", len(_price_cache))

    all_window_results: list[pd.DataFrame] = []
    all_validations: list[dict[str, Any]] = []

    for wi, window in enumerate(tqdm(windows, desc="Walk-forward windows")):
        logger.info(
            "Window %d: Train %s → %s | Test %s → %s",
            wi,
            window["train_start"].date(),
            window["train_end"].date(),
            window["test_start"].date(),
            window["test_end"].date(),
        )

        symbols = all_symbols_flat
        if not symbols:
            logger.warning("No symbols for window %d – skipping", wi)
            continue

        # ── Precompute signals once for this window ────────────────
        train_signals = _precompute_window_signals(
            cfg, symbols, window["train_start"], window["train_end"],
            symbol_to_asset=symbol_to_asset,
        )

        if not train_signals:
            logger.warning("Window %d: 0 precomputed signals – skipping", wi)
            continue

        # ── Optuna study (training phase) ─────────────────────────
        window_study_name = f"{study_name}_w{wi}"
        study = optuna.create_study(
            study_name=window_study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        objective = _build_objective(
            cfg, train_signals,
            window_index=wi, results_dir=results_dir,
            symbol_to_asset=symbol_to_asset,
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=_OPTUNA_WORKERS, show_progress_bar=True)

        # ── Out-of-sample test with best params ──────────────────
        best_params = study.best_trial.params
        best_params["style_weights"] = {
            "day": best_params.pop("weight_day", 1.0),
        }

        # ── Precompute OOS signals ──────────────────────────────────
        oos_all_signals = _precompute_window_signals(
            cfg, symbols, window["test_start"], window["test_end"],
            symbol_to_asset=symbol_to_asset,
        )

        # Filter OOS signals with best params (structure TP preserved)
        best_alignment = best_params.get("alignment_threshold", 0.65)
        best_min_rr = best_params.get("risk_reward", 2.0)
        oos_signals: list[TradeSignal] = []
        for sig in oos_all_signals:
            if sig.alignment_score < best_alignment:
                continue
            if sig.risk_reward < best_min_rr:
                continue
            oos_signals.append(TradeSignal(
                timestamp=sig.timestamp, symbol=sig.symbol,
                direction=sig.direction, style=sig.style,
                entry_price=sig.entry_price, stop_loss=sig.stop_loss,
                take_profit=sig.take_profit, risk_reward=sig.risk_reward,
                position_size=sig.position_size,
                leverage=best_params.get("leverage", 10),
                alignment_score=sig.alignment_score, meta=sig.meta,
            ))

        # Group OOS signals by asset class for correct commissions
        oos_by_class: dict[str, list[TradeSignal]] = {}
        for sig in oos_signals:
            ac = symbol_to_asset.get(sig.symbol, "crypto")
            oos_by_class.setdefault(ac, []).append(sig)

        best_rpt = best_params.get("risk_per_trade", None)
        max_eq = account_size * 2  # Cap compound growth at 2× initial

        oos_parts: list[pd.DataFrame] = []
        for ac, sigs in oos_by_class.items():
            t = simulate_trades(
                sigs,
                commission_pct=ASSET_COMMISSION.get(ac, cfg["backtest"]["commission_pct"]),
                slippage_pct=cfg["backtest"]["slippage_pct"],
                account_size=account_size,
                use_circuit_breaker=True,
                asset_class=ac,
                risk_per_trade_override=best_rpt,
                max_equity_for_sizing=max_eq,
            )
            if not t.empty:
                oos_parts.append(t)

        oos_trades = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
        if not oos_trades.empty:
            oos_trades = oos_trades.sort_values("timestamp").reset_index(drop=True)
        oos_metrics = compute_metrics(oos_trades, account_size=account_size)

        # Save OOS trades
        if not oos_trades.empty:
            oos_trades.sort_values("timestamp").to_csv(
                results_dir / f"oos_trades_w{wi}.csv", index=False,
            )

        # ── Monte Carlo check (always — it's a hard gate) ─────────
        mc_result = None
        if not oos_trades.empty:
            mc_result = monte_carlo_check(oos_trades, account_size)
            mc_path = results_dir / f"monte_carlo_w{wi}.json"
            with open(mc_path, "w") as fh:
                json.dump(mc_result, fh, indent=2, default=str)
            logger.info(
                "Monte Carlo W%d: robust=%s median_pnl=%.0f CI=[%.0f, %.0f] profitable=%.1f%% worst_dd_95=%.1f%%",
                wi, mc_result["robust"], mc_result["median_pnl"],
                mc_result["ci_lower"], mc_result["ci_upper"],
                mc_result["pct_profitable"] * 100,
                mc_result.get("worst_dd_95pct", 0) * 100,
            )

        # ── Parameter stability check (always — it's a hard gate) ─
        stability_result = check_parameter_stability(
            study, best_params, cfg, oos_all_signals,
            asset_class="crypto",  # Global mode defaults to crypto
        )
        logger.info(
            "Stability W%d: stable=%s max_change=%.1f%%",
            wi, stability_result["stable"],
            stability_result["max_pf_change_pct"],
        )

        # ── Deflated Sharpe Ratio (post-Optuna selection-bias haircut) ─
        dsr_result = compute_dsr_for_oos(study, oos_metrics)
        if dsr_result is not None:
            logger.info(
                "DSR W%d: %.3f (observed SR=%.2f, trial_var=%.3f, n_trials=%d, n_obs=%d)",
                wi, dsr_result["dsr"], dsr_result["observed_sharpe"],
                dsr_result["trial_sharpe_variance"], dsr_result["n_trials"],
                dsr_result["n_obs_daily"],
            )

        # ── Validation gates (8 gates: PF, trades, Sharpe, MC, DD, stability, quality, DSR) ──
        validation = validate_oos_results(
            oos_metrics, mc_result,
            stability_result=stability_result,
            dsr_result=dsr_result,
            max_dd_limit=-0.10,  # Funded account: max -10% all-time DD
            dsr_threshold=0.70,  # Research phase; 0.95 required before funded deploy
            asset_class="crypto",  # Global mode — default to crypto trade gate
        )
        validation["window"] = wi
        validation["stability"] = stability_result
        validation["dsr"] = dsr_result
        all_validations.append(validation)

        logger.info(
            "Window %d OOS: PF=%.2f(real) Sharpe=%.2f WR=%.1f%%(real) BE=%.0f%% "
            "Trades=%d(%dW/%dL/%dBE) DD=%.2f%% | %s",
            wi, oos_metrics.get("pf_real", oos_metrics["profit_factor"]),
            oos_metrics["sharpe"],
            oos_metrics.get("winrate_real", oos_metrics["winrate"]) * 100,
            oos_metrics.get("be_rate", 0) * 100,
            oos_metrics["total_trades"],
            oos_metrics.get("n_wins", 0), oos_metrics.get("n_losses", 0),
            oos_metrics.get("n_breakeven", 0),
            oos_metrics["max_drawdown"] * 100,
            validation["verdict"],
        )

        # Save window results
        window_result = {"window": wi, **oos_metrics, **best_params}
        all_window_results.append(pd.DataFrame([window_result]))

        # ── Top 20% extraction ─────────────────────────────────────
        top_df = extract_top_params(study, top_pct=top_pct)
        if not top_df.empty:
            top_df.to_csv(results_dir / f"top_params_w{wi}.csv", index=False)

        # ── Parameter importance ───────────────────────────────────
        compute_param_importance(study, results_dir)

        # ── Free memory between windows ──────────────────────────
        del train_signals, oos_all_signals, oos_signals
        del oos_by_class, oos_parts, oos_trades, study
        gc.collect()

    # ── Aggregate results ─────────────────────────────────────────
    if all_window_results:
        summary = pd.concat(all_window_results, ignore_index=True)
        summary.to_csv(results_dir / "wfo_summary.csv", index=False)

        global_top = summary.nlargest(
            max(1, int(len(summary) * top_pct)), "total_pnl"
        )
        global_top.to_csv(results_dir / "global_top_params.csv", index=False)

        # Summary stats
        stats = {
            "total_windows": len(windows),
            "mean_pnl": float(summary["total_pnl"].mean()),
            "mean_sharpe": float(summary["sharpe"].mean()),
            "mean_winrate": float(summary["winrate"].mean()),
            "mean_winrate_real": float(summary["winrate_real"].mean()) if "winrate_real" in summary.columns else 0.0,
            "mean_profit_factor": float(summary["profit_factor"].mean()),
            "mean_pf_real": float(summary["pf_real"].mean()) if "pf_real" in summary.columns else 0.0,
            "mean_be_rate": float(summary["be_rate"].mean()) if "be_rate" in summary.columns else 0.0,
            "worst_drawdown": float(summary["max_drawdown"].min()),
            "mean_trades_per_window": float(summary["total_trades"].mean()),
            "total_trades_all_windows": int(summary["total_trades"].sum()),
        }

        # Add validation summary
        n_passed = sum(1 for v in all_validations if v["passed"])
        stats["validation_passed"] = n_passed
        stats["validation_total"] = len(all_validations)
        stats["validation_pass_rate"] = n_passed / len(all_validations) if all_validations else 0

        with open(results_dir / "backtest_stats.json", "w") as fh:
            json.dump(stats, fh, indent=2)

        # Save validation results
        with open(results_dir / "validation_results.json", "w") as fh:
            json.dump(all_validations, fh, indent=2, default=str)

        logger.info("Summary stats: %s", json.dumps(stats, indent=2))

    logger.info("Walk-forward backtest complete. Results in %s", results_dir)


# ═══════════════════════════════════════════════════════════════════
#  Helper: simulate with specific params (DRY — used by objective,
#  OOS evaluation, stability, cross-window validation)
# ═══════════════════════════════════════════════════════════════════

def _simulate_with_params(
    params: dict[str, Any],
    precomputed_signals: list[TradeSignal],
    config: dict[str, Any],
    symbol_to_asset: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Filter precomputed signals with params, simulate trades, return (trades_df, metrics).

    This consolidates the repeated pattern from objective, OOS eval, and stability check.
    """
    alignment_th = params.get("alignment_threshold", 0.65)
    min_rr = params.get("risk_reward", 2.0)
    leverage = params.get("leverage", 10)
    rpt = params.get("risk_per_trade", None)
    be_ratchet_r = params.get("be_ratchet_r", 1.5)
    timeout_rr_threshold = params.get("timeout_rr_threshold", 0.5)
    max_hold_bars = params.get("max_hold_bars", SCALP_MAX_HOLD_BARS)
    account_size = config["account"]["size"]
    max_eq = account_size * 2

    # Filter signals
    filtered: list[TradeSignal] = []
    for sig in precomputed_signals:
        if sig.alignment_score < alignment_th:
            continue
        if sig.risk_reward < min_rr:
            continue
        filtered.append(TradeSignal(
            timestamp=sig.timestamp, symbol=sig.symbol,
            direction=sig.direction, style=sig.style,
            entry_price=sig.entry_price, stop_loss=sig.stop_loss,
            take_profit=sig.take_profit, risk_reward=sig.risk_reward,
            position_size=sig.position_size,
            leverage=leverage,
            alignment_score=sig.alignment_score, meta=sig.meta,
        ))

    if not filtered:
        return pd.DataFrame(), compute_metrics(pd.DataFrame(), account_size=account_size)

    # Group by asset class for correct commissions
    by_class: dict[str, list[TradeSignal]] = {}
    for sig in filtered:
        ac = (symbol_to_asset or {}).get(sig.symbol, "crypto")
        by_class.setdefault(ac, []).append(sig)

    all_trades: list[pd.DataFrame] = []
    for ac, sigs in by_class.items():
        t = simulate_trades(
            sigs,
            commission_pct=ASSET_COMMISSION.get(ac, config["backtest"]["commission_pct"]),
            slippage_pct=ASSET_SLIPPAGE.get(ac, config["backtest"]["slippage_pct"]),
            account_size=account_size,
            use_circuit_breaker=True,
            asset_class=ac,
            risk_per_trade_override=rpt,
            max_equity_for_sizing=max_eq,
            be_ratchet_r=be_ratchet_r,
            timeout_rr_threshold=timeout_rr_threshold,
            max_hold_bars=max_hold_bars,
        )
        if not t.empty:
            all_trades.append(t)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not trades.empty:
        trades = trades.sort_values("timestamp").reset_index(drop=True)
    metrics = compute_metrics(trades, account_size=account_size)
    return trades, metrics


# ═══════════════════════════════════════════════════════════════════
#  Cross-Window Evergreen Validation
# ═══════════════════════════════════════════════════════════════════

def cross_window_validate(
    candidate_params: list[dict[str, Any]],
    oos_signals_per_window: list[list[TradeSignal]],
    asset_class: str,
    config: dict[str, Any],
    min_pf: float = 1.5,
) -> dict[str, Any] | None:
    """
    Test each candidate param set on ALL OOS windows.

    Only params with pf_real >= min_pf on EVERY window qualify as "evergreen".
    Returns the best evergreen set (highest min PF across windows), or None.
    """
    symbol_to_asset: dict[str, str] = {}
    for window_sigs in oos_signals_per_window:
        for sig in window_sigs:
            symbol_to_asset[sig.symbol] = asset_class

    results: list[dict[str, Any]] = []

    for params in candidate_params:
        window_metrics: list[dict[str, float]] = []
        all_pass = True

        for wi, oos_sigs in enumerate(oos_signals_per_window):
            if not oos_sigs:
                all_pass = False
                break
            _, metrics = _simulate_with_params(
                params, oos_sigs, config,
                symbol_to_asset=symbol_to_asset,
            )
            pf = metrics.get("pf_real", metrics.get("profit_factor", 0))
            trades = metrics.get("total_trades", 0)
            # Reject: too few trades (metrics unreliable)
            if trades < 5:
                all_pass = False
                break
            if pf < min_pf:
                all_pass = False
            window_metrics.append(metrics)

        if not all_pass or not window_metrics:
            continue

        pf_values = [m.get("pf_real", m.get("profit_factor", 0)) for m in window_metrics]
        results.append({
            "params": params,
            "window_metrics": {
                f"w{i}": {
                    "pf_real": m.get("pf_real", 0),
                    "winrate_real": m.get("winrate_real", 0),
                    "trades": m.get("total_trades", 0),
                    "dd": m.get("max_drawdown", 0),
                    "sharpe": m.get("sharpe", 0),
                }
                for i, m in enumerate(window_metrics)
            },
            "min_pf": min(pf_values),
            "mean_pf": sum(pf_values) / len(pf_values),
        })

    if not results:
        return None

    # Best = highest minimum PF across all windows
    results.sort(key=lambda r: r["min_pf"], reverse=True)
    best = results[0]

    # Confidence: low if any window has < 10 trades
    min_trades = min(
        m.get("trades", 0)
        for wm in best["window_metrics"].values()
        for m in [wm]
    )
    confidence = "low" if min_trades < 10 else ("medium" if min_trades < 20 else "high")

    return {
        "asset_class": asset_class,
        "params": best["params"],
        "cross_window_metrics": best["window_metrics"],
        "min_pf": best["min_pf"],
        "mean_pf": best["mean_pf"],
        "is_evergreen": True,
        "confidence": confidence,
        "n_candidates_tested": len(candidate_params),
        "n_evergreen_found": len(results),
    }


# ═══════════════════════════════════════════════════════════════════
#  Per-Asset-Class Walk-Forward Pipeline
# ═══════════════════════════════════════════════════════════════════

# Default conservative params for asset classes with too few signals
_DEFAULT_CONSERVATIVE_PARAMS = {
    "alignment_threshold": 0.88,
    "risk_reward": 3.0,
    "leverage": 5,
    "risk_per_trade": 0.005,
}


def run_per_asset_class(
    config_path: str = "config/default_config.yaml",
) -> dict[str, dict[str, Any]]:
    """
    Walk-Forward Optuna backtest per asset class.

    For each class: optimize params independently, then cross-validate
    across ALL windows to find evergreen params.

    Returns: {asset_class: evergreen_result_dict}
    """
    cfg = load_config(config_path)
    results_dir = Path(cfg["backtest"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_months = cfg["backtest"]["train_months"]
    test_months = cfg["backtest"]["test_months"]
    n_trials = cfg["backtest"]["n_trials"]
    top_pct = cfg["backtest"]["top_percent"]
    study_name = cfg["backtest"]["study_name"]
    storage = cfg["backtest"]["storage"]
    account_size = cfg["account"]["size"]

    start = pd.Timestamp(cfg["data"]["start_date"], tz="UTC")
    end = pd.Timestamp(datetime.now(timezone.utc))

    # Per-class tuning ranges
    tuning_per_class = cfg.get("tuning_per_class", {})

    # Discover symbols per class
    multi_assets = get_multi_asset_symbols(cfg)
    if not multi_assets:
        logger.error("No instruments found in any data directory")
        return {}

    windows = generate_wf_windows(start, end, train_months, test_months)
    if not windows:
        logger.error("No walk-forward windows generated")
        return {}
    logger.info("Walk-forward windows: %d", len(windows))

    evergreen_results: dict[str, dict[str, Any]] = {}

    for ac, symbols in multi_assets.items():
        logger.info(
            "═══ Asset Class: %s (%d instruments) ═══",
            ac.upper(), len(symbols),
        )

        ac_results_dir = results_dir / ac
        ac_results_dir.mkdir(parents=True, exist_ok=True)

        # Symbol-to-asset mapping (single class)
        symbol_to_asset = {s: ac for s in symbols}

        # Pre-load price data for this class
        logger.info("Loading price data for %d %s instruments...", len(symbols), ac)
        load_price_data_for_symbols(symbols, cfg)

        # Per-class leverage range
        class_tuning = tuning_per_class.get(ac, {})
        lev_min = class_tuning.get("leverage_min", cfg["leverage"]["min"])
        lev_max = class_tuning.get("leverage_max", cfg["leverage"]["max"])

        # Collect best params + OOS signals per window for cross-validation
        best_params_per_window: list[dict[str, Any]] = []
        top_params_per_window: list[list[dict[str, Any]]] = []
        oos_signals_per_window: list[list[TradeSignal]] = []

        for wi, window in enumerate(windows):
            logger.info(
                "%s Window %d: Train %s → %s | Test %s → %s",
                ac.upper(), wi,
                window["train_start"].date(), window["train_end"].date(),
                window["test_start"].date(), window["test_end"].date(),
            )

            # ── Signal precomputation (per-class, fewer symbols = less RAM) ──
            train_signals = _precompute_window_signals(
                cfg, symbols, window["train_start"], window["train_end"],
                symbol_to_asset=symbol_to_asset,
            )

            if len(train_signals) < 50:
                logger.warning(
                    "%s W%d: Only %d signals (< 50) — using conservative defaults",
                    ac, wi, len(train_signals),
                )
                best_params_per_window.append(dict(_DEFAULT_CONSERVATIVE_PARAMS))

                # Still precompute OOS signals for cross-validation
                oos_sigs = _precompute_window_signals(
                    cfg, symbols, window["test_start"], window["test_end"],
                    symbol_to_asset=symbol_to_asset,
                )
                oos_signals_per_window.append(oos_sigs)
                top_params_per_window.append([dict(_DEFAULT_CONSERVATIVE_PARAMS)])
                del train_signals
                gc.collect()
                continue

            # ── Optuna study (per-class, per-window) ─────────────────
            window_study_name = f"{study_name}_{ac}_w{wi}"

            # Build objective with per-class leverage range
            def _make_objective(cfg, train_sigs, wi, res_dir, s2a, lmin, lmax):
                """Create objective with class-specific leverage range."""
                signals_by_class: dict[str, list[TradeSignal]] = {}
                for sig in train_sigs:
                    c = (s2a or {}).get(sig.symbol, "crypto")
                    signals_by_class.setdefault(c, []).append(sig)

                def objective(trial: optuna.Trial) -> float:
                    tuning = cfg.get("tuning", {})
                    alignment_threshold = trial.suggest_float(
                        "alignment_threshold",
                        tuning.get("alignment_threshold_min", 0.60),
                        tuning.get("alignment_threshold_max", 0.90),
                        step=0.05,
                    )
                    min_rr = trial.suggest_categorical(
                        "risk_reward", cfg["risk_reward"]["options"],
                    )
                    leverage = trial.suggest_int("leverage", lmin, lmax)
                    risk_per_trade = trial.suggest_float(
                        "risk_per_trade",
                        cfg["risk_per_trade"]["min"],
                        cfg["risk_per_trade"]["max"],
                        step=0.001,
                    )

                    params = {
                        "alignment_threshold": alignment_threshold,
                        "risk_reward": min_rr,
                        "leverage": leverage,
                        "risk_per_trade": risk_per_trade,
                    }
                    _, metrics = _simulate_with_params(
                        params, train_sigs, cfg,
                        symbol_to_asset=s2a,
                    )

                    if res_dir is not None and trial.number < 10:
                        pass  # skip per-trial CSV saves for per-class mode

                    # Phase-C objective (2026-04-19): daily Sharpe + hard guards.
                    # See _build_objective for the rationale (same contract).
                    trades_count = metrics["total_trades"]
                    pf_real = metrics["pf_real"]
                    max_dd = metrics["max_drawdown"]
                    sharpe = metrics["sharpe"]

                    if trades_count < 20:
                        score = 0.0
                    elif pf_real < 1.3:
                        score = 0.0
                    elif max_dd < -0.15:
                        score = 0.0
                    else:
                        wr = metrics.get("winrate_real", 0.5)
                        wr_factor = 1.0 if wr <= 0.80 else max(0.3, 1.0 - (wr - 0.80) * 5)
                        score = max(0.0, sharpe) * wr_factor

                    for k, v in metrics.items():
                        trial.set_user_attr(k, v)
                    return score

                return objective

            study = optuna.create_study(
                study_name=window_study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True,
            )
            objective = _make_objective(
                cfg, train_signals, wi, ac_results_dir,
                symbol_to_asset, lev_min, lev_max,
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=_OPTUNA_WORKERS, show_progress_bar=True)

            best = study.best_trial.params
            best_params_per_window.append(dict(best))

            # Collect top-20% param sets for cross-validation
            top_df = extract_top_params(study, top_pct=top_pct)
            top_sets: list[dict[str, Any]] = []
            if not top_df.empty:
                for _, row in top_df.iterrows():
                    p = {}
                    for col in ("alignment_threshold", "risk_reward", "leverage", "risk_per_trade"):
                        if col in row:
                            p[col] = row[col]
                    top_sets.append(p)
                top_df.to_csv(ac_results_dir / f"top_params_w{wi}.csv", index=False)
            top_params_per_window.append(top_sets)

            # ── OOS signals ──────────────────────────────────────────
            oos_sigs = _precompute_window_signals(
                cfg, symbols, window["test_start"], window["test_end"],
                symbol_to_asset=symbol_to_asset,
            )
            oos_signals_per_window.append(oos_sigs)

            # ── OOS evaluation with best params ──────────────────────
            oos_trades, oos_metrics = _simulate_with_params(
                best, oos_sigs, cfg,
                symbol_to_asset=symbol_to_asset,
            )

            if not oos_trades.empty:
                oos_trades.to_csv(ac_results_dir / f"oos_trades_w{wi}.csv", index=False)

            # Monte Carlo
            mc_result = None
            if not oos_trades.empty:
                mc_result = monte_carlo_check(oos_trades, account_size)

            # Stability
            stability_result = check_parameter_stability(
                study, best, cfg, oos_sigs,
                asset_class=ac,
            )

            # DSR haircut + Validation (per-class min trade gates)
            dsr_result = compute_dsr_for_oos(study, oos_metrics)
            validation = validate_oos_results(
                oos_metrics, mc_result,
                stability_result=stability_result,
                dsr_result=dsr_result,
                dsr_threshold=0.70,  # Research phase; 0.95 required before funded deploy
                asset_class=ac,
            )
            validation["dsr"] = dsr_result

            logger.info(
                "%s W%d OOS: PF=%.2f(real) WR=%.1f%%(real) BE=%.0f%% "
                "Trades=%d DD=%.2f%% Stability=%.1f%% | %s",
                ac.upper(), wi,
                oos_metrics.get("pf_real", 0),
                oos_metrics.get("winrate_real", 0) * 100,
                oos_metrics.get("be_rate", 0) * 100,
                oos_metrics.get("total_trades", 0),
                oos_metrics.get("max_drawdown", 0) * 100,
                stability_result.get("max_pf_change_pct", 0),
                validation["verdict"],
            )

            # Param importance
            compute_param_importance(study, ac_results_dir)

            # Cleanup
            del train_signals, oos_sigs, oos_trades, study
            gc.collect()

        # ── Cross-window validation ──────────────────────────────────
        # Collect ALL unique candidate param sets from all windows
        all_candidates: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for params_list in [best_params_per_window] + top_params_per_window:
            for p in (params_list if isinstance(params_list, list) else [params_list]):
                key = json.dumps(p, sort_keys=True)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_candidates.append(p)

        logger.info(
            "%s: Cross-window validation with %d candidates on %d OOS windows",
            ac.upper(), len(all_candidates), len(oos_signals_per_window),
        )

        evergreen = cross_window_validate(
            all_candidates, oos_signals_per_window, ac, cfg,
        )

        if evergreen is not None:
            # Save evergreen params
            eg_path = ac_results_dir / "evergreen_params.json"
            with open(eg_path, "w") as f:
                json.dump(evergreen, f, indent=2, default=str)
            logger.info(
                "✓ %s EVERGREEN: alignment=%.2f rr=%.1f lev=%d risk=%.3f | min_PF=%.2f (%d/%d candidates)",
                ac.upper(),
                evergreen["params"]["alignment_threshold"],
                evergreen["params"]["risk_reward"],
                evergreen["params"]["leverage"],
                evergreen["params"]["risk_per_trade"],
                evergreen["min_pf"],
                evergreen["n_evergreen_found"],
                evergreen["n_candidates_tested"],
            )
            evergreen_results[ac] = evergreen
        else:
            # No evergreen found — use conservative defaults
            logger.warning(
                "✗ %s: No evergreen params found! Using conservative defaults.",
                ac.upper(),
            )
            default_result = {
                "asset_class": ac,
                "params": dict(_DEFAULT_CONSERVATIVE_PARAMS),
                "cross_window_metrics": {},
                "min_pf": 0.0,
                "mean_pf": 0.0,
                "is_evergreen": False,
                "n_candidates_tested": len(all_candidates),
                "n_evergreen_found": 0,
            }
            eg_path = ac_results_dir / "evergreen_params.json"
            with open(eg_path, "w") as f:
                json.dump(default_result, f, indent=2, default=str)
            evergreen_results[ac] = default_result

        # Free OOS signals memory
        del oos_signals_per_window
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────
    summary = {}
    for ac, eg in evergreen_results.items():
        summary[ac] = {
            "is_evergreen": eg.get("is_evergreen", False),
            "params": eg["params"],
            "min_pf": eg.get("min_pf", 0),
        }
    with open(results_dir / "evergreen_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Per-asset-class backtest complete. Results in %s", results_dir)
    for ac, s in summary.items():
        status = "EVERGREEN" if s["is_evergreen"] else "DEFAULT"
        logger.info(
            "  %s: [%s] alignment=%.2f rr=%.1f lev=%d risk=%.3f min_pf=%.2f",
            ac, status,
            s["params"]["alignment_threshold"],
            s["params"]["risk_reward"],
            s["params"]["leverage"],
            s["params"]["risk_per_trade"],
            s["min_pf"],
        )

    return evergreen_results


# ═══════════════════════════════════════════════════════════════════
#  Brute-Force Evergreen Grid Search
# ═══════════════════════════════════════════════════════════════════

# Module-level shared state for forked workers (copy-on-write)
_BF_SIGNALS: list[list] = []
_BF_CFG: dict[str, Any] = {}
_BF_SYM2ASSET: dict[str, str] = {}
_BF_MIN_TRADES: int = 10
_BF_N_WINDOWS: int = 3


def _eval_combo_worker(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Evaluate one param combo across all windows. Top-level for multiprocessing."""
    window_metrics_list = []
    all_pass = True

    for wi, oos_sigs in enumerate(_BF_SIGNALS):
        if not oos_sigs:
            all_pass = False
            break
        _, metrics = _simulate_with_params(
            params, oos_sigs, _BF_CFG,
            symbol_to_asset=_BF_SYM2ASSET,
        )
        window_metrics_list.append(metrics)

        pf = metrics.get("pf_real", 0)
        trades = metrics.get("total_trades", 0)
        dd = metrics.get("max_drawdown", 0)
        wr = metrics.get("winrate_real", 0)
        sharpe = metrics.get("sharpe", 0)
        if (pf < 2.0 or pf > 15.0   # PF sanity range (tightened from 20)
            or trades < _BF_MIN_TRADES
            or dd < -0.08            # stricter DD gate
            or wr > 0.80             # unrealistic WR
            or sharpe > 8.0):        # impossible Sharpe
            all_pass = False
            break

    n_windows = _BF_N_WINDOWS
    row: dict[str, Any] = dict(params)
    for wi_r in range(n_windows):
        if wi_r < len(window_metrics_list):
            m = window_metrics_list[wi_r]
            row[f"w{wi_r}_pf"] = m.get("pf_real", 0)
            row[f"w{wi_r}_trades"] = m.get("total_trades", 0)
            row[f"w{wi_r}_dd"] = m.get("max_drawdown", 0)
            row[f"w{wi_r}_wr"] = m.get("winrate_real", 0)
            row[f"w{wi_r}_sharpe"] = m.get("sharpe", 0)
        else:
            row[f"w{wi_r}_pf"] = 0
            row[f"w{wi_r}_trades"] = 0
            row[f"w{wi_r}_dd"] = 0
            row[f"w{wi_r}_wr"] = 0
            row[f"w{wi_r}_sharpe"] = 0

    if all_pass and len(window_metrics_list) == n_windows:
        pfs = [m.get("pf_real", 0) for m in window_metrics_list]
        pf_ratio = max(pfs) / min(pfs) if min(pfs) > 0 else 999
        if pf_ratio > 5.0:
            all_pass = False

    eg_entry = None
    if all_pass and len(window_metrics_list) == n_windows:
        pfs = [m.get("pf_real", 0) for m in window_metrics_list]
        row["min_pf"] = min(pfs)
        row["mean_pf"] = sum(pfs) / len(pfs)
        row["is_evergreen"] = True
        eg_entry = {
            "params": dict(params),
            "min_pf": min(pfs),
            "mean_pf": sum(pfs) / len(pfs),
            "window_metrics": {
                f"w{i}": {
                    "pf_real": m.get("pf_real", 0),
                    "winrate_real": m.get("winrate_real", 0),
                    "trades": m.get("total_trades", 0),
                    "dd": m.get("max_drawdown", 0),
                    "sharpe": m.get("sharpe", 0),
                }
                for i, m in enumerate(window_metrics_list)
            },
        }
    else:
        pf_vals = [m.get("pf_real", 0) for m in window_metrics_list] if window_metrics_list else [0]
        row["min_pf"] = min(pf_vals)
        row["mean_pf"] = sum(pf_vals) / len(pf_vals)
        row["is_evergreen"] = False

    return row, eg_entry


def _build_grid(asset_class: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build exhaustive parameter grid for a given asset class.

    V17: Optimized grid — removed parameters with zero PF impact:
    - leverage: fixed per class (doesn't affect PF, only position sizing)
    - max_hold_bars: fixed at SCALP_MAX_HOLD_BARS (Scalp-Day Hybrid: 4h on 5m)
    - timeout_rr_threshold: fixed at 0.5 (no PF difference)
    - alignment 0.92 removed (produces 0% evergreen across all classes)
    - be_ratchet_r floor raised to 1.5 (1.0R too easily triggered = inflated WR)
    """
    # Fixed leverage per class (conservative — tune in Paper Grid instead)
    FIXED_LEVERAGE = {"crypto": 5, "stocks": 1, "commodities": 5, "forex": 10}
    lev = FIXED_LEVERAGE.get(asset_class, 5)

    grid = []
    for align, rr, risk, be_r in itertools.product(
        [0.70, 0.75, 0.80, 0.85, 0.88, 0.90],  # 6 values (dropped 0.92)
        [2.0, 2.5, 3.0, 3.5],                    # 4 values
        [0.005, 0.010, 0.015],                    # 3 values
        [1.5, 2.0, 2.5],                          # 3 BE ratchet R (floor 1.5)
    ):
        grid.append({
            "alignment_threshold": align,
            "risk_reward": rr,
            "leverage": lev,
            "risk_per_trade": risk,
            "be_ratchet_r": be_r,
            "timeout_rr_threshold": 0.5,              # Fixed — zero PF impact
            "max_hold_bars": SCALP_MAX_HOLD_BARS,      # Scalp-Day Hybrid (4h on 5m)
        })
    return grid


def run_bruteforce(config_path: str = "config/default_config.yaml") -> dict[str, dict[str, Any]]:
    """
    Exhaustive grid search over ALL OOS windows — no Optuna, no train optimization.

    For each asset class:
      1. Precompute signals for each OOS window (reuse cache)
      2. Test EVERY param combo on ALL 3 OOS windows
      3. Evergreen filter: PF ≥ 1.5 + min trades + max DD on EVERY window
      4. Rank by min(PF) — best worst-case
      5. Monte Carlo check on top result
      6. Save results (JSON + full CSV)

    Anti-overfitting: no train/test bias, ranking by worst window, exhaustive search.
    """
    cfg = load_config(config_path)
    results_dir = Path(cfg["backtest"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_months = cfg["backtest"]["train_months"]
    test_months = cfg["backtest"]["test_months"]
    account_size = cfg["account"]["size"]

    start = pd.Timestamp(cfg["data"]["start_date"], tz="UTC")
    end = pd.Timestamp(datetime.now(timezone.utc))

    multi_assets = get_multi_asset_symbols(cfg)
    if not multi_assets:
        logger.error("No instruments found in any data directory")
        return {}

    windows = generate_wf_windows(start, end, train_months, test_months)
    if not windows:
        logger.error("No walk-forward windows generated")
        return {}
    logger.info("Walk-forward windows: %d", len(windows))

    # Min trades per window (asset-class specific)
    MIN_TRADES = {"crypto": 15, "stocks": 15, "forex": 5, "commodities": 10}

    evergreen_results: dict[str, dict[str, Any]] = {}

    for ac, symbols in multi_assets.items():
        logger.info(
            "═══ BRUTEFORCE: %s (%d instruments) ═══",
            ac.upper(), len(symbols),
        )

        ac_results_dir = results_dir / ac
        ac_results_dir.mkdir(parents=True, exist_ok=True)
        symbol_to_asset = {s: ac for s in symbols}

        # Load price data
        logger.info("Loading price data for %d %s instruments...", len(symbols), ac)
        load_price_data_for_symbols(symbols, cfg)

        # ── Precompute OOS signals for all windows ──────────────────
        oos_signals_per_window: list[list[TradeSignal]] = []
        for wi, window in enumerate(windows):
            logger.info(
                "%s Window %d: OOS %s → %s (train %s → %s for warmup)",
                ac.upper(), wi,
                window["test_start"].date(), window["test_end"].date(),
                window["train_start"].date(), window["train_end"].date(),
            )
            # Signal gen uses train period for indicator warmup, emits only OOS signals
            oos_sigs = _precompute_window_signals(
                cfg, symbols, window["test_start"], window["test_end"],
                symbol_to_asset=symbol_to_asset,
            )
            oos_signals_per_window.append(oos_sigs)
            logger.info("%s W%d: %d OOS signals", ac.upper(), wi, len(oos_sigs))

        # ── Build parameter grid ────────────────────────────────────
        grid = _build_grid(ac, cfg)
        logger.info("%s: Grid size = %d combinations", ac.upper(), len(grid))

        min_trades_per_window = MIN_TRADES.get(ac, 10)

        # ── Evaluate ALL combos on ALL windows (parallel) ───────────
        # Store shared state in module-level vars for forked workers (COW)
        global _BF_SIGNALS, _BF_CFG, _BF_SYM2ASSET, _BF_MIN_TRADES, _BF_N_WINDOWS
        _BF_SIGNALS = oos_signals_per_window
        _BF_CFG = cfg
        _BF_SYM2ASSET = symbol_to_asset
        _BF_MIN_TRADES = min_trades_per_window
        _BF_N_WINDOWS = len(windows)

        import multiprocessing as mp
        with mp.Pool(processes=4) as pool:
            results = list(tqdm(
                pool.imap_unordered(_eval_combo_worker, grid, chunksize=32),
                total=len(grid), desc=f"{ac} bruteforce",
            ))

        all_grid_results = [r[0] for r in results]
        eg_candidates = [r[1] for r in results if r[1] is not None]

        # ── Save full grid CSV ──────────────────────────────────────
        grid_df = pd.DataFrame(all_grid_results)
        grid_df.to_csv(ac_results_dir / "grid_search_results.csv", index=False)
        logger.info(
            "%s: %d/%d evergreen candidates found",
            ac.upper(), len(eg_candidates), len(grid),
        )

        # ── Rank + save ─────────────────────────────────────────────
        if eg_candidates:
            eg_candidates.sort(key=lambda r: r["min_pf"], reverse=True)
            best = eg_candidates[0]

            # Sanity warnings
            warnings: list[str] = []
            for wi_key, wm in best["window_metrics"].items():
                if wm["pf_real"] > 10:
                    warnings.append(f"{wi_key}: PF={wm['pf_real']:.1f} (>10)")
                if wm["winrate_real"] > 0.80:
                    warnings.append(f"{wi_key}: WR={wm['winrate_real']*100:.0f}% (>80%)")

            if warnings:
                logger.warning(
                    "%s SANITY WARNINGS: %s",
                    ac.upper(), "; ".join(warnings),
                )

            # Confidence based on min trades across all windows
            min_t = min(
                wm["trades"] for wm in best["window_metrics"].values()
            )
            confidence = "low" if min_t < 10 else ("medium" if min_t < 20 else "high")

            evergreen = {
                "asset_class": ac,
                "params": best["params"],
                "cross_window_metrics": best["window_metrics"],
                "min_pf": best["min_pf"],
                "mean_pf": best["mean_pf"],
                "is_evergreen": True,
                "confidence": confidence,
                "n_candidates_tested": len(grid),
                "n_evergreen_found": len(eg_candidates),
                "warnings": warnings,
            }

            eg_path = ac_results_dir / "evergreen_params.json"
            with open(eg_path, "w") as f:
                json.dump(evergreen, f, indent=2, default=str)

            logger.info(
                "✓ %s EVERGREEN: alignment=%.2f rr=%.1f lev=%d risk=%.3f | "
                "min_PF=%.2f (%d/%d candidates) confidence=%s",
                ac.upper(),
                best["params"]["alignment_threshold"],
                best["params"]["risk_reward"],
                best["params"]["leverage"],
                best["params"]["risk_per_trade"],
                best["min_pf"],
                len(eg_candidates), len(grid),
                confidence,
            )
            evergreen_results[ac] = evergreen
        else:
            logger.warning(
                "✗ %s: No evergreen params found! Using conservative defaults.",
                ac.upper(),
            )
            default_result = {
                "asset_class": ac,
                "params": dict(_DEFAULT_CONSERVATIVE_PARAMS),
                "cross_window_metrics": {},
                "min_pf": 0.0,
                "mean_pf": 0.0,
                "is_evergreen": False,
                "confidence": "none",
                "n_candidates_tested": len(grid),
                "n_evergreen_found": 0,
            }
            eg_path = ac_results_dir / "evergreen_params.json"
            with open(eg_path, "w") as f:
                json.dump(default_result, f, indent=2, default=str)
            evergreen_results[ac] = default_result

        # Cleanup — free memory before next asset class
        del oos_signals_per_window, all_grid_results
        _price_cache.clear()
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────
    summary = {}
    for ac, eg in evergreen_results.items():
        summary[ac] = {
            "is_evergreen": eg.get("is_evergreen", False),
            "params": eg["params"],
            "min_pf": eg.get("min_pf", 0),
            "confidence": eg.get("confidence", "none"),
        }
    with open(results_dir / "evergreen_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Brute-force grid search complete. Results in %s", results_dir)
    for ac, s in summary.items():
        status = "EVERGREEN" if s["is_evergreen"] else "DEFAULT"
        logger.info(
            "  %s: [%s] alignment=%.2f rr=%.1f lev=%d risk=%.3f min_pf=%.2f conf=%s",
            ac, status,
            s["params"]["alignment_threshold"],
            s["params"]["risk_reward"],
            s["params"]["leverage"],
            s["params"]["risk_per_trade"],
            s["min_pf"],
            s.get("confidence", "?"),
        )

    return evergreen_results


# ═══════════════════════════════════════════════════════════════════
#  Paper Grid Variant Generator
# ═══════════════════════════════════════════════════════════════════

# Asset-class leverage caps (must match live_multi_bot.py)
_ASSET_MAX_LEVERAGE = {
    "crypto": 20, "forex": 30, "stocks": 4, "commodities": 20,
}


def generate_paper_grid_variants(
    results_dir: str = "backtest/results",
    output_path: str = "paper_grid_results/variants.json",
) -> list[dict[str, Any]]:
    """
    Generate 20 parameter variants per asset class from evergreen params.

    Reads evergreen_params.json from each class's results directory,
    creates 20 variants spanning conservative to aggressive.
    """
    results_path = Path(results_dir)
    all_variants: list[dict[str, Any]] = []

    for ac in ("crypto", "forex", "stocks", "commodities"):
        eg_file = results_path / ac / "evergreen_params.json"
        if not eg_file.exists():
            logger.warning("No evergreen params for %s — skipping", ac)
            continue

        with open(eg_file) as f:
            eg = json.load(f)

        p = eg["params"]
        al = p["alignment_threshold"]
        rr = p["risk_reward"]
        lev = int(p["leverage"])
        risk = p["risk_per_trade"]
        max_lev = _ASSET_MAX_LEVERAGE.get(ac, 20)

        def _v(name, a, r, l, rk):
            """Create variant dict, clamping values to valid ranges."""
            return {
                "name": f"{ac}-{name}",
                "alignment_threshold": round(min(max(a, 0.60), 0.95), 2),
                "min_rr": round(max(r, 1.5), 1),
                "leverage": max(1, min(int(round(l)), max_lev)),
                "risk_per_trade": round(min(max(rk, 0.002), 0.025), 4),
                "asset_class": ac,
            }

        variants = [
            _v("Base",            al,        rr,       lev,           risk),
            _v("Conservative-1",  al + 0.02, rr + 0.5, lev * 0.6,    risk * 0.5),
            _v("Conservative-2",  al + 0.02, rr,       lev * 0.6,    risk * 0.7),
            _v("Risk-Low",        al,        rr,       lev,           risk * 0.5),
            _v("Risk-High",       al,        rr,       lev,           risk * 1.5),
            _v("Risk-Max",        al,        rr,       lev,           min(risk * 2.0, 0.02)),
            _v("Lev-Low",         al,        rr,       max(lev*0.5,1),risk),
            _v("Lev-High",        al,        rr,       lev * 1.5,     risk),
            _v("Lev-Max",         al,        rr,       max_lev,       risk),
            _v("RR-Relaxed",      al,        rr - 0.5, lev,           risk),
            _v("RR-Strict",       al,        rr + 0.5, lev,           risk),
            _v("Align-Relaxed",   al - 0.05, rr,       lev,           risk),
            _v("Align-Strict",    al + 0.02, rr,       lev,           risk),
            _v("Fallback-Base",   0.78,      2.0,      lev,           risk),
            _v("Aggressive",      al - 0.05, rr - 0.5, lev * 1.5,    risk * 1.5),
            _v("Defensive",       al + 0.02, rr + 0.5, lev * 0.5,    risk * 0.5),
            _v("Wild-Max",        0.78,      2.0,      max_lev,       0.02),
            _v("Wild-Min",        0.92,      3.5,      max(1, int(max_lev * 0.2)), 0.003),
            _v("Balanced",        al,        rr,       lev,           risk * 0.8),
            _v("Turbo",           al - 0.03, rr,       lev * 1.3,    risk * 1.3),
        ]
        all_variants.extend(variants)
        logger.info("Generated 20 variants for %s (base: al=%.2f rr=%.1f lev=%d risk=%.3f)",
                     ac, al, rr, lev, risk)

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_variants, f, indent=2)
    logger.info("Saved %d paper grid variants → %s", len(all_variants), out_path)

    return all_variants


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna Walk-Forward Backtester (Scalp-Day Hybrid, alignment-threshold gate)",
    )
    parser.add_argument(
        "--config", default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--per-class", action="store_true",
        help="Run per-asset-class optimization (separate Optuna per class + cross-window evergreen)",
    )
    parser.add_argument(
        "--generate-paper-grid", action="store_true",
        help="Generate 20 paper grid variants per asset class from evergreen params",
    )
    parser.add_argument(
        "--bruteforce", action="store_true",
        help="Exhaustive grid search on ALL OOS windows (no Optuna, anti-overfitting)",
    )
    args = parser.parse_args()

    if args.generate_paper_grid:
        cfg = load_config(args.config)
        generate_paper_grid_variants(
            results_dir=cfg["backtest"]["results_dir"],
        )
    elif args.bruteforce:
        run_bruteforce(config_path=args.config)
    elif args.per_class:
        run_per_asset_class(config_path=args.config)
    else:
        run(config_path=args.config)
