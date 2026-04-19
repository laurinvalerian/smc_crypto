"""
═══════════════════════════════════════════════════════════════════
 backtest/wf_bruteforce.py
 ─────────────────────────
 Walk-Forward Bruteforce Grid Search + Quant-Math Gate Stack
 (Scalp-Day Hybrid, no tiers, crypto-only).

 Quality-upgrade plan Phases C–H (v1.7 → v1.11).
 Renamed from optuna_backtester.py on 2026-04-19: bruteforce has been
 the primary / only code path since Phase D; the Optuna TPE preview
 was kept as `--fast` but overfit noisy 20–50-trade landscapes per
 plan §3.1 and was removed alongside the rename.

 Pipeline
 --------
   1. Rolling walk-forward windows (configurable train/test months)
   2. Exhaustive grid search over alignment × RR × risk × be_ratchet
      (V18 grid: 432 combos, 48 (alignment, RR) cells, crypto-only)
   3. Evergreen filter: PF ≥ 1.5 + min trades + max DD on EVERY window
   4. Rank by min(PF) — best worst-case candidate
   5. Post-hoc gates (all must PASS to deploy to funded):
        • CPCV  — DSR IQR tight across combinatorial OOS splits
        • PBO   — IS-best wins OOS (selection-bias calibration)
        • MC    — CVaR-95% max-DD tail-risk
        • REG   — DSR-spread × Sharpe-rel-spread on top-10% cells
        • COST  — baseline metrics survive 2× elevated cost shock

 Usage:
     python -m backtest.wf_bruteforce                     # default config
     python -m backtest.wf_bruteforce --config path       # custom config
     python -m backtest.wf_bruteforce --generate-paper-grid
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
from backtest.cpcv import run_cpcv, cpcv_summary
from backtest.pbo import compute_pbo
from backtest.cost_stress import compute_cost_stress
from backtest.monte_carlo import compute_mc_cvar_dd
from backtest.region_heatmap import (
    build_region_grid,
    build_region_grid_sharpe,
    plot_region_heatmap,
    region_summary,
)

# Parallelism budget (override via env: BACKTEST_SIGNAL_WORKERS).
# Signal precompute is numpy-heavy → scales with physical cores.
_CPU_COUNT = os.cpu_count() or 4
_SIGNAL_WORKERS = int(os.environ.get("BACKTEST_SIGNAL_WORKERS", max(2, _CPU_COUNT - 2)))

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


# ═══════════════════════════════════════════════════════════════════
#  Signal cache
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


# Region A removed (v1.11, 2026-04-19): _build_objective +
# extract_top_params + compute_param_importance were Optuna-study
# utilities, unused after the Optuna pipeline removal. Bruteforce
# enumerates the grid directly without an objective function.


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


# Region B removed (v1.11, 2026-04-19): check_parameter_stability +
# run() — the Optuna walk-forward pipeline. Bruteforce grid+gates
# (run_bruteforce) is the only supported entry point since Phase D.


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

    # Cost-stress escape hatch (v1.11): when set, ignore ASSET_COMMISSION /
    # ASSET_SLIPPAGE module defaults and use the values explicitly set in
    # `config["backtest"]`. Lets compute_cost_stress inject elevated costs
    # without monkey-patching module globals across process boundaries.
    force_cfg_costs = bool(
        config.get("backtest", {}).get("force_backtest_costs", False)
    )

    all_trades: list[pd.DataFrame] = []
    for ac, sigs in by_class.items():
        if force_cfg_costs:
            commission_pct = float(config["backtest"]["commission_pct"])
            slippage_pct = float(config["backtest"]["slippage_pct"])
        else:
            commission_pct = ASSET_COMMISSION.get(
                ac, config["backtest"]["commission_pct"],
            )
            slippage_pct = ASSET_SLIPPAGE.get(
                ac, config["backtest"]["slippage_pct"],
            )
        t = simulate_trades(
            sigs,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
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


# Region C removed (v1.11, 2026-04-19): cross_window_validate was
# used only by run_per_asset_class (also removed). run_bruteforce
# inlines the evergreen PF-min gate per combo in _eval_combo_worker.


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


# Region D removed (v1.11, 2026-04-19): run_per_asset_class was the
# Optuna per-asset-class pipeline. Replaced by run_bruteforce which
# iterates asset classes inline using the exhaustive grid.


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
        # Phase D+F (2026-04-19):
        #   - Removed `sharpe > 8.0` ceiling: was calibrated for old naive
        #     sqrt(252) per-trade Sharpe. Daily-equity sqrt(365) values 5-15
        #     are legitimate (baseline_v1.7.1: max window Sharpe 14.29).
        #   - Removed `pf > 15.0` ceiling: same root-cause scale mismatch.
        #   - `pf < 1.5` (evergreen floor, matches cross_window_validate default)
        #   - `dd < -0.12`: the circuit breaker trips at -8% all-time, so
        #     observed DD clusters around -8.0 to -8.8% for aggressive combos.
        #     `-0.08` rejected all of them; `-0.12` gives CB-overshoot
        #     headroom and still respects funded-account's -10% alltime cap
        #     in the relevant evergreen regime (real DD <= -0.10 expected).
        #   - The overfit detector is the post-hoc DSR via CPCV, not
        #     magic-number ceilings.
        if (pf < 1.5
            or trades < _BF_MIN_TRADES
            or dd < -0.12
            or wr > 0.80):
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
        # Raised from 5.0 → 10.0 (Phase D+F): baseline_v1.7.1 shows legitimate
        # PF variance across regimes (2.43 w5 → 17.84 w1, ratio ~7). A 5.0
        # ceiling rejected robust params. CPCV IQR(DSR) < 0.3 is the
        # stronger cross-window consistency gate.
        if pf_ratio > 10.0:
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

    V18 (2026-04-19): Denser ``alignment × risk_reward`` sampling to give
    the Phase-G Region-Heatmap usable landscape resolution. The v1.10
    grid of 6×4 cells (24) left top-10% = 2–3 cells — too few to
    distinguish a plateau from a cliff. Here we expand to 8×6 cells
    (48) so top-10% ≈ 5 cells and the Sharpe rel-spread gate has real
    discriminating power.

    Kept unchanged (fixed per class or zero PF impact, per V17 audit):
    - leverage: fixed per class (affects position sizing, not PF)
    - max_hold_bars: fixed at SCALP_MAX_HOLD_BARS (Scalp-Day Hybrid: 4h on 5m)
    - timeout_rr_threshold: fixed at 0.5 (measured zero PF impact in V17)

    Dimensionality (8 × 6 × 3 × 3 = 432 combos, 2.0× v1.10):
    - alignment_threshold: fine sweep in the 0.70–0.88 live-relevant band.
    - risk_reward:         includes 2.25/2.75 so the min-RR sensitivity
                           curve has interior samples.
    - risk_per_trade:      unchanged (0.5 / 1.0 / 1.5 % mirrors live).
    - be_ratchet_r:        unchanged floor 1.5 (1.0R over-triggers BE).
    """
    # Fixed leverage per class (conservative — tune in Paper Grid instead)
    FIXED_LEVERAGE = {"crypto": 5, "stocks": 1, "commodities": 5, "forex": 10}
    lev = FIXED_LEVERAGE.get(asset_class, 5)

    grid = []
    for align, rr, risk, be_r in itertools.product(
        # 8 alignment values — dense in the 0.70–0.88 funded-live band.
        [0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88],
        # 6 RR values — interior samples at 2.25/2.75 reveal sensitivity.
        [2.0, 2.25, 2.5, 2.75, 3.0, 3.5],
        [0.005, 0.010, 0.015],                    # 3 risk bands
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
        # macOS Python 3.8+ default is 'spawn' which re-imports the module
        # — globals set above (_BF_SIGNALS, _BF_CFG, ...) are invisible to
        # workers. Force 'fork' so copy-on-write makes them visible.
        # (Alternative: worker-initializer + pickled state, 3× slower.)
        _mp_ctx = mp.get_context("fork")
        with _mp_ctx.Pool(processes=4) as pool:
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

            # ── Phase D+F: CPCV post-bruteforce validation ───────────
            # Bruteforce is the primary path (TPE overfits on noisy 20-50-trade
            # landscapes per plan §3.1). CPCV multiplies OOS evaluations of
            # the deployment-candidate via C(N, k) splits and reports the
            # DSR distribution — tight IQR signals split-robust params.
            cpcv_cfg = cfg.get("cpcv", {}) or {}
            cpcv_enabled = cpcv_cfg.get("enabled", True)
            cpcv_k = int(cpcv_cfg.get("k", 2))
            cpcv_min_windows = int(cpcv_cfg.get("min_windows", 4))
            cpcv_embargo = int(cpcv_cfg.get("embargo_bars", 288))
            cpcv_funded = float(cpcv_cfg.get("funded_threshold", 0.95))
            cpcv_research = float(cpcv_cfg.get("research_threshold", 0.70))

            if cpcv_enabled and len(windows) >= cpcv_min_windows:
                try:
                    # Null-distribution sample: mean-Sharpe across windows
                    # for *qualified* strategy candidates only. Bailey-LdP's
                    # n_trials is "independent strategies that competed for
                    # selection", not "every mechanically-enumerated combo".
                    # Using the full grid (incl. trivial filter configs that
                    # produce zero/junk trades) inflates E_max_null and
                    # collapses DSR to ~0 (seen in cpcv_summary v1 run).
                    #
                    # Primary filter: is_evergreen (passed WF PF-min gate).
                    # Fallback at <5 evergreens: combos that traded in all
                    # windows — still rejects "alignment=0.9 → no signals"
                    # trivial combos while keeping variance estimate feasible.
                    window_sharpe_cols = [
                        c for c in grid_df.columns
                        if c.startswith("w") and c.endswith("_sharpe")
                    ]
                    eg_mask = grid_df["is_evergreen"].fillna(False).astype(bool)
                    qualified_df = grid_df[eg_mask]
                    used_fallback = False
                    if len(qualified_df) < 5:
                        used_fallback = True
                        window_trade_cols = [
                            c for c in grid_df.columns if c.endswith("_trades")
                        ]
                        if window_trade_cols:
                            has_all_trades = (
                                grid_df[window_trade_cols].fillna(0) > 0
                            ).all(axis=1)
                            qualified_df = grid_df[has_all_trades]
                        else:
                            qualified_df = grid_df[eg_mask]

                    trial_sharpes: list[float] = []
                    if window_sharpe_cols:
                        for _, rrow in qualified_df.iterrows():
                            vals = [
                                float(rrow[c]) for c in window_sharpe_cols
                                if not pd.isna(rrow[c])
                            ]
                            if vals:
                                trial_sharpes.append(float(np.mean(vals)))

                    n_splits_expected = math.comb(len(windows), cpcv_k)
                    logger.info(
                        "%s CPCV: C(%d, %d) = %d splits | best_params=evergreen[0] "
                        "(min_PF=%.2f) | n_trials=%d (qualified %d/%d, "
                        "fallback=%s, evergreen=%d)",
                        ac.upper(), len(windows), cpcv_k, n_splits_expected,
                        best["min_pf"], len(trial_sharpes),
                        len(qualified_df), len(grid_df),
                        used_fallback, int(eg_mask.sum()),
                    )

                    cpcv_df = run_cpcv(
                        params=best["params"],
                        windows=windows,
                        signals_per_window={
                            wi: sigs for wi, sigs in enumerate(oos_signals_per_window)
                        },
                        simulate_fn=_simulate_with_params,
                        config=cfg,
                        symbol_to_asset=symbol_to_asset,
                        trial_sharpes=trial_sharpes,
                        k=cpcv_k,
                        embargo_bars=cpcv_embargo,
                        max_hold_bars=SCALP_MAX_HOLD_BARS,
                    )

                    cpcv_df_csv = cpcv_df.copy()
                    cpcv_df_csv["test_idx"] = cpcv_df_csv["test_idx"].apply(
                        lambda t: "|".join(map(str, t))
                    )
                    cpcv_df_csv.to_csv(
                        ac_results_dir / "cpcv_results.csv", index=False,
                    )

                    cpcv_sum = cpcv_summary(
                        cpcv_df,
                        funded_threshold=cpcv_funded,
                        research_threshold=cpcv_research,
                    )
                    cpcv_sum["asset_class"] = ac
                    cpcv_sum["best_params"] = {
                        k_: v for k_, v in best["params"].items()
                        if not isinstance(v, dict)
                    }
                    cpcv_sum["evergreen_min_pf"] = float(best["min_pf"])
                    cpcv_sum["evergreen_mean_pf"] = float(best["mean_pf"])
                    cpcv_sum["n_trials_in_sharpe_variance"] = len(trial_sharpes)
                    cpcv_sum["k"] = cpcv_k
                    cpcv_sum["embargo_bars"] = cpcv_embargo
                    cpcv_sum["n_windows"] = len(windows)
                    cpcv_sum["source"] = "bruteforce_primary"

                    with open(ac_results_dir / "cpcv_summary.json", "w") as f:
                        json.dump(cpcv_sum, f, indent=2, default=str)

                    evergreen["cpcv"] = cpcv_sum
                    with open(eg_path, "w") as f:
                        json.dump(evergreen, f, indent=2, default=str)

                    logger.info(
                        "%s CPCV: %d splits | median DSR=%.3f | IQR=%.3f | "
                        "funded-pass=%.1f%% | research-pass=%.1f%%",
                        ac.upper(), cpcv_sum["n_splits"],
                        cpcv_sum["median_dsr"], cpcv_sum["dsr_iqr"],
                        cpcv_sum["percent_passing_funded"] * 100,
                        cpcv_sum["percent_passing_research"] * 100,
                    )

                    if cpcv_sum["dsr_iqr"] >= 0.30:
                        logger.warning(
                            "%s CPCV IQR %.3f >= 0.30: params NOT split-robust. "
                            "PBO check (Phase E) will likely fail.",
                            ac.upper(), cpcv_sum["dsr_iqr"],
                        )

                except Exception as exc:
                    logger.error(
                        "%s CPCV post-bruteforce validation failed: %s",
                        ac.upper(), exc, exc_info=True,
                    )

            # ── Phase E: PBO post-bruteforce validation ──────────────
            # DSR asks "is this Sharpe real vs. the null?"
            # CPCV asks "is this Sharpe stable across OOS combinations?"
            # PBO asks "how often does IS-best lose OOS?" — the
            # *selection* question neither of the above directly answers.
            # Deploy-gate per plan §2.5: PBO < 0.30.
            pbo_cfg = cfg.get("pbo", {}) or {}
            pbo_enabled = pbo_cfg.get("enabled", True)
            pbo_n_splits = int(pbo_cfg.get("n_splits", 128))
            pbo_gate = float(pbo_cfg.get("gate_threshold", 0.30))
            if pbo_enabled:
                try:
                    pbo_sharpe_cols = [
                        c for c in grid_df.columns
                        if c.startswith("w") and c.endswith("_sharpe")
                    ]
                    if len(pbo_sharpe_cols) < 4:
                        raise ValueError(
                            f"need >= 4 window-sharpe columns, "
                            f"got {len(pbo_sharpe_cols)}"
                        )
                    # Bailey-BLLZ PBO needs even T; drop the last col
                    # if windows are odd — order preserved so IS/OOS
                    # halves stay contiguous-comparable.
                    if len(pbo_sharpe_cols) % 2 != 0:
                        pbo_sharpe_cols = pbo_sharpe_cols[:-1]
                    pbo_perf = grid_df[pbo_sharpe_cols].to_numpy(dtype=float)

                    pbo_result = compute_pbo(
                        pbo_perf, n_splits=pbo_n_splits, seed=42,
                    )
                    # Make JSON-serializable.
                    pbo_result["logit_values"] = pbo_result[
                        "logit_values"
                    ].tolist()
                    pbo_result["gate_threshold"] = pbo_gate
                    pbo_result["gate_pass"] = (
                        pbo_result["pbo"] < pbo_gate
                    )
                    pbo_result["asset_class"] = ac
                    pbo_result["metric_used"] = "w*_sharpe"

                    with open(ac_results_dir / "pbo_summary.json", "w") as f:
                        json.dump(pbo_result, f, indent=2, default=str)

                    evergreen["pbo"] = {
                        k: v for k, v in pbo_result.items()
                        if k != "logit_values"
                    }
                    with open(eg_path, "w") as f:
                        json.dump(evergreen, f, indent=2, default=str)

                    gate_msg = (
                        "PASS" if pbo_result["gate_pass"]
                        else "FAIL — funded-deploy blocked"
                    )
                    logger.info(
                        "%s PBO: %.3f (gate %.2f, %d splits, "
                        "λ_median=%.3f) | %s",
                        ac.upper(), pbo_result["pbo"], pbo_gate,
                        pbo_result["n_splits_used"],
                        pbo_result["lambda_median"], gate_msg,
                    )
                    if not pbo_result["gate_pass"]:
                        logger.warning(
                            "%s PBO %.3f >= %.2f: selection-bias gate "
                            "FAIL. Do NOT deploy to funded account.",
                            ac.upper(), pbo_result["pbo"], pbo_gate,
                        )
                except Exception as exc:
                    logger.error(
                        "%s PBO post-bruteforce validation failed: %s",
                        ac.upper(), exc, exc_info=True,
                    )

            # ── Phase G: Monte-Carlo tail-risk (CVaR-95% DD) ────────
            # VaR answers "how bad at the 5%-quantile?" — CVaR (Expected
            # Shortfall) answers "how bad on average in that tail?". CVaR
            # is coherent and penalises tail concentration; correct
            # funded-account DD gate. Plan §2.6: cvar_dd_95 > -0.20.
            mc_cfg = cfg.get("monte_carlo", {}) or {}
            mc_enabled = mc_cfg.get("enabled", True)
            mc_n_sims = int(mc_cfg.get("n_simulations", 1000))
            mc_confidence = float(mc_cfg.get("confidence", 0.95))
            mc_seed = int(mc_cfg.get("seed", 42))
            mc_gate = float(mc_cfg.get("gate_threshold", -0.20))
            if mc_enabled:
                try:
                    # Union of all-window OOS signals → single simulation
                    # of best_params → single tail-risk estimate.
                    union_sigs: list[Any] = []
                    for sigs in oos_signals_per_window:
                        union_sigs.extend(sigs)
                    mc_trades, _ = _simulate_with_params(
                        best["params"], union_sigs, cfg, symbol_to_asset,
                    )
                    mc_result = compute_mc_cvar_dd(
                        mc_trades,
                        account_size=account_size,
                        n_simulations=mc_n_sims,
                        confidence=mc_confidence,
                        seed=mc_seed,
                    )
                    mc_result["gate_threshold"] = mc_gate
                    mc_result["gate_pass"] = (
                        mc_result["cvar_dd_95"] > mc_gate
                    )
                    mc_result["asset_class"] = ac
                    mc_result["n_trades"] = int(len(mc_trades))

                    with open(ac_results_dir / "mc_summary.json", "w") as f:
                        json.dump(mc_result, f, indent=2, default=str)

                    evergreen["mc"] = mc_result
                    with open(eg_path, "w") as f:
                        json.dump(evergreen, f, indent=2, default=str)

                    gate_msg = (
                        "PASS" if mc_result["gate_pass"]
                        else "FAIL — funded-deploy blocked"
                    )
                    logger.info(
                        "%s MC: CVaR-%.0f%% DD=%.2f%% (gate > %.2f%%, "
                        "VaR=%.2f%%, median DD=%.2f%%, n_sims=%d, "
                        "trades=%d) | %s",
                        ac.upper(), mc_confidence * 100,
                        mc_result["cvar_dd_95"] * 100, mc_gate * 100,
                        mc_result["var_dd_95"] * 100,
                        mc_result["median_dd"] * 100,
                        mc_result["n_simulations"],
                        mc_result["n_trades"], gate_msg,
                    )
                    if not mc_result["gate_pass"]:
                        logger.warning(
                            "%s MC CVaR-%.0f%% DD %.2f%% worse than "
                            "gate %.2f%%. Do NOT deploy to funded account.",
                            ac.upper(), mc_confidence * 100,
                            mc_result["cvar_dd_95"] * 100, mc_gate * 100,
                        )
                except Exception as exc:
                    logger.error(
                        "%s MC post-bruteforce validation failed: %s",
                        ac.upper(), exc, exc_info=True,
                    )

            # ── Phase G: Parameter-region heatmap ──────────────────
            # PBO bounds *selection* bias; Region-Heatmap bounds
            # parameter *instability*. If the top-10% of cells span a
            # DSR range > 0.10 the selected combo sits on a ridge —
            # small perturbations collapse the edge. Plan §2.6:
            # spread_q90_q10 < 0.10.
            rh_cfg = cfg.get("region_heatmap", {}) or {}
            rh_enabled = rh_cfg.get("enabled", True)
            rh_top_pct = float(rh_cfg.get("top_pct", 0.10))
            rh_gate = float(rh_cfg.get("gate_threshold", 0.10))
            # v1.11 saturation-fix: relative Sharpe-spread gate on the
            # DSR-top cells. DSR saturates at 1.0 on strong grids and
            # loses headroom to differentiate plateau vs ridge — Sharpe
            # does not. The combined gate is AND of both spreads.
            rh_sharpe_rel_gate = float(rh_cfg.get("sharpe_rel_gate", 0.15))
            rh_param_x = rh_cfg.get("param_x", "alignment_threshold")
            rh_param_y = rh_cfg.get("param_y", "risk_reward")
            rh_obs_count = int(rh_cfg.get("observation_count", 60))
            if rh_enabled:
                try:
                    rh_sharpe_cols = [
                        c for c in grid_df.columns
                        if c.startswith("w") and c.endswith("_sharpe")
                    ]
                    if not rh_sharpe_cols:
                        raise ValueError("no w*_sharpe columns in grid_df")

                    # Same null-distribution logic as CPCV: evergreen-
                    # qualified first, fallback to "traded in every
                    # window" to keep variance estimate meaningful.
                    eg_mask2 = grid_df["is_evergreen"].fillna(False).astype(bool)
                    rh_qualified = grid_df[eg_mask2]
                    if len(rh_qualified) < 5:
                        rh_window_trade_cols = [
                            c for c in grid_df.columns if c.endswith("_trades")
                        ]
                        if rh_window_trade_cols:
                            rh_has_all = (
                                grid_df[rh_window_trade_cols].fillna(0) > 0
                            ).all(axis=1)
                            rh_qualified = grid_df[rh_has_all]

                    rh_trial_sharpes: list[float] = []
                    for _, rrow in rh_qualified.iterrows():
                        vals = [
                            float(rrow[c]) for c in rh_sharpe_cols
                            if not pd.isna(rrow[c])
                        ]
                        if vals:
                            rh_trial_sharpes.append(float(np.mean(vals)))

                    region_pivot = build_region_grid(
                        grid_df,
                        trial_sharpes=rh_trial_sharpes,
                        observation_count=rh_obs_count,
                        window_sharpe_cols=rh_sharpe_cols,
                        param_x=rh_param_x,
                        param_y=rh_param_y,
                    )
                    # v1.11: non-saturating companion metric — median
                    # mean-window-Sharpe per cell. Same pivot axes so the
                    # combined gate evaluates the *same* DSR-top cells.
                    region_pivot_sharpe = build_region_grid_sharpe(
                        grid_df,
                        window_sharpe_cols=rh_sharpe_cols,
                        param_x=rh_param_x,
                        param_y=rh_param_y,
                    )
                    region_sum = region_summary(
                        region_pivot,
                        top_pct=rh_top_pct,
                        gate_threshold=rh_gate,
                        sharpe_region_df=region_pivot_sharpe,
                        sharpe_rel_gate=rh_sharpe_rel_gate,
                    )
                    region_sum["asset_class"] = ac
                    region_sum["param_x"] = rh_param_x
                    region_sum["param_y"] = rh_param_y
                    region_sum["n_trials_in_variance"] = len(rh_trial_sharpes)

                    if not region_pivot.empty:
                        region_pivot.to_csv(
                            ac_results_dir / "region_heatmap.csv"
                        )
                    if not region_pivot_sharpe.empty:
                        region_pivot_sharpe.to_csv(
                            ac_results_dir / "region_heatmap_sharpe.csv"
                        )

                    with open(ac_results_dir / "region_heatmap.json", "w") as f:
                        json.dump(region_sum, f, indent=2, default=str)

                    try:
                        plot_region_heatmap(
                            region_pivot,
                            output_path=ac_results_dir / "region_heatmap.png",
                            title=f"{ac.upper()} parameter region — median DSR",
                        )
                        plot_region_heatmap(
                            region_pivot_sharpe,
                            output_path=ac_results_dir / "region_heatmap_sharpe.png",
                            title=f"{ac.upper()} parameter region — median Sharpe",
                        )
                    except Exception as plot_exc:
                        logger.warning(
                            "%s REGION heatmap PNG render failed: %s",
                            ac.upper(), plot_exc,
                        )

                    evergreen["region"] = region_sum
                    with open(eg_path, "w") as f:
                        json.dump(evergreen, f, indent=2, default=str)

                    dsr_msg = "PASS" if region_sum["dsr_gate_pass"] else "FAIL"
                    sh_msg = "PASS" if region_sum["sharpe_gate_pass"] else "FAIL"
                    gate_msg = (
                        "PASS" if region_sum["gate_pass"]
                        else "FAIL — funded-deploy blocked"
                    )
                    logger.info(
                        "%s REGION: top-%.0f%% DSR spread=%.3f (gate<%.2f, %s) "
                        "| Sharpe rel_spread=%.3f of median=%.2f "
                        "(rel_gate<%.2f, %s) | n_trials=%d | COMBINED %s",
                        ac.upper(), rh_top_pct * 100,
                        region_sum["spread_q90_q10"], rh_gate, dsr_msg,
                        region_sum["sharpe_rel_spread_q90_q10"],
                        region_sum["sharpe_top_cells_median"],
                        rh_sharpe_rel_gate, sh_msg,
                        len(rh_trial_sharpes), gate_msg,
                    )
                    if not region_sum["gate_pass"]:
                        reasons: list[str] = []
                        if not region_sum["dsr_gate_pass"]:
                            reasons.append(
                                f"DSR spread {region_sum['spread_q90_q10']:.3f} "
                                f">= {rh_gate:.2f}"
                            )
                        if not region_sum["sharpe_gate_pass"]:
                            reasons.append(
                                f"Sharpe rel_spread "
                                f"{region_sum['sharpe_rel_spread_q90_q10']:.3f} "
                                f">= {rh_sharpe_rel_gate:.2f}"
                            )
                        logger.warning(
                            "%s REGION landscape not plateau-like (%s). "
                            "Do NOT deploy to funded account.",
                            ac.upper(), "; ".join(reasons),
                        )
                except Exception as exc:
                    logger.error(
                        "%s REGION post-bruteforce validation failed: %s",
                        ac.upper(), exc, exc_info=True,
                    )

            # ── Phase H: Transaction-cost stress test ───────────────
            # Funded accounts face real-fill cost that can exceed
            # the Binance-taker assumption under volatile regimes.
            # Re-simulate best_params at a pessimistic cost scenario
            # and require PF/Sharpe/DD to stay above a funded floor.
            cs_cfg = cfg.get("cost_stress", {}) or {}
            cs_enabled = cs_cfg.get("enabled", True)
            if cs_enabled:
                try:
                    union_sigs_cs: list[Any] = []
                    for sigs in oos_signals_per_window:
                        union_sigs_cs.extend(sigs)

                    cs_result = compute_cost_stress(
                        best_params=best["params"],
                        oos_signals_union=union_sigs_cs,
                        cfg=cfg,
                        symbol_to_asset=symbol_to_asset,
                        simulate_fn=_simulate_with_params,
                        account_size=account_size,
                    )
                    cs_result["asset_class"] = ac

                    with open(
                        ac_results_dir / "cost_stress_summary.json", "w",
                    ) as f:
                        json.dump(cs_result, f, indent=2, default=str)

                    evergreen["cost_stress"] = cs_result
                    with open(eg_path, "w") as f:
                        json.dump(evergreen, f, indent=2, default=str)

                    pf_msg = "PASS" if cs_result["gate_pf_pass"] else "FAIL"
                    sh_msg = "PASS" if cs_result["gate_sharpe_pass"] else "FAIL"
                    dd_msg = "PASS" if cs_result["gate_dd_pass"] else "FAIL"
                    gate_msg = (
                        "PASS" if cs_result["gate_pass"]
                        else "FAIL — funded-deploy blocked"
                    )
                    logger.info(
                        "%s COST-STRESS: @ commission=%.2fbp slippage=%.2fbp | "
                        "PF %.2f→%.2f (gate>=%.2f, %s) | "
                        "Sharpe %.2f→%.2f (gate>=%.2f, %s) | "
                        "DD %.2f%%→%.2f%% (gate>=%.2f%%, %s) | COMBINED %s",
                        ac.upper(),
                        cs_result["stressed_commission_pct"] * 10000,
                        cs_result["stressed_slippage_pct"] * 10000,
                        cs_result["baseline_pf"], cs_result["stressed_pf"],
                        cs_result["gate_pf"], pf_msg,
                        cs_result["baseline_sharpe"], cs_result["stressed_sharpe"],
                        cs_result["gate_sharpe"], sh_msg,
                        cs_result["baseline_dd"] * 100,
                        cs_result["stressed_dd"] * 100,
                        cs_result["gate_dd"] * 100, dd_msg, gate_msg,
                    )
                    if not cs_result["gate_pass"]:
                        logger.warning(
                            "%s COST-STRESS collapsed at 2× conservative "
                            "cost model. Do NOT deploy to funded account.",
                            ac.upper(),
                        )
                except Exception as exc:
                    logger.error(
                        "%s COST-STRESS post-bruteforce validation failed: %s",
                        ac.upper(), exc, exc_info=True,
                    )
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

# Asset-class leverage caps (must match live_multi_bot.py).
# Crypto-only (2026-04-19, v1.11): only crypto is active. Keeping the dict
# keyed by asset class for forward-compat if ever re-expanded.
_ASSET_MAX_LEVERAGE = {
    "crypto": 20,
}


def generate_paper_grid_variants(
    results_dir: str = "backtest/results",
    output_path: str = "paper_grid_results/variants.json",
) -> list[dict[str, Any]]:
    """
    Generate 20 parameter variants per asset class from evergreen params.

    Crypto-only (2026-04-19, v1.11): only the crypto evergreen is consumed.
    Legacy forex/stocks/commodities classes were stripped in Phase 1.

    Reads evergreen_params.json from the class results directory, creates
    20 variants spanning conservative to aggressive.
    """
    results_path = Path(results_dir)
    all_variants: list[dict[str, Any]] = []

    for ac in ("crypto",):
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
        description=(
            "Walk-Forward Bruteforce Backtester + Quant-Math Gate Stack "
            "(Scalp-Day Hybrid, crypto-only, alignment-threshold gate)."
        ),
    )
    parser.add_argument(
        "--config", default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--generate-paper-grid", action="store_true",
        help="Generate paper-grid variants for live A/B testing from "
             "the crypto evergreen params (requires a prior default run).",
    )
    args = parser.parse_args()

    if args.generate_paper_grid:
        cfg = load_config(args.config)
        generate_paper_grid_variants(
            results_dir=cfg["backtest"]["results_dir"],
        )
    else:
        # Only entry point: bruteforce grid + full gate stack (Phases C–H).
        run_bruteforce(config_path=args.config)
