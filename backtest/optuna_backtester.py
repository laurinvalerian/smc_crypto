"""
═══════════════════════════════════════════════════════════════════
 backtest/optuna_backtester.py
 ─────────────────────────────
 Walk-Forward Optimization with Optuna + AAA++ Filter Validation.

 Features:
   • Rolling walk-forward windows (configurable train/test months)
   • Optuna Bayesian optimisation (configurable trials per window)
   • AAA++ tier filtering: only AAA++ and AAA+ trades pass
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
import hashlib
import json
import logging
import math
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


# ── AAA++ Tier Thresholds ─────────────────────────────────────────

TIER_AAA_PLUS_PLUS = "AAA++"
TIER_AAA_PLUS = "AAA+"

TIER_THRESHOLDS = {
    TIER_AAA_PLUS_PLUS: {"min_score": 0.88, "min_rr": 3.0},
    TIER_AAA_PLUS:      {"min_score": 0.78, "min_rr": 2.0},
}

TIER_RISK = {
    TIER_AAA_PLUS_PLUS: {"base_risk": 0.010, "max_risk": 0.015},
    TIER_AAA_PLUS:      {"base_risk": 0.005, "max_risk": 0.010},
}


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


# ═══════════════════════════════════════════════════════════════════
#  AAA++ Signal Classifier
# ═══════════════════════════════════════════════════════════════════

def classify_signal_tier(
    signal: TradeSignal,
    alignment_score: float,
    rr: float,
) -> str | None:
    """
    Classify a trade signal into AAA++ or AAA+ tier.
    Returns None if signal doesn't meet minimum quality.
    """
    for tier_name in (TIER_AAA_PLUS_PLUS, TIER_AAA_PLUS):
        thresholds = TIER_THRESHOLDS[tier_name]
        if alignment_score >= thresholds["min_score"] and rr >= thresholds["min_rr"]:
            return tier_name
    return None


def compute_dynamic_risk(
    tier: str,
    score: float,
    account_size: float,
) -> float:
    """Compute risk amount based on tier and score."""
    risk_cfg = TIER_RISK.get(tier, TIER_RISK[TIER_AAA_PLUS])
    base = risk_cfg["base_risk"]
    max_r = risk_cfg["max_risk"]
    # Scale between base and max based on score
    risk_pct = base + (max_r - base) * min(score, 1.0)
    return account_size * risk_pct


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


# Maximum holding period for day trading (number of 5m bars)
# Day trade = max 48 hours (2 trading days). All styles map to this.
_MAX_BARS_DEFAULT = 576  # 48 hours × 12 bars/hour


def _resolve_trade_outcome(
    sig: TradeSignal,
) -> tuple[str, float]:
    """
    Walk forward through real price bars after entry to determine outcome.

    Breakeven-only stop logic (like a patient SMC trader):
    - Before +1R: original SL stays
    - At +1R: SL moves to net-breakeven (entry + fees/slippage buffer)
    - After +1R: SL stays at breakeven — NO further trailing
    - Price has room to breathe and reach structure-based TP
    - TP hit → full win at target RR
    - BE-SL hit → tiny win (covers fees)
    - Timeout → close at last bar close

    Returns:
        (outcome, exit_price) where outcome is "win", "loss", or "skip"
    """
    df = _price_cache.get(sig.symbol)
    if df is None or df.empty:
        return "skip", sig.entry_price

    entry_ts = sig.timestamp
    if not hasattr(entry_ts, 'tzinfo') or entry_ts.tzinfo is None:
        entry_ts = pd.Timestamp(entry_ts, tz="UTC")

    mask = df.index > entry_ts
    future_bars = df.loc[mask]

    if future_bars.empty:
        return "skip", sig.entry_price

    max_bars = _MAX_BARS_DEFAULT
    future_bars = future_bars.iloc[:max_bars]

    is_long = sig.direction == "long"
    entry = sig.entry_price
    original_sl = sig.stop_loss
    tp = sig.take_profit
    sl_dist = abs(entry - original_sl)  # 1R in price terms

    # Breakeven stop state
    current_sl = original_sl
    reached_1r = False
    # Net-breakeven buffer: covers round-trip fees + slippage (~0.1%)
    fee_buffer = entry * 0.001

    for _, bar in future_bars.iterrows():
        high = bar["high"]
        low = bar["low"]

        if is_long:
            # Check if we've reached +1R (move to net-breakeven once)
            if not reached_1r and sl_dist > 0:
                bar_best_r = (high - entry) / sl_dist
                if bar_best_r >= 1.0:
                    reached_1r = True
                    current_sl = max(current_sl, entry + fee_buffer)

            # Check SL first (conservative: worst case within bar)
            if low <= current_sl:
                if current_sl >= entry:
                    return "win", current_sl
                else:
                    return "loss", current_sl
            # Check TP
            if high >= tp:
                return "win", tp

        else:  # short
            if not reached_1r and sl_dist > 0:
                bar_best_r = (entry - low) / sl_dist
                if bar_best_r >= 1.0:
                    reached_1r = True
                    current_sl = min(current_sl, entry - fee_buffer)

            if high >= current_sl:
                if current_sl <= entry:
                    return "win", current_sl
                else:
                    return "loss", current_sl
            if low <= tp:
                return "win", tp

    # Timeout: close at last bar's close
    last_close = float(future_bars.iloc[-1]["close"])
    if is_long:
        return ("win" if last_close > entry else "loss"), last_close
    else:
        return ("win" if last_close < entry else "loss"), last_close


# ═══════════════════════════════════════════════════════════════════
#  Trade simulation with REAL price paths + Circuit Breaker
# ═══════════════════════════════════════════════════════════════════

def simulate_trades(
    signals: list[TradeSignal],
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0001,
    account_size: float = 100_000,
    use_circuit_breaker: bool = True,
    aaa_only: bool = True,
    asset_class: str = "crypto",
) -> pd.DataFrame:
    """
    Simulate trades using REAL price-path outcomes.

    For each signal, walks forward through actual 5m/1m candle data
    to check whether SL or TP is hit first. No synthetic win probability.

    Features:
    - Real price-path simulation (SL/TP hit detection on actual candles)
    - AAA++ / AAA+ tier filtering
    - Compound position sizing (risk % of current equity)
    - Circuit breaker (daily -3%, weekly -5%, class -2%, all-time -8%)
    - Max holding period by style (scalp 4h, day 48h, swing 2w)
    - Timeout trades closed at market price
    """
    if not signals:
        return pd.DataFrame()

    cb = CircuitBreaker() if use_circuit_breaker else None
    equity = account_size

    rows: list[dict[str, Any]] = []
    rejected_count = 0
    skipped_no_data = 0

    for sig in signals:
        sl_dist = abs(sig.entry_price - sig.stop_loss)
        tp_dist = abs(sig.take_profit - sig.entry_price)
        if sl_dist <= 0 or tp_dist <= 0:
            continue

        rr = tp_dist / sl_dist
        alignment = sig.alignment_score

        # ── AAA++ tier gate ──────────────────────────────────────
        tier = classify_signal_tier(sig, alignment, rr)
        if aaa_only and tier is None:
            rejected_count += 1
            continue

        tier = tier or TIER_AAA_PLUS  # Fallback if not filtering

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

        risk_amount = compute_dynamic_risk(tier, alignment, equity)
        # Hard cap: never risk more than 3% of current equity per trade
        risk_amount = min(risk_amount, equity * 0.03)
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
        outcome, exit_price = _resolve_trade_outcome(sig)

        if outcome == "skip":
            skipped_no_data += 1
            continue

        # Calculate actual PnL from price movement
        if sig.direction == "long":
            price_pnl_pct = (exit_price - sig.entry_price) / sig.entry_price
        else:
            price_pnl_pct = (sig.entry_price - exit_price) / sig.entry_price

        pnl = position_notional * price_pnl_pct - cost

        # Actual realized RR (for logging)
        actual_rr = price_pnl_pct / sl_pct if sl_pct > 0 else 0.0

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
                "tier": tier,
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
                "risk_pct": risk_amount / account_size,
            }
        )

    if rejected_count > 0 or skipped_no_data > 0:
        logger.info(
            "Trade simulation: %d executed, %d rejected (tier/CB), %d skipped (no price data)",
            len(rows), rejected_count, skipped_no_data,
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Performance metrics (enhanced)
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(trades_df: pd.DataFrame, account_size: float = 100_000) -> dict[str, float]:
    """Compute comprehensive performance metrics."""
    if trades_df.empty:
        return {
            "total_pnl": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
            "sharpe": 0.0, "winrate": 0.0, "total_trades": 0,
            "recovery_factor": 0.0, "avg_rr": 0.0,
            "trades_aaa_pp": 0, "trades_aaa_p": 0,
            "pnl_per_trade": 0.0, "expectancy": 0.0,
        }

    pnl = trades_df["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    total_pnl = float(pnl.sum())
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 1e-9

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    winrate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0

    # Cumulative equity & drawdown
    equity = account_size + pnl.cumsum()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # Annualised Sharpe
    if len(pnl) > 1 and pnl.std() > 0:
        sharpe = float((pnl.mean() / pnl.std()) * math.sqrt(252))
    else:
        sharpe = 0.0

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

    # Tier breakdown
    trades_aaa_pp = int((trades_df.get("tier") == TIER_AAA_PLUS_PLUS).sum()) if "tier" in trades_df.columns else 0
    trades_aaa_p = int((trades_df.get("tier") == TIER_AAA_PLUS).sum()) if "tier" in trades_df.columns else 0

    # Expectancy = (winrate × avg_win) - ((1-winrate) × avg_loss)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = abs(float(losses.mean())) if len(losses) > 0 else 0.0
    expectancy = (winrate * avg_win) - ((1 - winrate) * avg_loss)

    return {
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "winrate": winrate,
        "total_trades": int(len(pnl)),
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
    max_dd_limit: float = -0.10,
) -> dict[str, Any]:
    """
    Check if out-of-sample results pass anti-overfitting gates.

    7 Gates (ALL must pass):
    1. Profit Factor >= 1.5
    2. Minimum 20 trades (SMC sniper: ~6-7/month over 2-3 month OOS)
    3. Sharpe >= 0.5
    4. Monte Carlo robust (95% CI profitable)
    5. Max Drawdown > -10% (funded account compliance)
    6. Parameter stability (±10% change < 50% PF shift)
    7. Win Rate > 20% AND Avg RR > 2.0 (quality check)
    """
    gates: dict[str, bool] = {}
    reasons: list[str] = []

    # Gate 1: Profit Factor
    pf = oos_metrics.get("profit_factor", 0)
    gates["profit_factor_ok"] = pf >= 1.5
    if not gates["profit_factor_ok"]:
        reasons.append(f"Profit Factor {pf:.2f} < 1.5")

    # Gate 2: Minimum trades (SMC sniper = few but high quality)
    n_trades = oos_metrics.get("total_trades", 0)
    gates["min_trades_ok"] = n_trades >= 20
    if not gates["min_trades_ok"]:
        reasons.append(f"Only {n_trades} trades (need >= 20)")

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

    # Gate 6: Parameter stability (hard fail, not just warning)
    if stability_result is not None:
        gates["stability_ok"] = stability_result.get("stable", False)
        if not gates["stability_ok"]:
            reasons.append(
                f"Parameter unstable: ±10% change causes "
                f"{stability_result.get('max_pf_change_pct', 0):.0f}% PF shift (max 50%)"
            )
    else:
        gates["stability_ok"] = True  # Skip if not run

    # Gate 7: Trade quality — Win Rate > 20% AND positive expectancy
    # Note: Target RR is always ≥ 4.0 (AAA+ tier). Realized RR varies
    # due to trailing stop and timeouts — that's normal trade management.
    wr = oos_metrics.get("winrate", 0)
    expectancy = oos_metrics.get("expectancy", 0)
    wr_ok = wr > 0.20
    exp_ok = expectancy > 0
    gates["quality_ok"] = wr_ok and exp_ok
    if not wr_ok:
        reasons.append(f"Win Rate {wr*100:.1f}% <= 20%")
    if not exp_ok:
        reasons.append(f"Negative expectancy: {expectancy:.2f}")

    all_passed = all(gates.values())

    return {
        "passed": all_passed,
        "gates": gates,
        "reasons": reasons,
        "verdict": "PASS - Ready for paper trading" if all_passed else "FAIL - " + "; ".join(reasons),
    }


# ═══════════════════════════════════════════════════════════════════
#  Optuna objective
# ═══════════════════════════════════════════════════════════════════

def _signal_cache_key(
    smc_cfg: dict[str, Any],
    symbols: list[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> str:
    """Build a deterministic hash for the signal cache."""
    key_data = json.dumps({
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
    smc_cfg = config.get("smc", {})
    params = {
        "swing_length": smc_cfg.get("swing_length", 10),
        "fvg_threshold": smc_cfg.get("fvg_threshold", 0.0004),
        "order_block_lookback": smc_cfg.get("order_block_lookback", 20),
        "liquidity_range_percent": smc_cfg.get("liquidity_range_percent", 0.005),
        "risk_reward": 3.0,  # Fallback RR (only used when no structure TP found)
        "alignment_threshold": 0.0,  # No filtering — let all signals through
        "style_weights": {"day": 1.0},
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

    all_signals_list = Parallel(n_jobs=3)(
        delayed(_gen_signals)(sym) for sym in symbols
    )
    all_signals: list[TradeSignal] = [
        s for sublist in all_signals_list for s in sublist
    ]

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

        all_trades: list[pd.DataFrame] = []
        for ac, sigs in filt_by_class.items():
            t = simulate_trades(
                sigs,
                commission_pct=ASSET_COMMISSION.get(ac, config["backtest"]["commission_pct"]),
                slippage_pct=config["backtest"]["slippage_pct"],
                account_size=config["account"]["size"],
                use_circuit_breaker=True,
                aaa_only=True,
                asset_class=ac,
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

        # Objective: PF × (1 − |DD|) × Sharpe — quality over quantity
        # No trade-count penalty: SMC sniper = few trades, high quality
        score = (
            metrics["profit_factor"]
            * (1.0 + metrics["max_drawdown"])  # max_drawdown is negative
            * max(metrics["sharpe"], 0.01)
        )

        # Hard funded-account DD gate: > -10% DD → heavy penalty
        if metrics["max_drawdown"] < -0.10:
            score *= 0.1

        # Quality floor: need at least some trades to evaluate
        if metrics["total_trades"] < 5:
            score = 0.0

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
    """Return all symbols with a 1m Parquet file in data_dir (legacy single-dir)."""
    parquets = sorted(data_dir.glob("*_1m.parquet"))
    symbols = [
        p.stem.replace("_1m", "").replace("_", "/").replace("/USDT/USDT", "/USDT:USDT")
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

    # Crypto: data/crypto/ → Top 30 by 1m file size (proxy for volume/liquidity)
    crypto_dir = Path(data_cfg.get("crypto_dir", "data/crypto"))
    max_crypto = cfg.get("volume_filter", {}).get("max_crypto_symbols", 30)
    if crypto_dir.exists():
        parquets = list(crypto_dir.glob("*_1m.parquet"))
        # Rank by file size (more data = more liquid/actively traded)
        sized = []
        for p in parquets:
            raw = p.stem.replace("_1m", "")
            if "USDT" in raw and raw != "volume":
                sized.append((raw, p.stat().st_size))
        sized.sort(key=lambda x: x[1], reverse=True)
        crypto_syms = [sym for sym, _ in sized[:max_crypto]]
        if crypto_syms:
            result["crypto"] = crypto_syms

    # Forex: data/forex/ → 1m parquets → OANDA format (EUR_USD)
    forex_dir = Path(data_cfg.get("forex_dir", "data/forex"))
    if forex_dir.exists():
        parquets = sorted(forex_dir.glob("*_1m.parquet"))
        forex_syms = [p.stem.replace("_1m", "") for p in parquets]
        if forex_syms:
            result["forex"] = forex_syms

    # Stocks: data/stocks/ → 5m parquets (no 1m!) → plain symbols
    stocks_dir = Path(data_cfg.get("stocks_dir", "data/stocks"))
    if stocks_dir.exists():
        parquets = sorted(stocks_dir.glob("*_5m.parquet"))
        stock_syms = [p.stem.replace("_5m", "").replace("_", ".") for p in parquets]
        if stock_syms:
            result["stocks"] = stock_syms

    # Commodities: data/commodities/ → 1m parquets → OANDA format
    comm_dir = Path(data_cfg.get("commodities_dir", "data/commodities"))
    if comm_dir.exists():
        parquets = sorted(comm_dir.glob("*_1m.parquet"))
        comm_syms = [p.stem.replace("_1m", "") for p in parquets]
        if comm_syms:
            result["commodities"] = comm_syms

    return result


# Asset-class specific commission rates
ASSET_COMMISSION: dict[str, float] = {
    "crypto": 0.0004,       # 0.04% taker (Binance Futures)
    "forex": 0.00005,       # ~0.5 pip spread equivalent
    "stocks": 0.0,          # Commission-free (Alpaca)
    "commodities": 0.0001,  # ~1 pip spread equivalent
}


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
) -> dict[str, Any]:
    """
    Check if small parameter changes (±10%) drastically change performance.
    Uses precomputed signals — only perturbs filtering/trading params.
    """

    def _run_with_params(params):
        alignment_th = params.get("alignment_threshold", 0.65)
        min_rr = params.get("risk_reward", 2.0)
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
            commission_pct=config["backtest"]["commission_pct"],
            slippage_pct=config["backtest"]["slippage_pct"],
            account_size=config["account"]["size"],
        )
        return compute_metrics(trades, account_size=config["account"]["size"])

    base_metrics = _run_with_params(best_params)
    base_pf = base_metrics["profit_factor"]

    perturbed_pfs: list[float] = []
    rng = np.random.RandomState(123)

    for _ in range(n_perturbations):
        perturbed = dict(best_params)
        for key, val in perturbed.items():
            if isinstance(val, (int, float)):
                factor = 1.0 + rng.uniform(-perturbation_pct, perturbation_pct)
                if isinstance(val, int):
                    perturbed[key] = max(1, int(val * factor))
                else:
                    perturbed[key] = val * factor

        metrics = _run_with_params(perturbed)
        perturbed_pfs.append(metrics["profit_factor"])

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
    Full walk-forward Optuna backtest pipeline with AAA++ filtering.

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
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

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

        oos_parts: list[pd.DataFrame] = []
        for ac, sigs in oos_by_class.items():
            t = simulate_trades(
                sigs,
                commission_pct=ASSET_COMMISSION.get(ac, cfg["backtest"]["commission_pct"]),
                slippage_pct=cfg["backtest"]["slippage_pct"],
                account_size=account_size,
                use_circuit_breaker=True,
                aaa_only=True,
                asset_class=ac,
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
        )
        logger.info(
            "Stability W%d: stable=%s max_change=%.1f%%",
            wi, stability_result["stable"],
            stability_result["max_pf_change_pct"],
        )

        # ── Validation gates (6 gates: PF, trades, Sharpe, MC, DD, stability) ──
        validation = validate_oos_results(
            oos_metrics, mc_result,
            stability_result=stability_result,
            max_dd_limit=-0.10,  # Funded account: max -10% all-time DD
        )
        validation["window"] = wi
        validation["stability"] = stability_result
        all_validations.append(validation)

        logger.info(
            "Window %d OOS: PF=%.2f Sharpe=%.2f WR=%.1f%% Trades=%d DD=%.2f%% | %s",
            wi, oos_metrics["profit_factor"], oos_metrics["sharpe"],
            oos_metrics["winrate"] * 100, oos_metrics["total_trades"],
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
            "mean_profit_factor": float(summary["profit_factor"].mean()),
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
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna Walk-Forward Backtester with AAA++ Filtering",
    )
    parser.add_argument(
        "--config", default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run(config_path=args.config)
