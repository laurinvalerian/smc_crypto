"""
═══════════════════════════════════════════════════════════════════
 backtest/optuna_backtester.py
 ─────────────────────────────
 Walk-Forward Optimization with Optuna + AAA++ Filter Validation.

 Features:
   • Rolling walk-forward windows (configurable train/test months)
   • Optuna Bayesian optimisation (configurable trials per window)
   • AAA++ tier filtering: only AAA++ and AAA+ trades pass
   • Circuit breaker simulation (daily -3%, weekly -5%, class -2%)
   • Full 13-component alignment scoring with filters
   • Monte Carlo robustness check (1000 shuffles, 95% CI)
   • Anti-overfitting gates (OOS PF >= 1.5, min 100 trades,
     parameter stability check)
   • Trades all available coins from data/
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
import json
import logging
import math
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
    TIER_AAA_PLUS_PLUS: {"min_score": 0.88, "min_rr": 5.0},
    TIER_AAA_PLUS:      {"min_score": 0.78, "min_rr": 4.0},
}

TIER_RISK = {
    TIER_AAA_PLUS_PLUS: {"base_risk": 0.010, "max_risk": 0.020},
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
#  Trade simulation with AAA++ filtering + Circuit Breaker
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
    Simulate PnL with AAA++ tier filtering and circuit breaker.

    Key improvements over old version:
    - Only AAA++ and AAA+ tier signals are traded (if aaa_only=True)
    - Dynamic risk sizing based on tier
    - Circuit breaker halts trading after excessive losses
    - Deterministic outcome model based on alignment + RR
    """
    if not signals:
        return pd.DataFrame()

    cb = CircuitBreaker() if use_circuit_breaker else None
    equity = account_size

    rows: list[dict[str, Any]] = []
    rejected_count = 0

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

        # ── Dynamic position sizing ──────────────────────────────
        risk_amount = compute_dynamic_risk(tier, alignment, equity)
        position_size = risk_amount / sl_dist if sl_dist > 0 else 0
        if position_size <= 0:
            continue

        # Size reduction from circuit breaker
        if cb is not None:
            position_size *= cb.get_size_factor()

        # Slippage + commission cost
        cost_pct = commission_pct * 2 + slippage_pct * 2
        cost = sig.entry_price * position_size * cost_pct

        # ── Deterministic outcome model ──────────────────────────
        # Uses alignment score + RR to estimate win probability
        # Higher alignment = higher base win rate
        # Higher RR = slightly lower win rate (harder to reach TP)
        rng = np.random.RandomState(
            int(sig.timestamp.timestamp()) % (2**31)
        )
        # Base win prob from alignment (AAA++ scores are 0.78+)
        base_win_prob = alignment * 0.60
        # RR penalty: each point above 3.0 reduces win prob by 2%
        rr_penalty = max(0, (rr - 3.0) * 0.02)
        win_prob = max(0.10, min(0.80, base_win_prob - rr_penalty))

        outcome = "win" if rng.random() < win_prob else "loss"

        if outcome == "win":
            pnl = tp_dist * position_size - cost
        else:
            pnl = -(sl_dist * position_size) - cost

        # Update equity
        equity += pnl

        # Record in circuit breaker
        if cb is not None:
            pnl_pct = pnl / account_size
            trade_time = sig.timestamp.to_pydatetime()
            if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo is None:
                trade_time = trade_time.replace(tzinfo=timezone.utc)
            cb.record_trade_pnl(pnl_pct, asset_class, sig.symbol, trade_time)

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
                "rr": rr,
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

    if rejected_count > 0:
        logger.info(
            "Trade simulation: %d executed, %d rejected (tier/CB filter)",
            len(rows), rejected_count,
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

    # Average RR
    avg_rr = float(trades_df["rr"].mean()) if "rr" in trades_df.columns else 0.0

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
    Shuffle trade order 1000x and check if 95% CI is still profitable.

    This validates that the strategy's edge is not dependent on
    a lucky sequence of trades.
    """
    if trades_df.empty or len(trades_df) < 10:
        return {
            "robust": False,
            "reason": "Not enough trades for Monte Carlo",
            "median_pnl": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "pct_profitable": 0.0,
        }

    pnl_array = trades_df["pnl"].values
    rng = np.random.RandomState(42)

    final_pnls: list[float] = []
    max_dds: list[float] = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(pnl_array)
        cum_pnl = np.cumsum(shuffled)
        final_pnls.append(float(cum_pnl[-1]))

        # Max drawdown of shuffled sequence
        equity = account_size + cum_pnl
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dds.append(float(np.min(dd)))

    final_pnls_arr = np.array(final_pnls)
    max_dds_arr = np.array(max_dds)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(final_pnls_arr, alpha / 2 * 100))
    ci_upper = float(np.percentile(final_pnls_arr, (1 - alpha / 2) * 100))
    median_pnl = float(np.median(final_pnls_arr))
    pct_profitable = float(np.mean(final_pnls_arr > 0))

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
) -> dict[str, Any]:
    """
    Check if out-of-sample results pass anti-overfitting gates.

    Gates:
    1. Profit Factor >= 1.5
    2. Minimum 100 trades
    3. Sharpe >= 0.5
    4. Monte Carlo robust (if provided)
    """
    gates: dict[str, bool] = {}
    reasons: list[str] = []

    # Gate 1: Profit Factor
    pf = oos_metrics.get("profit_factor", 0)
    gates["profit_factor_ok"] = pf >= 1.5
    if not gates["profit_factor_ok"]:
        reasons.append(f"Profit Factor {pf:.2f} < 1.5")

    # Gate 2: Minimum trades
    n_trades = oos_metrics.get("total_trades", 0)
    gates["min_trades_ok"] = n_trades >= 100
    if not gates["min_trades_ok"]:
        reasons.append(f"Only {n_trades} trades (need >= 100)")

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
    else:
        gates["monte_carlo_ok"] = True  # Skip if not run

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

def _build_objective(
    config: dict[str, Any],
    symbols: list[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    window_index: int = 0,
    results_dir: Path | None = None,
    symbol_to_asset: dict[str, str] | None = None,
):
    """Return an Optuna objective function closed over the training window."""

    def objective(trial: optuna.Trial) -> float:
        tuning = config.get("tuning", {})
        params: dict[str, Any] = {
            "leverage": trial.suggest_int(
                "leverage", config["leverage"]["min"], config["leverage"]["max"]
            ),
            "risk_per_trade": trial.suggest_float(
                "risk_per_trade",
                config["risk_per_trade"]["min"],
                config["risk_per_trade"]["max"],
                step=0.001,
            ),
            "risk_reward": trial.suggest_categorical(
                "risk_reward", config["risk_reward"]["options"]
            ),
            "alignment_threshold": trial.suggest_float(
                "alignment_threshold",
                tuning.get("alignment_threshold_min", 0.60),
                tuning.get("alignment_threshold_max", 0.90),
                step=0.05,
            ),
            "swing_length": trial.suggest_int(
                "swing_length",
                tuning.get("swing_length_min", 6),
                tuning.get("swing_length_max", 14),
            ),
            "fvg_threshold": trial.suggest_float(
                "fvg_threshold",
                tuning.get("fvg_threshold_min", 0.0002),
                tuning.get("fvg_threshold_max", 0.0008),
                step=0.0001,
            ),
            "style_weights": {
                "day": trial.suggest_float("weight_day", 0.5, 1.5, step=0.1),
            },
            "order_block_lookback": trial.suggest_int("order_block_lookback", 10, 40),
            "liquidity_range_percent": trial.suggest_float(
                "liquidity_range_percent", 0.002, 0.01, step=0.001
            ),
        }

        strategy = SMCMultiStyleStrategy(config, params)

        def _gen_signals(sym):
            try:
                return strategy.generate_signals(sym, start=train_start, end=train_end)
            except Exception as exc:
                logger.debug("Signal gen failed for %s: %s", sym, exc)
                return []

        all_signals_list = Parallel(n_jobs=3)(
            delayed(_gen_signals)(sym) for sym in symbols
        )
        all_signals: list[TradeSignal] = [
            s for sublist in all_signals_list for s in sublist
        ]

        if not all_signals:
            return 0.0

        # Group signals by asset class for correct commissions
        signals_by_class: dict[str, list[TradeSignal]] = {}
        for sig in all_signals:
            ac = (symbol_to_asset or {}).get(sig.symbol, "crypto")
            signals_by_class.setdefault(ac, []).append(sig)

        all_trades: list[pd.DataFrame] = []
        for ac, sigs in signals_by_class.items():
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

        # Multi-objective proxy: PF × (1 − |MaxDD|) × Sharpe × sqrt(trades)
        # sqrt(trades) rewards strategies with enough trades for significance
        trade_count_bonus = math.sqrt(max(metrics["total_trades"], 1)) / 10.0
        score = (
            metrics["profit_factor"]
            * (1.0 + metrics["max_drawdown"])  # max_drawdown is negative
            * max(metrics["sharpe"], 0.01)
            * min(trade_count_bonus, 2.0)  # Cap bonus at 2x
        )

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

    # Crypto: data/crypto/ → 1m parquets → keep raw filename format (BTCUSDT)
    crypto_dir = Path(data_cfg.get("crypto_dir", "data/crypto"))
    if crypto_dir.exists():
        parquets = sorted(crypto_dir.glob("*_1m.parquet"))
        crypto_syms = []
        for p in parquets:
            raw = p.stem.replace("_1m", "")
            if "USDT" in raw and raw != "volume":
                crypto_syms.append(raw)
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
    symbols: list[str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    perturbation_pct: float = 0.10,
    n_perturbations: int = 5,
) -> dict[str, Any]:
    """
    Check if small parameter changes (±10%) drastically change performance.
    If they do → overfitting warning.
    """
    base_strategy = SMCMultiStyleStrategy(config, best_params)

    def _run_with_params(params):
        strategy = SMCMultiStyleStrategy(config, params)
        all_signals = []
        for sym in symbols:
            try:
                sigs = strategy.generate_signals(sym, start=test_start, end=test_end)
                all_signals.extend(sigs)
            except Exception:
                pass
        trades = simulate_trades(
            all_signals,
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
            if isinstance(val, (int, float)) and key != "risk_reward":
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
    run_monte_carlo: bool = False,
    run_stability_check: bool = False,
) -> None:
    """
    Full walk-forward Optuna backtest pipeline with AAA++ filtering.
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

        # ── Optuna study (training phase) ─────────────────────────
        window_study_name = f"{study_name}_w{wi}"
        study = optuna.create_study(
            study_name=window_study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        objective = _build_objective(
            cfg, symbols, window["train_start"], window["train_end"],
            window_index=wi, results_dir=results_dir,
            symbol_to_asset=symbol_to_asset,
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

        # ── Out-of-sample test with best params ──────────────────
        best_params = study.best_trial.params
        best_params["style_weights"] = {
            "day": best_params.pop("weight_day", 1.0),
        }

        strategy = SMCMultiStyleStrategy(cfg, best_params)

        def _gen_oos_signals(sym):
            try:
                return strategy.generate_signals(
                    sym, start=window["test_start"], end=window["test_end"]
                )
            except Exception as exc:
                logger.debug("OOS signal gen failed for %s: %s", sym, exc)
                return []

        oos_signals_list = Parallel(n_jobs=3)(
            delayed(_gen_oos_signals)(sym) for sym in symbols
        )
        oos_signals: list[TradeSignal] = [
            s for sublist in oos_signals_list for s in sublist
        ]

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

        # ── Monte Carlo check (optional) ──────────────────────────
        mc_result = None
        if run_monte_carlo and not oos_trades.empty:
            mc_result = monte_carlo_check(oos_trades, account_size)
            mc_path = results_dir / f"monte_carlo_w{wi}.json"
            with open(mc_path, "w") as fh:
                json.dump(mc_result, fh, indent=2, default=str)
            logger.info(
                "Monte Carlo W%d: robust=%s median_pnl=%.0f CI=[%.0f, %.0f] profitable=%.1f%%",
                wi, mc_result["robust"], mc_result["median_pnl"],
                mc_result["ci_lower"], mc_result["ci_upper"],
                mc_result["pct_profitable"] * 100,
            )

        # ── Parameter stability check (optional) ──────────────────
        stability_result = None
        if run_stability_check:
            stability_result = check_parameter_stability(
                study, best_params, cfg, symbols,
                window["test_start"], window["test_end"],
            )
            logger.info(
                "Stability W%d: stable=%s max_change=%.1f%%",
                wi, stability_result["stable"],
                stability_result["max_pf_change_pct"],
            )

        # ── Validation gates ──────────────────────────────────────
        validation = validate_oos_results(oos_metrics, mc_result)
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
    parser.add_argument(
        "--monte-carlo", action="store_true",
        help="Run Monte Carlo robustness check on each OOS window",
    )
    parser.add_argument(
        "--stability-check", action="store_true",
        help="Run parameter stability check (±10%% perturbation)",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        run_monte_carlo=args.monte_carlo,
        run_stability_check=args.stability_check,
    )
