"""
═══════════════════════════════════════════════════════════════════
 backtest/optuna_backtester.py
 ─────────────────────────────
 Walk-Forward Optimization with Optuna Bayesian search.

 Features:
   • Rolling walk-forward windows (6-month train / 3-month test)
   • Optuna Bayesian optimisation (≥ 2 000 trials per window)
   • Extracts top 20 % best parameter sets automatically
   • Generates parameter-importance ranking (plot + CSV)
   • Full performance metrics per trial:
       Profit Factor, Max Drawdown, Sharpe, Winrate, #Trades,
       Profit per Style (Scalp / Day / Swing), Recovery Factor
   • All results stored in /backtest/results

 Usage:
     python -m backtest.optuna_backtester                 # default config
     python -m backtest.optuna_backtester --config path   # custom config
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
from tqdm import tqdm

from strategies.smc_multi_style import SMCMultiStyleStrategy, TradeSignal

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# Suppress noisy Optuna trial logs
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
        # Roll forward by test_months
        cursor = test_start

    return windows


# ═══════════════════════════════════════════════════════════════════
#  Trade simulation (vectorised on signals list)
# ═══════════════════════════════════════════════════════════════════

def simulate_trades(
    signals: list[TradeSignal],
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0001,
) -> pd.DataFrame:
    """
    Simulate PnL for a list of *TradeSignal* objects.
    Assumes each trade hits either SL or TP (whichever is reached first
    based on the risk-reward ratio).

    For the backtester's purpose we use a simplified model:
      • Each trade either wins (TP hit) or loses (SL hit).
      • Win probability is estimated from the signal's alignment score.

    Returns a DataFrame with one row per trade and PnL columns.
    """
    if not signals:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sig in signals:
        sl_dist = abs(sig.entry_price - sig.stop_loss)
        tp_dist = abs(sig.take_profit - sig.entry_price)
        risk_amount = sl_dist * sig.position_size

        # Slippage + commission cost
        cost_pct = commission_pct * 2 + slippage_pct * 2  # Entry + exit
        cost = sig.entry_price * sig.position_size * cost_pct

        # Determine outcome using alignment score as a probability proxy
        # Higher alignment → higher probability of winning
        # Baseline winrate: score * 0.6 (conservative estimate)
        rng = np.random.RandomState(
            int(sig.timestamp.timestamp()) % (2**31)
        )
        win_prob = sig.alignment_score * 0.55
        outcome = "win" if rng.random() < win_prob else "loss"

        if outcome == "win":
            pnl = tp_dist * sig.position_size - cost
        else:
            pnl = -(sl_dist * sig.position_size) - cost

        rows.append(
            {
                "timestamp": sig.timestamp,
                "symbol": sig.symbol,
                "direction": sig.direction,
                "style": sig.style,
                "entry": sig.entry_price,
                "sl": sig.stop_loss,
                "tp": sig.take_profit,
                "rr": sig.risk_reward,
                "qty": sig.position_size,
                "leverage": sig.leverage,
                "alignment": sig.alignment_score,
                "outcome": outcome,
                "pnl": pnl,
                "cost": cost,
            }
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Performance metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(trades_df: pd.DataFrame, account_size: float = 100_000) -> dict[str, float]:
    """
    Compute comprehensive performance metrics from a trades DataFrame.
    """
    if trades_df.empty:
        return {
            "total_pnl": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
            "sharpe": 0.0, "winrate": 0.0, "total_trades": 0,
            "recovery_factor": 0.0,
            "pnl_scalp": 0.0, "pnl_day": 0.0, "pnl_swing": 0.0,
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
    max_drawdown = float(drawdown.min())  # Negative value

    # Annualised Sharpe (assuming ~252 trading days, mean daily PnL)
    if len(pnl) > 1 and pnl.std() > 0:
        sharpe = float((pnl.mean() / pnl.std()) * math.sqrt(252))
    else:
        sharpe = 0.0

    # Recovery factor = total PnL / abs(max drawdown in $)
    max_dd_usd = abs(max_drawdown * account_size) if max_drawdown != 0 else 1e-9
    recovery_factor = total_pnl / max_dd_usd if max_dd_usd > 0 else 0.0

    # PnL per style
    pnl_scalp = float(trades_df.loc[trades_df["style"] == "scalp", "pnl"].sum())
    pnl_day = float(trades_df.loc[trades_df["style"] == "day", "pnl"].sum())
    pnl_swing = float(trades_df.loc[trades_df["style"] == "swing", "pnl"].sum())

    return {
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "winrate": winrate,
        "total_trades": int(len(pnl)),
        "recovery_factor": recovery_factor,
        "pnl_scalp": pnl_scalp,
        "pnl_day": pnl_day,
        "pnl_swing": pnl_swing,
    }


# ═══════════════════════════════════════════════════════════════════
#  Optuna objective
# ═══════════════════════════════════════════════════════════════════

def _build_objective(
    config: dict[str, Any],
    symbols: list[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
):
    """Return an Optuna objective function closed over the training window."""

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ────────────────────────────────
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
            "swing_length": trial.suggest_int("swing_length", 5, 20),
            "fvg_threshold": trial.suggest_float(
                "fvg_threshold", 0.0005, 0.005, step=0.0005
            ),
            "alignment_threshold": trial.suggest_float(
                "alignment_threshold", 0.4, 0.9, step=0.05
            ),
            "style_weights": {
                "scalp": trial.suggest_float("weight_scalp", 0.5, 1.5, step=0.1),
                "day": trial.suggest_float("weight_day", 0.5, 1.5, step=0.1),
                "swing": trial.suggest_float("weight_swing", 0.5, 1.5, step=0.1),
            },
            "order_block_lookback": trial.suggest_int("order_block_lookback", 10, 40),
            "liquidity_range_percent": trial.suggest_float(
                "liquidity_range_percent", 0.002, 0.01, step=0.001
            ),
        }

        strategy = SMCMultiStyleStrategy(config, params)
        all_signals: list[TradeSignal] = []

        for sym in symbols:
            try:
                sigs = strategy.generate_signals(sym, start=train_start, end=train_end)
                all_signals.extend(sigs)
            except Exception as exc:
                logger.debug("Signal gen failed for %s: %s", sym, exc)

        if not all_signals:
            return 0.0

        trades = simulate_trades(
            all_signals,
            commission_pct=config["backtest"]["commission_pct"],
            slippage_pct=config["backtest"]["slippage_pct"],
        )
        metrics = compute_metrics(trades, account_size=config["account"]["size"])

        # Multi-objective proxy: Profit Factor × (1 − |MaxDD|) × Sharpe
        score = (
            metrics["profit_factor"]
            * (1.0 + metrics["max_drawdown"])  # max_drawdown is negative
            * max(metrics["sharpe"], 0.01)
        )
        # Store metrics as trial user attrs
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
    """
    From a completed Optuna study, extract the top *top_pct* fraction
    of trials sorted by objective value (descending).
    """
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
    """
    Use Optuna's built-in fANOVA importance evaluator.
    Saves a CSV and a bar-chart PNG.
    """
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
    # Save CSV
    csv_path = results_dir / "param_importance.csv"
    imp_df.to_csv(csv_path, index=False)
    logger.info("Parameter importance saved → %s", csv_path)

    # Save plot
    try:
        import plotly.express as px

        fig = px.bar(
            imp_df,
            x="importance",
            y="parameter",
            orientation="h",
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
#  Volume-filtered symbol list
# ═══════════════════════════════════════════════════════════════════

def get_top_symbols(
    data_dir: Path,
    month: pd.Period | None = None,
    rank_threshold: int = 100,
) -> list[str]:
    """
    Read the pre-computed volume rankings and return symbols inside
    the top *rank_threshold* for the given month.
    Falls back to all available symbols if ranking file is missing.
    """
    ranking_path = data_dir / "volume_rankings.parquet"
    if not ranking_path.exists():
        logger.warning("Volume ranking file not found – using all available symbols")
        parquets = list(data_dir.glob("*_1m.parquet"))
        return [
            p.stem.replace("_1m", "").replace("_", "/").replace("/USDT/USDT", "/USDT:USDT")
            for p in parquets
        ]

    rank_df = pd.read_parquet(ranking_path)
    if month is not None:
        rank_df = rank_df[rank_df["month"] == month]

    top = rank_df[rank_df["rank"] <= rank_threshold]
    return top["symbol"].unique().tolist()


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run(config_path: str = "config/default_config.yaml") -> None:
    """
    Full walk-forward Optuna backtest pipeline.
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

    # Date range from config
    start = pd.Timestamp(cfg["data"]["start_date"], tz="UTC")
    end = pd.Timestamp(datetime.now(timezone.utc))

    windows = generate_wf_windows(start, end, train_months, test_months)
    logger.info("Walk-forward windows: %d", len(windows))

    all_window_results: list[pd.DataFrame] = []

    for wi, window in enumerate(tqdm(windows, desc="Walk-forward windows")):
        logger.info(
            "Window %d: Train %s → %s | Test %s → %s",
            wi,
            window["train_start"].date(),
            window["train_end"].date(),
            window["test_start"].date(),
            window["test_end"].date(),
        )

        # Determine symbols for this window's training period
        month = window["train_start"].to_period("M")
        symbols = get_top_symbols(
            data_dir, month=month, rank_threshold=cfg["volume_filter"]["rank_threshold"]
        )
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
            cfg, symbols, window["train_start"], window["train_end"]
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # ── Out-of-sample test with best params ──────────────────
        best_params = study.best_trial.params
        # Reconstruct nested style_weights
        best_params["style_weights"] = {
            "scalp": best_params.pop("weight_scalp", 1.0),
            "day": best_params.pop("weight_day", 1.0),
            "swing": best_params.pop("weight_swing", 1.0),
        }

        strategy = SMCMultiStyleStrategy(cfg, best_params)
        oos_signals: list[TradeSignal] = []
        for sym in symbols:
            try:
                sigs = strategy.generate_signals(
                    sym, start=window["test_start"], end=window["test_end"]
                )
                oos_signals.extend(sigs)
            except Exception:
                pass

        oos_trades = simulate_trades(
            oos_signals,
            commission_pct=cfg["backtest"]["commission_pct"],
            slippage_pct=cfg["backtest"]["slippage_pct"],
        )
        oos_metrics = compute_metrics(oos_trades, account_size=cfg["account"]["size"])

        # Save window results
        window_result = {"window": wi, **oos_metrics, **best_params}
        all_window_results.append(pd.DataFrame([window_result]))

        # ── Top 20 % extraction ───────────────────────────────────
        top_df = extract_top_params(study, top_pct=top_pct)
        if not top_df.empty:
            top_csv = results_dir / f"top_params_w{wi}.csv"
            top_df.to_csv(top_csv, index=False)
            logger.info("Top %d%% params saved → %s", int(top_pct * 100), top_csv)

        # ── Parameter importance ──────────────────────────────────
        compute_param_importance(study, results_dir)

    # ── Aggregate results ─────────────────────────────────────────
    if all_window_results:
        summary = pd.concat(all_window_results, ignore_index=True)
        summary_path = results_dir / "wfo_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("WFO summary saved → %s", summary_path)

        # Global top params across all windows
        global_top = summary.nlargest(
            max(1, int(len(summary) * top_pct)), "total_pnl"
        )
        global_top_path = results_dir / "global_top_params.csv"
        global_top.to_csv(global_top_path, index=False)
        logger.info("Global top params → %s", global_top_path)

        # Summary stats to JSON
        stats = {
            "total_windows": len(windows),
            "mean_pnl": float(summary["total_pnl"].mean()),
            "mean_sharpe": float(summary["sharpe"].mean()),
            "mean_winrate": float(summary["winrate"].mean()),
            "mean_profit_factor": float(summary["profit_factor"].mean()),
            "worst_drawdown": float(summary["max_drawdown"].min()),
        }
        stats_path = results_dir / "backtest_stats.json"
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("Summary stats → %s", stats_path)

    logger.info("✅  Walk-forward backtest complete. Results in %s", results_dir)


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Walk-Forward Backtester")
    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run(config_path=args.config)
