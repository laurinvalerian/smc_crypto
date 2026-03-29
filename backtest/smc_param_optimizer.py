"""
SMC Parameter Optimizer -- Finds optimal detection params per instrument cluster.

Currently, SMC params (swing_length, fvg_threshold, ob_lookback, liquidity_range)
are manually set per asset class. This tool clusters instruments by behavior and
optimizes params per cluster using Optuna + walk-forward validation.

Usage:
    python3 -m backtest.smc_param_optimizer --cluster          # Step 1: cluster instruments
    python3 -m backtest.smc_param_optimizer --optimize          # Step 2: optimize per cluster
    python3 -m backtest.smc_param_optimizer --compare           # Step 3: compare old vs new
    python3 -m backtest.smc_param_optimizer --export            # Step 4: export to config
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.smc_multi_style import (
    SMCMultiStyleStrategy,
    TradeSignal,
)
from backtest.optuna_backtester import (
    load_config,
    generate_wf_windows,
    simulate_trades,
    load_price_data_for_symbols,
)

# ── Logging ──────────────────────────────────────────────────────
_results_dir = Path("backtest/results/smc_optimization")
_results_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_results_dir / "smc_optimizer.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

DATA_DIRS = {
    "crypto": Path("data/crypto"),
    "forex": Path("data/forex"),
    "stocks": Path("data/stocks"),
    "commodities": Path("data/commodities"),
}

# Default SMC profiles (mirrors config/default_config.yaml smc_profiles)
DEFAULT_SMC_PARAMS: dict[str, dict[str, Any]] = {
    "crypto": {
        "swing_length": 8, "fvg_threshold": 0.0006,
        "order_block_lookback": 20, "liquidity_range_percent": 0.01,
    },
    "forex": {
        "swing_length": 20, "fvg_threshold": 0.001,
        "order_block_lookback": 30, "liquidity_range_percent": 0.008,
    },
    "stocks": {
        "swing_length": 10, "fvg_threshold": 0.0003,
        "order_block_lookback": 20, "liquidity_range_percent": 0.005,
    },
    "commodities": {
        "swing_length": 10, "fvg_threshold": 0.0004,
        "order_block_lookback": 20, "liquidity_range_percent": 0.005,
    },
}

# Walk-forward windows for optimization (3 months train, 1 month test)
OPT_WINDOWS = [
    ("2024-01-01", "2024-04-01", "2024-04-01", "2024-05-01"),
    ("2024-04-01", "2024-07-01", "2024-07-01", "2024-08-01"),
    ("2024-07-01", "2024-10-01", "2024-10-01", "2024-11-01"),
]


# ═══════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstrumentStats:
    """Per-instrument behavioral statistics for clustering."""
    symbol: str
    asset_class: str
    avg_daily_atr_pct: float = 0.0
    avg_daily_volume_usd: float = 0.0
    avg_spread_pct: float = 0.0
    avg_bars_per_day: float = 0.0


@dataclass
class ClusterInfo:
    """Cluster assignment with metadata."""
    cluster_id: int
    name: str
    instruments: list[str] = field(default_factory=list)
    centroid: dict[str, float] = field(default_factory=dict)
    asset_classes: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  Symbol Discovery
# ═══════════════════════════════════════════════════════════════════

def get_all_symbols() -> list[tuple[str, str]]:
    """Discover all available symbols across asset classes.

    Returns list of (symbol, asset_class) tuples.
    """
    results: list[tuple[str, str]] = []
    for ac, data_dir in DATA_DIRS.items():
        if not data_dir.exists():
            continue
        parquets = sorted(data_dir.glob("*_5m.parquet"))
        for p in parquets:
            sym = p.stem.replace("_5m", "")
            results.append((sym, ac))
    return results


# ═══════════════════════════════════════════════════════════════════
#  Step 1: Clustering
# ═══════════════════════════════════════════════════════════════════

def compute_instrument_stats(symbol: str, asset_class: str) -> InstrumentStats | None:
    """Compute behavioral statistics for one instrument from 5m data."""
    data_dir = DATA_DIRS.get(asset_class)
    if data_dir is None:
        return None

    safe = symbol.replace("/", "_").replace(":", "_")
    path = data_dir / f"{safe}_5m.parquet"
    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.debug("Failed loading %s: %s", path, e)
        return None

    if len(df) < 100:
        return None

    # Ensure numeric columns
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    # ATR as % of price (14-period)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    # Simple rolling ATR approximation
    atr_period = min(14, len(tr))
    if atr_period > 0 and len(tr) >= atr_period:
        atr_vals = np.convolve(tr, np.ones(atr_period) / atr_period, mode="valid")
        close_aligned = close[atr_period:]
        mask = close_aligned > 0
        if mask.any():
            atr_pct = atr_vals[mask] / close_aligned[mask]
            # Scale from 5m ATR to approximate daily ATR (multiply by sqrt(288))
            avg_daily_atr_pct = float(np.mean(atr_pct)) * np.sqrt(288)
        else:
            avg_daily_atr_pct = 0.0
    else:
        avg_daily_atr_pct = 0.0

    # Average daily volume in USD (0 for forex/tick-volume)
    avg_daily_volume_usd = 0.0
    if "volume" in df.columns:
        vol = df["volume"].values.astype(float)
        # Forex tick volume is not real USD volume
        if asset_class not in ("forex", "commodities"):
            vol_usd = vol * close
            # Approximate daily: sum 288 bars (24h of 5m)
            if len(vol_usd) >= 288:
                daily_chunks = len(vol_usd) // 288
                daily_vols = [
                    np.sum(vol_usd[i * 288:(i + 1) * 288])
                    for i in range(daily_chunks)
                ]
                avg_daily_volume_usd = float(np.mean(daily_vols)) if daily_vols else 0.0

    # Average spread % (high-low range as % of close)
    ranges = high - low
    mask = close > 0
    if mask.any():
        spread_pct = ranges[mask] / close[mask]
        avg_spread_pct = float(np.mean(spread_pct))
    else:
        avg_spread_pct = 0.0

    # Average bars per day (trading hours proxy)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        days = (ts.max() - ts.min()).days
        if days > 0:
            avg_bars_per_day = len(df) / days
        else:
            avg_bars_per_day = len(df)
    else:
        avg_bars_per_day = 288.0  # default 24h

    return InstrumentStats(
        symbol=symbol,
        asset_class=asset_class,
        avg_daily_atr_pct=avg_daily_atr_pct,
        avg_daily_volume_usd=avg_daily_volume_usd,
        avg_spread_pct=avg_spread_pct,
        avg_bars_per_day=avg_bars_per_day,
    )


def _auto_name_cluster(
    centroid: dict[str, float],
    asset_classes: list[str],
) -> str:
    """Generate a descriptive name for a cluster based on its centroid."""
    # Determine dominant asset class
    from collections import Counter
    ac_counts = Counter(asset_classes)
    dominant_ac = ac_counts.most_common(1)[0][0] if ac_counts else "mixed"

    # Volatility descriptor
    atr = centroid.get("avg_daily_atr_pct", 0)
    if atr > 0.04:
        vol_desc = "high_vol"
    elif atr > 0.015:
        vol_desc = "med_vol"
    else:
        vol_desc = "low_vol"

    # Trading hours descriptor
    bars = centroid.get("avg_bars_per_day", 288)
    if bars > 250:
        hours_desc = "24h"
    elif bars > 150:
        hours_desc = "extended"
    else:
        hours_desc = "market_hours"

    return f"{vol_desc}_{dominant_ac}_{hours_desc}"


def run_clustering(n_clusters: int = 8) -> dict[str, Any]:
    """Cluster all instruments by behavioral statistics.

    Returns cluster assignment dict suitable for JSON export.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    logger.info("=== Step 1: Clustering instruments ===")
    all_symbols = get_all_symbols()
    logger.info("Found %d instruments across all asset classes", len(all_symbols))

    # Compute stats for all instruments
    stats_list: list[InstrumentStats] = []
    for sym, ac in all_symbols:
        st = compute_instrument_stats(sym, ac)
        if st is not None:
            stats_list.append(st)

    if len(stats_list) < n_clusters:
        logger.warning(
            "Only %d instruments with valid stats, reducing clusters to %d",
            len(stats_list), max(2, len(stats_list) // 2),
        )
        n_clusters = max(2, len(stats_list) // 2)

    logger.info("Computed stats for %d instruments", len(stats_list))

    # Build feature matrix
    feature_names = ["avg_daily_atr_pct", "avg_daily_volume_usd",
                     "avg_spread_pct", "avg_bars_per_day"]
    X = np.array([
        [s.avg_daily_atr_pct, s.avg_daily_volume_usd,
         s.avg_spread_pct, s.avg_bars_per_day]
        for s in stats_list
    ])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Build cluster assignments
    clusters: dict[str, ClusterInfo] = {}
    for i in range(n_clusters):
        mask = labels == i
        members = [stats_list[j] for j in range(len(stats_list)) if mask[j]]
        if not members:
            continue

        # Centroid in original scale
        centroid_scaled = kmeans.cluster_centers_[i]
        centroid_original = scaler.inverse_transform(centroid_scaled.reshape(1, -1))[0]
        centroid_dict = {fn: float(centroid_original[k]) for k, fn in enumerate(feature_names)}

        ac_list = [m.asset_class for m in members]
        name = _auto_name_cluster(centroid_dict, ac_list)

        clusters[str(i)] = ClusterInfo(
            cluster_id=i,
            name=name,
            instruments=[m.symbol for m in members],
            centroid=centroid_dict,
            asset_classes=list(set(ac_list)),
        )

    # Build output dict
    result = {
        "n_clusters": n_clusters,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "feature_names": feature_names,
        "clusters": {
            k: {
                "name": v.name,
                "instruments": v.instruments,
                "centroid": v.centroid,
                "asset_classes": v.asset_classes,
            }
            for k, v in clusters.items()
        },
        # Reverse lookup: symbol -> cluster_id
        "symbol_to_cluster": {
            sym: int(cid)
            for cid, info in clusters.items()
            for sym in info.instruments
        },
    }

    # Save
    out_path = Path("config/instrument_clusters.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved cluster assignments to %s", out_path)

    # Print summary
    print("\n=== Cluster Summary ===")
    print(f"{'ID':<4} {'Name':<35} {'#Instr':<8} {'ATR%':<10} {'Vol($)':<14} {'Spread%':<10} {'Bars/Day':<10}")
    print("-" * 95)
    for cid, info in sorted(clusters.items(), key=lambda x: int(x[0])):
        c = info.centroid
        print(
            f"{cid:<4} {info.name:<35} {len(info.instruments):<8} "
            f"{c.get('avg_daily_atr_pct', 0):.4f}    "
            f"{c.get('avg_daily_volume_usd', 0):>12,.0f}  "
            f"{c.get('avg_spread_pct', 0):.6f}  "
            f"{c.get('avg_bars_per_day', 0):>8.1f}"
        )
    print()

    return result


# ═══════════════════════════════════════════════════════════════════
#  Step 2: Optimization
# ═══════════════════════════════════════════════════════════════════

def _get_asset_class_for_symbol(symbol: str) -> str:
    """Determine asset class for a symbol by checking data directories."""
    for ac, data_dir in DATA_DIRS.items():
        safe = symbol.replace("/", "_").replace(":", "_")
        if (data_dir / f"{safe}_5m.parquet").exists():
            return ac
    return "crypto"  # fallback


def _generate_signals_with_params(
    symbol: str,
    asset_class: str,
    smc_params: dict[str, Any],
    config: dict[str, Any],
    train_start: str,
    train_end: str,
) -> list[TradeSignal]:
    """Generate signals for one symbol using specific SMC detection params.

    Creates a strategy instance with the given SMC params and runs signal
    generation over the specified time window.
    """
    params = {
        "swing_length": smc_params["swing_length"],
        "fvg_threshold": smc_params["fvg_threshold"],
        "order_block_lookback": smc_params["order_block_lookback"],
        "liquidity_range_percent": smc_params["liquidity_range_percent"],
        "leverage": 5,
        "risk_per_trade": 0.005,
        "risk_reward": 2.0,
        "alignment_threshold": 0.65,
        "asset_class": asset_class,
    }

    strategy = SMCMultiStyleStrategy(config, params)
    start_ts = pd.Timestamp(train_start, tz="UTC")
    end_ts = pd.Timestamp(train_end, tz="UTC")

    try:
        signals = strategy.generate_signals(symbol, start=start_ts, end=end_ts)
        return signals
    except Exception as e:
        logger.debug("Signal gen failed for %s: %s", symbol, e)
        return []


def _evaluate_params(
    smc_params: dict[str, Any],
    representative_symbols: list[tuple[str, str]],
    config: dict[str, Any],
    windows: list[tuple[str, str, str, str]],
) -> float:
    """Evaluate a set of SMC params across representative symbols and windows.

    Returns the average OOS profit factor across all windows.
    """
    oos_profit_factors: list[float] = []

    for train_start, train_end, test_start, test_end in windows:
        all_signals: list[TradeSignal] = []

        for sym, ac in representative_symbols:
            sigs = _generate_signals_with_params(
                sym, ac, smc_params, config, test_start, test_end,
            )
            all_signals.extend(sigs)

        if not all_signals:
            oos_profit_factors.append(0.0)
            continue

        # Load price data for simulation
        syms = list({s.symbol for s in all_signals})
        load_price_data_for_symbols(syms, config)

        # Simulate with default trading params (we only vary detection params)
        results_df = simulate_trades(
            all_signals,
            commission_pct=config["backtest"].get("commission_pct", 0.0004),
            slippage_pct=config["backtest"].get("slippage_pct", 0.0001),
            account_size=config["account"]["size"],
            use_circuit_breaker=False,
            aaa_only=False,  # we want to see raw signal quality
            asset_class=representative_symbols[0][1] if representative_symbols else "crypto",
        )

        # Column is "pnl" (absolute $), not "pnl_pct"
        pnl_col = "pnl" if "pnl" in results_df.columns else "pnl_pct"
        if results_df.empty or pnl_col not in results_df.columns:
            oos_profit_factors.append(0.0)
            continue

        # Compute profit factor
        gross_profit = results_df.loc[results_df[pnl_col] > 0, pnl_col].sum()
        gross_loss = abs(results_df.loc[results_df[pnl_col] < 0, pnl_col].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

        # Cap PF at 10 to avoid outlier influence
        oos_profit_factors.append(min(pf, 10.0))

    return float(np.mean(oos_profit_factors)) if oos_profit_factors else 0.0


def _select_representative_symbols(
    instruments: list[str],
    max_symbols: int = 5,
) -> list[tuple[str, str]]:
    """Select representative symbols from a cluster for optimization.

    Picks up to max_symbols instruments that have sufficient historical data
    covering the optimization windows (2024-01 to 2024-11).
    Returns list of (symbol, asset_class) tuples.
    """
    MIN_OPT_DATE = pd.Timestamp("2024-01-01", tz="UTC")
    MIN_ROWS = 50_000  # ~2 months of 5m data

    candidates: list[tuple[str, str, int]] = []
    for sym in instruments:
        ac = _get_asset_class_for_symbol(sym)
        safe = sym.replace("/", "_").replace(":", "_")
        data_dir = DATA_DIRS.get(ac)
        if data_dir is None:
            continue
        path = data_dir / f"{safe}_5m.parquet"
        if not path.exists():
            continue

        # Check data actually covers the optimization window
        try:
            df = pd.read_parquet(path, columns=["timestamp"])
            if len(df) < MIN_ROWS:
                continue
            earliest = pd.Timestamp(df["timestamp"].min())
            if earliest > MIN_OPT_DATE:
                logger.debug("Skipping %s: data starts %s (need < %s)", sym, earliest, MIN_OPT_DATE)
                continue
            candidates.append((sym, ac, len(df)))
        except Exception:
            continue

    # Sort by row count (more data = more representative)
    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = [(sym, ac) for sym, ac, _ in candidates[:max_symbols]]
    if not selected:
        logger.warning("No instruments with sufficient data for optimization windows")
    return selected


def run_optimization(
    n_trials: int = 30,
    max_symbols_per_cluster: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run Optuna optimization for SMC params per cluster.

    Returns dict of cluster_id -> optimized params.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load cluster assignments
    cluster_path = Path("config/instrument_clusters.json")
    if not cluster_path.exists():
        logger.error("No cluster file found. Run --cluster first.")
        return {}

    with open(cluster_path) as f:
        cluster_data = json.load(f)

    config = load_config()
    optimized: dict[str, dict[str, Any]] = {}

    for cid, cinfo in cluster_data["clusters"].items():
        instruments = cinfo["instruments"]
        cluster_name = cinfo["name"]
        asset_classes = cinfo.get("asset_classes", ["crypto"])

        logger.info(
            "=== Optimizing cluster %s: %s (%d instruments) ===",
            cid, cluster_name, len(instruments),
        )

        # Select representative symbols
        rep_symbols = _select_representative_symbols(
            instruments, max_symbols=max_symbols_per_cluster,
        )
        if not rep_symbols:
            logger.warning("  No valid symbols for cluster %s, skipping", cid)
            continue

        logger.info(
            "  Representatives: %s",
            ", ".join(f"{s}({ac})" for s, ac in rep_symbols),
        )

        # Determine default params (from dominant asset class)
        dominant_ac = asset_classes[0] if asset_classes else "crypto"
        old_params = DEFAULT_SMC_PARAMS.get(dominant_ac, DEFAULT_SMC_PARAMS["crypto"]).copy()

        # Optuna study
        def objective(trial: optuna.Trial) -> float:
            smc_params = {
                "swing_length": trial.suggest_int("swing_length", 5, 30),
                "fvg_threshold": trial.suggest_float(
                    "fvg_threshold", 0.0001, 0.003, log=True,
                ),
                "order_block_lookback": trial.suggest_int(
                    "order_block_lookback", 10, 50,
                ),
                "liquidity_range_percent": trial.suggest_float(
                    "liquidity_range_percent", 0.003, 0.02, log=True,
                ),
            }
            return _evaluate_params(
                smc_params, rep_symbols, config, OPT_WINDOWS,
            )

        study = optuna.create_study(
            direction="maximize",
            study_name=f"smc_cluster_{cid}",
        )

        # Seed with current defaults as first trial
        study.enqueue_trial(old_params)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best_pf = study.best_value

        # Evaluate old params for comparison
        old_pf = _evaluate_params(old_params, rep_symbols, config, OPT_WINDOWS)

        optimized[cid] = {
            "cluster_name": cluster_name,
            "old_params": old_params,
            "optimized_params": best,
            "old_pf": old_pf,
            "optimized_pf": best_pf,
            "improvement_pct": ((best_pf - old_pf) / old_pf * 100) if old_pf > 0 else 0,
            "n_trials": n_trials,
            "representative_symbols": [s for s, _ in rep_symbols],
        }

        # Save per-cluster results
        result_path = _results_dir / f"cluster_{cid}_params.json"
        with open(result_path, "w") as f:
            json.dump(optimized[cid], f, indent=2)
        logger.info(
            "  Cluster %s: old PF=%.2f -> optimized PF=%.2f (%+.1f%%)",
            cid, old_pf, best_pf, optimized[cid]["improvement_pct"],
        )

        gc.collect()

    # Print summary
    print("\n=== Optimization Summary ===")
    print(f"{'Cluster':<8} {'Name':<35} {'Old PF':<10} {'New PF':<10} {'Change':<10}")
    print("-" * 75)
    for cid, info in sorted(optimized.items(), key=lambda x: int(x[0])):
        change = info["improvement_pct"]
        marker = " ***" if change > 10 else ""
        print(
            f"{cid:<8} {info['cluster_name']:<35} "
            f"{info['old_pf']:<10.2f} {info['optimized_pf']:<10.2f} "
            f"{change:>+.1f}%{marker}"
        )
    print()

    return optimized


# ═══════════════════════════════════════════════════════════════════
#  Step 3: Comparison
# ═══════════════════════════════════════════════════════════════════

def run_comparison() -> None:
    """Compare old vs optimized params on holdout data."""
    # Load cluster assignments
    cluster_path = Path("config/instrument_clusters.json")
    if not cluster_path.exists():
        logger.error("No cluster file found. Run --cluster first.")
        return

    with open(cluster_path) as f:
        cluster_data = json.load(f)

    config = load_config()

    # Holdout window (not used in optimization)
    holdout_windows = [
        ("2024-10-01", "2025-01-01", "2025-01-01", "2025-02-01"),
    ]

    print("\n=== Holdout Comparison (Jan 2025) ===")
    print(
        f"{'Cluster':<8} {'Name':<30} "
        f"{'Old PF':<9} {'Old #T':<8} {'Old WR':<8} "
        f"{'New PF':<9} {'New #T':<8} {'New WR':<8} "
        f"{'Delta':<8}"
    )
    print("-" * 110)

    for cid, cinfo in sorted(cluster_data["clusters"].items(), key=lambda x: int(x[0])):
        # Load optimized params
        result_path = _results_dir / f"cluster_{cid}_params.json"
        if not result_path.exists():
            print(f"{cid:<8} {cinfo['name']:<30} [no optimization results]")
            continue

        with open(result_path) as f:
            opt_data = json.load(f)

        old_params = opt_data["old_params"]
        new_params = opt_data["optimized_params"]

        rep_symbols = _select_representative_symbols(
            cinfo["instruments"], max_symbols=5,
        )
        if not rep_symbols:
            continue

        # Evaluate both param sets on holdout
        old_pf = _evaluate_params(old_params, rep_symbols, config, holdout_windows)
        new_pf = _evaluate_params(new_params, rep_symbols, config, holdout_windows)

        delta = new_pf - old_pf
        marker = " ***" if delta > 0.5 else ""

        print(
            f"{cid:<8} {cinfo['name']:<30} "
            f"{old_pf:<9.2f} {'--':<8} {'--':<8} "
            f"{new_pf:<9.2f} {'--':<8} {'--':<8} "
            f"{delta:>+.2f}{marker}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════
#  Step 4: Export
# ═══════════════════════════════════════════════════════════════════

def run_export() -> None:
    """Export optimized params to a YAML config file."""
    cluster_path = Path("config/instrument_clusters.json")
    if not cluster_path.exists():
        logger.error("No cluster file found. Run --cluster first.")
        return

    with open(cluster_path) as f:
        cluster_data = json.load(f)

    smc_profiles: dict[str, Any] = {}

    for cid, cinfo in sorted(cluster_data["clusters"].items(), key=lambda x: int(x[0])):
        result_path = _results_dir / f"cluster_{cid}_params.json"
        if not result_path.exists():
            logger.warning("No optimization results for cluster %s, using defaults", cid)
            dominant_ac = cinfo.get("asset_classes", ["crypto"])[0]
            params = DEFAULT_SMC_PARAMS.get(dominant_ac, DEFAULT_SMC_PARAMS["crypto"]).copy()
        else:
            with open(result_path) as f:
                opt_data = json.load(f)
            params = opt_data["optimized_params"]

        profile_key = cinfo["name"]
        smc_profiles[profile_key] = {
            "cluster_id": int(cid),
            "instruments": cinfo["instruments"],
            "swing_length": int(params["swing_length"]),
            "fvg_threshold": float(params["fvg_threshold"]),
            "order_block_lookback": int(params["order_block_lookback"]),
            "liquidity_range_percent": float(params["liquidity_range_percent"]),
        }

    output = {
        "# Generated by smc_param_optimizer.py": None,
        "# Review and merge into config/default_config.yaml manually": None,
        "smc_profiles_optimized": smc_profiles,
    }

    out_path = Path("config/smc_profiles_optimized.yaml")
    with open(out_path, "w") as f:
        # Write header comments
        f.write("# ═══════════════════════════════════════════════════════════════════\n")
        f.write("#  SMC Profiles (Optimized per Instrument Cluster)\n")
        f.write(f"#  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write("#  Review and merge into config/default_config.yaml manually.\n")
        f.write("# ═══════════════════════════════════════════════════════════════════\n\n")
        yaml.dump(
            {"smc_profiles_optimized": smc_profiles},
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    logger.info("Exported optimized profiles to %s", out_path)
    print(f"\nExported to: {out_path}")
    print("Review the file and merge relevant sections into config/default_config.yaml")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SMC Parameter Optimizer -- per-cluster detection param tuning",
    )
    parser.add_argument("--cluster", action="store_true",
                        help="Step 1: Cluster instruments by behavior")
    parser.add_argument("--optimize", action="store_true",
                        help="Step 2: Optimize SMC params per cluster (Optuna)")
    parser.add_argument("--compare", action="store_true",
                        help="Step 3: Compare old vs new on holdout data")
    parser.add_argument("--export", action="store_true",
                        help="Step 4: Export optimized params to YAML")
    parser.add_argument("--n-clusters", type=int, default=8,
                        help="Number of clusters (default 8)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials per cluster (default 30)")
    parser.add_argument("--max-symbols", type=int, default=5,
                        help="Representative symbols per cluster (default 5)")
    args = parser.parse_args()

    if not any([args.cluster, args.optimize, args.compare, args.export]):
        parser.print_help()
        return

    if args.cluster:
        run_clustering(n_clusters=args.n_clusters)

    if args.optimize:
        run_optimization(
            n_trials=args.n_trials,
            max_symbols_per_cluster=args.max_symbols,
        )

    if args.compare:
        run_comparison()

    if args.export:
        run_export()


if __name__ == "__main__":
    main()
