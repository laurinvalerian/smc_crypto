"""
Compare 5 XGBoost entry filter variants with walk-forward validation.

Variants:
  A: threshold=0.50, alignment_score excluded (previous live config)
  B: threshold=0.78, alignment_score excluded
  C: threshold=0.78, alignment_score included as feature
  D: threshold=0.50, alignment_score included as feature
  E: threshold=0.65, alignment_score included as feature

Walk-forward: train on windows 0..N-1, test on window N (last 3 windows as OOS).
Anti-overfitting: results shown per OOS window + aggregate.

Usage:
  python3 -m backtest.compare_variants
  # Monitor: tail -f backtest/results/variant_comparison.log
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────
DATA_DIR = Path("data/rl_training")
OUT_DIR = Path("backtest/results/variant_comparison")
N_OOS_WINDOWS = 3  # last 3 windows as OOS test sets
N_JOBS = max(1, os.cpu_count() - 1)  # leave 1 core free

ASSET_CLASSES = ["crypto", "stocks"]  # forex/commodities paused per roadmap

# 41 features matching the current entry_quality model (schema v2)
BASE_FEATURES = [
    "struct_1d", "struct_4h", "struct_1h", "struct_15m", "struct_5m",
    "decay_1d", "decay_4h", "decay_1h", "decay_15m", "decay_5m",
    "premium_discount",
    "h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
    "precision_trigger", "volume_ok",
    "ema20_dist_5m", "ema50_dist_5m", "ema20_dist_1h", "ema50_dist_1h",
    "atr_5m_norm", "atr_1h_norm", "atr_daily_norm",
    "rsi_5m", "rsi_1h",
    "volume_ratio", "adx_1h",
    "hour_sin", "hour_cos",
    "fvg_bull_active", "fvg_bear_active",
    "ob_bull_active", "ob_bear_active",
    "liq_above_count", "liq_below_count",
    "symbol_volatility_rank", "symbol_liquidity_rank", "symbol_spread_rank",
    "asset_class_id",
    "style_id",
]

# XGBoost hyperparams (matching continuous_learner.py)
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=2,  # per-model parallelism (outer loop handles variant parallelism)
)

# Commission rates (matching live_multi_bot.py ASSET_COMMISSION)
ASSET_COMMISSION = {
    "crypto": 0.0004,
    "forex": 0.00005,
    "stocks": 0.0,
    "commodities": 0.0001,
}


@dataclass
class Variant:
    name: str
    threshold: float
    include_score: bool


VARIANTS = [
    Variant("A_t050_noscore", 0.50, False),
    Variant("B_t078_noscore", 0.78, False),
    Variant("C_t078_score",   0.78, True),
    Variant("D_t050_score",   0.50, True),
    Variant("E_t065_score",   0.65, True),
]


def _features_for_variant(v: Variant) -> list[str]:
    feats = list(BASE_FEATURES)
    if v.include_score:
        feats.append("alignment_score")
    return feats


def _ensure_asset_class_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add asset_class_id if missing."""
    if "asset_class_id" not in df.columns:
        ac_map = {"crypto": 0.0, "forex": 0.33, "stocks": 0.66, "commodities": 1.0}
        df["asset_class_id"] = df["asset_class"].map(ac_map).fillna(0.5)
    return df


def _ensure_style_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add style_id if missing (default to day=0.5)."""
    if "style_id" not in df.columns:
        df["style_id"] = 0.5
    return df


def _simulate_trades(df_oos: pd.DataFrame, preds: np.ndarray, proba: np.ndarray,
                     conf_threshold: float, asset_class: str) -> dict:
    """Simulate trades on OOS data and return performance metrics."""
    comm = ASSET_COMMISSION.get(asset_class, 0.0004)
    comm_roundtrip = comm * 2

    # Only take trades the model accepts (predict=1) with confidence >= threshold
    mask = (preds == 1) & (proba >= conf_threshold)
    trades = df_oos[mask].copy()

    if len(trades) == 0:
        return {
            "trades": 0, "wins": 0, "wr": 0.0, "pf": 0.0,
            "avg_rr": 0.0, "total_pnl_pct": 0.0, "sharpe": 0.0,
            "avg_conf": 0.0,
        }

    # PnL per trade: label_rr * direction, minus commission
    # label_rr is signed (positive = favorable move in trade direction)
    rr_values = trades["label_rr"].values.astype(float)
    profitable = trades["label_profitable"].values.astype(int)

    # Approximate PnL: each trade risks ~0.5% of equity
    risk_pct = 0.005
    pnl_per_trade = []
    for rr, prof in zip(rr_values, profitable):
        if prof:
            pnl = abs(rr) * risk_pct - comm_roundtrip
        else:
            pnl = -1.0 * risk_pct - comm_roundtrip
        pnl_per_trade.append(pnl)

    pnl_arr = np.array(pnl_per_trade)
    wins = int((pnl_arr > 0).sum())
    losses = int((pnl_arr <= 0).sum())
    gross_profit = float(pnl_arr[pnl_arr > 0].sum()) if wins > 0 else 0.0
    gross_loss = float(abs(pnl_arr[pnl_arr <= 0].sum())) if losses > 0 else 0.001

    wr = wins / len(pnl_arr) * 100 if len(pnl_arr) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    avg_rr = float(np.mean(np.abs(rr_values)))
    total_pnl = float(pnl_arr.sum()) * 100  # as percentage
    sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)) if np.std(pnl_arr) > 0 else 0.0

    return {
        "trades": len(trades),
        "wins": wins,
        "wr": round(wr, 1),
        "pf": round(min(pf, 99.9), 2),
        "avg_rr": round(avg_rr, 2),
        "total_pnl_pct": round(total_pnl, 3),
        "sharpe": round(sharpe, 2),
        "avg_conf": round(float(proba[mask].mean()), 3),
    }


def run_variant_asset(args: tuple) -> dict:
    """Run one variant × asset class combination. Called by multiprocessing."""
    variant_name, threshold, include_score, asset_class = args

    v = Variant(variant_name, threshold, include_score)
    feat_cols = _features_for_variant(v)

    # Load data
    path = DATA_DIR / f"{asset_class}_samples.parquet"
    if not path.exists():
        return {"variant": variant_name, "asset_class": asset_class, "error": "no data"}

    df = pd.read_parquet(path)
    df = _ensure_asset_class_id(df)
    df = _ensure_style_id(df)

    # Filter by alignment threshold
    df = df[df["alignment_score"] >= threshold].copy()
    if len(df) < 100:
        return {"variant": variant_name, "asset_class": asset_class, "error": f"only {len(df)} samples after threshold"}

    windows = sorted(df["window"].unique())
    n_windows = len(windows)
    if n_windows < 4:
        return {"variant": variant_name, "asset_class": asset_class, "error": f"only {n_windows} windows"}

    # Walk-forward: last N_OOS_WINDOWS as test
    oos_results = []
    for oos_idx in range(n_windows - N_OOS_WINDOWS, n_windows):
        oos_window = windows[oos_idx]
        train_windows = windows[:oos_idx]

        df_train = df[df["window"].isin(train_windows)]
        df_test = df[df["window"] == oos_window]

        if len(df_train) < 50 or len(df_test) < 10:
            continue

        # Ensure all feature columns exist
        for c in feat_cols:
            if c not in df_train.columns:
                df_train[c] = 0.0
                df_test[c] = 0.0

        X_train = df_train[feat_cols].values.astype(np.float32)
        y_train = df_train["label_profitable"].values.astype(int)
        X_test = df_test[feat_cols].values.astype(np.float32)
        y_test = df_test["label_profitable"].values.astype(int)

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Train XGBoost
        # Balance classes via scale_pos_weight
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            scale_pos_weight=spw,
            verbosity=0,
            **XGB_PARAMS,
        )

        # 10% of training as eval set for early stopping
        split_idx = int(len(X_train) * 0.9)
        model.fit(
            X_train[:split_idx], y_train[:split_idx],
            eval_set=[(X_train[split_idx:], y_train[split_idx:])],
            verbose=False,
        )

        # Predict on OOS
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # Simulate trades at confidence threshold 0.55 (matches live)
        result = _simulate_trades(df_test, preds, proba, 0.55, asset_class)
        result["oos_window"] = int(oos_window)
        result["train_samples"] = len(df_train)
        result["test_samples"] = len(df_test)
        result["best_iteration"] = model.best_iteration if hasattr(model, "best_iteration") else -1
        oos_results.append(result)

    # Aggregate across OOS windows
    if not oos_results:
        return {"variant": variant_name, "asset_class": asset_class, "error": "no OOS results"}

    total_trades = sum(r["trades"] for r in oos_results)
    total_wins = sum(r["wins"] for r in oos_results)
    total_pnl = sum(r["total_pnl_pct"] for r in oos_results)
    avg_pf = np.mean([r["pf"] for r in oos_results if r["trades"] > 0]) if total_trades > 0 else 0.0
    avg_sharpe = np.mean([r["sharpe"] for r in oos_results if r["trades"] > 0]) if total_trades > 0 else 0.0
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0.0

    return {
        "variant": variant_name,
        "asset_class": asset_class,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "wr": round(avg_wr, 1),
        "pf": round(float(avg_pf), 2),
        "sharpe": round(float(avg_sharpe), 2),
        "total_pnl_pct": round(total_pnl, 3),
        "per_window": oos_results,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "comparison.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("XGBoost Entry Filter Variant Comparison")
    log.info("=" * 70)
    log.info("Variants: %d | Asset classes: %s | OOS windows: %d | Workers: %d",
             len(VARIANTS), ASSET_CLASSES, N_OOS_WINDOWS, N_JOBS)

    # Build task list
    tasks = []
    for v in VARIANTS:
        for ac in ASSET_CLASSES:
            tasks.append((v.name, v.threshold, v.include_score, ac))

    log.info("Total tasks: %d (%d variants × %d asset classes)", len(tasks), len(VARIANTS), len(ASSET_CLASSES))

    # Run with multiprocessing + progress bar
    t0 = time.time()
    results = []
    with mp.Pool(N_JOBS) as pool:
        for result in tqdm(pool.imap_unordered(run_variant_asset, tasks),
                           total=len(tasks), desc="Variants", unit="task"):
            results.append(result)
            if "error" not in result:
                log.info("DONE %s / %s: %d trades, WR=%.1f%%, PF=%.2f, Sharpe=%.2f, PnL=%.3f%%",
                         result["variant"], result["asset_class"],
                         result["total_trades"], result["wr"],
                         result["pf"], result["sharpe"], result["total_pnl_pct"])
            else:
                log.warning("SKIP %s / %s: %s", result["variant"], result["asset_class"], result["error"])

    elapsed = time.time() - t0
    log.info("Completed in %.1f minutes", elapsed / 60)

    # ── Summary Table ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("VARIANT COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Variant':<22} {'Asset':<10} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'PnL%':>8}")
    print("-" * 90)

    for v in VARIANTS:
        for ac in ASSET_CLASSES:
            r = next((r for r in results if r.get("variant") == v.name and r.get("asset_class") == ac), None)
            if r and "error" not in r:
                print(f"{v.name:<22} {ac:<10} {r['total_trades']:>7} {r['wr']:>5.1f}% {r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['total_pnl_pct']:>+7.3f}%")
            elif r:
                print(f"{v.name:<22} {ac:<10} {'ERROR: ' + r['error']}")
        print()

    # ── Per-window breakdown for best variant ─────────────────────────
    print("\n" + "=" * 90)
    print("PER-WINDOW BREAKDOWN (anti-overfitting check: consistent across windows?)")
    print("=" * 90)

    for v in VARIANTS:
        print(f"\n--- {v.name} ---")
        for ac in ASSET_CLASSES:
            r = next((r for r in results if r.get("variant") == v.name and r.get("asset_class") == ac), None)
            if r and "per_window" in r:
                print(f"  {ac}:")
                for pw in r["per_window"]:
                    print(f"    W{pw['oos_window']:>2}: {pw['trades']:>5} trades, WR={pw['wr']:>5.1f}%, "
                          f"PF={pw['pf']:>6.2f}, Sharpe={pw['sharpe']:>6.2f}, PnL={pw['total_pnl_pct']:>+7.3f}%")

    # Save results
    results_clean = [r for r in results if "error" not in r]
    if results_clean:
        df_out = pd.DataFrame([{
            "variant": r["variant"],
            "asset_class": r["asset_class"],
            "trades": r["total_trades"],
            "wins": r["total_wins"],
            "wr": r["wr"],
            "pf": r["pf"],
            "sharpe": r["sharpe"],
            "pnl_pct": r["total_pnl_pct"],
        } for r in results_clean])
        csv_path = OUT_DIR / "comparison_results.csv"
        df_out.to_csv(csv_path, index=False)
        log.info("Results saved to %s", csv_path)

    print("\n✓ Done. Log: %s" % log_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
