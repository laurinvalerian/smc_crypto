"""
Compare exit model combinations against baseline (VECTORIZED).

Replays OOS trades with different model combos and computes PF, Sharpe, WR,
max DD with bootstrap confidence intervals. All simulations use NumPy batch
operations — no iterrows.

Usage:
    python3 -m backtest.compare_exit_combos
    python3 -m backtest.compare_exit_combos --classes crypto stocks
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("backtest/results/rl")
DATA_DIR = Path("data/rl_training")

OOS_WINDOWS = [10, 11]

COMBOS = {
    0: {"name": "Baseline", "tp": False, "sl": False, "be": False},
    1: {"name": "TP only", "tp": True, "sl": False, "be": False},
    2: {"name": "SL only", "tp": False, "sl": True, "be": False},
    3: {"name": "BE only", "tp": False, "sl": False, "be": True},
    4: {"name": "TP + BE", "tp": True, "sl": False, "be": True},
    5: {"name": "TP + SL", "tp": True, "sl": True, "be": False},
    6: {"name": "SL + BE", "tp": False, "sl": True, "be": True},
    7: {"name": "TP + SL + BE", "tp": True, "sl": True, "be": True},
}


def _load_model(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _build_feature_matrix(df: pd.DataFrame, model_data: dict) -> np.ndarray:
    """Build feature matrix for batch prediction. Returns (n, n_features) array."""
    feat_names = model_data.get("feat_names", [])
    n = len(df)
    X = np.zeros((n, len(feat_names)), dtype=np.float32)
    for i, fn in enumerate(feat_names):
        if fn in df.columns:
            vals = df[fn].values.astype(np.float32)
            mask = np.isfinite(vals)
            X[mask, i] = vals[mask]
    return X


def _batch_predict_tp(model_data: dict, X: np.ndarray, planned_tp: np.ndarray) -> np.ndarray:
    """Batch predict adjusted TP. Returns adjusted_tp array."""
    predicted_mfe = model_data["model"].predict(X).astype(np.float64)
    predicted_mfe = np.maximum(predicted_mfe, 0.5)
    adjusted = planned_tp.copy()
    # Reduce: predicted MFE much lower than planned → take profit earlier
    reduce_mask = predicted_mfe < (planned_tp * 0.7)
    adjusted[reduce_mask] = np.maximum(predicted_mfe[reduce_mask] * 0.85, 0.5)
    # Extend: predicted MFE much higher → let it run
    extend_mask = predicted_mfe > (planned_tp * 1.3)
    adjusted[extend_mask] = np.minimum(predicted_mfe[extend_mask] * 0.8, planned_tp[extend_mask] * 1.5)
    return adjusted


def _batch_predict_sl(model_data: dict, X: np.ndarray) -> np.ndarray:
    """Batch predict SL multiplier. Returns sl_mult array."""
    predicted = model_data["model"].predict(X).astype(np.float64)
    sl_mult = np.ones(len(predicted), dtype=np.float64)
    sl_mult[predicted > 0.75] = 0.5    # tighten significantly
    sl_mult[(predicted > 0.25) & (predicted <= 0.75)] = 0.7  # tighten slightly
    sl_mult[predicted < -0.25] = 1.3   # widen
    return sl_mult


def _batch_predict_be(model_data: dict, X: np.ndarray) -> np.ndarray:
    """Batch predict BE level. Returns be_level array."""
    predicted = model_data["model"].predict(X).astype(np.float64)
    return np.clip(predicted, 0.0, 3.0)


def _simulate_combo_vectorized(
    actual_rr: np.ndarray,
    planned_tp: np.ndarray,
    actual_mfe: np.ndarray,
    eval_mfe: np.ndarray,
    cost_rr: np.ndarray,
    outcome: np.ndarray,
    mae: np.ndarray,
    adjusted_tp: np.ndarray | None,
    sl_mult: np.ndarray | None,
    be_level: np.ndarray | None,
) -> np.ndarray:
    """Vectorized simulation of a combo. Returns adjusted RR array."""
    rr = actual_rr.copy()
    n = len(rr)

    # SL effect
    if sl_mult is not None:
        changed = np.abs(sl_mult - 1.0) > 0.01
        tighten = changed & (sl_mult < 1.0)
        widen = changed & (sl_mult > 1.0)

        # Tighten: stopped out if MAE > new SL distance
        stopped = tighten & (mae > sl_mult)
        rr[stopped] = -sl_mult[stopped] - cost_rr[stopped]
        # Tighten: survived → RR scales inversely (wins only)
        survived_win = tighten & ~stopped & (outcome == 1)
        rr[survived_win] = actual_rr[survived_win] / sl_mult[survived_win]

        # Widen: loss saved if MAE < new wider SL and had good MFE
        widen_saved = widen & (outcome == 2) & (mae < sl_mult) & (actual_mfe >= 1.0)
        rr[widen_saved] = (actual_mfe[widen_saved] * 0.5) / sl_mult[widen_saved] - cost_rr[widen_saved]
        # Widen: win → RR smaller
        widen_win = widen & (outcome == 1)
        rr[widen_win] = actual_rr[widen_win] / sl_mult[widen_win]

    # TP effect
    if adjusted_tp is not None:
        tp_changed = np.abs(adjusted_tp - planned_tp) > 0.01
        # Reduced TP: hit if eval_mfe >= adjusted_tp
        tp_reduced = tp_changed & (adjusted_tp < planned_tp) & (eval_mfe >= adjusted_tp)
        rr[tp_reduced] = adjusted_tp[tp_reduced] - cost_rr[tp_reduced]
        # Extended TP: hit if eval_mfe >= adjusted_tp
        tp_extended = tp_changed & (adjusted_tp > planned_tp) & (eval_mfe >= adjusted_tp)
        rr[tp_extended] = adjusted_tp[tp_extended] - cost_rr[tp_extended]

    # BE effect
    if be_level is not None:
        be_saves = (be_level > 0.1) & (outcome == 2) & (actual_mfe >= be_level)
        rr[be_saves] = -cost_rr[be_saves]

    return np.clip(rr, -2.0, 20.0)


def _calc_metrics(rr_arr: np.ndarray) -> dict:
    if len(rr_arr) == 0:
        return {"pf": 0, "sharpe": 0, "wr": 0, "max_dd": 0, "avg_rr": 0, "n": 0}
    wins = rr_arr[rr_arr > 0].sum()
    losses = abs(rr_arr[rr_arr < 0].sum())
    pf = float(wins / max(losses, 0.001))
    sharpe = float(np.mean(rr_arr) / max(np.std(rr_arr), 1e-6))
    wr = float((rr_arr > 0).mean())
    cumsum = np.cumsum(rr_arr)
    peak = np.maximum.accumulate(cumsum)
    dd = cumsum - peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    avg_rr = float(np.mean(rr_arr))
    return {"pf": pf, "sharpe": sharpe, "wr": wr, "max_dd": max_dd, "avg_rr": avg_rr, "n": len(rr_arr)}


def _bootstrap_pf_ci(rr_arr: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    if len(rr_arr) < 10:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    pfs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(rr_arr, size=len(rr_arr), replace=True)
        w = sample[sample > 0].sum()
        l = abs(sample[sample < 0].sum())
        pfs[i] = w / max(l, 0.001)
    alpha = (1 - ci) / 2
    return (float(np.percentile(pfs, alpha * 100)), float(np.percentile(pfs, (1 - alpha) * 100)))


def run_comparison(classes: list[str] | None = None) -> dict:
    if classes is None:
        classes = ["crypto", "stocks", "forex", "commodities"]

    tp_model = _load_model("models/rl_tp_optimizer.pkl")
    sl_model = _load_model("models/rl_sl_adjuster.pkl")
    be_model = _load_model("models/rl_be_manager.pkl")

    loaded = []
    if tp_model: loaded.append("TP")
    if sl_model: loaded.append("SL")
    if be_model: loaded.append("BE")
    logger.info("Models loaded: %s", ", ".join(loaded) if loaded else "NONE")

    from backtest.generate_rl_data import WINDOWS
    all_entries = []
    for ac in classes:
        path = DATA_DIR / f"{ac}_samples.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        entries = df[df["label_action"] > 0].copy()
        entries["asset_class"] = ac
        for wi in OOS_WINDOWS:
            if wi >= len(WINDOWS):
                continue
            w_start, w_end = WINDOWS[wi]
            mask = (entries["timestamp"] >= w_start) & (entries["timestamp"] < w_end)
            window_entries = entries[mask]
            if len(window_entries) > 0:
                window_entries = window_entries.copy()
                window_entries["oos_window"] = wi
                all_entries.append(window_entries)
                logger.info("  %s W%d: %d OOS entries", ac, wi, len(window_entries))

    if not all_entries:
        logger.error("No OOS entries found!")
        return {}

    oos_df = pd.concat(all_entries, ignore_index=True)
    logger.info("Total OOS entries: %d", len(oos_df))
    del all_entries; gc.collect()

    # Pre-extract arrays (shared across all combos)
    actual_rr = oos_df["label_rr"].values.astype(np.float64)
    planned_tp = oos_df["label_tp_rr"].values.astype(np.float64)
    actual_mfe = oos_df["label_max_favorable_rr"].values.astype(np.float64)
    cost_rr = oos_df["label_cost_rr"].values.astype(np.float64)
    outcome = oos_df["label_outcome"].values.astype(np.int32)
    mae = oos_df["label_mae_rr"].values.astype(np.float64) if "label_mae_rr" in oos_df.columns else np.zeros(len(oos_df), dtype=np.float64)
    post_tp = oos_df["label_post_tp_max_rr"].values.astype(np.float64) if "label_post_tp_max_rr" in oos_df.columns else np.full(len(oos_df), np.nan)
    eval_mfe = np.where(np.isnan(post_tp), actual_mfe, post_tp)
    ac_arr = oos_df["asset_class"].values
    window_arr = oos_df["oos_window"].values

    # Pre-compute batch predictions (once per model, reused across combos)
    logger.info("Pre-computing model predictions...")
    tp_adjusted = None
    if tp_model:
        X_tp = _build_feature_matrix(oos_df, tp_model)
        tp_adjusted = _batch_predict_tp(tp_model, X_tp, planned_tp)
        del X_tp
        logger.info("  TP predictions done")

    sl_mult_arr = None
    if sl_model:
        X_sl = _build_feature_matrix(oos_df, sl_model)
        sl_mult_arr = _batch_predict_sl(sl_model, X_sl)
        del X_sl
        logger.info("  SL predictions done")

    be_level_arr = None
    if be_model:
        X_be = _build_feature_matrix(oos_df, be_model)
        be_level_arr = _batch_predict_be(be_model, X_be)
        del X_be
        logger.info("  BE predictions done")

    gc.collect()

    # Run each combo (vectorized — seconds, not minutes)
    results = {}
    for combo_id, combo in COMBOS.items():
        if combo["tp"] and tp_model is None:
            continue
        if combo["sl"] and sl_model is None:
            continue
        if combo["be"] and be_model is None:
            continue

        logger.info("--- Combo %d: %s ---", combo_id, combo["name"])

        rr_arr = _simulate_combo_vectorized(
            actual_rr, planned_tp, actual_mfe, eval_mfe, cost_rr, outcome, mae,
            adjusted_tp=tp_adjusted if combo["tp"] else None,
            sl_mult=sl_mult_arr if combo["sl"] else None,
            be_level=be_level_arr if combo["be"] else None,
        )

        metrics = _calc_metrics(rr_arr)
        pf_lo, pf_hi = _bootstrap_pf_ci(rr_arr)
        metrics["pf_ci_lo"] = pf_lo
        metrics["pf_ci_hi"] = pf_hi

        # Per-asset-class breakdown
        per_class = {}
        for ac in classes:
            ac_mask = ac_arr == ac
            if ac_mask.sum() > 0:
                per_class[ac] = _calc_metrics(rr_arr[ac_mask])
        per_window = {}
        for wi in OOS_WINDOWS:
            w_mask = window_arr == wi
            if w_mask.sum() > 0:
                per_window[f"W{wi}"] = _calc_metrics(rr_arr[w_mask])

        metrics["per_class"] = per_class
        metrics["per_window"] = per_window
        results[combo_id] = {"combo": combo["name"], **metrics}

        logger.info("  PF=%.2f [%.2f-%.2f] Sharpe=%.3f WR=%.1f%% DD=%.1f AvgRR=%.3f n=%d",
                    metrics["pf"], pf_lo, pf_hi, metrics["sharpe"],
                    metrics["wr"] * 100, metrics["max_dd"], metrics["avg_rr"], metrics["n"])

    # Summary table + gate evaluation
    if 0 in results:
        baseline = results[0]
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON RESULTS vs BASELINE")
        logger.info("=" * 80)
        logger.info("%-15s %8s %14s %8s %8s %8s %10s",
                    "Combo", "PF", "PF 95% CI", "Sharpe", "WR", "MaxDD", "vs Base")
        logger.info("-" * 80)

        for cid, res in sorted(results.items()):
            vs = "---" if cid == 0 else f"{(res['pf'] / max(baseline['pf'], 0.001) - 1) * 100:+.1f}%"
            logger.info("%-15s %8.2f [%5.2f-%5.2f] %8.3f %7.1f%% %8.1f %10s",
                        res["combo"], res["pf"],
                        res.get("pf_ci_lo", 0), res.get("pf_ci_hi", 0),
                        res["sharpe"], res["wr"] * 100, res["max_dd"], vs)

        logger.info("\n" + "=" * 80)
        logger.info("DEPLOY GATES")
        logger.info("=" * 80)
        best_combo = None
        best_score = 0

        for cid, res in sorted(results.items()):
            if cid == 0:
                continue
            pf_sig = res.get("pf_ci_lo", 0) > baseline["pf"]
            sharpe_ok = res["sharpe"] > baseline["sharpe"]
            dd_ok = res["max_dd"] >= baseline["max_dd"] * 1.20

            consistent = True
            for wk, wv in res.get("per_window", {}).items():
                bw = baseline.get("per_window", {}).get(wk, {})
                if bw and wv.get("pf", 0) < bw.get("pf", 0) * 0.9:
                    consistent = False

            passed = pf_sig and sharpe_ok and dd_ok and consistent
            reasons = []
            if not pf_sig: reasons.append(f"PF CI lo {res.get('pf_ci_lo',0):.2f} <= base {baseline['pf']:.2f}")
            if not sharpe_ok: reasons.append(f"Sharpe {res['sharpe']:.3f} <= base {baseline['sharpe']:.3f}")
            if not dd_ok: reasons.append(f"DD {res['max_dd']:.1f} > 120% base {baseline['max_dd']:.1f}")
            if not consistent: reasons.append("Inconsistent across windows")

            logger.info("  [%s] %-15s %s", "PASS" if passed else "FAIL", res["combo"],
                        " | ".join(reasons) if reasons else "All gates passed")

            if passed:
                score = (res["pf"] / max(baseline["pf"], 0.001)) * \
                        (res["sharpe"] / max(baseline["sharpe"], 1e-6)) * \
                        (baseline["max_dd"] / min(res["max_dd"], -0.001))
                if score > best_score:
                    best_score = score
                    best_combo = cid

        if best_combo is not None:
            logger.info("\nRECOMMENDATION: Deploy combo %d (%s) — best score %.2f",
                        best_combo, results[best_combo]["combo"], best_score)
        else:
            logger.info("\nRECOMMENDATION: No combo passed all gates — keep baseline")

    # Per-class detail for best combo
    if best_combo and best_combo in results:
        logger.info("\n" + "=" * 80)
        logger.info("BEST COMBO PER-CLASS BREAKDOWN: %s", results[best_combo]["combo"])
        logger.info("=" * 80)
        for ac, m in results[best_combo].get("per_class", {}).items():
            bm = baseline.get("per_class", {}).get(ac, {})
            b_pf = bm.get("pf", 0)
            logger.info("  %-15s PF=%.2f (base %.2f, %+.1f%%) Sharpe=%.3f WR=%.1f%% DD=%.1f n=%d",
                        ac, m["pf"], b_pf, (m["pf"] / max(b_pf, 0.001) - 1) * 100,
                        m["sharpe"], m["wr"] * 100, m["max_dd"], m["n"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "combo_comparison.json"
    def _j(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_j)
    logger.info("Results saved: %s", out_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=None)
    args = parser.parse_args()
    run_comparison(classes=args.classes)
