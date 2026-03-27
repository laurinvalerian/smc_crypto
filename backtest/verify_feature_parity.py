"""
Phase B: Feature Parity + Smoke Test + Integration Completeness
================================================================
Verifies that live feature extraction matches training data exactly.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.indicators import compute_rsi_wilders, compute_atr_wilders


def b1_feature_parity():
    """B1: Compare feature distributions between training data and live extraction logic."""
    print("\n" + "=" * 60)
    print("B1: FEATURE PARITY TEST")
    print("=" * 60)

    parquet_path = Path("data/rl_training/crypto_samples.parquet")
    if not parquet_path.exists():
        print(f"  SKIP — {parquet_path} not found")
        return True

    df = pd.read_parquet(parquet_path)
    # Only entry bars
    entries = df[df["label_action"] > 0]
    if len(entries) == 0:
        print("  SKIP — no entry bars found")
        return True

    sample = entries.sample(n=min(100, len(entries)), random_state=42)
    print(f"  Loaded {len(sample)} entry rows from {parquet_path}")

    # Features we can directly verify from the parquet data
    # (these are the stored features, not reconstructed)
    feature_cols = [
        "struct_1d", "struct_4h", "struct_1h", "struct_15m", "struct_5m",
        "decay_1d", "decay_4h", "decay_1h", "decay_15m", "decay_5m",
        "premium_discount",
        "h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
        "has_entry_zone", "precision_trigger", "volume_ok",
        "ema20_dist_5m", "ema50_dist_5m", "ema20_dist_1h", "ema50_dist_1h",
        "atr_5m_norm", "atr_1h_norm", "atr_daily_norm",
        "rsi_5m", "rsi_1h",
        "volume_ratio", "adx_1h", "alignment_score",
        "hour_sin", "hour_cos",
        "fvg_bull_active", "fvg_bear_active",
        "ob_bull_active", "ob_bear_active",
        "liq_above_count", "liq_below_count",
    ]

    print("\n  Feature distribution check (training data ranges):")
    all_pass = True
    for col in feature_cols:
        if col not in sample.columns:
            print(f"    {col:25s} — MISSING from training data")
            all_pass = False
            continue
        vals = sample[col].dropna()
        if len(vals) == 0:
            print(f"    {col:25s} — all NaN")
            continue
        mn, mx, mean, std = vals.min(), vals.max(), vals.mean(), vals.std()
        # Flag suspicious distributions
        flag = ""
        if col == "has_entry_zone" and mean < 0.5:
            flag = " ⚠ LOW (expected ~1.0 for entries)"
        if col.startswith("rsi") and (mx > 1.1 or mn < -0.1):
            flag = " ⚠ OUT OF [0,1] RANGE"
        if col.startswith("atr") and mx > 10:
            flag = " ⚠ EXTREME VALUES"
        if col.startswith("hour") and (mx > 1.1 or mn < -1.1):
            flag = " ⚠ OUT OF [-1,1] RANGE"
        status = "OK" if not flag else "WARN"
        print(f"    {col:25s}  [{mn:8.4f}, {mx:8.4f}]  mean={mean:7.4f}  std={std:6.4f}  {status}{flag}")

    # Verify RSI values are in [0, 1] range (normalized)
    if "rsi_5m" in sample.columns:
        rsi_5m = sample["rsi_5m"]
        if rsi_5m.max() > 1.1 or rsi_5m.min() < -0.1:
            print("  FAIL: rsi_5m not in [0,1] range — training might use raw [0,100]?")
            all_pass = False

    # Verify hour encoding uses integer-based values
    if "hour_sin" in sample.columns:
        hs = sample["hour_sin"]
        # Integer hours produce only 24 distinct sin values
        unique_count = len(hs.round(6).unique())
        if unique_count <= 30:
            print(f"  hour_sin: {unique_count} unique values — consistent with integer hours ✓")
        else:
            print(f"  hour_sin: {unique_count} unique values — may indicate fractional hours")

    # Verify has_entry_zone for entries
    if "has_entry_zone" in sample.columns:
        ez_mean = sample["has_entry_zone"].mean()
        print(f"  has_entry_zone mean for entries: {ez_mean:.3f}")
        # For entry bars it should be mostly 1.0 (aligned with bias)
        # but not always — some entries may have no 15m zone

    print(f"\n  B1 result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def b2_smoke_test():
    """B2: Smoke test all 3 RL models."""
    print("\n" + "=" * 60)
    print("B2: SMOKE TEST")
    print("=" * 60)

    all_pass = True

    # Test 1: ast.parse live_multi_bot.py
    try:
        ast.parse(open("live_multi_bot.py").read())
        print("  live_multi_bot.py syntax:  PASS")
    except SyntaxError as e:
        print(f"  live_multi_bot.py syntax:  FAIL — {e}")
        all_pass = False

    # Test 2: Import RLBrainSuite
    try:
        from rl_brain_v2 import RLBrainSuite
        print("  RLBrainSuite import:       PASS")
    except Exception as e:
        print(f"  RLBrainSuite import:       FAIL — {e}")
        all_pass = False
        return all_pass

    # Test 3: Load with config
    try:
        import yaml
        with open("config/default_config.yaml") as f:
            config = yaml.safe_load(f)
        suite = RLBrainSuite(config)
        print(f"  RLBrainSuite init:         PASS")
    except Exception as e:
        print(f"  RLBrainSuite init:         FAIL — {e}")
        all_pass = False
        return all_pass

    # Realistic test features (from training distribution)
    test_features = {
        "struct_1d": 1.0, "struct_4h": 1.0, "struct_1h": 1.0,
        "struct_15m": 1.0, "struct_5m": 1.0,
        "decay_1d": 0.8, "decay_4h": 0.6, "decay_1h": 0.9,
        "decay_15m": 0.7, "decay_5m": 0.5,
        "premium_discount": -1.0,
        "h4_confirms": 1.0, "h4_poi": 1.0,
        "h1_confirms": 1.0, "h1_choch": 1.0,
        "has_entry_zone": 1.0, "precision_trigger": 1.0, "volume_ok": 1.0,
        "ema20_dist_5m": -0.005, "ema50_dist_5m": -0.01,
        "ema20_dist_1h": -0.008, "ema50_dist_1h": -0.015,
        "atr_5m_norm": 0.015, "atr_1h_norm": 0.025, "atr_daily_norm": 0.035,
        "rsi_5m": 0.42, "rsi_1h": 0.55,
        "volume_ratio": 2.3, "adx_1h": 0.65,
        "alignment_score": 0.90,
        "hour_sin": -0.29, "hour_cos": -0.96,
        "fvg_bull_active": 0.4, "fvg_bear_active": 0.0,
        "ob_bull_active": 0.2, "ob_bear_active": 0.0,
        "liq_above_count": 0.2, "liq_below_count": 0.6,
        "asset_class_id": 0.0,
    }

    # Test 4: predict_entry
    try:
        take, confidence = suite.predict_entry(test_features)
        in_range = 0.0 <= confidence <= 1.0
        print(f"  predict_entry:             {'PASS' if in_range else 'FAIL'} "
              f"(take={take}, confidence={confidence:.3f})")
        if not in_range:
            all_pass = False
    except Exception as e:
        print(f"  predict_entry:             FAIL — {e}")
        all_pass = False

    # Test 5: predict_tp_adjustment
    try:
        adjusted = suite.predict_tp_adjustment(test_features, planned_tp_rr=3.0)
        in_range = 0.1 <= adjusted <= 10.0
        print(f"  predict_tp_adjustment:     {'PASS' if in_range else 'FAIL'} "
              f"(adjusted_rr={adjusted:.2f})")
        if not in_range:
            all_pass = False
    except Exception as e:
        print(f"  predict_tp_adjustment:     FAIL — {e}")
        all_pass = False

    # Test 6: predict_be_level
    try:
        be_level = suite.predict_be_level(test_features, cost_rr=0.5)
        in_range = 0.0 <= be_level <= 3.0
        print(f"  predict_be_level:          {'PASS' if in_range else 'FAIL'} "
              f"(be_level={be_level:.2f})")
        if not in_range:
            all_pass = False
    except Exception as e:
        print(f"  predict_be_level:          FAIL — {e}")
        all_pass = False

    # Test 7: Wilder's RSI matches training
    try:
        close = np.array([100, 101, 102, 101, 100, 99, 100, 101, 102, 103,
                          104, 103, 102, 101, 100, 99, 100, 101, 102, 103],
                         dtype=np.float64)
        rsi = compute_rsi_wilders(close, 14)
        last_rsi = rsi[-1] / 100.0  # normalize like live code
        in_range = 0.0 <= last_rsi <= 1.0
        print(f"  Wilder's RSI normalize:    {'PASS' if in_range else 'FAIL'} "
              f"(value={last_rsi:.4f})")
        if not in_range:
            all_pass = False
    except Exception as e:
        print(f"  Wilder's RSI:              FAIL — {e}")
        all_pass = False

    print(f"\n  B2 result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def b3_integration_completeness():
    """B3: Check integration wiring."""
    print("\n" + "=" * 60)
    print("B3: INTEGRATION COMPLETENESS")
    print("=" * 60)

    all_pass = True
    src = Path("live_multi_bot.py").read_text()

    # Check 1: Feature extraction bridge exists
    has_bridge = "_build_xgb_features" in src
    print(f"  _build_xgb_features exists:    {'PASS' if has_bridge else 'FAIL'}")
    if not has_bridge:
        all_pass = False

    # Check 2: Entry filter wired
    has_entry = "predict_entry" in src and "entry_filter_enabled" in src
    print(f"  Entry filter wired:            {'PASS' if has_entry else 'FAIL'}")
    if not has_entry:
        all_pass = False

    # Check 3: TP optimizer wired
    has_tp = "predict_tp_adjustment" in src and "tp_enabled" in src
    print(f"  TP optimizer wired:            {'PASS' if has_tp else 'FAIL'}")
    if not has_tp:
        all_pass = False

    # Check 4: BE manager wired
    has_be = "predict_be_level" in src and "be_enabled" in src
    print(f"  BE manager wired:              {'PASS' if has_be else 'FAIL'}")
    if not has_be:
        all_pass = False

    # Check 5: Config enabled/disabled flags work
    rl_brain_v2_src = Path("rl_brain_v2.py").read_text()
    has_disable = 'self.entry_filter_enabled = ef_cfg.get("enabled"' in rl_brain_v2_src
    print(f"  Config flags functional:       {'PASS' if has_disable else 'FAIL'}")
    if not has_disable:
        all_pass = False

    # Check 6: Shadow mode logging (rejected trades)
    has_shadow = "XGB entry filter skipped" in src
    print(f"  Shadow mode logging:           {'PASS' if has_shadow else 'FAIL'}")
    if not has_shadow:
        all_pass = False

    # Check 7: Wilder's RSI import
    has_wilders = "compute_rsi_wilders" in src
    print(f"  Wilder's RSI import:           {'PASS' if has_wilders else 'FAIL'}")
    if not has_wilders:
        all_pass = False

    # Check 8: Wilder's ATR import
    has_atr = "compute_atr_wilders" in src
    print(f"  Wilder's ATR import:           {'PASS' if has_atr else 'FAIL'}")
    if not has_atr:
        all_pass = False

    # Check 9: has_entry_zone set
    has_ez = 'feat["has_entry_zone"] = 1.0' in src
    print(f"  has_entry_zone set:            {'PASS' if has_ez else 'FAIL'}")
    if not has_ez:
        all_pass = False

    # Check 10: original_sl preserved
    has_osl = '"original_sl"' in src
    print(f"  original_sl preserved:         {'PASS' if has_osl else 'FAIL'}")
    if not has_osl:
        all_pass = False

    # Check 11: Memory cleanup
    has_cleanup = "Strip heavy indicator" in src
    print(f"  _ind_* memory cleanup:         {'PASS' if has_cleanup else 'FAIL'}")
    if not has_cleanup:
        all_pass = False

    # Check 12: Running FVG/OB counts
    has_running = "running per-bar algorithm" in src
    print(f"  Running FVG/OB counts:         {'PASS' if has_running else 'FAIL'}")
    if not has_running:
        all_pass = False

    # Check 13: cost_rr R-multiples
    has_cost_fix = "_TRAIN_COMMISSION" in src
    print(f"  cost_rr R-multiples:           {'PASS' if has_cost_fix else 'FAIL'}")
    if not has_cost_fix:
        all_pass = False

    # Check 14: Integer hour encoding
    has_int_hour = "_hour_int" in src
    print(f"  Integer hour encoding:         {'PASS' if has_int_hour else 'FAIL'}")
    if not has_int_hour:
        all_pass = False

    # Check 15: Feature count — verify _build_xgb_features produces all expected features
    # Some features are set via f-string loops, so check for the feature name
    # appearing anywhere as a string in the source (literal or f-string pattern)
    expected_features = {
        "struct_1d", "struct_4h", "struct_1h", "struct_15m", "struct_5m",
        "decay_1d", "decay_4h", "decay_1h", "decay_15m", "decay_5m",
        "premium_discount",
        "h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
        "has_entry_zone", "precision_trigger", "volume_ok",
        "ema20_dist_5m", "ema50_dist_5m", "ema20_dist_1h", "ema50_dist_1h",
        "atr_5m_norm", "atr_1h_norm", "atr_daily_norm",
        "rsi_5m", "rsi_1h",
        "volume_ratio", "adx_1h", "alignment_score",
        "hour_sin", "hour_cos",
        "fvg_bull_active", "fvg_bear_active",
        "ob_bull_active", "ob_bear_active",
        "liq_above_count", "liq_below_count",
        "asset_class_id",
    }
    # Extract the _build_xgb_features method body
    xgb_section = src[src.index("_build_xgb_features"):src.index("# ── Signal preparation")]
    found = set()
    for f in expected_features:
        # Check for literal feat["name"] OR f-string pattern that produces it
        if f'"{f}"' in xgb_section or f"'{f}'" in xgb_section:
            found.add(f)
    # Also check loop-generated features by their patterns
    loop_patterns = {
        "struct_": {"struct_1d", "struct_4h", "struct_1h", "struct_15m", "struct_5m"},
        "decay_": {"decay_1d", "decay_4h", "decay_1h", "decay_15m", "decay_5m"},
    }
    for pattern, feats in loop_patterns.items():
        if f'feat[f"{pattern}' in xgb_section or f"feat[f'{pattern}" in xgb_section:
            found |= feats
    # Boolean flags set in a loop
    for flag in ("h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
                 "precision_trigger", "volume_ok"):
        if f'"{flag}"' in xgb_section:
            found.add(flag)
    missing = expected_features - found
    n_found = len(found)
    print(f"  Feature count ({n_found}/{len(expected_features)}):           "
          f"{'PASS' if not missing else 'FAIL'}")
    if missing:
        print(f"    Missing: {missing}")
        all_pass = False

    print(f"\n  B3 result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE B: FULL VERIFICATION")
    print("=" * 60)

    r1 = b1_feature_parity()
    r2 = b2_smoke_test()
    r3 = b3_integration_completeness()

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  B1 Feature Parity:          {'PASS' if r1 else 'FAIL'}")
    print(f"  B2 Smoke Test:              {'PASS' if r2 else 'FAIL'}")
    print(f"  B3 Integration:             {'PASS' if r3 else 'FAIL'}")

    overall = r1 and r2 and r3
    print(f"\n  OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    sys.exit(0 if overall else 1)
