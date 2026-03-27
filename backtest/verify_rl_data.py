"""
═══════════════════════════════════════════════════════════════════
 backtest/verify_rl_data.py
 ─────────────────────────
 Verify RL training data integrity before training.
 GATE: All checks must pass before any model training proceeds.

 Usage:
     python3 -m backtest.verify_rl_data
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/rl_training")

ALL_CLASSES = ["crypto", "forex", "stocks", "commodities"]

EXPECTED_LABEL_COLS = [
    "label_action", "label_outcome", "label_profitable",
    "label_rr", "label_exit_mechanism", "label_exit_bar",
    "label_max_favorable_rr", "label_tp_rr", "label_cost_rr",
]

EXPECTED_FEATURE_COLS = [
    "struct_1d", "struct_4h", "struct_1h", "struct_15m", "struct_5m",
    "decay_1d", "decay_4h", "decay_1h", "decay_15m", "decay_5m",
    "bias_strong", "daily_bias", "premium_discount",
    "h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
    "has_entry_zone", "precision_trigger", "volume_ok",
    "ema20_dist_5m", "ema50_dist_5m", "ema20_dist_1h", "ema50_dist_1h",
    "atr_5m_norm", "atr_1h_norm", "atr_daily_norm",
    "rsi_5m", "rsi_1h", "volume_ratio", "adx_1h",
    "alignment_score", "hour_sin", "hour_cos",
    "fvg_bull_active", "fvg_bear_active",
    "ob_bull_active", "ob_bear_active",
    "liq_above_count", "liq_below_count",
]

META_COLS = ["timestamp", "symbol", "asset_class", "window"]

# Expected commission rates per class (for cost_rr sanity checks)
EXPECTED_COMMISSION = {
    "crypto": 0.0004, "forex": 0.0003, "stocks": 0.0001, "commodities": 0.0003,
}


class VerificationResult:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []  # (name, passed, detail)

    def check(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.checks.append((name, passed, detail))
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def all_passed(self) -> bool:
        return all(p for _, p, _ in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for _, p, _ in self.checks if p)

    @property
    def n_failed(self) -> int:
        return sum(1 for _, p, _ in self.checks if not p)


def verify_schema(df: pd.DataFrame, asset_class: str, result: VerificationResult) -> None:
    """Check that all expected columns exist."""
    expected = set(EXPECTED_FEATURE_COLS + EXPECTED_LABEL_COLS + META_COLS)
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected

    result.check(
        f"{asset_class}: all expected columns present",
        len(missing) == 0,
        f"missing: {missing}" if missing else f"{len(actual)} columns",
    )
    if extra:
        print(f"    (extra columns: {extra})")


def verify_no_nan_inf(df: pd.DataFrame, asset_class: str, result: VerificationResult) -> None:
    """Check for NaN/inf in label columns."""
    entries = df[df["label_action"] > 0]
    for col in EXPECTED_LABEL_COLS:
        if col not in entries.columns:
            continue
        n_nan = int(entries[col].isna().sum())
        n_inf = int(np.isinf(entries[col].astype(float)).sum()) if entries[col].dtype in (np.float32, np.float64) else 0
        result.check(
            f"{asset_class}: {col} no NaN/inf",
            n_nan == 0 and n_inf == 0,
            f"{n_nan} NaN, {n_inf} inf in {len(entries)} entries" if (n_nan > 0 or n_inf > 0) else "",
        )


def verify_fee_inclusion(df: pd.DataFrame, asset_class: str, result: VerificationResult) -> None:
    """Verify label_cost_rr > 0 for all entry bars (fees are included)."""
    entries = df[df["label_action"] > 0]
    cost_rr = entries["label_cost_rr"]
    n_zero_cost = int((cost_rr == 0).sum())
    n_negative_cost = int((cost_rr < 0).sum())
    mean_cost = float(cost_rr.mean())
    median_cost = float(cost_rr.median())

    result.check(
        f"{asset_class}: label_cost_rr > 0 for all entries",
        n_zero_cost == 0 and n_negative_cost == 0,
        f"{n_zero_cost} zero, {n_negative_cost} negative out of {len(entries)} entries. "
        f"mean={mean_cost:.4f}, median={median_cost:.4f}",
    )


def verify_label_distributions(df: pd.DataFrame, asset_class: str, result: VerificationResult) -> None:
    """Check that label distributions are reasonable."""
    entries = df[df["label_action"] > 0]
    n_entries = len(entries)

    # Win rate should be 30-60%
    n_win = int((entries["label_outcome"] == 1).sum())
    n_loss = int((entries["label_outcome"] == 2).sum())
    n_be = int((entries["label_outcome"] == 3).sum())
    real_trades = n_win + n_loss
    wr = n_win / max(real_trades, 1)
    # WR = wins/(wins+losses) can be high when most exits are timeouts (forex)
    # Use label_profitable rate (wins/all) as secondary check
    profitable_rate = n_win / max(n_entries, 1)
    result.check(
        f"{asset_class}: win rate realistic (WR 15-85%, profitable rate 15-70%)",
        (0.15 <= wr <= 0.85) and (0.15 <= profitable_rate <= 0.70),
        f"WR={wr:.1%} ({n_win}W/{n_loss}L/{n_be}BE), profitable_rate={profitable_rate:.1%}",
    )

    # label_rr mean should be slightly negative or near zero after fees
    rr_mean = float(entries["label_rr"].mean())
    rr_std = float(entries["label_rr"].std())
    result.check(
        f"{asset_class}: label_rr mean is realistic (< 2.0)",
        rr_mean < 2.0,
        f"mean={rr_mean:.3f}, std={rr_std:.3f}",
    )

    # label_max_favorable_rr >= 0 always
    mfe = entries["label_max_favorable_rr"]
    n_neg_mfe = int((mfe < 0).sum())
    result.check(
        f"{asset_class}: label_max_favorable_rr >= 0",
        n_neg_mfe == 0,
        f"{n_neg_mfe} negative values" if n_neg_mfe > 0 else f"min={float(mfe.min()):.3f}",
    )

    # label_exit_mechanism has all 4 exit types (1=TP, 2=SL, 3=BE, 4=timeout)
    exit_mechs = set(entries["label_exit_mechanism"].unique())
    expected_mechs = {1, 2, 3, 4}
    present_mechs = exit_mechs & expected_mechs
    result.check(
        f"{asset_class}: all exit mechanisms present (TP/SL/BE/timeout)",
        expected_mechs.issubset(exit_mechs),
        f"present: {sorted(present_mechs)}, missing: {sorted(expected_mechs - exit_mechs)}",
    )

    # label_cost_rr varies by asset class (crypto should be highest due to funding)
    cost_mean = float(entries["label_cost_rr"].mean())
    result.check(
        f"{asset_class}: cost_rr mean > 0",
        cost_mean > 0,
        f"mean_cost_rr={cost_mean:.5f}",
    )

    # Print distribution summary
    exit_counts = entries["label_exit_mechanism"].value_counts().to_dict()
    exit_names = {0: "none", 1: "TP", 2: "SL", 3: "BE", 4: "timeout"}
    exit_str = ", ".join(f"{exit_names.get(k, k)}={v}" for k, v in sorted(exit_counts.items()))
    print(f"    Exit distribution: {exit_str}")
    print(f"    RR: mean={rr_mean:.3f}, median={float(entries['label_rr'].median()):.3f}, "
          f"p5={float(entries['label_rr'].quantile(0.05)):.3f}, p95={float(entries['label_rr'].quantile(0.95)):.3f}")
    print(f"    MFE: mean={float(mfe.mean()):.3f}, median={float(mfe.median()):.3f}, "
          f"max={float(mfe.max()):.3f}")


def verify_window_coverage(df: pd.DataFrame, asset_class: str, result: VerificationResult) -> None:
    """Check all 12 windows (W0-W11) are present."""
    windows = sorted(df["window"].unique())
    expected = list(range(12))
    result.check(
        f"{asset_class}: all 12 windows present",
        windows == expected,
        f"found: {windows}" if windows != expected else "",
    )

    # Print per-window entry counts
    entries = df[df["label_action"] > 0]
    per_window = entries.groupby("window").size()
    counts_str = ", ".join(f"W{w}={c}" for w, c in per_window.items())
    print(f"    Entries per window: {counts_str}")


def verify_row_counts(df: pd.DataFrame, asset_class: str, result: VerificationResult) -> None:
    """Check row counts are reasonable."""
    n_total = len(df)
    n_entries = int((df["label_action"] > 0).sum())
    n_symbols = df["symbol"].nunique()

    expected_symbols = {"crypto": 30, "forex": 28, "stocks": 50, "commodities": 4}
    exp_sym = expected_symbols.get(asset_class, 0)

    result.check(
        f"{asset_class}: symbol count matches expected ({exp_sym})",
        n_symbols == exp_sym,
        f"found {n_symbols} symbols" if n_symbols != exp_sym else "",
    )

    result.check(
        f"{asset_class}: has substantial entry count",
        n_entries > 1000,
        f"{n_entries} entries from {n_total} total rows across {n_symbols} symbols",
    )


def verify_feature_ranges(df: pd.DataFrame, asset_class: str, result: VerificationResult) -> None:
    """Spot-check feature value ranges for sanity."""
    entries = df[df["label_action"] > 0]

    # alignment_score should be in [0, 1]
    if "alignment_score" in entries.columns:
        a_min = float(entries["alignment_score"].min())
        a_max = float(entries["alignment_score"].max())
        result.check(
            f"{asset_class}: alignment_score in [0, 1]",
            a_min >= -0.01 and a_max <= 1.01,
            f"range=[{a_min:.3f}, {a_max:.3f}]",
        )

    # RSI should be in [0, 1] (normalized)
    for rsi_col in ["rsi_5m", "rsi_1h"]:
        if rsi_col in entries.columns:
            r_min = float(entries[rsi_col].min())
            r_max = float(entries[rsi_col].max())
            result.check(
                f"{asset_class}: {rsi_col} in [0, 1]",
                r_min >= -0.01 and r_max <= 1.01,
                f"range=[{r_min:.3f}, {r_max:.3f}]",
            )

    # Volume ratio should be positive
    if "volume_ratio" in entries.columns:
        v_min = float(entries["volume_ratio"].min())
        result.check(
            f"{asset_class}: volume_ratio >= 0",
            v_min >= 0,
            f"min={v_min:.3f}",
        )


def verify_cross_class_costs(class_stats: dict[str, float], result: VerificationResult) -> None:
    """Verify that cost_rr varies appropriately across asset classes."""
    if len(class_stats) < 2:
        return

    # Crypto should have highest costs (funding + commission)
    if "crypto" in class_stats and "stocks" in class_stats:
        result.check(
            "cross-class: crypto costs > stocks costs",
            class_stats["crypto"] > class_stats["stocks"],
            f"crypto={class_stats['crypto']:.5f}, stocks={class_stats['stocks']:.5f}",
        )


def main() -> int:
    print("=" * 70)
    print(" RL Training Data Verification")
    print("=" * 70)

    result = VerificationResult()
    cost_means: dict[str, float] = {}
    total_entries = 0

    for ac in ALL_CLASSES:
        path = DATA_DIR / f"{ac}_samples.parquet"
        print(f"\n{'─' * 50}")
        print(f" {ac.upper()} — {path}")
        print(f"{'─' * 50}")

        # Check file exists
        result.check(f"{ac}: parquet file exists", path.exists())
        if not path.exists():
            continue

        # Load
        df = pd.read_parquet(path)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  Loaded: {len(df):,} rows, {len(df.columns)} cols, {size_mb:.1f} MB")

        # Run all checks
        verify_schema(df, ac, result)
        verify_no_nan_inf(df, ac, result)
        verify_fee_inclusion(df, ac, result)
        verify_label_distributions(df, ac, result)
        verify_window_coverage(df, ac, result)
        verify_row_counts(df, ac, result)
        verify_feature_ranges(df, ac, result)

        # Track for cross-class
        entries = df[df["label_action"] > 0]
        cost_means[ac] = float(entries["label_cost_rr"].mean())
        total_entries += len(entries)

        del df

    # Cross-class checks
    print(f"\n{'─' * 50}")
    print(" CROSS-CLASS CHECKS")
    print(f"{'─' * 50}")
    verify_cross_class_costs(cost_means, result)

    result.check(
        "total entries across all classes",
        total_entries > 100_000,
        f"{total_entries:,} entries",
    )

    # Summary
    print(f"\n{'=' * 70}")
    print(f" VERIFICATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed: {result.n_passed}/{len(result.checks)}")
    print(f"  Failed: {result.n_failed}/{len(result.checks)}")

    if result.n_failed > 0:
        print(f"\n  FAILED CHECKS:")
        for name, passed, detail in result.checks:
            if not passed:
                print(f"    ✗ {name}: {detail}")

    if result.all_passed:
        print(f"\n  ✓ ALL CHECKS PASSED — data is ready for training")
        return 0
    else:
        print(f"\n  ✗ {result.n_failed} CHECK(S) FAILED — fix data before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
