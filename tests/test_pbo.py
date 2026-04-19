"""Tests for backtest/pbo.py — Probability of Backtest Overfitting.

Reference
---------
Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014).
"The Probability of Backtest Overfitting". SSRN.

Algorithm summary
-----------------
Given an N-trials × T-windows performance matrix M:
  1. Enumerate (or sample) S splits of the T columns into equal-size
     in-sample (IS) and out-of-sample (OOS) halves.
  2. For each split s, n*(s) = argmax over trials of mean(M[n, IS_s]).
  3. omega_s = rank of trial n*(s) in OOS (normalized to (0, 1)).
  4. lambda_s = logit(omega_s) = log(omega_s / (1 - omega_s)).
  5. PBO = fraction of splits where omega_s < 0.5 (equivalently lambda_s < 0).

PBO → 0: best-IS consistently best-OOS (no overfit).
PBO → 1: best-IS consistently worst-OOS (pure overfit).
PBO ≈ 0.5: random; backtest tells you nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.pbo import compute_pbo


class TestPboBasic:
    """Fundamental invariants."""

    def test_pbo_zero_when_ranks_identical_across_windows(self):
        """When every window has the same trial order, IS-best = OOS-best → PBO = 0."""
        # 5 trials × 6 windows; trial 0 always best, trial 4 always worst.
        perf = np.tile(np.array([[5.0, 4.0, 3.0, 2.0, 1.0]]).T, (1, 6))
        result = compute_pbo(perf, n_splits=10, seed=42)
        assert result["pbo"] == 0.0

    def test_pbo_one_when_is_and_oos_are_reversed(self):
        """When IS-best is OOS-worst on every split, PBO → 1."""
        # Construct so that the first 3 cols (any "IS") rank trial 0 highest,
        # and the last 3 cols rank trial 0 lowest.
        is_half = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]]).T  # trial 0 best
        oos_half = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).T  # trial 0 worst
        perf = np.hstack([np.tile(is_half, (1, 3)), np.tile(oos_half, (1, 3))])
        # With a fixed split [0,1,2] IS / [3,4,5] OOS every trial-0 is IS-best
        # but OOS-worst → omega = 1/(N+1) → lambda < 0 → PBO = 1.
        result = compute_pbo(perf, n_splits=1, seed=42, fixed_split=(0, 1, 2))
        assert result["pbo"] == 1.0

    def test_pbo_accepts_dataframe_input(self):
        """DataFrame input is coerced to ndarray transparently."""
        rng = np.random.default_rng(0)
        perf_df = pd.DataFrame(rng.normal(size=(20, 8)))
        result = compute_pbo(perf_df, n_splits=8, seed=42)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_output_keys(self):
        perf = np.random.default_rng(0).normal(size=(10, 6))
        result = compute_pbo(perf, n_splits=4, seed=42)
        for key in ("pbo", "n_splits_used", "n_trials", "n_windows",
                    "lambda_median", "logit_values"):
            assert key in result, f"missing key: {key}"

    def test_pbo_in_unit_interval(self):
        rng = np.random.default_rng(1)
        perf = rng.normal(size=(30, 10))
        result = compute_pbo(perf, n_splits=20, seed=42)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_n_windows_reported_correctly(self):
        perf = np.zeros((5, 8))
        result = compute_pbo(perf, n_splits=4, seed=42)
        assert result["n_windows"] == 8
        assert result["n_trials"] == 5


class TestPboSplits:
    """Split enumeration behaviour."""

    def test_rejects_odd_window_count_without_explicit_k(self):
        perf = np.random.default_rng(0).normal(size=(5, 7))
        with pytest.raises(ValueError, match="even"):
            compute_pbo(perf, n_splits=4, seed=42)

    def test_rejects_n_splits_zero(self):
        perf = np.zeros((5, 6))
        with pytest.raises(ValueError, match="n_splits"):
            compute_pbo(perf, n_splits=0, seed=42)

    def test_rejects_too_few_trials(self):
        perf = np.zeros((1, 6))
        with pytest.raises(ValueError, match="trials"):
            compute_pbo(perf, n_splits=4, seed=42)

    def test_rejects_too_few_windows(self):
        perf = np.zeros((5, 2))
        with pytest.raises(ValueError, match="windows"):
            compute_pbo(perf, n_splits=4, seed=42)

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(7)
        perf = rng.normal(size=(15, 8))
        r1 = compute_pbo(perf, n_splits=16, seed=123)
        r2 = compute_pbo(perf, n_splits=16, seed=123)
        assert r1["pbo"] == r2["pbo"]
        assert np.array_equal(r1["logit_values"], r2["logit_values"])

    def test_caps_n_splits_at_combinatorial_limit(self):
        """For T=4 → C(4,2)=6 possible splits. Asking for 100 caps at 6."""
        perf = np.random.default_rng(0).normal(size=(10, 4))
        result = compute_pbo(perf, n_splits=100, seed=42)
        assert result["n_splits_used"] <= 6

    def test_fixed_split_is_honoured(self):
        """Explicit fixed_split overrides random sampling."""
        perf = np.arange(24, dtype=float).reshape(4, 6)
        result = compute_pbo(perf, n_splits=1, seed=42, fixed_split=(0, 1, 2))
        assert result["n_splits_used"] == 1


class TestPboLogit:
    """Logit transform edge cases."""

    def test_logit_handles_boundary_omega(self):
        """Omega at 0 or 1 would give ±inf; implementation must clamp."""
        # Perfect agreement → omega = 1 for every split. Without clamp
        # logit = +inf and median blows up. Compute PBO anyway.
        perf = np.tile(np.array([[5.0, 4.0, 3.0, 2.0, 1.0]]).T, (1, 6))
        result = compute_pbo(perf, n_splits=5, seed=42)
        assert np.isfinite(result["lambda_median"])

    def test_logit_values_length_matches_splits(self):
        perf = np.random.default_rng(0).normal(size=(10, 8))
        result = compute_pbo(perf, n_splits=12, seed=42)
        assert len(result["logit_values"]) == result["n_splits_used"]


class TestPboKnownMatrix:
    """Hand-computable cases."""

    def test_monotone_performance_gives_zero_pbo(self):
        """Trial i has score (i+1) * window_i_factor with positive factors."""
        n_trials, n_windows = 6, 8
        factors = np.linspace(1.0, 2.0, n_windows)
        perf = np.outer(np.arange(1, n_trials + 1), factors)
        # Best trial in any IS subset is trial 5; best OOS too → PBO = 0.
        result = compute_pbo(perf, n_splits=16, seed=42)
        assert result["pbo"] == 0.0
