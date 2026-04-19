"""Tests for core.metrics — Sharpe + Deflated-Sharpe Ratio.

Coverage targets:
    - equity_curve: construction, timestamp ordering, empty input
    - daily_returns: resample + pct_change semantics
    - sharpe_daily: annualization factor, zero-std guard, empty
    - expected_max_sharpe_null: monotonicity in n_trials and variance
    - deflated_sharpe_ratio: edge cases + Bailey-LdP monotonicity properties
    - trial_sharpe_variance: ddof=1 and NaN-skip
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.metrics import (
    BARS_PER_YEAR_CRYPTO,
    EULER_MASCHERONI,
    daily_returns,
    deflated_sharpe_ratio,
    equity_curve,
    expected_max_sharpe_null,
    return_moments,
    sharpe_daily,
    trial_sharpe_variance,
)


def _trades(pnls, start="2024-01-01", freq="1D") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(pnls), freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": idx, "pnl": pnls})


# ─────────────────────────────────────────────────────────────────
#  equity_curve
# ─────────────────────────────────────────────────────────────────

class TestEquityCurve:
    def test_cumulative_sum_is_correct(self):
        trades = _trades([100, -50, 200, -25])
        curve = equity_curve(trades, account_size=10_000)
        assert list(curve.values) == [10_100, 10_050, 10_250, 10_225]

    def test_sorts_by_timestamp(self):
        idx = pd.to_datetime(
            ["2024-01-03", "2024-01-01", "2024-01-02"], utc=True
        )
        trades = pd.DataFrame({"timestamp": idx, "pnl": [30.0, 10.0, 20.0]})
        curve = equity_curve(trades, account_size=1_000)
        # Sorted → pnls apply in order 10, 20, 30
        assert list(curve.values) == [1_010, 1_030, 1_060]

    def test_empty_trades_returns_empty(self):
        empty = pd.DataFrame({"timestamp": [], "pnl": []})
        curve = equity_curve(empty, account_size=1_000)
        assert curve.empty

    def test_exit_time_column_preferred(self):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)
        trades = pd.DataFrame({"exit_time": idx, "pnl": [50, 100]})
        curve = equity_curve(trades, account_size=1_000)
        assert list(curve.values) == [1_050, 1_150]

    def test_missing_column_raises(self):
        bad = pd.DataFrame({"timestamp": ["2024-01-01"], "wrong": [1]})
        with pytest.raises(ValueError, match="pnl"):
            equity_curve(bad, account_size=1_000)


# ─────────────────────────────────────────────────────────────────
#  daily_returns
# ─────────────────────────────────────────────────────────────────

class TestDailyReturns:
    def test_single_trade_yields_empty(self):
        trades = _trades([100])
        ret = daily_returns(trades, account_size=10_000)
        assert ret.empty  # Only one day, no pct_change possible

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame({"timestamp": [], "pnl": []})
        assert daily_returns(empty, account_size=10_000).empty

    def test_positive_pnl_yields_positive_returns(self):
        trades = _trades([100, 100, 100])
        ret = daily_returns(trades, account_size=10_000)
        assert (ret > 0).all()


# ─────────────────────────────────────────────────────────────────
#  sharpe_daily
# ─────────────────────────────────────────────────────────────────

class TestSharpeDaily:
    def test_empty_returns_zero(self):
        empty = pd.DataFrame({"timestamp": [], "pnl": []})
        assert sharpe_daily(empty, account_size=10_000) == 0.0

    def test_single_trade_returns_zero(self):
        # daily_returns has <2 entries → 0.0
        trades = _trades([100])
        assert sharpe_daily(trades, account_size=10_000) == 0.0

    def test_two_equal_returns_zero_std_zero_sharpe(self):
        # Two equal equity levels → pct_change has single 0 → < 2 obs
        trades = _trades([100, 0])
        # Equity: 10100, 10100 → pct_change: [0] → len 1 < 2 → 0.0
        assert sharpe_daily(trades, account_size=10_000) == 0.0

    def test_sharpe_positive_for_uptrend(self):
        trades = _trades([100, 80, 120, 60, 140, 40, 160, 50, 180, 30])
        sr = sharpe_daily(trades, account_size=10_000)
        assert sr > 0.0

    def test_annualization_uses_crypto_365(self):
        """Manual Sharpe = mean(r)/std(r) * sqrt(365) for crypto default."""
        # Construct trades with predictable returns
        trades = _trades([100, 50, -20, 80, -10, 90, 30, -5, 70, 40])
        returns = daily_returns(trades, account_size=10_000)
        expected = (
            returns.mean() / returns.std(ddof=1)
        ) * math.sqrt(BARS_PER_YEAR_CRYPTO)
        got = sharpe_daily(trades, account_size=10_000)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_stocks_annualization_lower(self):
        """Same returns, stocks periods-per-year (252) → lower Sharpe than crypto (365)."""
        trades = _trades([100, 50, -20, 80, -10, 90, 30])
        crypto = sharpe_daily(
            trades, account_size=10_000, periods_per_year=365
        )
        stocks = sharpe_daily(
            trades, account_size=10_000, periods_per_year=252
        )
        # sqrt(365)/sqrt(252) ≈ 1.203
        assert crypto == pytest.approx(stocks * math.sqrt(365 / 252), rel=1e-9)

    def test_risk_free_rate_reduces_sharpe(self):
        trades = _trades([50, 60, 45, 55, 40, 65])
        sr_zero = sharpe_daily(trades, account_size=10_000, risk_free_rate=0.0)
        sr_high = sharpe_daily(trades, account_size=10_000, risk_free_rate=0.20)
        assert sr_zero > sr_high


# ─────────────────────────────────────────────────────────────────
#  expected_max_sharpe_null
# ─────────────────────────────────────────────────────────────────

class TestExpectedMaxSharpe:
    def test_single_trial_returns_zero(self):
        assert expected_max_sharpe_null(sharpe_variance=0.1, n_trials=1) == 0.0

    def test_zero_variance_returns_zero(self):
        assert expected_max_sharpe_null(sharpe_variance=0.0, n_trials=30) == 0.0

    def test_monotonic_in_n_trials(self):
        """More trials → higher expected max-SR under null."""
        low = expected_max_sharpe_null(sharpe_variance=0.1, n_trials=5)
        mid = expected_max_sharpe_null(sharpe_variance=0.1, n_trials=50)
        high = expected_max_sharpe_null(sharpe_variance=0.1, n_trials=500)
        assert low < mid < high

    def test_monotonic_in_variance(self):
        """Higher Sharpe variance across trials → higher E[max]."""
        low = expected_max_sharpe_null(sharpe_variance=0.01, n_trials=30)
        high = expected_max_sharpe_null(sharpe_variance=1.0, n_trials=30)
        assert low < high

    def test_matches_formula(self):
        """Bailey-LdP Eq. 5: sqrt(V) × [(1-γ)×Φ⁻¹(1-1/N) + γ×Φ⁻¹(1-1/(N·e))]."""
        from scipy.stats import norm
        n, var = 30, 0.1
        expected = math.sqrt(var) * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n)
            + EULER_MASCHERONI * norm.ppf(1 - 1 / (n * math.e))
        )
        assert expected_max_sharpe_null(var, n) == pytest.approx(expected, rel=1e-12)


# ─────────────────────────────────────────────────────────────────
#  deflated_sharpe_ratio
# ─────────────────────────────────────────────────────────────────

class TestDSR:
    def test_single_trial_returns_zero(self):
        assert deflated_sharpe_ratio(
            observed_sharpe=2.0, sharpe_variance=0.1,
            n_trials=1, observation_count=252,
        ) == 0.0

    def test_zero_observations_returns_zero(self):
        assert deflated_sharpe_ratio(
            observed_sharpe=2.0, sharpe_variance=0.1,
            n_trials=30, observation_count=1,
        ) == 0.0

    def test_zero_variance_returns_zero(self):
        assert deflated_sharpe_ratio(
            observed_sharpe=2.0, sharpe_variance=0.0,
            n_trials=30, observation_count=252,
        ) == 0.0

    def test_high_sr_many_obs_yields_high_dsr(self):
        """SR 3.0 over 252 days with few trials and low null-variance → DSR ≈ 1."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=3.0, sharpe_variance=0.1,
            n_trials=10, observation_count=252,
            skewness=0.0, kurtosis=3.0,
        )
        assert dsr > 0.99

    def test_low_sr_low_dsr(self):
        """SR 0.3 with many trials → almost certainly noise."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.3, sharpe_variance=0.5,
            n_trials=100, observation_count=100,
            skewness=0.0, kurtosis=3.0,
        )
        assert dsr < 0.5

    def test_dsr_monotonic_decreasing_in_trials(self):
        """Same SR, more trials explored → higher E[max_null], lower DSR."""
        shared = dict(
            observed_sharpe=1.5, sharpe_variance=0.2,
            observation_count=252, skewness=0.0, kurtosis=3.0,
        )
        low = deflated_sharpe_ratio(n_trials=5, **shared)
        mid = deflated_sharpe_ratio(n_trials=50, **shared)
        high = deflated_sharpe_ratio(n_trials=500, **shared)
        assert low > mid > high

    def test_dsr_monotonic_increasing_in_observations(self):
        """Same SR, more return observations → tighter estimate → higher DSR."""
        shared = dict(
            observed_sharpe=1.5, sharpe_variance=0.2,
            n_trials=30, skewness=0.0, kurtosis=3.0,
        )
        short = deflated_sharpe_ratio(observation_count=30, **shared)
        medium = deflated_sharpe_ratio(observation_count=252, **shared)
        long_ = deflated_sharpe_ratio(observation_count=2520, **shared)
        assert short < medium < long_

    def test_negative_skew_hurts_dsr(self):
        """Negative skew (rare big losses) penalizes DSR."""
        # Choose params so the DSR lands in the sensitive norm.cdf range
        # (not saturated at 1.0) — SR just above E[max_null], moderate obs.
        shared = dict(
            observed_sharpe=1.5, sharpe_variance=0.3,
            n_trials=30, observation_count=60, kurtosis=3.0,
        )
        symmetric = deflated_sharpe_ratio(skewness=0.0, **shared)
        negative = deflated_sharpe_ratio(skewness=-1.5, **shared)
        # Sanity: neither should saturate to 1.0
        assert symmetric < 0.999, f"symmetric DSR saturated: {symmetric}"
        assert symmetric > negative, f"symmetric={symmetric}, negative={negative}"

    def test_fat_tails_hurt_dsr(self):
        """Higher kurtosis → more tail risk → lower DSR."""
        shared = dict(
            observed_sharpe=2.0, sharpe_variance=0.1,
            n_trials=30, observation_count=252, skewness=0.0,
        )
        normal = deflated_sharpe_ratio(kurtosis=3.0, **shared)
        fat = deflated_sharpe_ratio(kurtosis=10.0, **shared)
        assert normal > fat

    def test_dsr_output_in_valid_probability_range(self):
        """DSR ∈ [0, 1] regardless of inputs."""
        for sr in (-2.0, 0.0, 0.5, 2.0, 10.0):
            for n in (5, 30, 500):
                for obs in (30, 252, 1000):
                    dsr = deflated_sharpe_ratio(
                        observed_sharpe=sr, sharpe_variance=0.2,
                        n_trials=n, observation_count=obs,
                        skewness=0.0, kurtosis=3.0,
                    )
                    assert 0.0 <= dsr <= 1.0, (sr, n, obs, dsr)


# ─────────────────────────────────────────────────────────────────
#  trial_sharpe_variance
# ─────────────────────────────────────────────────────────────────

class TestTrialSharpeVariance:
    def test_too_few_returns_zero(self):
        assert trial_sharpe_variance([]) == 0.0
        assert trial_sharpe_variance([1.5]) == 0.0

    def test_uses_ddof_1(self):
        vals = [1.0, 2.0, 3.0]
        got = trial_sharpe_variance(vals)
        assert got == pytest.approx(np.var(vals, ddof=1), rel=1e-12)

    def test_skips_nans(self):
        vals = [1.0, float("nan"), 2.0, float("nan"), 3.0]
        got = trial_sharpe_variance(vals)
        assert got == pytest.approx(np.var([1.0, 2.0, 3.0], ddof=1), rel=1e-12)


# ─────────────────────────────────────────────────────────────────
#  return_moments
# ─────────────────────────────────────────────────────────────────

class TestReturnMoments:
    def test_empty_returns_safe_defaults(self):
        empty = pd.DataFrame({"timestamp": [], "pnl": []})
        n, skew, kurt = return_moments(empty, account_size=10_000)
        assert n == 0 and skew == 0.0 and kurt == 3.0

    def test_normal_distribution_gives_expected_shape(self):
        # Build returns from a normal distribution with enough obs
        rng = np.random.default_rng(42)
        pnls = rng.normal(10, 100, size=300).tolist()
        trades = _trades(pnls)
        n, skew, kurt = return_moments(trades, account_size=100_000)
        assert n >= 250
        # Normal → skew ≈ 0, kurt ≈ 3. With 300 samples ± 0.5 tolerance.
        assert abs(skew) < 1.0
        assert 2.0 < kurt < 5.0
