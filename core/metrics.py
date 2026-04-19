"""Quantitative performance metrics — Bailey & Lopez de Prado standard.

Phase B of .omc/plans/quality-upgrade-plan.md.

Replaces the inflated trade-PnL Sharpe in `backtest/optuna_backtester.compute_metrics`
(which used sqrt(252) on per-trade PnL — wrong for 5m-entry/4h-hold regime).

All functions pure, deterministic, test-covered. No side-effects.

References
----------
- Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality". The Journal of
  Portfolio Management, 40 (5) 94-107.
- Lopez de Prado (2018): "Advances in Financial Machine Learning", Chs. 7 & 14.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI: float = 0.5772156649015329

BARS_PER_YEAR_CRYPTO: int = 365
BARS_PER_YEAR_FX: int = 260
BARS_PER_YEAR_EQUITIES: int = 252


def equity_curve(
    trades_df: pd.DataFrame,
    account_size: float,
) -> pd.Series:
    """Timestamped equity curve from a trades DataFrame.

    Expected columns: either `exit_time` or `timestamp`, and `pnl`.
    Returns equity indexed by UTC timestamp, sorted ascending.
    """
    if trades_df.empty:
        return pd.Series(dtype=float, name="equity")

    ts_col = "exit_time" if "exit_time" in trades_df.columns else "timestamp"
    if ts_col not in trades_df.columns or "pnl" not in trades_df.columns:
        raise ValueError(
            f"equity_curve expects columns '{ts_col}' and 'pnl' — got {list(trades_df.columns)}"
        )

    ordered = trades_df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(ordered[ts_col], utc=True)
    equity = account_size + ordered["pnl"].cumsum().to_numpy()
    return pd.Series(equity, index=ts, name="equity")


def daily_returns(
    trades_df: pd.DataFrame,
    account_size: float,
) -> pd.Series:
    """Equity-curve resampled to 1D, then pct-change.

    Days without trades keep the previous day's equity (ffill); they
    contribute a zero-return observation to the annualized Sharpe.
    """
    curve = equity_curve(trades_df, account_size)
    if curve.empty:
        return pd.Series(dtype=float)
    daily = curve.resample("1D").last().ffill()
    return daily.pct_change().dropna()


def sharpe_daily(
    trades_df: pd.DataFrame,
    account_size: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = BARS_PER_YEAR_CRYPTO,
) -> float:
    """Daily-frequency Sharpe, annualized.

    This is the correct way to annualize for 5m-entry / 4h-hold strategies:
    aggregate trade PnL into daily equity returns, then scale by sqrt(N_days).

    `risk_free_rate` is an annual percentage (e.g. 0.05 for 5% p.a.).
    """
    returns = daily_returns(trades_df, account_size)
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def return_moments(
    trades_df: pd.DataFrame,
    account_size: float,
) -> tuple[int, float, float]:
    """Return `(N_obs, skewness, kurtosis)` of the daily-return series.

    `kurtosis` is *non-excess* (Normal distribution = 3.0). DSR expects this
    convention; `pd.Series.kurtosis()` returns excess kurtosis so we add 3.
    """
    returns = daily_returns(trades_df, account_size)
    if len(returns) < 4:
        return len(returns), 0.0, 3.0
    skew = float(returns.skew())
    kurt = float(returns.kurtosis()) + 3.0
    return len(returns), skew, kurt


def expected_max_sharpe_null(
    sharpe_variance: float,
    n_trials: int,
) -> float:
    """Expected maximum Sharpe under the null hypothesis (all strategies random).

    Derived from the distribution of the maximum of `n_trials` i.i.d. normal
    Sharpe estimators. See Bailey & Lopez de Prado (2014), Eq. 5.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    sqrt_var = math.sqrt(sharpe_variance)
    term_a = (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    term_b = EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sqrt_var * (term_a + term_b))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_variance: float,
    n_trials: int,
    observation_count: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that `observed_sharpe` reflects genuine edge.

    Returns a value in [0, 1]. Interpret as: confidence that the observed
    Sharpe is *not* a selection-bias / data-mining artifact produced by
    running `n_trials` random strategies. DSR ≥ 0.95 is the conventional
    threshold for "statistically significant after overfit deflation".

    Parameters
    ----------
    observed_sharpe
        Annualized Sharpe of the selected strategy.
    sharpe_variance
        Variance of annualized Sharpes across all evaluated trials (an
        estimate of the null-distribution spread).
    n_trials
        Number of independent strategy configurations tested.
    observation_count
        Number of return observations the Sharpe was estimated from
        (e.g. daily returns count).
    skewness
        Third standardized moment of the return distribution.
    kurtosis
        Fourth standardized moment (non-excess; Normal = 3.0).

    Notes
    -----
    Bailey & Lopez de Prado (2014), Eqs. 4–7. For skewness and kurtosis we
    use the non-standardized moment convention γ3, γ4 to match the paper.
    """
    if n_trials < 2 or observation_count < 2 or sharpe_variance <= 0:
        return 0.0

    e_max = expected_max_sharpe_null(sharpe_variance, n_trials)

    # Higher-moment haircut for non-Normality of the return distribution
    denom_sq = (
        1.0
        - skewness * observed_sharpe
        + (kurtosis - 1.0) / 4.0 * observed_sharpe ** 2
    )
    if denom_sq <= 0:
        return 0.0

    z = (
        (observed_sharpe - e_max)
        * math.sqrt(observation_count - 1)
        / math.sqrt(denom_sq)
    )
    return float(norm.cdf(z))


def trial_sharpe_variance(trial_sharpes: Sequence[float]) -> float:
    """Variance of annualized Sharpes across trials (sample variance, ddof=1)."""
    arr = np.asarray([s for s in trial_sharpes if not np.isnan(s)], dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(arr.var(ddof=1))
