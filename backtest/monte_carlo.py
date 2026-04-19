"""Monte-Carlo tail-risk gate — Conditional VaR (Expected Shortfall) on max-DDs.

Phase G of `.omc/plans/quality-upgrade-plan.md`.

Motivation
----------
The pre-Phase-G ``monte_carlo_check`` returns ``worst_dd_95pct`` = the
5th-percentile max-DD across simulated equity curves — a *Value at Risk*
(VaR). VaR ignores the shape of the tail: two strategies with the same
5th-percentile can have very different expected shortfalls beyond it.

CVaR (Expected Shortfall) is the **mean** max-DD across the worst
``1 − confidence`` fraction of simulated paths. It is:

  - *coherent* (sub-additive; penalises tail concentration),
  - *more conservative* than VaR (CVaR ≤ VaR on a common sample),
  - *insensitive* to quantile choice within the tail.

For a real-money deploy, CVaR is the correct max-DD gate.

Algorithm
---------
1. Build R-multiples from ``trades_df`` (``actual_rr`` or derive from
   ``pnl / (equity_before × risk_pct)``). Clip to keep a single outlier
   from dominating the distribution.
2. For each of ``n_simulations`` runs:
     - shuffle the R-multiple vector,
     - compound equity through the shuffle at the median per-trade risk_pct,
     - record final equity and max-DD of that path.
3. Sort max-DDs ascending (most-negative first). With tail size
   ``k = ceil((1 − confidence) × n)``:
     - ``var_dd  = sorted_dds[k − 1]``        (the VaR cutoff)
     - ``cvar_dd = mean(sorted_dds[:k])``     (Expected Shortfall)

Gate (plan §2.6): ``cvar_dd_95 > −0.20``.

Reference
---------
Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional
value-at-risk." *Journal of Risk*, 2, 21–42.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_MIN_TRADES_FOR_MC: int = 10
_R_CLIP_LOW: float = -2.0
_R_CLIP_HIGH: float = 15.0


# ═══════════════════════════════════════════════════════════════════
#  Internals
# ═══════════════════════════════════════════════════════════════════

def _extract_r_multiples(trades_df: pd.DataFrame) -> np.ndarray:
    """R-multiple per trade: prefer ``actual_rr``, else derive from PnL."""
    if "actual_rr" in trades_df.columns:
        return np.asarray(trades_df["actual_rr"].values, dtype=float).copy()

    required = {"pnl", "risk_pct", "equity"}
    missing = required - set(trades_df.columns)
    if missing:
        raise ValueError(
            f"trades_df missing columns for R-multiple derivation: {missing}"
        )

    pnl = np.asarray(trades_df["pnl"].values, dtype=float)
    risk_pct = np.asarray(trades_df["risk_pct"].values, dtype=float)
    equity_after = np.asarray(trades_df["equity"].values, dtype=float)
    eq_before = equity_after - pnl
    risk_amt = np.where(eq_before > 0, eq_before * risk_pct, 1.0)
    return np.where(risk_amt > 0, pnl / risk_amt, 0.0)


def _empty_result(reason: str, seed: int, confidence: float) -> dict[str, Any]:
    """Default output for degenerate inputs (too few trades, empty df)."""
    return {
        "cvar_dd_95": 0.0,
        "var_dd_95": 0.0,
        "median_dd": 0.0,
        "mean_dd": 0.0,
        "median_final_pnl": 0.0,
        "ci_lower_pnl": 0.0,
        "ci_upper_pnl": 0.0,
        "pct_profitable": 0.0,
        "n_simulations": 0,
        "seed": int(seed),
        "confidence": float(confidence),
        "reason": reason,
    }


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

def compute_mc_cvar_dd(
    trades_df: pd.DataFrame,
    account_size: float = 100_000.0,
    n_simulations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte-Carlo tail-risk gate via R-multiple resampling.

    Returns a dict holding:

    - ``cvar_dd_95``      Expected Shortfall of max-DDs at ``confidence``.
    - ``var_dd_95``       VaR quantile of max-DDs at ``confidence``.
    - ``median_dd``       Median max-DD across simulations.
    - ``mean_dd``         Mean max-DD across simulations.
    - ``median_final_pnl``  Median terminal PnL.
    - ``ci_lower_pnl``    Lower percentile of terminal PnL at ``confidence``.
    - ``ci_upper_pnl``    Upper percentile of terminal PnL at ``confidence``.
    - ``pct_profitable``  Fraction of simulations ending above start.
    - ``n_simulations``   Simulations actually run (0 if short-circuited).
    - ``seed``            RNG seed used.
    - ``confidence``      Tail-width parameter used.
    - ``reason``          ``"ok"`` or a short degenerate-input tag.

    Drawdowns are returned as signed fractions: ``−0.20`` means 20% DD.
    Since DD is a percentage of running-max equity, results are
    near-invariant in ``account_size``.

    Parameters
    ----------
    trades_df
        Trades DataFrame. Must supply either ``actual_rr`` OR the
        triple (``pnl``, ``risk_pct``, ``equity``).
    account_size
        Starting equity.
    n_simulations
        Number of resampled equity curves.
    confidence
        Tail width: ``0.95`` → CVaR/VaR over the worst 5% of paths.
    seed
        RNG seed for reproducibility.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1 (got {n_simulations})")

    if trades_df.empty or len(trades_df) < _MIN_TRADES_FOR_MC:
        return _empty_result(
            "not_enough_trades", seed=seed, confidence=confidence,
        )

    r_multiples = _extract_r_multiples(trades_df)
    r_multiples = np.clip(r_multiples, _R_CLIP_LOW, _R_CLIP_HIGH)

    if "risk_pct" in trades_df.columns:
        median_risk_pct = float(np.median(trades_df["risk_pct"].values))
    else:
        median_risk_pct = 0.01  # 1% fallback

    rng = np.random.default_rng(seed)
    n_trades = len(r_multiples)
    max_dds = np.empty(n_simulations, dtype=float)
    final_equities = np.empty(n_simulations, dtype=float)

    for s in range(n_simulations):
        shuffled = r_multiples[rng.permutation(n_trades)]

        equity = float(account_size)
        curve = np.empty(n_trades, dtype=float)
        for i in range(n_trades):
            risk_amount = equity * median_risk_pct
            equity = max(equity + risk_amount * shuffled[i], 0.0)
            curve[i] = equity
        final_equities[s] = equity

        running_max = np.maximum.accumulate(curve)
        dd = np.where(
            running_max > 0,
            (curve - running_max) / running_max,
            0.0,
        )
        max_dds[s] = float(dd.min())

    # Tail statistics — sort max_dds ascending (most-negative first).
    sorted_dds = np.sort(max_dds)
    tail_size = max(1, int(np.ceil((1.0 - confidence) * n_simulations)))
    cvar_dd = float(sorted_dds[:tail_size].mean())
    var_dd = float(sorted_dds[tail_size - 1])

    final_pnls = final_equities - account_size
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(final_pnls, alpha / 2.0 * 100.0))
    ci_upper = float(np.percentile(final_pnls, (1.0 - alpha / 2.0) * 100.0))

    return {
        "cvar_dd_95": cvar_dd,
        "var_dd_95": var_dd,
        "median_dd": float(np.median(max_dds)),
        "mean_dd": float(np.mean(max_dds)),
        "median_final_pnl": float(np.median(final_pnls)),
        "ci_lower_pnl": ci_lower,
        "ci_upper_pnl": ci_upper,
        "pct_profitable": float(np.mean(final_pnls > 0)),
        "n_simulations": int(n_simulations),
        "seed": int(seed),
        "confidence": float(confidence),
        "reason": "ok",
    }


__all__ = ["compute_mc_cvar_dd"]
