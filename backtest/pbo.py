"""Probability of Backtest Overfitting (PBO).

Phase E of `.omc/plans/quality-upgrade-plan.md`.

Reference
---------
Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014).
"The Probability of Backtest Overfitting". SSRN.

Motivation
----------
DSR (Phase C) asks "is this Sharpe real vs. the null?". CPCV (Phase D)
asks "is this Sharpe stable across OOS combinations?". Neither directly
asks "how often does the IS winner lose OOS?" — that is the selection
question PBO answers.

Algorithm
---------
Given an ``N × T`` performance matrix ``M`` (N trials × T windows):

  1. Enumerate (or sample) ``S`` splits of the T columns into equal
     in-sample (IS) and out-of-sample (OOS) halves of size ``T/2``.
  2. For each split ``s``:
     - ``n*(s) = argmax over trials of mean(M[n, IS_s])``.
     - ``omega_s`` = relative OOS rank of ``n*(s)`` in ``(0, 1)``.
     - ``lambda_s = log(omega_s / (1 - omega_s))``.
  3. ``PBO = P[omega_s < 0.5]`` (fraction of splits where IS-best beats
     fewer than half of competitors OOS).

``PBO → 0``: best-IS consistently best-OOS (selection robust).
``PBO → 1``: best-IS consistently worst-OOS (pure overfit).
``PBO ≈ 0.5``: backtest carries no selection information.

Deploy-Gate (plan §2.5): ``PBO < 0.3`` for funded-account readiness.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  Rank helper — numpy-only, tie-safe average rank
# ═══════════════════════════════════════════════════════════════════

def _avg_rank(values: np.ndarray, idx: int) -> float:
    """1-based average rank of ``values[idx]``.

    For ties (equal values), returns the average of tied rank positions
    — matches ``scipy.stats.rankdata(method='average')`` at a single point.
    """
    target = values[idx]
    less = float(np.sum(values < target))
    equal = float(np.sum(values == target))
    return less + (equal + 1.0) / 2.0


# ═══════════════════════════════════════════════════════════════════
#  PBO
# ═══════════════════════════════════════════════════════════════════

def compute_pbo(
    perf_matrix: np.ndarray | pd.DataFrame,
    n_splits: int = 16,
    seed: int = 42,
    fixed_split: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Compute the Probability of Backtest Overfitting.

    Parameters
    ----------
    perf_matrix
        N-trials × T-windows matrix of a performance metric (Sharpe, DSR,
        PnL, …). Higher = better is assumed — pass ``-max_drawdown`` if
        the metric is a cost.
    n_splits
        Number of equal-size IS/OOS splits to evaluate. Capped at
        ``C(T, T/2)`` when the combinatorial universe is smaller.
    seed
        RNG seed for the random sample of splits (ignored if
        ``n_splits >= C(T, T/2)`` — all splits are then enumerated).
    fixed_split
        Optional explicit IS column tuple. If set, PBO is evaluated on
        this single split only. Intended for tests and reproducibility
        probes, not production.

    Returns
    -------
    dict
        ``pbo`` (float in [0, 1]), ``n_splits_used`` (int),
        ``n_trials`` (int), ``n_windows`` (int),
        ``lambda_median`` (float), ``logit_values`` (np.ndarray).
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"perf_matrix must be 2-D, got shape {M.shape}")
    n_trials, n_windows = M.shape

    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1 (got {n_splits})")
    if n_trials < 2:
        raise ValueError(f"need >= 2 trials, got {n_trials}")
    if n_windows < 4:
        raise ValueError(f"need >= 4 windows, got {n_windows}")
    if n_windows % 2 != 0:
        raise ValueError(
            f"n_windows must be even (PBO uses equal IS/OOS halves), "
            f"got {n_windows}"
        )

    k = n_windows // 2
    all_cols = tuple(range(n_windows))

    if fixed_split is not None:
        splits: list[tuple[int, ...]] = [tuple(int(c) for c in fixed_split)]
    else:
        universe = list(combinations(all_cols, k))
        if len(universe) <= n_splits:
            splits = universe
        else:
            rng = np.random.default_rng(seed)
            pick_idx = rng.choice(len(universe), size=n_splits, replace=False)
            splits = [universe[i] for i in pick_idx]

    EPS = 1e-6  # clamp to keep logit finite at omega ∈ {0, 1}
    omegas: list[float] = []
    logits: list[float] = []
    for is_cols in splits:
        is_set = set(is_cols)
        oos_cols = tuple(c for c in all_cols if c not in is_set)

        is_means = M[:, list(is_cols)].mean(axis=1)
        oos_means = M[:, list(oos_cols)].mean(axis=1)

        n_star = int(np.argmax(is_means))
        rank_oos = _avg_rank(oos_means, n_star)
        omega = rank_oos / (n_trials + 1)
        omega = float(np.clip(omega, EPS, 1.0 - EPS))
        omegas.append(omega)
        logits.append(float(np.log(omega / (1.0 - omega))))

    omegas_arr = np.asarray(omegas, dtype=float)
    logits_arr = np.asarray(logits, dtype=float)
    pbo = float((omegas_arr < 0.5).mean())

    return {
        "pbo": pbo,
        "n_splits_used": len(splits),
        "n_trials": int(n_trials),
        "n_windows": int(n_windows),
        "lambda_median": float(np.median(logits_arr)),
        "logit_values": logits_arr,
    }


__all__ = ["compute_pbo"]
