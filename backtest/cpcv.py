"""Combinatorial Purged Cross-Validation (CPCV).

Phase D of `.omc/plans/quality-upgrade-plan.md`.

Reference
---------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
Ch. 7 ("Cross-Validation in Finance").

Motivation
----------
A single walk-forward run gives ``N`` out-of-sample evaluations for
``N`` windows. That is a thin variance estimate on which to gate a
real-money deployment. CPCV enumerates every way of choosing
``k`` test windows out of ``N`` total windows:

    N = 12,  k = 2  →  C(12, 2) = 66 OOS evaluations

With *purging* (remove train-side samples whose holding period
overlaps a test window) and an *embargo* (gap to suppress serial
autocorrelation), this yields a defensible distribution of the
metric under OOS conditions — sharper DSR variance estimate, and
an IQR that signals parameter robustness.

Usage
-----
>>> splits = list(cpcv_splits(wf_windows, k=2, embargo_bars=288))
>>> cpcv_df = run_cpcv(
...     params=best_params,
...     windows=wf_windows,
...     signals_per_window={wi: precomputed[wi] for wi in range(len(wf_windows))},
...     simulate_fn=_simulate_with_params,
...     config=cfg,
...     symbol_to_asset=symbol_to_asset,
...     trial_sharpes=study_trial_sharpes,
...     k=2,
... )
>>> summary = cpcv_summary(cpcv_df, funded_threshold=0.95)

Notes on scope
--------------
This implementation evaluates a *single fixed* parameter set across
all CPCV splits — a post-hoc validation of the parameters selected
by the Optuna walk-forward run. Re-optimizing per split is feasible
but ~N× more expensive and is deferred to Phase E (PBO) where rank
stability across sub-matrices is the explicit target.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from core.constants import SCALP_MAX_HOLD_BARS
from core.metrics import (
    deflated_sharpe_ratio as _deflated_sharpe,
    trial_sharpe_variance as _trial_sharpe_variance,
)


# ═══════════════════════════════════════════════════════════════════
#  Split enumeration
# ═══════════════════════════════════════════════════════════════════

def cpcv_splits(
    windows: list[dict[str, Any]],
    k: int = 2,
    embargo_bars: int = 288,
) -> Iterator[dict[str, Any]]:
    """Yield every ``C(N, k)`` combinatorial split of walk-forward windows.

    Each yielded dict carries the partitioning plus the ``embargo_bars``
    setting so downstream purging can apply it without extra arguments.

    Parameters
    ----------
    windows
        Walk-forward windows as produced by ``generate_wf_windows``. Each
        entry must expose ``train_start``, ``train_end``, ``test_start``,
        ``test_end`` timestamps.
    k
        Number of windows reserved for test in each split. Must satisfy
        ``1 <= k <= len(windows)``.
    embargo_bars
        Gap (in 5-minute bars) between the end of a signal's holding
        period and the test-window start required to keep a train
        signal. Propagated verbatim in the yielded dict.

    Yields
    ------
    dict
        ``{"train_windows": [...], "test_windows": [...],
           "test_idx": (i, j, ...), "embargo_bars": int}``
    """
    n = len(windows)
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    if k > n:
        raise ValueError(f"k={k} exceeds window count {n}")

    for test_idx in combinations(range(n), k):
        test_set = set(test_idx)
        train = [w for i, w in enumerate(windows) if i not in test_set]
        test = [windows[i] for i in test_idx]
        yield {
            "train_windows": train,
            "test_windows": test,
            "test_idx": test_idx,
            "embargo_bars": embargo_bars,
        }


# ═══════════════════════════════════════════════════════════════════
#  Purging
# ═══════════════════════════════════════════════════════════════════

def purge_train_signals(
    train_signals: Iterable[Any],
    test_windows: list[dict[str, Any]],
    max_hold_bars: int = SCALP_MAX_HOLD_BARS,
    embargo_bars: int = 288,
    bar_minutes: int = 5,
) -> list[Any]:
    """Remove train signals whose holding period + embargo overlaps any test window.

    A signal is *purged* when the closed interval
    ``[sig.timestamp, sig.timestamp + max_hold + embargo]`` intersects
    the closed interval ``[test_start, test_end]`` of *any* test window.

    The purge protects a future train-side optimiser from reading labels
    whose outcome was determined by data that appears inside a test
    window (leakage), and the embargo suppresses serial autocorrelation
    between adjacent train/test boundaries.

    Parameters
    ----------
    train_signals
        Iterable of objects with ``.timestamp`` (tz-aware).
    test_windows
        List of dicts with ``test_start`` and ``test_end`` timestamps.
    max_hold_bars
        Maximum bars a signal remains open post-entry.
    embargo_bars
        Gap bars added after the hold-end before the signal is "safe".
    bar_minutes
        Minutes per bar (5 for the Scalp-Day Hybrid 5m-entry regime).
    """
    kept: list[Any] = []
    if not test_windows:
        kept.extend(train_signals)
        return kept

    hold_minutes = max_hold_bars * bar_minutes
    embargo_minutes = embargo_bars * bar_minutes
    tail_minutes = hold_minutes + embargo_minutes

    test_bounds = [
        (pd.Timestamp(w["test_start"]), pd.Timestamp(w["test_end"]))
        for w in test_windows
    ]

    for sig in train_signals:
        sig_start = pd.Timestamp(sig.timestamp)
        sig_end = sig_start + pd.Timedelta(minutes=tail_minutes)

        overlaps = any(
            sig_start <= t_end and sig_end >= t_start
            for t_start, t_end in test_bounds
        )
        if not overlaps:
            kept.append(sig)

    return kept


# ═══════════════════════════════════════════════════════════════════
#  Evaluator
# ═══════════════════════════════════════════════════════════════════

_SimulateFn = Callable[
    [dict[str, Any], list[Any], dict[str, Any], dict[str, str]],
    tuple[pd.DataFrame, dict[str, float]],
]


def run_cpcv(
    params: dict[str, Any],
    windows: list[dict[str, Any]],
    signals_per_window: dict[int, list[Any]],
    simulate_fn: _SimulateFn,
    config: dict[str, Any],
    symbol_to_asset: dict[str, str],
    trial_sharpes: list[float],
    k: int = 2,
    embargo_bars: int = 288,
    max_hold_bars: int = SCALP_MAX_HOLD_BARS,
    bar_minutes: int = 5,
) -> pd.DataFrame:
    """Evaluate ``params`` on every CPCV split and return per-split metrics.

    For each split the function:
      1. Forms the test signal pool as the union of signals from the
         test windows (by index).
      2. Calls ``simulate_fn(params, test_sigs, config, symbol_to_asset)``
         to obtain trades and metrics.
      3. Haircuts the Sharpe via the Deflated Sharpe Ratio using
         ``trial_sharpes`` as the null-distribution sample.

    The training signal pool is produced but NOT used by ``simulate_fn``
    in this baseline implementation — parameters are fixed. It is still
    computed and purged so the protocol is complete and available to a
    future re-training extension.

    Parameters
    ----------
    params
        Parameter dict compatible with ``simulate_fn``.
    windows
        Walk-forward windows defining the CPCV universe.
    signals_per_window
        Pre-computed signals keyed by window index. Missing keys are
        treated as empty lists.
    simulate_fn
        Callable returning ``(trades_df, metrics_dict)``. The metrics
        dict must include at least ``sharpe``, ``pf_real``,
        ``max_drawdown``, ``total_pnl``, ``total_trades``,
        ``n_obs_daily``, ``skew``, ``kurt_nonexcess``.
    config
        Config dict forwarded to ``simulate_fn``.
    symbol_to_asset
        Symbol→asset-class map forwarded to ``simulate_fn``.
    trial_sharpes
        Sharpe values collected from the original Optuna study. Variance
        across these is the null-distribution estimate for the DSR.
    k
        Test-windows per split.
    embargo_bars
        Purge embargo in 5m bars.
    max_hold_bars
        Max holding period used by the purge.
    bar_minutes
        Minutes per bar (default 5 for crypto 5m).
    """
    clean_trials = [
        float(s) for s in trial_sharpes
        if s is not None and not (isinstance(s, float) and np.isnan(s))
    ]
    trial_var = _trial_sharpe_variance(clean_trials)
    n_trials = len(clean_trials)

    rows: list[dict[str, Any]] = []
    for split_idx, split in enumerate(cpcv_splits(windows, k=k, embargo_bars=embargo_bars)):
        test_idx: tuple[int, ...] = split["test_idx"]

        # Union test signals
        test_sigs: list[Any] = []
        for wi in test_idx:
            test_sigs.extend(signals_per_window.get(wi, []))

        if not test_sigs:
            continue  # split carries no information

        # Build train pool (present for protocol completeness; purged but unused downstream)
        train_sigs: list[Any] = []
        for wi, sigs in signals_per_window.items():
            if wi in test_idx:
                continue
            train_sigs.extend(sigs)
        train_sigs = purge_train_signals(
            train_sigs,
            test_windows=split["test_windows"],
            max_hold_bars=max_hold_bars,
            embargo_bars=embargo_bars,
            bar_minutes=bar_minutes,
        )

        _, metrics = simulate_fn(params, test_sigs, config, symbol_to_asset)

        observed = float(metrics.get("sharpe", 0.0))
        n_obs = int(metrics.get("n_obs_daily", 0))
        skew = float(metrics.get("skew", 0.0))
        kurt = float(metrics.get("kurt_nonexcess", 3.0))

        if trial_var > 0 and n_trials >= 2 and n_obs >= 2:
            dsr = _deflated_sharpe(
                observed_sharpe=observed,
                sharpe_variance=trial_var,
                n_trials=n_trials,
                observation_count=n_obs,
                skewness=skew,
                kurtosis=kurt,
            )
        else:
            dsr = 0.0

        rows.append({
            "split_idx": split_idx,
            "test_idx": test_idx,
            "n_test_signals": len(test_sigs),
            "n_train_signals_purged": len(train_sigs),
            "total_pnl": float(metrics.get("total_pnl", 0.0)),
            "sharpe": observed,
            "pf_real": float(metrics.get("pf_real", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "trades": int(metrics.get("total_trades", 0)),
            "winrate_real": float(metrics.get("winrate_real", 0.0)),
            "skew": skew,
            "kurt_nonexcess": kurt,
            "n_obs_daily": n_obs,
            "dsr": float(dsr),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════

def cpcv_summary(
    cpcv_df: pd.DataFrame,
    funded_threshold: float = 0.95,
    research_threshold: float = 0.70,
) -> dict[str, Any]:
    """Aggregate a CPCV results DataFrame into a JSON-safe summary dict.

    IQR < 0.3 is the Phase-D robustness criterion: tight dispersion of
    DSR across splits signals parameters that are not split-specific.

    ``funded_threshold`` defaults to 0.95 per Bailey & Lopez de Prado
    (2014); ``research_threshold`` defaults to 0.70 to stay consistent
    with the existing research-phase gate.
    """
    n = int(len(cpcv_df))
    if n == 0:
        return {
            "n_splits": 0,
            "median_dsr": 0.0,
            "mean_dsr": 0.0,
            "dsr_q25": 0.0,
            "dsr_q75": 0.0,
            "dsr_iqr": 0.0,
            "median_sharpe": 0.0,
            "median_pf_real": 0.0,
            "median_max_drawdown": 0.0,
            "percent_passing_funded": 0.0,
            "percent_passing_research": 0.0,
            "funded_threshold": float(funded_threshold),
            "research_threshold": float(research_threshold),
        }

    dsr = cpcv_df["dsr"].astype(float)
    q25, q75 = float(dsr.quantile(0.25)), float(dsr.quantile(0.75))
    return {
        "n_splits": n,
        "median_dsr": float(dsr.median()),
        "mean_dsr": float(dsr.mean()),
        "dsr_q25": q25,
        "dsr_q75": q75,
        "dsr_iqr": q75 - q25,
        "median_sharpe": float(cpcv_df["sharpe"].astype(float).median()),
        "median_pf_real": float(cpcv_df["pf_real"].astype(float).median()),
        "median_max_drawdown": float(cpcv_df["max_drawdown"].astype(float).median()),
        "percent_passing_funded": float((dsr >= funded_threshold).mean()),
        "percent_passing_research": float((dsr >= research_threshold).mean()),
        "funded_threshold": float(funded_threshold),
        "research_threshold": float(research_threshold),
    }


__all__ = [
    "cpcv_splits",
    "purge_train_signals",
    "run_cpcv",
    "cpcv_summary",
]
