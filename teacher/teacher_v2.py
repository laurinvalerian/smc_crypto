"""
Teacher v2 — retrospective "what should have been done" labeler.

Given the post-entry MFE (max favorable excursion in R-units) and
MAE (max adverse excursion in R-units) of a realised or simulated trade,
produce the four supervised targets used to train the Student brain:

    optimal_entry   (binary)   — was this setup worth taking in hindsight?
    optimal_sl_rr   (regr.)    — where should the SL have been placed?
    optimal_tp_rr   (regr.)    — where should the TP have been placed?
    optimal_size    (regr.)    — Sharpe-like position-size multiplier

The labels are *hindsight-optimal* — they use knowledge of what the
market actually did after entry. The Student learns to predict these
targets from entry-time features only, which lets it set SL / TP / size
correctly in one pass at entry instead of stacking 4 post-hoc adjuster
models on top of the strategy.

Why this matters (audit 2026-04-17):
  * The old SL adjuster learned from its own adjustments (self-loop →
    WIDEN predicted in 0 of 5.3 M backtest samples).
  * The old TP optimizer pauschal-reduced TP and ate PF in 4/5 folds.
  * Direct supervision on MFE/MAE-derived optima removes the feedback
    loop and the "nudge-from-baseline" framing entirely.

The four label formulas here are the single source of truth. They are
used both by the backtest data generator (vectorised, at label-gen
time) and by `live_teacher` (per trade, at journal close). Keeping the
definition in one place keeps train-time labels identical to inference-
time labels, which matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Tunable constants — keep conservative; adjust only with evidence.
# ---------------------------------------------------------------------------
# SL bounds. Floor at 1 R gives every trade room to breathe; cap at 3 R
# prevents the student from learning "always wide SL" on setups that
# happened to go 10 R against before recovery.
SL_FLOOR: float = 1.0
SL_CAP: float = 3.0
SL_BUFFER: float = 1.10  # place SL 10 % deeper than observed MAE

# TP bounds. Floor at 0.5 R ensures at least a minimum R:R ≥ 0.5.
# Cap at 10 R avoids encoding implausible catch-the-tail outliers.
TP_FLOOR: float = 0.5
TP_CAP: float = 10.0
TP_CAPTURE: float = 0.85  # capture 85 % of observed peak MFE

# Size multiplier bounds. 0.5× lets the student dial down risky setups;
# 2.0× rewards high-quality ones without letting a single outlier blow
# the risk budget.
SIZE_FLOOR: float = 0.5
SIZE_CAP: float = 2.0
SIZE_MAE_FLOOR: float = 0.3  # avoid /0 when MAE was tiny

# Entry gate parameters.
ENTRY_MFE_MIN_R: float = 1.0       # must have reached ≥ 1 R favorable
ENTRY_MFE_OVER_MAE_RATIO: float = 1.5  # and MFE must exceed MAE × 1.5


@dataclass(frozen=True)
class TeacherLabels:
    """Structured single-trade output of the teacher (used by live_teacher).

    The vectorised `compute_teacher_labels` returns numpy arrays instead.
    """
    optimal_entry: int
    optimal_sl_rr: float
    optimal_tp_rr: float
    optimal_size: float


def _to_array(x) -> np.ndarray:
    """Accept scalar, list, or ndarray — always return a float32 ndarray."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def compute_teacher_labels(
    mfe_rr,
    mae_rr,
    outcome=None,
) -> dict[str, np.ndarray]:
    """Vectorised computation of the four teacher-v2 targets.

    Parameters
    ----------
    mfe_rr : array-like
        Max favorable excursion in R-multiples (e.g. 2.3 = went 2.3× the
        planned SL distance in the profitable direction).
    mae_rr : array-like
        Max adverse excursion in R-multiples (positive = worst drawdown
        relative to the planned SL distance).
    outcome : array-like of int, optional
        Per-trade outcome (1 = win, 2 = loss, 3 = breakeven). When
        provided it is used to gate entry-quality on the requirement
        that the trade reached real profit — it blocks the "tiny MFE"
        wins from being labelled high quality.

    Returns
    -------
    dict with keys ``optimal_entry``, ``optimal_sl_rr``, ``optimal_tp_rr``,
    ``optimal_size``. All arrays have the same shape as the input.
    """
    mfe = np.maximum(_to_array(mfe_rr), 0.0)
    mae = np.maximum(_to_array(mae_rr), 0.0)

    if mfe.shape != mae.shape:
        raise ValueError(f"mfe/mae shape mismatch: {mfe.shape} vs {mae.shape}")

    # ── SL: deepen MAE by 10 %, clamp into a sensible range ──
    optimal_sl = np.clip(mae * SL_BUFFER, SL_FLOOR, SL_CAP).astype(np.float32)

    # ── TP: capture 85 % of peak, clamp into a sensible range ──
    optimal_tp = np.clip(mfe * TP_CAPTURE, TP_FLOOR, TP_CAP).astype(np.float32)

    # ── Entry: binary "worth taking in hindsight" ──
    entry_mask = (mfe >= ENTRY_MFE_MIN_R) & (mfe > mae * ENTRY_MFE_OVER_MAE_RATIO)
    if outcome is not None:
        outcome_arr = np.asarray(outcome)
        if outcome_arr.shape == mfe.shape:
            # Explicit losses and breakevens never count as "worth taking"
            # even if MFE crossed 1 R — the trade was closed out first.
            entry_mask &= outcome_arr == 1
    optimal_entry = entry_mask.astype(np.int32)

    # ── Size: Sharpe-like reward/risk, clamped ──
    denom = np.maximum(mae, SIZE_MAE_FLOOR)
    optimal_size = np.clip(mfe / denom, SIZE_FLOOR, SIZE_CAP).astype(np.float32)

    return {
        "optimal_entry": optimal_entry,
        "optimal_sl_rr": optimal_sl,
        "optimal_tp_rr": optimal_tp,
        "optimal_size": optimal_size,
    }


def compute_single(mfe_rr: float, mae_rr: float, outcome: int | None = None) -> TeacherLabels:
    """Convenience wrapper for the per-trade live path."""
    labels = compute_teacher_labels(
        np.asarray([mfe_rr]), np.asarray([mae_rr]),
        outcome=np.asarray([outcome]) if outcome is not None else None,
    )
    return TeacherLabels(
        optimal_entry=int(labels["optimal_entry"][0]),
        optimal_sl_rr=float(labels["optimal_sl_rr"][0]),
        optimal_tp_rr=float(labels["optimal_tp_rr"][0]),
        optimal_size=float(labels["optimal_size"][0]),
    )
