"""
core.sizing — Confidence-based risk sizing as Single Source of Truth.

Replaces the tier-based AAA++/AAA+ risk dispatch that previously lived in
backtest/wf_bruteforce.py::compute_dynamic_risk (Optuna backtester, now
retired) and the aaa_plus_plus.tiers config block.

Model:
  risk_fraction(score) scales LINEARLY between DEFAULT_RISK_PER_TRADE
  (at ALIGNMENT_THRESHOLD) and MAX_RISK_PER_TRADE (at score == 1.0).
  Below threshold: 0.0 (signal would be rejected by the gate anyway).

Live pipeline multiplies this by the Student size-head output
(student_brain size_floor / size_cap). Backtest uses raw output.

Usage:
    from core.sizing import compute_risk_fraction, compute_risk_amount
    risk_pct = compute_risk_fraction(alignment_score=0.85)
    risk_usd = compute_risk_amount(score=0.85, equity=100_000)
"""

from __future__ import annotations

from core.constants import (
    ALIGNMENT_THRESHOLD,
    DEFAULT_RISK_PER_TRADE,
    MAX_RISK_PER_TRADE,
)


def compute_risk_fraction(alignment_score: float) -> float:
    """
    Return risk as a fraction of equity for a given alignment score.

    Scales linearly from DEFAULT_RISK_PER_TRADE (0.25% at score 0.78)
    to MAX_RISK_PER_TRADE (1.0% at score 1.0). Returns 0.0 below the
    entry gate — callers should never reach sizing for rejected signals
    but we fail safely.
    """
    if alignment_score < ALIGNMENT_THRESHOLD:
        return 0.0

    span = 1.0 - ALIGNMENT_THRESHOLD
    if span <= 0:
        return DEFAULT_RISK_PER_TRADE

    scale = (alignment_score - ALIGNMENT_THRESHOLD) / span
    if scale < 0.0:
        scale = 0.0
    elif scale > 1.0:
        scale = 1.0

    return DEFAULT_RISK_PER_TRADE + (MAX_RISK_PER_TRADE - DEFAULT_RISK_PER_TRADE) * scale


def compute_risk_amount(
    alignment_score: float,
    equity: float,
    student_size_multiplier: float = 1.0,
    max_risk_override: float | None = None,
) -> float:
    """
    Convert an alignment score + equity into a USD risk amount.

    Live pipeline passes `student_size_multiplier` from the Student size-head
    prediction (clamped to the size_floor/size_cap band). Backtest passes 1.0.

    `max_risk_override` lets Optuna tune a per-variant hard cap below 1.5%.
    """
    if equity <= 0:
        return 0.0

    fraction = compute_risk_fraction(alignment_score)
    fraction *= max(student_size_multiplier, 0.0)

    cap = MAX_RISK_PER_TRADE if max_risk_override is None else max_risk_override
    if fraction > cap:
        fraction = cap

    return equity * fraction
