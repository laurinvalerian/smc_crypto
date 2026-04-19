"""
tests/test_sizing.py — regression for core.sizing.compute_risk_fraction.

Tier system (AAA++/AAA+) was killed 2026-04-19; risk now scales linearly
between DEFAULT_RISK_PER_TRADE at the alignment gate and MAX_RISK_PER_TRADE
at score 1.0. These tests pin the math so future tweaks don't silently
change live position sizing.
"""

from __future__ import annotations

import math

import pytest

from core.constants import (
    ALIGNMENT_THRESHOLD,
    DEFAULT_RISK_PER_TRADE,
    MAX_RISK_PER_TRADE,
)
from core.sizing import compute_risk_amount, compute_risk_fraction


def test_below_threshold_returns_zero():
    """Signals below the alignment gate get zero risk (would be rejected anyway)."""
    assert compute_risk_fraction(0.0) == 0.0
    assert compute_risk_fraction(0.5) == 0.0
    assert compute_risk_fraction(ALIGNMENT_THRESHOLD - 1e-9) == 0.0


def test_at_threshold_returns_default_risk():
    """Score exactly at threshold (0.78) → minimum base risk (0.5%)."""
    assert math.isclose(
        compute_risk_fraction(ALIGNMENT_THRESHOLD),
        DEFAULT_RISK_PER_TRADE,
        abs_tol=1e-12,
    )


def test_at_max_score_returns_max_risk():
    """Score = 1.0 → max risk (1.5%)."""
    assert math.isclose(
        compute_risk_fraction(1.0), MAX_RISK_PER_TRADE, abs_tol=1e-12
    )


def test_midpoint_halfway_between_base_and_max():
    """Score halfway between threshold and 1.0 → risk halfway between default and max."""
    mid_score = (ALIGNMENT_THRESHOLD + 1.0) / 2  # = 0.89
    expected = (DEFAULT_RISK_PER_TRADE + MAX_RISK_PER_TRADE) / 2  # = 0.01
    assert math.isclose(
        compute_risk_fraction(mid_score), expected, abs_tol=1e-12
    )


def test_monotonic_scaling():
    """Higher score → strictly higher risk fraction."""
    fractions = [compute_risk_fraction(s) for s in [0.78, 0.82, 0.88, 0.94, 1.0]]
    assert fractions == sorted(fractions)
    assert len(set(fractions)) == 5  # all distinct


def test_above_one_clamps_at_max():
    """Score > 1.0 (shouldn't happen but fail-safe) clamps at max risk."""
    assert math.isclose(
        compute_risk_fraction(1.2), MAX_RISK_PER_TRADE, abs_tol=1e-12
    )


def test_compute_risk_amount_basic():
    """USD risk = equity × fraction."""
    equity = 100_000.0
    assert math.isclose(
        compute_risk_amount(0.78, equity),
        equity * DEFAULT_RISK_PER_TRADE,
        abs_tol=1e-9,
    )
    assert math.isclose(
        compute_risk_amount(1.0, equity),
        equity * MAX_RISK_PER_TRADE,
        abs_tol=1e-9,
    )


def test_compute_risk_amount_zero_equity():
    """Zero or negative equity → zero risk."""
    assert compute_risk_amount(0.9, 0.0) == 0.0
    assert compute_risk_amount(0.9, -100.0) == 0.0


def test_student_multiplier_scales_fraction():
    """Live pipeline multiplies by Student size-head prediction."""
    equity = 100_000.0
    base = compute_risk_amount(0.88, equity, student_size_multiplier=1.0)
    half = compute_risk_amount(0.88, equity, student_size_multiplier=0.5)
    double = compute_risk_amount(0.88, equity, student_size_multiplier=2.0)
    assert math.isclose(half, base * 0.5, abs_tol=1e-9)
    # Doubled multiplier can push past MAX_RISK → clamped
    assert double <= equity * MAX_RISK_PER_TRADE + 1e-9


def test_student_multiplier_clamps_negative_to_zero():
    """Defensive: negative multiplier can't produce short position sizing."""
    assert compute_risk_amount(0.9, 100_000, student_size_multiplier=-1.0) == 0.0


def test_max_risk_override_caps_fraction():
    """Optuna can pass a tighter hard cap per variant."""
    equity = 100_000.0
    capped = compute_risk_amount(
        1.0, equity, max_risk_override=0.005
    )
    assert math.isclose(capped, equity * 0.005, abs_tol=1e-9)
