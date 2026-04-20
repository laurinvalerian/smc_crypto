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
    expected = (DEFAULT_RISK_PER_TRADE + MAX_RISK_PER_TRADE) / 2  # = 0.00625 (0.25%+1.0% /2)
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


# ═══════════════════════════════════════════════════════════════════
#  Track A1 regression — live_multi_bot must use core.sizing SSOT
#  (added 2026-04-20; replaces the hardcoded 0.2%-1.5% block that
#  produced 1.5% unconditional risk when Student/XGB were absent).
# ═══════════════════════════════════════════════════════════════════

def test_live_bot_uses_core_sizing_import():
    """live_multi_bot must import compute_risk_fraction from core.sizing.

    Regression guard: if someone reverts to an inline implementation,
    the import dependency disappears and this test fails loudly.
    """
    import live_multi_bot
    from core.sizing import compute_risk_fraction as _cr
    assert getattr(live_multi_bot, "compute_risk_fraction", None) is _cr


def test_sizing_at_current_live_params_alignment_082():
    """Pin the exact risk value at the v1.11 evergreen alignment (0.82).

    (0.82 - 0.78) / (1.0 - 0.78) = 0.181818..., so
    risk = 0.0025 + 0.0075 * 0.181818 = 0.00386363..., i.e. ~0.3864%.

    Before Track A1, the live bot produced 1.5% at every score because
    the hardcoded conf_factor=1.0 (no model → xgb_confidence=1.0) bypassed
    alignment entirely. This test pins the post-fix behaviour.
    """
    expected = DEFAULT_RISK_PER_TRADE + (MAX_RISK_PER_TRADE - DEFAULT_RISK_PER_TRADE) * (0.04 / 0.22)
    assert math.isclose(compute_risk_fraction(0.82), expected, rel_tol=1e-9)
    # Sanity: in [0.3%, 0.4%] window
    assert 0.003 < compute_risk_fraction(0.82) < 0.004
