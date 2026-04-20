"""
tests/test_bracket_sizing.py — integration-light tests for the sizing path
that _execute_bracket_order_with_risk_reduction uses after Track A1.

The real function takes many paperbot-internal attrs; these tests exercise
the exact formula the live path uses (core.sizing + size_multiplier + clamp)
without standing up a full bot. The regression guarantee is: if someone
changes the formula in live_multi_bot, the pinned values below drift and
the test fails.
"""

from __future__ import annotations

import inspect
import math

import pytest

from core.constants import DEFAULT_RISK_PER_TRADE, MAX_RISK_PER_TRADE
from core.sizing import compute_risk_fraction


def _live_bot_sizing(alignment_score: float, size_multiplier: float) -> float:
    """Mirror of the post-Track-A1 formula in _execute_bracket_order_with_risk_reduction.

    If live_multi_bot's formula changes, update this helper AND the tests below.
    """
    base = compute_risk_fraction(alignment_score)
    scaled = base * float(size_multiplier)
    return max(DEFAULT_RISK_PER_TRADE, min(scaled, MAX_RISK_PER_TRADE))


def test_bracket_sizing_no_student():
    """Without Student (size_mult=1.0) risk is pure alignment-based."""
    # At alignment gate (0.78): minimum risk 0.25%
    assert math.isclose(_live_bot_sizing(0.78, 1.0), DEFAULT_RISK_PER_TRADE, rel_tol=1e-9)
    # At max alignment (1.0): max risk 1.0%
    assert math.isclose(_live_bot_sizing(1.0, 1.0), MAX_RISK_PER_TRADE, rel_tol=1e-9)
    # At evergreen 0.82: 0.3864%
    assert 0.003 < _live_bot_sizing(0.82, 1.0) < 0.004
    # Below threshold clamps to floor (defence-in-depth; live path should
    # never reach here because alignment gate filters first).
    assert _live_bot_sizing(0.50, 1.0) == DEFAULT_RISK_PER_TRADE


def test_bracket_sizing_with_student():
    """Student size-head multiplies base risk, clamped to [MIN, MAX]."""
    # size_mult 1.5 at alignment 0.85 → below cap so preserved.
    base_085 = compute_risk_fraction(0.85)
    scaled = base_085 * 1.5
    assert _live_bot_sizing(0.85, 1.5) == pytest.approx(
        max(DEFAULT_RISK_PER_TRADE, min(scaled, MAX_RISK_PER_TRADE))
    )
    # size_mult 2.0 at alignment 1.0 → 2.0% would break funded cap; clamp to 1.0%.
    assert _live_bot_sizing(1.0, 2.0) == MAX_RISK_PER_TRADE
    # size_mult 0.3 at alignment 1.0 → 0.3% would fall below min; clamp to 0.25%.
    # (2026-04-20: student size_floor=0.5 already prevents this in practice,
    # but the clamp is defence-in-depth.)
    assert _live_bot_sizing(1.0, 0.1) == DEFAULT_RISK_PER_TRADE


def test_no_hardcoded_risk_bounds_in_bracket_function():
    """Function-scoped regression: the sizing method must not re-introduce
    inline 0.002 / 0.015 / conf_floor literals on the risk-computation path.

    Broader module-level grep lives in test_constants.py — this one pins the
    specific function body so a refactor that moves code around stays honest.
    """
    from live_multi_bot import PaperBot
    src = inspect.getsource(PaperBot._execute_bracket_order_with_risk_reduction)
    assert "min_risk = 0.002" not in src, "hardcoded min_risk back in bracket function"
    assert "max_risk = 0.015" not in src, "hardcoded max_risk back in bracket function"
    assert "conf_floor = " not in src or "self.rl_suite.confidence_threshold" not in src, (
        "confidence-only sizing path resurrected"
    )
    # Must import + call the SSOT.
    assert "compute_risk_fraction(" in src, "SSOT call missing from bracket function"
