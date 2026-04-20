"""
tests/test_alignment.py — SSOT regression for core.alignment.compute_alignment_score.

Phase 2.1 of the Crypto-Only Refocus extracted the alignment-score computation
from strategies/smc_multi_style.py into core/alignment.py. These tests lock in
the expected scoring behaviour so future refactors or weight changes do not
silently drift the Live vs Backtest gate (alignment_threshold = 0.78).

Scope:
  - Bit-parity across all 2^7 binary flag combinations (crypto & forex)
  - Known-value scenarios (max score, above-threshold, rejection, legacy boost)
  - Weight-table invariants (sum == 0.90)
  - Call-site consistency: generate_rl_data._compute_alignment_score is
    the identical function object as core.alignment.compute_alignment_score
"""

from __future__ import annotations

import itertools
import math

import pytest

from core.alignment import (
    CORE_WEIGHTS_CRYPTO,
    CORE_WEIGHTS_FOREX,
    compute_alignment_score,
)


# ════════════════════════════════════════════════════════════════════
#  Weight invariants
# ════════════════════════════════════════════════════════════════════

def test_crypto_weights_sum_to_max_core_score():
    """All 9 crypto weights must sum to 0.90 (max score before clamp)."""
    assert math.isclose(sum(CORE_WEIGHTS_CRYPTO.values()), 0.90, abs_tol=1e-12)


def test_forex_weights_sum_to_max_core_score():
    """Legacy forex weights must also sum to 0.90 (redistribution only)."""
    assert math.isclose(sum(CORE_WEIGHTS_FOREX.values()), 0.90, abs_tol=1e-12)


def test_crypto_weights_have_all_nine_keys():
    expected = {
        "bias", "bias_strong", "h4", "h4_poi",
        "h1", "h1_choch", "zone", "trigger", "volume",
    }
    assert set(CORE_WEIGHTS_CRYPTO.keys()) == expected
    assert set(CORE_WEIGHTS_FOREX.keys()) == expected


def test_crypto_weights_match_documented_training_values():
    """Documented training weights (Phase 2.2 SHAP report, line 114-120)."""
    assert CORE_WEIGHTS_CRYPTO["bias"] == 0.12
    assert CORE_WEIGHTS_CRYPTO["bias_strong"] == 0.08
    assert CORE_WEIGHTS_CRYPTO["h4"] == 0.08
    assert CORE_WEIGHTS_CRYPTO["h4_poi"] == 0.08
    assert CORE_WEIGHTS_CRYPTO["h1"] == 0.08
    assert CORE_WEIGHTS_CRYPTO["h1_choch"] == 0.06
    assert CORE_WEIGHTS_CRYPTO["zone"] == 0.15
    assert CORE_WEIGHTS_CRYPTO["trigger"] == 0.15
    assert CORE_WEIGHTS_CRYPTO["volume"] == 0.10


# ════════════════════════════════════════════════════════════════════
#  Known-value scenarios
# ════════════════════════════════════════════════════════════════════

def test_all_flags_true_scores_at_max():
    """All 9 flags True → score = 0.90 (saturates before clamp)."""
    score = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True,
        h4_confirms=True,
        h4_poi=True,
        h1_choch=True,
        volume_ok=True,
    )
    assert math.isclose(score, 0.90, abs_tol=1e-12)


def test_rejection_all_flags_false_scores_zero():
    score = compute_alignment_score("neutral", False, None, False)
    assert score == 0.0


def test_partial_flags_still_above_gate():
    """Enough flags to clear the 0.78 alignment gate without volume_ok."""
    # bias(0.12) + bias_strong(0.08) + h4(0.08) + h4_poi(0.08) + h1(0.08)
    # + h1_choch(0.06) + zone(0.15) + trigger(0.15) = 0.80
    score = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True,
        h4_confirms=True,
        h4_poi=True,
        h1_choch=True,
        volume_ok=False,
    )
    assert math.isclose(score, 0.80, abs_tol=1e-12)
    assert score >= 0.78, "must clear alignment gate"


def test_neutral_bias_skips_bias_weight_even_with_bias_strong_true():
    """bias_strong is only counted when daily_bias is bullish/bearish."""
    # Without the guard, we'd add bias_strong alone. Old code guards this.
    score = compute_alignment_score(
        "neutral", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True,
        h4_confirms=True,
        h4_poi=True,
        h1_choch=True,
        volume_ok=True,
    )
    # Skipped under neutral bias: w["bias"] (0.12), w["bias_strong"] (0.08)
    # Counted: h4(0.08) + h4_poi(0.08) + h1(0.08) + h1_choch(0.06)
    #        + zone(0.15) + trigger(0.15) + volume(0.10) = 0.70
    assert math.isclose(score, 0.70, abs_tol=1e-12)


def test_h1_choch_requires_h1_confirms():
    """CHoCH bonus only stacks on top of h1 structural confirm."""
    # h1_confirms=False → h1_choch ignored
    score_without = compute_alignment_score(
        "bullish", False, None, False,
        h1_choch=True,
    )
    # h1_confirms=True → CHoCH counted
    score_with = compute_alignment_score(
        "bullish", True, None, False,
        h1_choch=True,
    )
    assert math.isclose(score_with - score_without, 0.08 + 0.06, abs_tol=1e-12)


def test_old_style_backward_compat_boost():
    """4-arg call without new flags gets 1.3x boost (legacy behaviour)."""
    # bias(0.12) + h1(0.08) + zone(0.15) + trigger(0.15) = 0.50
    # Boosted: 0.50 * 1.3 = 0.65
    score = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
    )
    assert math.isclose(score, 0.65, abs_tol=1e-12)


def test_old_style_boost_clamps_at_one():
    """Legacy boost never exceeds 1.0 even on boundary inputs."""
    # Already at max of old-style (0.50) → 0.65, safely below clamp
    # Synthetic max: give it style_weight=2.0 → should clamp
    score = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        style_weight=2.0,
    )
    assert score <= 1.0


def test_style_weight_multiplier_applied():
    """Score scales linearly by style_weight then clamps at 1.0."""
    # Without style_weight: 0.90 (all flags true)
    base = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True, h4_confirms=True, h4_poi=True,
        h1_choch=True, volume_ok=True, style_weight=1.0,
    )
    # Half weight: expect 0.45
    half = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True, h4_confirms=True, h4_poi=True,
        h1_choch=True, volume_ok=True, style_weight=0.5,
    )
    assert math.isclose(base, 0.90, abs_tol=1e-12)
    assert math.isclose(half, 0.45, abs_tol=1e-12)


def test_style_weight_clamps_at_one():
    """style_weight > 1 never yields score > 1.0."""
    score = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True, h4_confirms=True, h4_poi=True,
        h1_choch=True, volume_ok=True, style_weight=2.0,
    )
    assert score == 1.0


# ════════════════════════════════════════════════════════════════════
#  Forex branch (legacy retention)
# ════════════════════════════════════════════════════════════════════

def test_forex_uses_redistributed_weights():
    """Forex branch redistributes: less on zone/trigger, more on h4/volume."""
    # Same flags, different asset_class
    crypto = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True, h4_confirms=True, h4_poi=True,
        h1_choch=True, volume_ok=True,
        asset_class="crypto",
    )
    forex = compute_alignment_score(
        "bullish", True, {"top": 100, "bottom": 99}, True,
        bias_strong=True, h4_confirms=True, h4_poi=True,
        h1_choch=True, volume_ok=True,
        asset_class="forex",
    )
    # Both hit 0.90 when all flags True (redistribution preserves sum)
    assert math.isclose(crypto, 0.90, abs_tol=1e-12)
    assert math.isclose(forex, 0.90, abs_tol=1e-12)


def test_forex_weights_differ_from_crypto_when_partial():
    """Partial flags expose the weight redistribution."""
    # Only bias + bias_strong + h4 + volume → crypto vs forex differ
    crypto = compute_alignment_score(
        "bullish", False, None, False,
        bias_strong=True, h4_confirms=True, volume_ok=True,
        asset_class="crypto",
    )
    forex = compute_alignment_score(
        "bullish", False, None, False,
        bias_strong=True, h4_confirms=True, volume_ok=True,
        asset_class="forex",
    )
    # Crypto: 0.12 + 0.08 + 0.08 + 0.10 = 0.38
    # Forex:  0.12 + 0.12 + 0.12 + 0.14 = 0.50
    assert math.isclose(crypto, 0.38, abs_tol=1e-12)
    assert math.isclose(forex, 0.50, abs_tol=1e-12)


# ════════════════════════════════════════════════════════════════════
#  Bit-parity across all binary flag combinations
# ════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "bias",
    ["bullish", "bearish", "neutral"],
)
@pytest.mark.parametrize(
    "asset",
    [None, "crypto", "forex", "stocks"],
)
def test_all_128_flag_combinations_match_manual_computation(bias, asset):
    """
    For every (bias × 2^7 flag combos × asset_class) combination the score
    must equal a direct weighted-sum computation. 3 × 4 × 128 = 1536 cases.
    """
    w = CORE_WEIGHTS_FOREX if asset == "forex" else CORE_WEIGHTS_CRYPTO
    zone_obj = {"top": 1.0, "bottom": 0.99}

    flags = ["bias_strong", "h4_confirms", "h4_poi",
             "h1_confirms", "h1_choch", "has_zone", "precision_trigger", "volume_ok"]
    for combo in itertools.product([False, True], repeat=len(flags)):
        kwargs = dict(zip(flags, combo))
        has_zone = kwargs.pop("has_zone")
        h1_confirms = kwargs.pop("h1_confirms")
        precision = kwargs.pop("precision_trigger")

        # Expected: replay the compute_alignment_score body manually
        expected = 0.0
        if bias in ("bullish", "bearish"):
            expected += w["bias"]
            if kwargs["bias_strong"]:
                expected += w["bias_strong"]
        if kwargs["h4_confirms"]:
            expected += w["h4"]
        if kwargs["h4_poi"]:
            expected += w["h4_poi"]
        if h1_confirms:
            expected += w["h1"]
            if kwargs["h1_choch"]:
                expected += w["h1_choch"]
        if has_zone:
            expected += w["zone"]
        if precision:
            expected += w["trigger"]
        if kwargs["volume_ok"]:
            expected += w["volume"]

        # Backward-compat boost
        old_style = (
            not kwargs["bias_strong"] and not kwargs["h4_confirms"]
            and not kwargs["h4_poi"] and not kwargs["h1_choch"]
            and not kwargs["volume_ok"]
        )
        if old_style and expected > 0:
            expected = min(expected * 1.3, 1.0)
        expected = min(expected, 1.0)

        actual = compute_alignment_score(
            bias, h1_confirms, zone_obj if has_zone else None, precision,
            asset_class=asset, **kwargs,
        )
        assert math.isclose(actual, expected, abs_tol=1e-12), (
            f"mismatch bias={bias} asset={asset} flags={kwargs} "
            f"zone={has_zone} h1={h1_confirms} trig={precision}: "
            f"actual={actual} expected={expected}"
        )


# ════════════════════════════════════════════════════════════════════
#  Call-site identity (Phase 2.1 SSOT verification)
# ════════════════════════════════════════════════════════════════════

def test_smc_multi_style_aliases_core_alignment():
    """strategies._compute_alignment_score MUST be core.compute_alignment_score."""
    from strategies.smc_multi_style import _compute_alignment_score as smc_alias
    assert smc_alias is compute_alignment_score


def test_generate_rl_data_imports_same_function():
    """backtest/generate_rl_data.py uses the same function object via re-export."""
    from backtest.generate_rl_data import _compute_alignment_score as gen_alias
    assert gen_alias is compute_alignment_score


def test_core_package_exports_compute_alignment_score():
    """The top-level `core` import surface must expose the function."""
    import core
    assert core.compute_alignment_score is compute_alignment_score
    assert core.CORE_WEIGHTS_CRYPTO is CORE_WEIGHTS_CRYPTO


# ════════════════════════════════════════════════════════════════════
#  Live-gate SSOT routing (regression for 2026-04-20 signal-drought fix)
# ════════════════════════════════════════════════════════════════════
#
# Before 2026-04-20 the live bot (PaperBot._prepare_signal) gated on the
# continuous 13-component _multi_tf_alignment_score, while the backtest and
# training parquet both gated on core.alignment.compute_alignment_score.
# The continuous multipliers (0.15×zone_quality instead of 0.15, 0.10×
# volume_score instead of 0.10, plus D/P and volatility penalties) dropped
# the live score ~0.07 below the SSOT form, causing 0 trades in 16h on 30
# symbols while SSOT-calibration projected ~24 gate-triggers in the same
# window. Fix: route live gate through core.alignment.compute_alignment_score.
# See .omc/research/alignment_drought_probe.py for the empirical evidence.

def test_live_bot_imports_compute_alignment_score():
    """PaperBot must import compute_alignment_score from core.alignment SSOT."""
    from live_multi_bot import compute_alignment_score as live_alias
    assert live_alias is compute_alignment_score


def test_live_bot_gate_matches_smc_strategy_gate():
    """
    The function used by live gate and the function used by backtest gate
    must be the same object — no drift, no divergent re-implementation.
    """
    from live_multi_bot import compute_alignment_score as live_f
    from strategies.smc_multi_style import _compute_alignment_score as bt_f
    assert live_f is bt_f is compute_alignment_score


def test_prepare_signal_gate_uses_ssot_not_rich_score():
    """
    Regression guard: the fix wired `score` (gate variable) to the SSOT
    function call, not the continuous rich_score. If someone reverts this,
    the gate-crossing drought returns. Check by reading the source file.
    """
    from pathlib import Path
    import re

    src = (Path(__file__).resolve().parents[1] / "live_multi_bot.py").read_text()
    # Find _prepare_signal method body
    m = re.search(r"def _prepare_signal\(.*?\n(?=    def )", src, re.DOTALL)
    assert m is not None, "could not locate _prepare_signal in live_multi_bot.py"
    body = m.group(0)

    # Must contain the SSOT call for the gate.
    assert "compute_alignment_score(" in body, (
        "live _prepare_signal must call compute_alignment_score (SSOT gate)"
    )
    # Must use `score = compute_alignment_score(...)` as the gate-score bind.
    assert re.search(r"score\s*=\s*compute_alignment_score\(", body), (
        "SSOT result must be assigned to `score` (the gate-check variable)"
    )
    # Rich score kept separately as a feature; sanity-check it is retained.
    assert "rich_score" in body, (
        "rich_score must be preserved for dashboard/XGB features"
    )


def test_gate_score_under_expected_distribution_scenarios():
    """
    Sanity scenarios that mimic the 2 observed NEAR-MISS events from
    2026-04-19/20 (APT score=0.700, INJ score=0.645). With the SSOT formula
    on the SAME boolean flags, APT should rise closer to the gate — proving
    the SSOT routing is what unblocks the drought, not any gate-lowering.
    """
    # APT NEAR-MISS flags: bias=T, bias_strong=T, h4_confirms=F, h4_poi=T,
    #   h1_confirms=T, h1_choch=F, entry_zone=exists, precision_trigger=T,
    #   volume_ok=T
    # Expected SSOT: 0.12 + 0.08 + 0 + 0.08 + 0.08 + 0 + 0.15 + 0.15 + 0.10 = 0.76
    apt_like = compute_alignment_score(
        "bearish", True, {"top": 0.937, "bottom": 0.932}, True,
        bias_strong=True,
        h4_confirms=False,
        h4_poi=True,
        h1_choch=False,
        volume_ok=True,
    )
    assert math.isclose(apt_like, 0.76, abs_tol=1e-12)
    # Below 0.78 — still a legitimate reject under SSOT, but 0.76 is closer
    # than the 0.70 that the continuous live score reported.
    assert apt_like < 0.78

    # If the same setup had either h4_confirms=True OR h1_choch=True,
    # the SSOT gate would fire.
    apt_with_h4 = compute_alignment_score(
        "bearish", True, {"top": 0.937, "bottom": 0.932}, True,
        bias_strong=True, h4_confirms=True, h4_poi=True,
        h1_choch=False, volume_ok=True,
    )
    assert apt_with_h4 >= 0.78

    apt_with_choch = compute_alignment_score(
        "bearish", True, {"top": 0.937, "bottom": 0.932}, True,
        bias_strong=True, h4_confirms=False, h4_poi=True,
        h1_choch=True, volume_ok=True,
    )
    assert math.isclose(apt_with_choch, 0.82, abs_tol=1e-12)
    assert apt_with_choch >= 0.78
