"""
core.alignment — Single Source of Truth for the alignment-score computation.

This module owns the core top-down multi-timeframe alignment score used in
the Crypto-Only Refocus. The backtest pathway (strategies/smc_multi_style.py)
calls `compute_alignment_score` directly. The live pathway
(live_multi_bot.py._multi_tf_alignment_score) imports CORE_WEIGHTS_CRYPTO and
composes a richer, graded score with 5 additional bonus components that are
NOT part of the training-set score. Those bonus weights stay in the live
module because the SHAP-analysis (2026-04-18) showed they are not empirically
calibrated yet and a full SSOT merge would require a Phase 2.3 rebalance.

This is the "Minimal-Extrakt / Option A" recommended in
`.omc/research/alignment-ablation.md` — bit-identical to the previous
`_compute_alignment_score` in `strategies/smc_multi_style.py:1341-1409`.

Phase 2.1 of the Crypto-Only Refocus.
"""

from __future__ import annotations

# ------------------------------------------------------------------------
# Core component weights — SSOT for both backtest and live
# ------------------------------------------------------------------------

CORE_WEIGHTS_CRYPTO: dict[str, float] = {
    "bias": 0.12,
    "bias_strong": 0.08,
    "h4": 0.08,
    "h4_poi": 0.08,
    "h1": 0.08,
    "h1_choch": 0.06,
    "zone": 0.15,
    "trigger": 0.15,
    "volume": 0.10,
}
"""
Training-calibrated weights for the 9 core alignment components.
Sum = 0.90 (max core-score before style_weight multiplier).

Used by:
  - strategies/smc_multi_style.py: backtest path, binary flags
  - live_multi_bot.py._multi_tf_alignment_score: live path, graded
    (multiplied by zone_quality / vol_score / adx_score etc.)

Both paths MUST use these identical weights for the alignment_threshold
gate (0.78) to behave consistently between backtest and live.
"""

CORE_WEIGHTS_FOREX: dict[str, float] = {
    "bias": 0.12,
    "bias_strong": 0.12,
    "h4": 0.12,
    "h4_poi": 0.08,
    "h1": 0.10,
    "h1_choch": 0.06,
    "zone": 0.08,
    "trigger": 0.08,
    "volume": 0.14,
}
"""
Legacy forex weights retained for backward compatibility.

Crypto-Only Refocus (Phase 1, 2026-04-18) removed forex trading but the
`_compute_alignment_score(asset_class="forex")` branch still exists in
case any archived training data is re-processed. Deferred cleanup.
Sum = 0.90.
"""


def compute_alignment_score(
    daily_bias: str,
    h1_confirms: bool,
    entry_zone: dict | None,
    precision_trigger: bool,
    style_weight: float = 1.0,
    *,
    bias_strong: bool = False,
    h4_confirms: bool = False,
    h4_poi: bool = False,
    h1_choch: bool = False,
    volume_ok: bool = False,
    asset_class: str | None = None,
) -> float:
    """
    Granular top-down alignment score (0–1).

    Default scoring (crypto/stocks/commodities):
      • Daily bias (1D) +0.12, strong +0.08, 4H +0.08, 4H POI +0.08,
        1H +0.08, CHoCH +0.06, entry_zone +0.15, trigger +0.15, volume +0.10
      Max = 0.90

    Forex scoring (redistributed — entry_zone/trigger unreliable with tick volume):
      • Daily bias +0.12, strong +0.12, 4H +0.12, 4H POI +0.08,
        1H +0.10, CHoCH +0.06, entry_zone +0.08, trigger +0.08, volume +0.14
      Max = 0.90

    Clamped to [0, 1].
    """
    # Forex-specific weights: less on entry_zone/trigger, more on HTF structure
    w = CORE_WEIGHTS_FOREX if asset_class == "forex" else CORE_WEIGHTS_CRYPTO

    score = 0.0

    if daily_bias in ("bullish", "bearish"):
        score += w["bias"]
        if bias_strong:
            score += w["bias_strong"]
    if h4_confirms:
        score += w["h4"]
    if h4_poi:
        score += w["h4_poi"]
    if h1_confirms:
        score += w["h1"]
        if h1_choch:
            score += w["h1_choch"]
    if entry_zone is not None:
        score += w["zone"]
    if precision_trigger:
        score += w["trigger"]
    if volume_ok:
        score += w["volume"]

    # Backward compat: if using old 4-arg call and no new flags,
    # fall back to roughly equivalent old scoring
    old_style = (
        not bias_strong and not h4_confirms and not h4_poi
        and not h1_choch and not volume_ok
    )
    if old_style and score > 0:
        # Boost to roughly match old 0.25-per-step scale
        score = min(score * 1.3, 1.0)

    return min(score * style_weight, 1.0)


__all__ = [
    "CORE_WEIGHTS_CRYPTO",
    "CORE_WEIGHTS_FOREX",
    "compute_alignment_score",
]
