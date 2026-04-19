"""
SMC Quality Filters
===================
Confluence modules that feed the alignment score (core.alignment) and the
confidence-based risk sizer (core.sizing).

Modules:
  - trend_strength : ADX, momentum confluence, multi-TF trend agreement
  - volume_liquidity : 3-layer volume scoring, dollar-volume floor, volume profile
  - session_filter : Session-aware scoring per asset class
  - zone_quality : Zone decay, unmitigated check, formation quality
"""

from filters.trend_strength import (
    compute_adx,
    check_momentum_confluence,
    multi_tf_trend_agreement,
    compute_trend_strength_score,
)
from filters.volume_liquidity import (
    compute_volume_score,
    check_dollar_volume_floor,
    compute_volume_profile_score,
)
from filters.session_filter import compute_session_score
from filters.zone_quality import compute_zone_quality

__all__ = [
    "compute_adx",
    "check_momentum_confluence",
    "multi_tf_trend_agreement",
    "compute_trend_strength_score",
    "compute_volume_score",
    "check_dollar_volume_floor",
    "compute_volume_profile_score",
    "compute_session_score",
    "compute_zone_quality",
]
