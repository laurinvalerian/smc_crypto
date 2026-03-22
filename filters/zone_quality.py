"""
Zone Quality Filter
===================
Continuous zone quality scoring with:
  - Exponential time decay
  - Unmitigated zone check
  - Zone size relative to ATR
  - Formation quality (impulse candle body/wick ratio)
  - HTF overlap bonus
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _compute_decay(age_bars: int, decay_rate: float = 0.15) -> float:
    """Exponential decay. Half-life ~4.6 bars at rate 0.15."""
    return math.exp(-decay_rate * age_bars)


def _check_unmitigated(
    zone_top: float,
    zone_bottom: float,
    closes: np.ndarray,
    zone_bar_idx: int,
    current_bar_idx: int,
    direction: str,
) -> float:
    """
    Check if price has NOT returned to and closed through this zone
    since its formation.

    Returns:
      1.0 = fully unmitigated (price never returned)
      0.5 = partially mitigated (price touched but didn't close through)
      0.0 = fully mitigated (price closed through the zone)
    """
    if zone_bar_idx >= current_bar_idx or current_bar_idx > len(closes):
        return 1.0

    # Check candles between zone formation and current bar
    check_start = zone_bar_idx + 1
    check_end = min(current_bar_idx, len(closes))

    touched = False

    for i in range(check_start, check_end):
        c = float(closes[i])
        if direction == "bullish":
            # For bullish zone (support): mitigated if price closed below bottom
            if c < zone_bottom:
                return 0.0
            if c < zone_top:
                touched = True
        else:
            # For bearish zone (resistance): mitigated if price closed above top
            if c > zone_top:
                return 0.0
            if c > zone_bottom:
                touched = True

    return 0.5 if touched else 1.0


def _zone_size_score(zone_range: float, atr: float) -> float:
    """
    Score zone size relative to ATR.
    Sweet spot is 0.5-2.0 ATR. Peak at 1.0 ATR.

    Too small (< 0.3 ATR) = noise, too large (> 3.0 ATR) = imprecise.
    """
    if atr <= 0 or zone_range <= 0:
        return 0.0

    ratio = zone_range / atr

    if ratio < 0.3:
        return 0.1  # Too small
    elif ratio <= 1.0:
        # Rising: 0.3 ATR → 0.5, 1.0 ATR → 1.0
        return 0.5 + 0.5 * (ratio - 0.3) / 0.7
    elif ratio <= 2.0:
        # Slightly declining: 1.0 ATR → 1.0, 2.0 ATR → 0.7
        return 1.0 - 0.3 * (ratio - 1.0)
    elif ratio <= 3.0:
        # More decline: 2.0 → 0.7, 3.0 → 0.3
        return 0.7 - 0.4 * (ratio - 2.0)
    else:
        return 0.1  # Too large


def _formation_quality(
    df: pd.DataFrame,
    zone_bar_idx: int,
) -> float:
    """
    Score the quality of the impulse candle that formed the zone.
    Strong body with small wicks = clean zone.

    Returns 0.0-1.0.
    """
    if zone_bar_idx < 0 or zone_bar_idx >= len(df):
        return 0.5

    row = df.iloc[zone_bar_idx]
    o = float(row.get("open", 0))
    h = float(row.get("high", 0))
    l = float(row.get("low", 0))
    c = float(row.get("close", 0))

    total_range = h - l
    if total_range <= 0:
        return 0.0

    body = abs(c - o)
    body_ratio = body / total_range

    # High body ratio = strong impulse candle = clean zone
    # 0.7+ body ratio is very clean
    if body_ratio >= 0.7:
        return 1.0
    elif body_ratio >= 0.5:
        return 0.7 + 0.3 * (body_ratio - 0.5) / 0.2
    elif body_ratio >= 0.3:
        return 0.4 + 0.3 * (body_ratio - 0.3) / 0.2
    else:
        return 0.2  # Doji/spinning top = weak zone


def _htf_overlap_bonus(
    zone_top: float,
    zone_bottom: float,
    htf_zones: list[dict] | None,
) -> float:
    """
    Check if a higher-timeframe zone (4H or 1D) overlaps with this entry zone.
    Overlapping zones are stronger.

    Returns 0.0 or 1.0.
    """
    if not htf_zones:
        return 0.0

    for hz in htf_zones:
        hz_top = hz.get("top", 0)
        hz_bottom = hz.get("bottom", 0)
        # Check overlap
        if hz_bottom <= zone_top and hz_top >= zone_bottom:
            return 1.0

    return 0.0


def compute_zone_quality(
    zone_data: dict | None,
    zone_bar_idx: int,
    current_bar_idx: int,
    closes_15m: np.ndarray | None = None,
    df_15m: pd.DataFrame | None = None,
    atr_15m: float = 0.0,
    htf_zones: list[dict] | None = None,
    decay_rate: float = 0.15,
) -> dict[str, Any]:
    """
    Compute comprehensive zone quality score.

    Parameters:
      zone_data: dict with "top", "bottom", "direction", "type"
      zone_bar_idx: bar index where zone was formed
      current_bar_idx: current bar index
      closes_15m: 15m close prices for unmitigated check
      df_15m: 15m DataFrame for formation quality
      atr_15m: ATR(14) on 15m for size scoring
      htf_zones: list of higher-timeframe zones for overlap check
      decay_rate: exponential decay rate (default 0.15)

    Returns dict with:
      - decay_factor: float (0-1)
      - unmitigated_score: float (0, 0.5, or 1.0)
      - size_score: float (0-1)
      - formation_score: float (0-1)
      - htf_overlap: float (0 or 1)
      - zone_quality: float (0-1, composite)
      - zone_quality_ok: bool (>= 0.7 threshold)
    """
    from typing import Any

    if zone_data is None:
        return {
            "decay_factor": 0.0,
            "unmitigated_score": 0.0,
            "size_score": 0.0,
            "formation_score": 0.0,
            "htf_overlap": 0.0,
            "zone_quality": 0.0,
            "zone_quality_ok": False,
        }

    zone_top = zone_data.get("top", 0)
    zone_bottom = zone_data.get("bottom", 0)
    direction = zone_data.get("direction", "bullish")
    zone_range = abs(zone_top - zone_bottom)

    # Age-based decay
    age_bars = max(0, current_bar_idx - zone_bar_idx)
    decay = _compute_decay(age_bars, decay_rate)

    # Unmitigated check
    unmitigated = 1.0
    if closes_15m is not None and len(closes_15m) > 0:
        unmitigated = _check_unmitigated(
            zone_top, zone_bottom, closes_15m,
            zone_bar_idx, current_bar_idx, direction,
        )

    # Size relative to ATR
    size_sc = _zone_size_score(zone_range, atr_15m)

    # Formation quality
    formation = 0.5
    if df_15m is not None:
        formation = _formation_quality(df_15m, zone_bar_idx)

    # HTF overlap
    htf = _htf_overlap_bonus(zone_top, zone_bottom, htf_zones)

    # Composite: decay × unmitigated × (weighted factors)
    # The decay and unmitigated act as hard multipliers (gate)
    # The rest are soft factors
    base_quality = size_sc * 0.35 + formation * 0.35 + (0.30 if htf > 0 else 0.0)
    zone_quality = decay * unmitigated * base_quality

    # Boost for HTF overlap (can push score higher)
    if htf > 0:
        zone_quality = min(zone_quality * 1.2, 1.0)

    zone_quality = float(np.clip(zone_quality, 0.0, 1.0))

    return {
        "decay_factor": decay,
        "unmitigated_score": unmitigated,
        "size_score": size_sc,
        "formation_score": formation,
        "htf_overlap": htf,
        "zone_quality": zone_quality,
        "zone_quality_ok": zone_quality >= 0.7,
    }
