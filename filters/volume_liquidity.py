"""
Volume & Liquidity Filter
=========================
3-layer volume scoring, dollar-volume floor, and volume profile analysis.

Ensures trades are only taken in liquid instruments with above-average
volume confirmation.
"""
from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  Layer 1: Relative Volume Score
# ═══════════════════════════════════════════════════════════════════

def compute_relative_volume_score(
    volumes: np.ndarray,
    min_ratio_threshold: float = 1.5,
) -> tuple[float, float]:
    """
    Compare current volume against 100-bar rolling average.

    Score = min(current_vol / avg_100, 3.0) / 3.0
    Minimum threshold: current must be >= min_ratio_threshold × avg_100.

    Returns (score 0-1, ratio).
    """
    if len(volumes) < 20:
        return 0.0, 0.0

    current_vol = float(volumes[-1])
    lookback = min(100, len(volumes) - 1)
    avg_vol = float(np.mean(volumes[-(lookback + 1):-1]))

    if avg_vol <= 0:
        return 0.0, 0.0

    ratio = current_vol / avg_vol
    score = min(ratio, 3.0) / 3.0

    # Below minimum threshold = hard reject (score 0)
    if ratio < min_ratio_threshold:
        return 0.0, ratio

    return float(np.clip(score, 0.0, 1.0)), ratio


# ═══════════════════════════════════════════════════════════════════
#  Layer 2: Dollar Volume Floor
# ═══════════════════════════════════════════════════════════════════

# Minimum USDT-equivalent volume per 5m bar
DOLLAR_VOL_FLOORS = {
    "crypto": 50_000,       # $50K per 5m bar
    "forex": 100_000,       # $100K per 5m bar
    "stocks": 100_000,      # $100K per 5m bar
    "commodities": 50_000,  # $50K per 5m bar
}


def check_dollar_volume_floor(
    price: float,
    volume: float,
    asset_class: str = "crypto",
) -> tuple[bool, float]:
    """
    Check if dollar volume (price × volume) meets the minimum floor.

    Returns (passes_floor, dollar_volume).
    """
    dollar_vol = price * volume
    floor = DOLLAR_VOL_FLOORS.get(asset_class, 50_000)
    return dollar_vol >= floor, dollar_vol


# ═══════════════════════════════════════════════════════════════════
#  Layer 3: Volume Profile Score
# ═══════════════════════════════════════════════════════════════════

def compute_volume_profile_score(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    entry_price: float,
    tp_price: float,
    direction: str,
    num_bins: int = 10,
) -> float:
    """
    Simplified volume profile on 1H data.

    Divides the price range into bins, sums volume per bin.
    Good trade = entry near high-volume node (support), TP toward
    low-volume area (path of least resistance).

    Returns score 0-1.
    """
    n = len(closes)
    if n < 20:
        return 0.5  # neutral if insufficient data

    # Use last 100 bars for profile
    lookback = min(100, n)
    h = highs[-lookback:].astype(np.float64)
    l = lows[-lookback:].astype(np.float64)
    v = volumes[-lookback:].astype(np.float64)

    price_min = float(np.min(l))
    price_max = float(np.max(h))
    price_range = price_max - price_min

    if price_range <= 0 or np.sum(v) <= 0:
        return 0.5

    # Create volume profile bins
    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_volumes = np.zeros(num_bins, dtype=np.float64)

    # Distribute each bar's volume across bins it spans
    for i in range(lookback):
        bar_low = float(l[i])
        bar_high = float(h[i])
        bar_vol = float(v[i])
        if bar_vol <= 0 or bar_high <= bar_low:
            continue
        for b in range(num_bins):
            overlap_low = max(bar_low, bin_edges[b])
            overlap_high = min(bar_high, bin_edges[b + 1])
            if overlap_high > overlap_low:
                fraction = (overlap_high - overlap_low) / (bar_high - bar_low)
                bin_volumes[b] += bar_vol * fraction

    total_vol = np.sum(bin_volumes)
    if total_vol <= 0:
        return 0.5

    # Normalize to density
    vol_density = bin_volumes / total_vol

    # Find entry bin and TP bin
    entry_bin = int(np.clip(
        (entry_price - price_min) / price_range * num_bins, 0, num_bins - 1
    ))
    tp_bin = int(np.clip(
        (tp_price - price_min) / price_range * num_bins, 0, num_bins - 1
    ))

    # Score: entry at high-volume node (support/resistance) = good
    entry_vol_score = min(vol_density[entry_bin] * num_bins, 2.0) / 2.0

    # Score: path to TP through low-volume = good (less resistance)
    if direction == "long":
        path_bins = range(entry_bin, min(tp_bin + 1, num_bins))
    else:
        path_bins = range(max(tp_bin, 0), entry_bin + 1)

    if len(list(path_bins)) > 0:
        path_vol = np.mean([vol_density[b] for b in path_bins])
        # Low path volume = high score (1/num_bins is "average" density)
        avg_density = 1.0 / num_bins
        path_score = max(0.0, 1.0 - (path_vol / avg_density - 0.5))
        path_score = float(np.clip(path_score, 0.0, 1.0))
    else:
        path_score = 0.5

    return float(np.clip(entry_vol_score * 0.4 + path_score * 0.6, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════
#  Combined Volume Score
# ═══════════════════════════════════════════════════════════════════

def compute_volume_score(
    volumes_5m: np.ndarray,
    price: float,
    current_volume: float,
    asset_class: str = "crypto",
    highs_1h: np.ndarray | None = None,
    lows_1h: np.ndarray | None = None,
    closes_1h: np.ndarray | None = None,
    volumes_1h: np.ndarray | None = None,
    entry_price: float | None = None,
    tp_price: float | None = None,
    direction: str = "long",
) -> dict[str, Any]:
    """
    Compute all volume/liquidity metrics.

    Returns dict with:
      - relative_score: float (0-1)
      - relative_ratio: float
      - dollar_vol_ok: bool
      - dollar_volume: float
      - profile_score: float (0-1)
      - volume_score: float (0-1, composite)
      - volume_ok: bool (passes minimum requirements)
    """
    from typing import Any

    # Layer 1: Relative volume
    rel_score, rel_ratio = compute_relative_volume_score(volumes_5m, min_ratio_threshold=1.5)

    # Layer 2: Dollar volume floor
    dollar_ok, dollar_vol = check_dollar_volume_floor(price, current_volume, asset_class)

    # Layer 3: Volume profile (optional, needs 1H data)
    profile_score = 0.5  # neutral default
    if (highs_1h is not None and lows_1h is not None and
            closes_1h is not None and volumes_1h is not None and
            entry_price is not None and tp_price is not None):
        profile_score = compute_volume_profile_score(
            highs_1h, lows_1h, closes_1h, volumes_1h,
            entry_price, tp_price, direction,
        )

    # Composite score
    composite = rel_score * 0.45 + profile_score * 0.30 + (1.0 if dollar_ok else 0.0) * 0.25
    volume_ok = rel_score > 0.0 and dollar_ok  # Both must pass

    return {
        "relative_score": rel_score,
        "relative_ratio": rel_ratio,
        "dollar_vol_ok": dollar_ok,
        "dollar_volume": dollar_vol,
        "profile_score": profile_score,
        "volume_score": float(np.clip(composite, 0.0, 1.0)),
        "volume_ok": volume_ok,
    }
