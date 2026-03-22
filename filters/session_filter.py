"""
Session Filter
==============
Session-aware scoring per asset class.

Each asset class has optimal trading windows where volume and liquidity
are highest. Trading outside these windows means wider spreads,
more stop-hunts, and lower probability setups.
"""
from __future__ import annotations

from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════
#  Session Definitions (UTC hours)
# ═══════════════════════════════════════════════════════════════════

# Each entry: (start_hour, end_hour, score)
# Ranges are inclusive on start, exclusive on end.

CRYPTO_SESSIONS = [
    # London open + NY open = peak crypto volume
    (7, 10, 1.0),    # London open
    (13, 17, 1.0),   # NY open + overlap
    (10, 13, 0.8),   # London midday
    (17, 21, 0.7),   # NY afternoon
    (0, 7, 0.5),     # Asian session
    (21, 24, 0.5),   # Late evening
]

FOREX_SESSIONS = [
    # London+NY overlap is king for forex
    (13, 17, 1.0),   # London-NY overlap
    (7, 13, 0.9),    # London session
    (17, 21, 0.6),   # NY afternoon (thinner)
    (0, 7, 0.4),     # Asian (good for JPY pairs only)
    (21, 24, 0.3),   # Dead zone
]

STOCKS_US_SESSIONS = [
    # Regular trading hours only (14:30-21:00 UTC)
    (14, 16, 1.0),   # Market open (highest volume)
    (16, 19, 0.8),   # Midday
    (19, 21, 0.9),   # Power hour / close
    # Outside hours = 0.0 (no trading)
]

COMMODITIES_SESSIONS = [
    # Gold/Silver: London + NY
    (7, 10, 0.9),    # London open
    (13, 17, 1.0),   # NY session (highest gold volume)
    (10, 13, 0.7),   # London midday
    (17, 21, 0.6),   # NY afternoon
    (0, 7, 0.3),     # Asian (low gold volume)
    (21, 24, 0.3),
]

SESSION_MAP = {
    "crypto": CRYPTO_SESSIONS,
    "forex": FOREX_SESSIONS,
    "stocks": STOCKS_US_SESSIONS,
    "commodities": COMMODITIES_SESSIONS,
}


def compute_session_score(
    asset_class: str = "crypto",
    utc_now: datetime | None = None,
) -> float:
    """
    Compute session quality score based on current UTC time and asset class.

    Returns score 0.0-1.0.
    For stocks outside trading hours, returns 0.0.
    """
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    hour = utc_now.hour
    sessions = SESSION_MAP.get(asset_class, CRYPTO_SESSIONS)

    # Find the matching session
    for start, end, score in sessions:
        if start <= hour < end:
            return score

    # Stocks: outside all defined sessions = no trading
    if asset_class == "stocks":
        return 0.0

    # Default fallback for undefined hours
    return 0.3


def is_tradeable_session(
    asset_class: str = "crypto",
    min_session_score: float = 0.5,
    utc_now: datetime | None = None,
) -> tuple[bool, float]:
    """
    Check if the current session meets minimum quality.

    Returns (is_tradeable, session_score).
    """
    score = compute_session_score(asset_class, utc_now)
    return score >= min_session_score, score
