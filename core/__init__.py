"""
core — Single Source of Truth for cross-cutting trading constants and logic.

Phase 2.1 of the Crypto-Only Refocus introduces this package to eliminate
duplication between live (live_multi_bot.py) and backtest (generate_rl_data.py)
code paths. Any "MUST match"-style comment that previously bound two files
together should be replaced by a single import from `core.*`.

Modules:
    constants — COMMISSION, SLIPPAGE, ALIGNMENT_THRESHOLD, asset metadata
    alignment — compute_alignment_score (pure, bit-identical SSOT since 2.1)
"""

from core.constants import (
    COMMISSION,
    SLIPPAGE,
    ALIGNMENT_THRESHOLD,
    AAA_PLUS_PLUS_THRESHOLD,
    AAA_PLUS_THRESHOLD,
    LEVERAGE_CAP,
    BINANCE_TAKER_FEE,
    DEFAULT_RISK_PER_TRADE,
    MAX_PORTFOLIO_HEAT,
    MAX_RISK_PER_TRADE,
)

from core.alignment import (
    compute_alignment_score,
    CORE_WEIGHTS_CRYPTO,
    CORE_WEIGHTS_FOREX,
)

__all__ = [
    "COMMISSION",
    "SLIPPAGE",
    "ALIGNMENT_THRESHOLD",
    "AAA_PLUS_PLUS_THRESHOLD",
    "AAA_PLUS_THRESHOLD",
    "LEVERAGE_CAP",
    "BINANCE_TAKER_FEE",
    "DEFAULT_RISK_PER_TRADE",
    "MAX_PORTFOLIO_HEAT",
    "MAX_RISK_PER_TRADE",
    "compute_alignment_score",
    "CORE_WEIGHTS_CRYPTO",
    "CORE_WEIGHTS_FOREX",
]
