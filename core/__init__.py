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
    LEVERAGE_CAP,
    BINANCE_TAKER_FEE,
    DEFAULT_RISK_PER_TRADE,
    MAX_PORTFOLIO_HEAT,
    MAX_RISK_PER_TRADE,
    SCALP_MAX_HOLD_BARS,
)

from core.alignment import (
    compute_alignment_score,
    CORE_WEIGHTS_CRYPTO,
    CORE_WEIGHTS_FOREX,
)

from core.sizing import (
    compute_risk_fraction,
    compute_risk_amount,
)

__all__ = [
    "COMMISSION",
    "SLIPPAGE",
    "ALIGNMENT_THRESHOLD",
    "LEVERAGE_CAP",
    "BINANCE_TAKER_FEE",
    "DEFAULT_RISK_PER_TRADE",
    "MAX_PORTFOLIO_HEAT",
    "MAX_RISK_PER_TRADE",
    "SCALP_MAX_HOLD_BARS",
    "compute_alignment_score",
    "CORE_WEIGHTS_CRYPTO",
    "CORE_WEIGHTS_FOREX",
    "compute_risk_fraction",
    "compute_risk_amount",
]
