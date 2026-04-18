"""
core.constants — Trading constants as the Single Source of Truth.

These values were previously duplicated across:
  - live_multi_bot.py (ASSET_COMMISSION, ASSET_SLIPPAGE, ASSET_SMC_PARAMS)
  - backtest/generate_rl_data.py (_TRAIN_COMMISSION, _TRAIN_SLIPPAGE)
  - backtest/optuna_backtester.py (ASSET_COMMISSION, ASSET_SLIPPAGE)
  - config/default_config.yaml (top_down.alignment_threshold)
  - CLAUDE.md (documentation)

The 3-fold alignment-threshold discrepancy (config 0.65 / code 0.78 /
CLAUDE.md 0.88) flagged in the 2026-04-18 audit is resolved here:
ALIGNMENT_THRESHOLD = 0.78 is the authoritative value.

Phase 2.1 (TODO 2.1c, .omc/plans/crypto-only-refocus.md) will migrate
all readers in the codebase to import from this module.
"""

# ------------------------------------------------------------------------
# Trading Costs (Crypto-Only, Binance USDT-M Futures)
# ------------------------------------------------------------------------

BINANCE_TAKER_FEE: float = 0.0004
"""Binance Futures taker fee (4bps). Applied at entry and exit."""

COMMISSION: float = BINANCE_TAKER_FEE
"""Total commission per side. Crypto-Only after Phase 1 strip."""

SLIPPAGE: float = 0.0002
"""Conservative slippage assumption (2bps). Applied to both legs."""


# ------------------------------------------------------------------------
# Tier Thresholds
# ------------------------------------------------------------------------

ALIGNMENT_THRESHOLD: float = 0.78
"""
Single Source of Truth for the alignment-score gate.

Previously inconsistent: config/default_config.yaml had 0.65 (never read),
live_multi_bot.py:213 hardcoded 0.78 (actual live value), CLAUDE.md
documented 0.88 (Sniper aspiration). 0.78 is the actual live behaviour
verified in the 2026-04-18 audit and is now authoritative.

Note: The 0.88 value lives on as AAA_PLUS_PLUS_THRESHOLD — the tier
classification ceiling, not the entry gate.
"""

AAA_PLUS_PLUS_THRESHOLD: float = 0.88
"""
Tier-classification ceiling for AAA++ (highest-conviction setups).

A trade is AAA++ if alignment_score >= 0.88 AND all 11 component flags
are True. AAA++ trades use the full risk allocation (1.0-1.5% per trade).
Below this but >= ALIGNMENT_THRESHOLD: AAA+ tier (0.5-1% risk).
"""

AAA_PLUS_THRESHOLD: float = ALIGNMENT_THRESHOLD
"""Tier-classification floor for AAA+. Same as the entry gate."""


# ------------------------------------------------------------------------
# Risk / Position Sizing
# ------------------------------------------------------------------------

DEFAULT_RISK_PER_TRADE: float = 0.005
"""Default risk per trade as fraction of equity (0.5%, conservative fallback)."""

MAX_RISK_PER_TRADE: float = 0.015
"""Hard cap on risk per trade (1.5%). Funded-account compliance limit."""

MAX_PORTFOLIO_HEAT: float = 0.06
"""
Maximum sum of open-position risk as fraction of equity (6%).

Enforced by risk/circuit_breaker.py before opening a new bracket order.
"""


# ------------------------------------------------------------------------
# Leverage
# ------------------------------------------------------------------------

LEVERAGE_CAP: int = 10
"""Maximum leverage for crypto (Binance USDT-M Futures, V14 reduced cap)."""


# ------------------------------------------------------------------------
# Asset Class (Crypto-Only after Phase 1 strip)
# ------------------------------------------------------------------------

ASSET_CLASS: str = "crypto"
"""Sole asset class after Crypto-Only Refocus (2026-04-18)."""

ASSET_CLASS_ID: int = 0
"""XGBoost asset_class_id feature value (matches rl_brain_v2.ASSET_CLASS_MAP)."""
