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
# Alignment Gate (single threshold, no tiers)
# ------------------------------------------------------------------------

ALIGNMENT_THRESHOLD: float = 0.78
"""
Single Source of Truth for the alignment-score gate.

Signals with score < ALIGNMENT_THRESHOLD are rejected outright.
Signals with score >= ALIGNMENT_THRESHOLD pass; risk sizing is then
scaled linearly by confidence (see core/sizing.py). No AAA++/AAA+ tier
classification — killed 2026-04-19 because Student size-head + linear
scaling already captures the confidence gradient the tier system tried
to hard-code.
"""


# ------------------------------------------------------------------------
# Risk / Position Sizing
# ------------------------------------------------------------------------

DEFAULT_RISK_PER_TRADE: float = 0.0025
"""
Default risk per trade as fraction of equity (0.25%, at alignment threshold).

Lowered from 0.5% → 0.25% (2026-04-19) because Scalp-Day Hybrid produces
~3-5× more trades than the old Sniper-Day approach. Smaller per-trade
risk is needed to keep portfolio-heat and daily-DD under funded limits.
"""

MAX_RISK_PER_TRADE: float = 0.010
"""
Hard cap on risk per trade (1.0%, at alignment score 1.0).

Lowered from 1.5% → 1.0% (2026-04-19) — see DEFAULT_RISK_PER_TRADE.
Still well inside funded-account compliance (<1.5%).
"""

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
# Holding period (Scalp-Day Hybrid, 2026-04-19 decision)
# ------------------------------------------------------------------------

SCALP_MAX_HOLD_BARS: int = 48
"""
Maximum holding period = 48 bars × 5m = 4 hours.

Scalp-Day Hybrid: 5m entry, SMC structure plays out in 1–4h, Commission
budget still healthy. Tighter than classical day (24h) for shorter RL
feedback loop; longer than pure scalp (<2h) so SMC targets have room.
"""


# ------------------------------------------------------------------------
# Asset Class (Crypto-Only after Phase 1 strip)
# ------------------------------------------------------------------------

ASSET_CLASS: str = "crypto"
"""Sole asset class after Crypto-Only Refocus (2026-04-18)."""

ASSET_CLASS_ID: int = 0
"""XGBoost asset_class_id feature value (matches rl_brain_v2.ASSET_CLASS_MAP)."""
