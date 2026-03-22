"""
Exchange Data Models
====================
Shared dataclasses used across all exchange adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstrumentMeta:
    """
    Normalised metadata for any tradeable instrument, regardless of exchange.

    Populated once when markets are loaded and cached per symbol.
    """

    symbol: str                     # Unified symbol, e.g. "BTC/USDT:USDT", "EUR/USD", "AAPL"
    exchange_symbol: str = ""       # Exchange-native symbol, e.g. "BTCUSDT", "EUR_USD"
    asset_class: str = "crypto"     # "crypto" | "forex" | "stocks" | "commodities"
    exchange_id: str = ""           # e.g. "binanceusdm", "oanda", "alpaca"

    # Precision & limits
    tick_size: float = 0.01         # Minimum price increment
    lot_size: float = 0.001         # Minimum quantity step
    min_qty: float = 0.0            # Minimum order quantity
    max_qty: float | None = None    # Maximum order quantity (None = no limit)
    min_notional: float = 5.0       # Minimum notional value (USDT/USD)
    max_notional: float | None = None

    # Leverage
    max_leverage: int = 1           # Exchange-allowed maximum
    default_leverage: int = 1

    # Trading hours (UTC)
    trading_hours: list[tuple[int, int]] | None = None  # [(start_h, end_h), ...]
    trades_24_7: bool = False       # Crypto = True

    # Costs
    spread_typical_pct: float = 0.0001   # Typical spread as fraction of price
    commission_pct: float = 0.0004       # One-way commission (taker)

    # Raw exchange info (for adapter-specific logic)
    raw_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """Normalised result from an order placement."""

    order_id: str | None = None
    symbol: str = ""
    side: str = ""                  # "buy" | "sell"
    order_type: str = ""            # "market" | "limit" | "stop_market" | "take_profit_market"
    qty: float = 0.0
    price: float | None = None      # Fill price (market) or trigger price (stop)
    status: str = "unknown"         # "filled" | "open" | "cancelled" | "failed"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionInfo:
    """Normalised open position."""

    symbol: str = ""
    side: str = ""                  # "long" | "short"
    qty: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    margin_mode: str = "cross"      # "cross" | "isolated"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BalanceInfo:
    """Normalised account balance."""

    currency: str = "USDT"
    total: float = 0.0
    free: float = 0.0
    used: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
