"""
Exchange Adapters Package
=========================
Provides a uniform interface for trading across different exchanges/brokers.

Usage:
    from exchanges import BinanceAdapter
"""
from exchanges.base import ExchangeAdapter
from exchanges.models import (
    BalanceInfo,
    InstrumentMeta,
    OrderResult,
    PositionInfo,
)

# Lazy imports for adapters (avoid ImportError if ccxt/v20/alpaca-py not installed)
try:
    from exchanges.binance_adapter import BinanceAdapter
except ImportError:
    BinanceAdapter = None  # type: ignore[assignment,misc]


def _get_binance_adapter():
    from exchanges.binance_adapter import BinanceAdapter
    return BinanceAdapter


__all__ = [
    "ExchangeAdapter",
    "InstrumentMeta",
    "OrderResult",
    "PositionInfo",
    "BalanceInfo",
]
