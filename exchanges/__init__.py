"""
Exchange Adapters Package
=========================
Provides a uniform interface for trading across different exchanges/brokers.

Usage:
    from exchanges import BinanceAdapter, OandaAdapter, AlpacaAdapter
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


def _get_oanda_adapter():
    from exchanges.oanda_adapter import OandaAdapter
    return OandaAdapter


def _get_alpaca_adapter():
    from exchanges.alpaca_adapter import AlpacaAdapter
    return AlpacaAdapter


__all__ = [
    "ExchangeAdapter",
    "InstrumentMeta",
    "OrderResult",
    "PositionInfo",
    "BalanceInfo",
]
