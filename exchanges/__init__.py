"""
Exchange Adapters Package
=========================
Provides a uniform interface for trading across different exchanges/brokers.

Usage:
    from exchanges import BinanceAdapter, OandaAdapter, AlpacaAdapter
"""
from exchanges.base import ExchangeAdapter
from exchanges.binance_adapter import BinanceAdapter
from exchanges.models import (
    BalanceInfo,
    InstrumentMeta,
    OrderResult,
    PositionInfo,
)

# Lazy imports for optional adapters (avoid ImportError if v20/alpaca-py not installed)


def _get_oanda_adapter():
    from exchanges.oanda_adapter import OandaAdapter
    return OandaAdapter


def _get_alpaca_adapter():
    from exchanges.alpaca_adapter import AlpacaAdapter
    return AlpacaAdapter


__all__ = [
    "ExchangeAdapter",
    "BinanceAdapter",
    "InstrumentMeta",
    "OrderResult",
    "PositionInfo",
    "BalanceInfo",
]
