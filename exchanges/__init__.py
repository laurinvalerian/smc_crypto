"""
Exchange Adapters Package
=========================
Provides a uniform interface for trading across different exchanges/brokers.

Usage:
    from exchanges import BinanceAdapter, ExchangeAdapter, InstrumentMeta
"""
from exchanges.base import ExchangeAdapter
from exchanges.binance_adapter import BinanceAdapter
from exchanges.models import (
    BalanceInfo,
    InstrumentMeta,
    OrderResult,
    PositionInfo,
)

__all__ = [
    "ExchangeAdapter",
    "BinanceAdapter",
    "InstrumentMeta",
    "OrderResult",
    "PositionInfo",
    "BalanceInfo",
]
