"""
Abstract Exchange Adapter
=========================
Defines the interface that every exchange adapter must implement.
This decouples the trading bot from any specific exchange/broker.
"""
from __future__ import annotations

import abc
import logging
from datetime import datetime
from typing import Any

from exchanges.models import (
    BalanceInfo,
    InstrumentMeta,
    OrderResult,
    PositionInfo,
)

logger = logging.getLogger(__name__)


class ExchangeAdapter(abc.ABC):
    """
    Abstract base class for exchange adapters.

    Every concrete adapter (Binance, OANDA, Alpaca, ...) implements these
    methods so the bot can trade any asset class through a uniform interface.
    """

    # ── Identity ────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def exchange_id(self) -> str:
        """Short identifier, e.g. 'binanceusdm', 'oanda', 'alpaca'."""

    @property
    @abc.abstractmethod
    def asset_class(self) -> str:
        """Primary asset class: 'crypto', 'forex', 'stocks', 'commodities'."""

    # ── Lifecycle ───────────────────────────────────────────────────

    @abc.abstractmethod
    async def connect(self) -> None:
        """Initialise connection, load markets, authenticate."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Graceful shutdown: close WS connections, release resources."""

    # ── Market Data ─────────────────────────────────────────────────

    @abc.abstractmethod
    async def load_markets(self) -> dict[str, InstrumentMeta]:
        """
        Load and cache all available instruments.

        Returns dict[unified_symbol, InstrumentMeta].
        """

    @abc.abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list[float]]:
        """
        Fetch historical OHLCV bars.

        Returns list of [timestamp_ms, open, high, low, close, volume].
        """

    async def fetch_ohlcv_sync(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list[float]]:
        """
        Synchronous OHLCV fetch (for history loading at startup).

        Default falls back to fetch_ohlcv. Override if a separate
        sync client is needed (e.g. Binance uses ccxt sync for startup).
        """
        return await self.fetch_ohlcv(symbol, timeframe, since, limit)

    @abc.abstractmethod
    async def watch_ohlcv(self, symbol: str, timeframe: str = "5m") -> list[list]:
        """Subscribe to live OHLCV candles via WebSocket. Returns new bars."""

    @abc.abstractmethod
    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        """Subscribe to live ticker (bid/ask/last) via WebSocket."""

    # ── Instrument Info ─────────────────────────────────────────────

    @abc.abstractmethod
    def get_instrument(self, symbol: str) -> InstrumentMeta | None:
        """Return cached InstrumentMeta for a symbol, or None."""

    def normalize_symbol(self, symbol: str) -> str:
        """
        Convert an external symbol to the adapter's internal format.

        Default: identity. Override for exchange-specific normalisation
        (e.g. 'BTC/USDT:USDT' → 'BTCUSDT' on Binance).
        """
        return symbol

    def get_exchange_symbol(self, symbol: str) -> str:
        """Return the exchange-native symbol string."""
        meta = self.get_instrument(symbol)
        return meta.exchange_symbol if meta else self.normalize_symbol(symbol)

    # ── Precision Helpers ───────────────────────────────────────────

    @abc.abstractmethod
    def price_to_precision(self, symbol: str, price: float) -> float:
        """Round price to exchange tick size."""

    @abc.abstractmethod
    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Round quantity to exchange lot size."""

    # ── Trading ─────────────────────────────────────────────────────

    @abc.abstractmethod
    async def create_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        """Place a market order. Returns normalised OrderResult."""

    @abc.abstractmethod
    async def create_stop_loss(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        """Place a stop-loss order (STOP_MARKET or equivalent)."""

    @abc.abstractmethod
    async def create_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        """Place a take-profit order (TAKE_PROFIT_MARKET or equivalent)."""

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order. Returns True on success."""

    @abc.abstractmethod
    async def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Return list of open orders for a symbol."""

    # ── Account ─────────────────────────────────────────────────────

    @abc.abstractmethod
    async def fetch_balance(self) -> BalanceInfo:
        """Return normalised account balance."""

    @abc.abstractmethod
    async def fetch_positions(self) -> list[PositionInfo]:
        """Return all open positions."""

    @abc.abstractmethod
    async def fetch_my_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return recent trades for a symbol."""

    # ── Leverage & Margin ───────────────────────────────────────────

    @abc.abstractmethod
    async def set_leverage(self, leverage: int, symbol: str) -> None:
        """Set leverage for a symbol."""

    @abc.abstractmethod
    async def set_margin_mode(self, mode: str, symbol: str) -> None:
        """Set margin mode ('cross' or 'isolated') for a symbol."""

    async def fetch_max_leverage(self, symbol: str) -> int:
        """
        Fetch the maximum allowed leverage for a symbol.

        Default: return max_leverage from InstrumentMeta.
        Override for exchanges that require API calls for bracket data.
        """
        meta = self.get_instrument(symbol)
        return meta.max_leverage if meta else 1

    # ── Trading Hours ───────────────────────────────────────────────

    def is_market_open(self, symbol: str, utc_now: datetime | None = None) -> bool:
        """
        Check if the market is open for trading.

        Default: always open (crypto). Override for stocks/forex.
        """
        meta = self.get_instrument(symbol)
        if meta and meta.trades_24_7:
            return True

        if utc_now is None:
            from datetime import timezone
            utc_now = datetime.now(timezone.utc)

        if meta and meta.trading_hours:
            hour = utc_now.hour
            for start_h, end_h in meta.trading_hours:
                if start_h <= hour < end_h:
                    return True
            return False

        return True  # Default: assume open
