"""
Alpaca Adapter
==============
Exchange adapter for Alpaca Markets (US Stocks).
Supports paper and live trading via Alpaca REST + WebSocket API.

Requires: pip install alpaca-py

Environment variables:
  ALPACA_API_KEY    — Your Alpaca API key
  ALPACA_SECRET_KEY — Your Alpaca secret key
  ALPACA_PAPER      — "true" for paper trading (default)
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any

from exchanges.base import ExchangeAdapter
from exchanges.models import (
    BalanceInfo,
    InstrumentMeta,
    OrderResult,
    PositionInfo,
)

logger = logging.getLogger(__name__)

# ── Top 50 US Stocks by Market Cap ─────────────────────────────────

US_STOCK_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK.B", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA",
    "JNJ", "PG", "COST", "HD", "ABBV",
    "MRK", "CRM", "AMD", "NFLX", "BAC",
    "CVX", "KO", "PEP", "LIN", "TMO",
    "ADBE", "WMT", "MCD", "CSCO", "ACN",
    "ABT", "DHR", "TXN", "NEE", "PM",
    "ORCL", "INTC", "CMCSA", "DIS", "VZ",
    "IBM", "QCOM", "AMGN", "GE", "INTU",
]

# Alpaca timeframe mapping
TF_MAP: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
}

# US Stock Market Hours (Eastern Time → UTC)
# Regular hours: 9:30-16:00 ET = 14:30-21:00 UTC (EST) or 13:30-20:00 UTC (EDT)
# Using UTC approximation (varies with DST)
US_MARKET_HOURS_UTC: list[tuple[int, int]] = [(13, 21)]  # Conservative range covering both EST/EDT


class AlpacaAdapter(ExchangeAdapter):
    """
    Exchange adapter for Alpaca Markets (US Stocks).

    Features:
    - REST API for orders and account management
    - WebSocket for real-time market data
    - Paper trading support (default)
    - Max leverage 4x (Reg T margin)
    - Position sizing in shares (fractional supported)
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        paper: bool = True,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper

        # Alpaca clients (lazy-loaded)
        self._trading_client: Any = None
        self._data_client: Any = None

        # Cached instruments
        self._instruments: dict[str, InstrumentMeta] = {}
        self._markets_loaded = False

    # ── Identity ────────────────────────────────────────────────────

    @property
    def exchange_id(self) -> str:
        return "alpaca"

    @property
    def asset_class(self) -> str:
        return "stocks"

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialise Alpaca REST clients."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError:
            raise ImportError(
                "Alpaca SDK required. Install with: pip install alpaca-py"
            )

        self._trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )

        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        logger.info(
            "AlpacaAdapter connected: paper=%s", self._paper,
        )

    async def close(self) -> None:
        """Alpaca uses REST — no persistent connection to close."""
        self._trading_client = None
        self._data_client = None

    # ── Market Data ─────────────────────────────────────────────────

    async def load_markets(self) -> dict[str, InstrumentMeta]:
        if self._markets_loaded:
            return self._instruments

        if self._trading_client is None:
            raise RuntimeError("AlpacaAdapter not connected. Call connect() first.")

        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass as AlpacaAssetClass

            request = GetAssetsRequest(
                asset_class=AlpacaAssetClass.US_EQUITY,
                status="active",
            )
            all_assets = await asyncio.to_thread(
                self._trading_client.get_all_assets, request,
            )
            asset_map = {a.symbol: a for a in all_assets}
        except Exception as exc:
            logger.warning("Failed to fetch Alpaca assets: %s", exc)
            asset_map = {}

        for symbol in US_STOCK_UNIVERSE:
            tick_size = 0.01  # US stocks: penny increments
            lot_size = 0.001  # Alpaca supports fractional shares
            min_qty = 0.001
            fractionable = False
            tradable = True

            if symbol in asset_map:
                asset = asset_map[symbol]
                fractionable = getattr(asset, "fractionable", False)
                tradable = getattr(asset, "tradable", True)
                if not tradable:
                    continue

            self._instruments[symbol] = InstrumentMeta(
                symbol=symbol,
                exchange_symbol=symbol,
                asset_class="stocks",
                exchange_id="alpaca",
                tick_size=tick_size,
                lot_size=lot_size if fractionable else 1.0,
                min_qty=min_qty if fractionable else 1.0,
                max_qty=None,
                min_notional=1.0,
                max_leverage=4,  # Reg T margin
                default_leverage=2,
                trades_24_7=False,
                trading_hours=US_MARKET_HOURS_UTC,
                spread_typical_pct=0.0005,
                commission_pct=0.0,  # Alpaca: commission-free
            )

        self._markets_loaded = True
        logger.info("AlpacaAdapter loaded %d stocks", len(self._instruments))
        return self._instruments

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list[float]]:
        """Fetch historical bars from Alpaca."""
        if self._data_client is None:
            raise RuntimeError("AlpacaAdapter not connected.")

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError:
            raise ImportError("alpaca-py required")

        # Map timeframe
        tf_mapping = {
            "1m": TimeFrame(1, TimeFrameUnit.Minute),
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
            "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "4h": TimeFrame(4, TimeFrameUnit.Hour),
            "1d": TimeFrame(1, TimeFrameUnit.Day),
        }
        alpaca_tf = tf_mapping.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))

        start = None
        if since is not None:
            start = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
        else:
            # Default: last N bars worth of data
            multipliers = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
            minutes_back = multipliers.get(timeframe, 5) * limit
            start = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_tf,
            start=start,
            limit=limit,
        )

        bars = await asyncio.to_thread(
            self._data_client.get_stock_bars, request,
        )

        candles: list[list[float]] = []
        if symbol in bars.data:
            for bar in bars.data[symbol]:
                ts = int(bar.timestamp.timestamp() * 1000)
                candles.append([
                    ts,
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(bar.close),
                    float(bar.volume),
                ])

        return candles

    async def watch_ohlcv(self, symbol: str, timeframe: str = "5m") -> list[list]:
        """Alpaca: poll REST for latest bars (WebSocket streams trades, not OHLCV)."""
        candles = await self.fetch_ohlcv(symbol, timeframe, limit=2)
        return candles

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch latest quote from Alpaca."""
        if self._data_client is None:
            raise RuntimeError("AlpacaAdapter not connected.")

        try:
            from alpaca.data.requests import StockLatestQuoteRequest
        except ImportError:
            raise ImportError("alpaca-py required")

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = await asyncio.to_thread(
            self._data_client.get_stock_latest_quote, request,
        )

        if symbol in quotes:
            q = quotes[symbol]
            bid = float(q.bid_price) if q.bid_price else 0
            ask = float(q.ask_price) if q.ask_price else 0
            return {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "last": (bid + ask) / 2 if bid and ask else 0,
                "spread": ask - bid if bid and ask else 0,
                "timestamp": datetime.now(timezone.utc),
            }

        return {"symbol": symbol, "bid": 0, "ask": 0, "last": 0}

    # ── Instrument Info ─────────────────────────────────────────────

    def get_instrument(self, symbol: str) -> InstrumentMeta | None:
        return self._instruments.get(symbol)

    def normalize_symbol(self, symbol: str) -> str:
        return symbol  # Alpaca uses plain ticker symbols

    # ── Precision Helpers ───────────────────────────────────────────

    def price_to_precision(self, symbol: str, price: float) -> float:
        return round(price, 2)  # US stocks: penny precision

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        meta = self.get_instrument(symbol)
        if meta and meta.lot_size >= 1.0:
            return float(int(amount))
        return round(amount, 3)  # Fractional shares: 3 decimal places

    # ── Trading ─────────────────────────────────────────────────────

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        if self._trading_client is None:
            raise RuntimeError("AlpacaAdapter not connected.")

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError:
            raise ImportError("alpaca-py required")

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=self.amount_to_precision(symbol, qty),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = await asyncio.to_thread(
            self._trading_client.submit_order, request,
        )

        return OrderResult(
            order_id=str(order.id) if order else None,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            price=float(order.filled_avg_price) if order and order.filled_avg_price else None,
            status=str(order.status) if order else "unknown",
            raw={"order": order},
        )

    async def create_stop_loss(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        if self._trading_client is None:
            raise RuntimeError("AlpacaAdapter not connected.")

        try:
            from alpaca.trading.requests import StopOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError:
            raise ImportError("alpaca-py required")

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        request = StopOrderRequest(
            symbol=symbol,
            qty=self.amount_to_precision(symbol, qty),
            side=order_side,
            stop_price=self.price_to_precision(symbol, stop_price),
            time_in_force=TimeInForce.GTC,
        )

        order = await asyncio.to_thread(
            self._trading_client.submit_order, request,
        )

        return OrderResult(
            order_id=str(order.id) if order else None,
            symbol=symbol,
            side=side,
            order_type="stop_market",
            qty=qty,
            price=stop_price,
            status=str(order.status) if order else "unknown",
            raw={"order": order},
        )

    async def create_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        if self._trading_client is None:
            raise RuntimeError("AlpacaAdapter not connected.")

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError:
            raise ImportError("alpaca-py required")

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        request = LimitOrderRequest(
            symbol=symbol,
            qty=self.amount_to_precision(symbol, qty),
            side=order_side,
            limit_price=self.price_to_precision(symbol, stop_price),
            time_in_force=TimeInForce.GTC,
        )

        order = await asyncio.to_thread(
            self._trading_client.submit_order, request,
        )

        return OrderResult(
            order_id=str(order.id) if order else None,
            symbol=symbol,
            side=side,
            order_type="take_profit_market",
            qty=qty,
            price=stop_price,
            status=str(order.status) if order else "unknown",
            raw={"order": order},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self._trading_client is None:
            return False
        try:
            await asyncio.to_thread(
                self._trading_client.cancel_order_by_id, order_id,
            )
            return True
        except Exception as exc:
            logger.warning("Alpaca cancel_order failed %s: %s", order_id, exc)
            return False

    async def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        if self._trading_client is None:
            return []

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol],
            )
            orders = await asyncio.to_thread(
                self._trading_client.get_orders, request,
            )
            return [
                {
                    "id": str(o.id),
                    "symbol": symbol,
                    "type": str(o.type),
                    "side": str(o.side),
                    "price": float(o.limit_price or o.stop_price or 0),
                    "amount": float(o.qty or 0),
                    "status": str(o.status),
                }
                for o in orders
            ]
        except Exception as exc:
            logger.warning("Alpaca fetch_open_orders failed %s: %s", symbol, exc)
            return []

    # ── Account ─────────────────────────────────────────────────────

    async def fetch_balance(self) -> BalanceInfo:
        if self._trading_client is None:
            raise RuntimeError("AlpacaAdapter not connected.")

        account = await asyncio.to_thread(
            self._trading_client.get_account,
        )

        return BalanceInfo(
            currency="USD",
            total=float(account.equity) if account.equity else 0,
            free=float(account.buying_power) if account.buying_power else 0,
            used=float(account.equity) - float(account.buying_power or 0)
            if account.equity else 0,
            raw={"account": account},
        )

    async def fetch_positions(self) -> list[PositionInfo]:
        if self._trading_client is None:
            return []

        positions = await asyncio.to_thread(
            self._trading_client.get_all_positions,
        )

        result: list[PositionInfo] = []
        for p in positions:
            qty = abs(float(p.qty or 0))
            if qty <= 0:
                continue
            result.append(PositionInfo(
                symbol=p.symbol,
                side="long" if float(p.qty or 0) > 0 else "short",
                qty=qty,
                entry_price=float(p.avg_entry_price or 0),
                unrealized_pnl=float(p.unrealized_pl or 0),
                leverage=1,  # Alpaca doesn't expose per-position leverage
                margin_mode="cross",
                raw={"position": p},
            ))
        return result

    async def fetch_my_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if self._trading_client is None:
            return []

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                limit=limit,
            )
            orders = await asyncio.to_thread(
                self._trading_client.get_orders, request,
            )

            trades = []
            for o in orders:
                if o.filled_qty and float(o.filled_qty) > 0:
                    trades.append({
                        "id": str(o.id),
                        "symbol": symbol,
                        "side": str(o.side),
                        "price": float(o.filled_avg_price or 0),
                        "amount": float(o.filled_qty),
                        "timestamp": o.filled_at.timestamp() * 1000 if o.filled_at else 0,
                    })
            return trades
        except Exception as exc:
            logger.warning("Alpaca fetch_my_trades failed %s: %s", symbol, exc)
            return []

    # ── Leverage & Margin ───────────────────────────────────────────

    async def set_leverage(self, leverage: int, symbol: str) -> None:
        """Alpaca: leverage is account-level (Reg T margin). No per-symbol setting."""
        pass  # No-op

    async def set_margin_mode(self, mode: str, symbol: str) -> None:
        """Alpaca: always portfolio margin or Reg T. No per-symbol mode."""
        pass  # No-op

    # ── Trading Hours ───────────────────────────────────────────────

    def is_market_open(self, symbol: str, utc_now: datetime | None = None) -> bool:
        """Check if US stock market is open (Regular Trading Hours only)."""
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        weekday = utc_now.weekday()
        # Closed on weekends
        if weekday >= 5:
            return False

        # Regular hours: ~14:30-21:00 UTC (varies with DST)
        # Use conservative window
        hour = utc_now.hour
        minute = utc_now.minute
        total_minutes = hour * 60 + minute

        # 13:30 UTC (EDT open) to 21:00 UTC (EST close) — covers both DST modes
        return 13 * 60 + 30 <= total_minutes < 21 * 60
