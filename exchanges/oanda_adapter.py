"""
OANDA v20 Adapter
=================
Exchange adapter for OANDA v20 REST API.
Handles Forex (28 major/minor/cross pairs) and Commodities (XAU, XAG, WTI).

Requires: pip install v20  (OANDA v20 Python library)

Environment variables:
  OANDA_ACCOUNT_ID   — Your OANDA account ID
  OANDA_ACCESS_TOKEN — Your OANDA API access token
  OANDA_ENVIRONMENT  — "practice" (demo) or "live"
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Any

from exchanges.base import ExchangeAdapter
from exchanges.models import (
    BalanceInfo,
    InstrumentMeta,
    OrderResult,
    PositionInfo,
)

logger = logging.getLogger(__name__)

# ── OANDA Instrument Universes ─────────────────────────────────────

FOREX_PAIRS: list[str] = [
    # Majors (7)
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CHF", "USD_CAD", "NZD_USD",
    # Crosses (21)
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
]

COMMODITY_INSTRUMENTS: list[str] = [
    "XAU_USD",   # Gold
    "XAG_USD",   # Silver
    "WTICO_USD", # WTI Crude Oil
    "BCO_USD",   # Brent Crude Oil
]

# OANDA timeframe mapping (our format → OANDA granularity)
TF_MAP: dict[str, str] = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "4h": "H4",
    "1d": "D",
}

# Forex trading hours: Sunday 17:00 EST to Friday 17:00 EST (24/5)
# In UTC: Sunday 22:00 to Friday 22:00
FOREX_TRADING_HOURS: list[tuple[int, int]] = [(0, 24)]  # 24h when market is open

# Commodity-specific hours vary; Gold/Silver trade nearly 23h/day on weekdays
COMMODITY_TRADING_HOURS: list[tuple[int, int]] = [(0, 24)]


class OandaAdapter(ExchangeAdapter):
    """
    Exchange adapter for OANDA v20 API.

    Supports Forex and Commodities trading with:
    - REST API for orders and account info
    - Polling-based price streaming (v20 streaming endpoint)
    - Separate SL/TP orders (no bracket orders, uses Trade-attached SL/TP)
    """

    def __init__(
        self,
        account_id: str = "",
        access_token: str = "",
        environment: str = "practice",  # "practice" or "live"
    ) -> None:
        self._account_id = account_id
        self._access_token = access_token
        self._environment = environment

        # v20 API context (lazy-loaded)
        self._api: Any = None

        # Cached instruments
        self._instruments: dict[str, InstrumentMeta] = {}
        self._markets_loaded = False

        # OANDA-native to unified symbol mapping
        self._oanda_to_unified: dict[str, str] = {}
        self._unified_to_oanda: dict[str, str] = {}

        # Concurrency limiter for candle fetches (practice account rate limits)
        self._candle_semaphore = asyncio.Semaphore(3)

    # ── Identity ────────────────────────────────────────────────────

    @property
    def exchange_id(self) -> str:
        return "oanda"

    @property
    def asset_class(self) -> str:
        return "forex"  # Primary; also handles commodities

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialise OANDA v20 API context."""
        try:
            import v20
        except ImportError:
            raise ImportError(
                "OANDA v20 library required. Install with: pip install v20"
            )

        hostname = (
            "api-fxpractice.oanda.com"
            if self._environment == "practice"
            else "api-fxtrade.oanda.com"
        )

        self._api = v20.Context(
            hostname=hostname,
            token=self._access_token,
            port=443,
            ssl=True,
        )

        logger.info(
            "OandaAdapter connected: account=%s env=%s",
            self._account_id, self._environment,
        )

    async def close(self) -> None:
        """OANDA v20 uses REST — no persistent connection to close."""
        self._api = None

    # ── Market Data ─────────────────────────────────────────────────

    async def load_markets(self) -> dict[str, InstrumentMeta]:
        if self._markets_loaded:
            return self._instruments

        if self._api is None:
            raise RuntimeError("OandaAdapter not connected. Call connect() first.")

        # Fetch instrument list from OANDA
        response = await asyncio.to_thread(
            self._api.account.instruments, self._account_id,
        )

        oanda_instruments = {}
        if hasattr(response, "body") and "instruments" in response.body:
            for inst in response.body["instruments"]:
                oanda_instruments[inst.name] = inst

        # Build our universe (only forex + commodities we care about)
        all_symbols = FOREX_PAIRS + COMMODITY_INSTRUMENTS

        for oanda_sym in all_symbols:
            # Convert OANDA format to unified: EUR_USD → EUR/USD
            unified = oanda_sym.replace("_", "/")
            asset_class = (
                "commodities" if oanda_sym in COMMODITY_INSTRUMENTS else "forex"
            )

            self._oanda_to_unified[oanda_sym] = unified
            self._unified_to_oanda[unified] = oanda_sym

            # Defaults (will be overridden if OANDA provides data)
            tick_size = 0.00001 if asset_class == "forex" else 0.01
            lot_size = 1.0  # OANDA uses units, not lots
            min_qty = 1.0
            max_leverage = 30 if asset_class == "forex" else 20
            commission = 0.0  # OANDA uses spread-based pricing
            spread_pct = 0.00015 if asset_class == "forex" else 0.0003

            # Override from OANDA API if available
            if oanda_sym in oanda_instruments:
                inst = oanda_instruments[oanda_sym]
                pip_loc = getattr(inst, "pipLocation", -4)
                tick_size = 10 ** pip_loc
                min_units = getattr(inst, "minimumTradeSize", "1")
                min_qty = float(min_units)
                max_units = getattr(inst, "maximumOrderUnits", None)
                max_qty = float(max_units) if max_units else None
                margin_rate = getattr(inst, "marginRate", "0.033")
                if margin_rate:
                    max_leverage = int(1.0 / float(margin_rate))

            trading_hours = (
                COMMODITY_TRADING_HOURS if asset_class == "commodities"
                else FOREX_TRADING_HOURS
            )

            self._instruments[unified] = InstrumentMeta(
                symbol=unified,
                exchange_symbol=oanda_sym,
                asset_class=asset_class,
                exchange_id="oanda",
                tick_size=tick_size,
                lot_size=lot_size,
                min_qty=min_qty,
                max_qty=max_qty if oanda_sym in oanda_instruments else None,
                min_notional=1.0,  # OANDA: 1 unit minimum
                max_leverage=max_leverage,
                default_leverage=max_leverage,
                trades_24_7=False,
                trading_hours=trading_hours,
                spread_typical_pct=spread_pct,
                commission_pct=commission,
            )

        self._markets_loaded = True
        logger.info(
            "OandaAdapter loaded %d instruments (%d forex, %d commodities)",
            len(self._instruments),
            len(FOREX_PAIRS),
            len(COMMODITY_INSTRUMENTS),
        )
        return self._instruments

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list[float]]:
        """Fetch OHLCV candles from OANDA v20 API."""
        if self._api is None:
            raise RuntimeError("OandaAdapter not connected.")

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))
        granularity = TF_MAP.get(timeframe, "M5")

        kwargs: dict[str, Any] = {
            "granularity": granularity,
            "count": min(limit, 5000),  # OANDA max 5000
            "price": "M",  # Midpoint prices
        }
        if since is not None:
            from_time = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
            kwargs["fromTime"] = from_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            kwargs.pop("count", None)

        try:
            async with self._candle_semaphore:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._api.instrument.candles, oanda_sym, **kwargs,
                    ),
                    timeout=30.0,  # 30s max (practice account slow during London open)
                )
        except asyncio.TimeoutError:
            logger.debug("OANDA candle request timed out for %s", oanda_sym)
            return []

        candles: list[list[float]] = []
        if hasattr(response, "body") and response.body and "candles" in response.body:
            for c in response.body["candles"]:
                if not c.complete:
                    continue
                mid = c.mid
                ts = int(
                    datetime.strptime(
                        c.time[:19] + "Z", "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc).timestamp() * 1000
                )
                candles.append([
                    ts,
                    float(mid.o),
                    float(mid.h),
                    float(mid.l),
                    float(mid.c),
                    float(c.volume),
                ])

        return candles

    async def watch_ohlcv(self, symbol: str, timeframe: str = "5m") -> list[list]:
        """OANDA doesn't have native WebSocket OHLCV — poll via REST."""
        # Poll at interval matching timeframe
        candles = await self.fetch_ohlcv(symbol, timeframe, limit=2)
        return candles

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current pricing via OANDA streaming/pricing endpoint."""
        if self._api is None:
            raise RuntimeError("OandaAdapter not connected.")

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))

        response = await asyncio.to_thread(
            self._api.pricing.get,
            self._account_id,
            instruments=oanda_sym,
        )

        if hasattr(response, "body") and "prices" in response.body:
            prices = response.body["prices"]
            if prices:
                p = prices[0]
                bid = float(p.bids[0].price) if p.bids else 0
                ask = float(p.asks[0].price) if p.asks else 0
                return {
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "last": (bid + ask) / 2,
                    "spread": ask - bid,
                    "timestamp": datetime.now(timezone.utc),
                }

        return {"symbol": symbol, "bid": 0, "ask": 0, "last": 0}

    # ── Instrument Info ─────────────────────────────────────────────

    def get_instrument(self, symbol: str) -> InstrumentMeta | None:
        return self._instruments.get(symbol)

    def normalize_symbol(self, symbol: str) -> str:
        """Convert unified to OANDA format: 'EUR/USD' → 'EUR_USD'."""
        return symbol.replace("/", "_")

    # ── Precision Helpers ───────────────────────────────────────────

    def price_to_precision(self, symbol: str, price: float) -> float:
        meta = self.get_instrument(symbol)
        if meta and meta.tick_size > 0:
            decimals = max(0, -int(math.log10(meta.tick_size)))
            return round(price, decimals)
        return round(price, 5)

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """OANDA uses integer units for forex, round to whole number."""
        return float(int(amount))

    # ── Trading ─────────────────────────────────────────────────────

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        if self._api is None:
            raise RuntimeError("OandaAdapter not connected.")

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))
        # OANDA: positive units = buy, negative = sell
        units = int(qty) if side == "buy" else -int(qty)

        order_data = {
            "type": "MARKET",
            "instrument": oanda_sym,
            "units": str(units),
            "timeInForce": "FOK",
        }

        # Attach SL/TP if provided in params
        if params:
            if "stopLossPrice" in params:
                order_data["stopLossOnFill"] = {
                    "price": str(self.price_to_precision(symbol, params["stopLossPrice"]))
                }
            if "takeProfitPrice" in params:
                order_data["takeProfitOnFill"] = {
                    "price": str(self.price_to_precision(symbol, params["takeProfitPrice"]))
                }
            if params.get("reduceOnly"):
                order_data["positionFill"] = "REDUCE_ONLY"

        response = await asyncio.to_thread(
            self._api.order.market, self._account_id, **{"order": order_data},
        )

        order_id = None
        fill_price = 0.0
        status = "unknown"

        if hasattr(response, "body"):
            if "orderFillTransaction" in response.body:
                fill = response.body["orderFillTransaction"]
                order_id = str(getattr(fill, "id", ""))
                fill_price = float(getattr(fill, "price", 0))
                status = "filled"
            elif "orderCreateTransaction" in response.body:
                create = response.body["orderCreateTransaction"]
                order_id = str(getattr(create, "id", ""))
                status = "open"

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            price=fill_price,
            status=status,
            raw=response.body if hasattr(response, "body") else {},
        )

    async def create_stop_loss(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        """Create a stop-loss order on OANDA.

        OANDA attaches SL to trades, not as standalone orders.
        This creates a STOP order as a workaround for standalone SL.
        """
        if self._api is None:
            raise RuntimeError("OandaAdapter not connected.")

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))
        units = int(qty) if side == "buy" else -int(qty)

        order_data = {
            "type": "STOP",
            "instrument": oanda_sym,
            "units": str(units),
            "price": str(self.price_to_precision(symbol, stop_price)),
            "triggerCondition": "DEFAULT",
            "timeInForce": "GTC",
        }

        response = await asyncio.to_thread(
            self._api.order.create, self._account_id, **{"order": order_data},
        )

        order_id = None
        status = "unknown"
        if hasattr(response, "body") and "orderCreateTransaction" in response.body:
            create = response.body["orderCreateTransaction"]
            order_id = str(getattr(create, "id", ""))
            status = "open"

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="stop_market",
            qty=qty,
            price=stop_price,
            status=status,
            raw=response.body if hasattr(response, "body") else {},
        )

    async def create_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        """Create a take-profit limit order on OANDA."""
        if self._api is None:
            raise RuntimeError("OandaAdapter not connected.")

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))
        units = int(qty) if side == "buy" else -int(qty)

        order_data = {
            "type": "LIMIT",
            "instrument": oanda_sym,
            "units": str(units),
            "price": str(self.price_to_precision(symbol, stop_price)),
            "triggerCondition": "DEFAULT",
            "timeInForce": "GTC",
        }

        response = await asyncio.to_thread(
            self._api.order.create, self._account_id, **{"order": order_data},
        )

        order_id = None
        status = "unknown"
        if hasattr(response, "body") and "orderCreateTransaction" in response.body:
            create = response.body["orderCreateTransaction"]
            order_id = str(getattr(create, "id", ""))
            status = "open"

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="take_profit_market",
            qty=qty,
            price=stop_price,
            status=status,
            raw=response.body if hasattr(response, "body") else {},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self._api is None:
            return False
        try:
            await asyncio.to_thread(
                self._api.order.cancel, self._account_id, order_id,
            )
            return True
        except Exception as exc:
            logger.warning("OANDA cancel_order failed %s: %s", order_id, exc)
            return False

    async def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        if self._api is None:
            return []

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))
        response = await asyncio.to_thread(
            self._api.order.list_pending, self._account_id,
        )

        orders = []
        if hasattr(response, "body") and "orders" in response.body:
            for o in response.body["orders"]:
                if getattr(o, "instrument", "") == oanda_sym:
                    orders.append({
                        "id": str(getattr(o, "id", "")),
                        "symbol": symbol,
                        "type": getattr(o, "type", ""),
                        "side": "buy" if float(getattr(o, "units", 0)) > 0 else "sell",
                        "price": float(getattr(o, "price", 0)),
                        "amount": abs(float(getattr(o, "units", 0))),
                    })
        return orders

    # ── Account ─────────────────────────────────────────────────────

    async def fetch_balance(self) -> BalanceInfo:
        if self._api is None:
            raise RuntimeError("OandaAdapter not connected.")

        response = await asyncio.to_thread(
            self._api.account.summary, self._account_id,
        )

        if hasattr(response, "body") and "account" in response.body:
            acc = response.body["account"]
            return BalanceInfo(
                currency=getattr(acc, "currency", "USD"),
                total=float(getattr(acc, "balance", 0)),
                free=float(getattr(acc, "marginAvailable", 0)),
                used=float(getattr(acc, "marginUsed", 0)),
                raw={"account": acc},
            )

        return BalanceInfo(currency="USD")

    async def fetch_positions(self) -> list[PositionInfo]:
        if self._api is None:
            return []

        response = await asyncio.to_thread(
            self._api.position.list_open, self._account_id,
        )

        positions: list[PositionInfo] = []
        if hasattr(response, "body") and "positions" in response.body:
            for p in response.body["positions"]:
                instrument = getattr(p, "instrument", "")
                unified = self._oanda_to_unified.get(
                    instrument, instrument.replace("_", "/")
                )

                long_units = float(getattr(p.long, "units", 0))
                short_units = abs(float(getattr(p.short, "units", 0)))

                if long_units > 0:
                    positions.append(PositionInfo(
                        symbol=unified,
                        side="long",
                        qty=long_units,
                        entry_price=float(getattr(p.long, "averagePrice", 0)),
                        unrealized_pnl=float(getattr(p.long, "unrealizedPL", 0)),
                        leverage=1,  # OANDA manages margin internally
                        margin_mode="cross",
                        raw={"position": p},
                    ))
                if short_units > 0:
                    positions.append(PositionInfo(
                        symbol=unified,
                        side="short",
                        qty=short_units,
                        entry_price=float(getattr(p.short, "averagePrice", 0)),
                        unrealized_pnl=float(getattr(p.short, "unrealizedPL", 0)),
                        leverage=1,
                        margin_mode="cross",
                        raw={"position": p},
                    ))

        return positions

    async def fetch_my_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if self._api is None:
            return []

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))
        kwargs: dict[str, Any] = {"instrument": oanda_sym, "count": limit}
        if since is not None:
            from_time = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
            kwargs["sinceTransactionID"] = None  # Would need transaction ID
            kwargs.pop("sinceTransactionID")

        response = await asyncio.to_thread(
            self._api.trade.list_open, self._account_id,
        )

        trades = []
        if hasattr(response, "body") and "trades" in response.body:
            for t in response.body["trades"]:
                if getattr(t, "instrument", "") == oanda_sym:
                    trades.append({
                        "id": str(getattr(t, "id", "")),
                        "symbol": symbol,
                        "side": "buy" if float(getattr(t, "currentUnits", 0)) > 0 else "sell",
                        "price": float(getattr(t, "price", 0)),
                        "amount": abs(float(getattr(t, "currentUnits", 0))),
                        "timestamp": getattr(t, "openTime", ""),
                    })
        return trades

    # ── Leverage & Margin ───────────────────────────────────────────

    async def set_leverage(self, leverage: int, symbol: str) -> None:
        """OANDA manages margin/leverage at account level, not per-symbol."""
        pass  # No-op — leverage is determined by margin rate on OANDA

    async def set_margin_mode(self, mode: str, symbol: str) -> None:
        """OANDA uses account-level margin — always 'cross'."""
        pass  # No-op

    # ── Trading Hours ───────────────────────────────────────────────

    def is_market_open(self, symbol: str, utc_now: datetime | None = None) -> bool:
        """Forex: open Sunday 22:00 UTC to Friday 22:00 UTC."""
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        weekday = utc_now.weekday()  # 0=Mon, 6=Sun
        hour = utc_now.hour

        # Closed: Friday 22:00 UTC → Sunday 22:00 UTC
        if weekday == 4 and hour >= 22:  # Friday after 22:00
            return False
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and hour < 22:  # Sunday before 22:00
            return False

        return True
