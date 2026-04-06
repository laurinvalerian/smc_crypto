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
    - Trade-attached SL/TP via stopLossOnFill/takeProfitOnFill
    - SL modification via Trade endpoint (set_dependent_orders)
    """

    # NOTE: All asyncio.to_thread calls are wrapped in asyncio.wait_for(timeout=30s).
    # asyncio.wait_for does NOT cancel the underlying thread on timeout — the OANDA SDK
    # call continues in the background. Socket-level timeouts in the v20 SDK handle
    # eventual thread cleanup. Monitor with threading.active_count() if issues arise.

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

        # Concurrency limiters (practice account rate limits ~30 req/s)
        # 8 concurrent candle fetches (was 3 — caused 6-min queue for 36 instruments)
        self._candle_semaphore = asyncio.Semaphore(8)
        # 8 concurrent ticker fetches (was uncontrolled — amplified rate-limit pressure)
        self._ticker_semaphore = asyncio.Semaphore(8)

    # ── Identity ────────────────────────────────────────────────────

    @property
    def exchange_id(self) -> str:
        return "oanda"

    @property
    def asset_class(self) -> str:
        return "forex"  # Primary; also handles commodities

    @property
    def supports_attached_sl_tp(self) -> bool:
        return True

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
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.account.instruments, self._account_id,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: load_markets")
            raise

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

        try:
            async with self._ticker_semaphore:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._api.pricing.get,
                        self._account_id,
                        instruments=oanda_sym,
                    ),
                    timeout=30.0,
                )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: watch_ticker")
            raise

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

    async def fetch_batch_pricing(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch pricing for multiple instruments in a single OANDA API call.

        OANDA pricing API accepts comma-separated instruments, allowing
        36 ticker requests to be batched into 1-2 API calls.
        """
        if self._api is None:
            raise RuntimeError("OandaAdapter not connected.")
        if not symbols:
            return {}

        # Convert unified symbols to OANDA format
        oanda_syms = []
        sym_map: dict[str, str] = {}  # oanda_sym -> unified_sym
        for s in symbols:
            oanda_sym = self._unified_to_oanda.get(s, s.replace("/", "_"))
            oanda_syms.append(oanda_sym)
            sym_map[oanda_sym] = s

        instruments_str = ",".join(oanda_syms)

        try:
            async with self._ticker_semaphore:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._api.pricing.get,
                        self._account_id,
                        instruments=instruments_str,
                    ),
                    timeout=30.0,
                )
        except asyncio.TimeoutError:
            logger.warning("OANDA batch pricing timed out for %d instruments", len(symbols))
            return {}

        result: dict[str, dict[str, Any]] = {}
        if hasattr(response, "body") and "prices" in response.body:
            for p in response.body["prices"]:
                oanda_name = p.instrument
                unified = sym_map.get(oanda_name, oanda_name.replace("_", "/"))
                bid = float(p.bids[0].price) if p.bids else 0
                ask = float(p.asks[0].price) if p.asks else 0
                result[unified] = {
                    "symbol": unified,
                    "bid": bid,
                    "ask": ask,
                    "last": (bid + ask) / 2,
                    "spread": ask - bid,
                    "timestamp": datetime.now(timezone.utc),
                }

        return result

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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.order.market, self._account_id, **order_data,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: create_market_order")
            raise

        order_id = None
        fill_price = 0.0
        status = "unknown"
        oanda_trade_id = None

        if hasattr(response, "body"):
            if "orderFillTransaction" in response.body:
                fill = response.body["orderFillTransaction"]
                order_id = str(getattr(fill, "id", ""))
                fill_price = float(getattr(fill, "price", 0))
                status = "filled"
                # Extract OANDA trade ID for trade-attached SL/TP management
                trade_opened = getattr(fill, "tradeOpened", None)
                if trade_opened:
                    oanda_trade_id = str(getattr(trade_opened, "tradeID", ""))
                if not oanda_trade_id:
                    # Fallback: check tradesClosed for reduce-only orders
                    trades_reduced = getattr(fill, "tradesReduced", None) or []
                    if trades_reduced:
                        oanda_trade_id = str(getattr(trades_reduced[0], "tradeID", ""))
            elif "orderCreateTransaction" in response.body:
                create = response.body["orderCreateTransaction"]
                order_id = str(getattr(create, "id", ""))
                status = "open"
            elif "orderRejectTransaction" in response.body:
                reject = response.body["orderRejectTransaction"]
                reason = getattr(reject, "rejectReason", "unknown")
                logger.warning("OANDA order REJECTED %s %s %s: %s", side, oanda_sym, qty, reason)
                status = "rejected"
            elif "orderCancelTransaction" in response.body:
                cancel = response.body["orderCancelTransaction"]
                reason = getattr(cancel, "reason", "unknown")
                logger.warning("OANDA order CANCELLED %s %s %s: %s", side, oanda_sym, qty, reason)
                status = "cancelled"
            else:
                # Log unexpected response for debugging
                keys = list(response.body.keys()) if hasattr(response.body, "keys") else str(type(response.body))
                logger.warning("OANDA unexpected response for %s %s: keys=%s", side, oanda_sym, keys)

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            price=fill_price,
            status=status,
            raw=response.body if hasattr(response, "body") else {},
            trade_id=oanda_trade_id,
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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.order.create, self._account_id, **{"order": order_data},
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: create_stop_loss")
            raise

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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.order.create, self._account_id, **{"order": order_data},
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: create_take_profit")
            raise

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

    async def modify_stop_loss(
        self,
        old_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        new_stop_price: float,
        *,
        trade_id: str | None = None,
    ) -> OrderResult | None:
        """Modify trade-attached SL via OANDA Trade endpoint (set_dependent_orders)."""
        if self._api is None:
            return None
        if not trade_id:
            # Fallback to base class cancel+replace for standalone SL orders
            return await super().modify_stop_loss(
                old_order_id, symbol, side, qty, new_stop_price,
            )
        try:
            price_str = str(self.price_to_precision(symbol, new_stop_price))
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.trade.set_dependent_orders,
                    self._account_id,
                    trade_id,
                    **{"stopLoss": {"price": price_str}},
                ),
                timeout=30.0,
            )
            logger.info("OANDA modify_stop_loss: trade %s SL → %s", trade_id, price_str)
            return OrderResult(
                order_id=trade_id,
                symbol=symbol,
                side=side,
                order_type="modify_sl",
                qty=qty,
                price=new_stop_price,
                status="modified",
                raw=response.body if hasattr(response, "body") else {},
                trade_id=trade_id,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA modify_stop_loss timed out for trade %s", trade_id)
            return None
        except Exception as exc:
            logger.warning("OANDA modify_stop_loss failed for trade %s: %s", trade_id, exc)
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self._api is None:
            return False
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.order.cancel, self._account_id, order_id,
                ),
                timeout=30.0,
            )
            return True
        except Exception as exc:
            logger.warning("OANDA cancel_order failed %s: %s", order_id, exc)
            return False

    async def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        if self._api is None:
            return []

        oanda_sym = self._unified_to_oanda.get(symbol, symbol.replace("/", "_"))
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.order.list_pending, self._account_id,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: fetch_open_orders")
            raise

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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.account.summary, self._account_id,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: fetch_balance")
            raise

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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.position.list_open, self._account_id,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: fetch_positions")
            raise

        positions: list[PositionInfo] = []
        if hasattr(response, "body") and "positions" in response.body:
            for p in response.body["positions"]:
                instrument = getattr(p, "instrument", "")
                # Use OANDA-native symbol (XAG_USD) — bots use this format
                sym = instrument

                long_units = float(getattr(p.long, "units", 0))
                short_units = abs(float(getattr(p.short, "units", 0)))

                if long_units > 0:
                    positions.append(PositionInfo(
                        symbol=sym,
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
                        symbol=sym,
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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._api.trade.list_open, self._account_id,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("OANDA API call timed out after 30s: fetch_my_trades")
            raise

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
