"""
Binance USDT-M Futures Adapter
==============================
Wraps ccxt/ccxt.pro for Binance USDT-M Futures (live + testnet).
Implements the ExchangeAdapter interface so the bot can trade crypto
through the same API used for forex/stocks/commodities later.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import ccxt as ccxt_sync

try:
    import ccxt.pro as ccxtpro
except ImportError:
    sys.exit("ccxt.pro is required.  Install with:  pip install 'ccxt[pro]'")

from exchanges.base import ExchangeAdapter
from exchanges.models import (
    BalanceInfo,
    InstrumentMeta,
    OrderResult,
    PositionInfo,
)

logger = logging.getLogger(__name__)


class BinanceAdapter(ExchangeAdapter):
    """
    Exchange adapter for Binance USDT-M Futures.

    Wraps both:
    - ccxt.pro (async) for WebSocket streams and trading
    - ccxt sync for startup history loading
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

        # Async exchange (ccxt.pro) — main trading client
        self._exchange: Any = None
        # Sync exchange — for startup OHLCV history loading
        self._sync_exchange: Any = None

        # Cached instrument metadata
        self._instruments: dict[str, InstrumentMeta] = {}
        self._markets_loaded = False

    # ── Identity ────────────────────────────────────────────────────

    @property
    def exchange_id(self) -> str:
        return "binanceusdm"

    @property
    def asset_class(self) -> str:
        return "crypto"

    # ── Raw exchange access (for migration period) ──────────────────

    @property
    def raw(self) -> Any:
        """Direct access to the underlying ccxt.pro exchange object.

        Use only during migration while PaperBot still calls
        self.exchange.xxx directly. Will be removed once all calls
        go through the adapter interface.
        """
        return self._exchange

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create ccxt.pro exchange and optionally enable demo trading."""
        self._exchange = ccxtpro.binanceusdm({
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if self._testnet:
            self._exchange.set_sandbox_mode(True)

        # Sync client for history
        self._sync_exchange = ccxt_sync.binanceusdm({"enableRateLimit": True})

        logger.info(
            "BinanceAdapter connected: %s (testnet=%s)",
            self._exchange.id, self._testnet,
        )

    async def close(self) -> None:
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception:
                pass

    # ── Market Data ─────────────────────────────────────────────────

    async def load_markets(self) -> dict[str, InstrumentMeta]:
        if self._markets_loaded:
            return self._instruments

        await self._exchange.load_markets()

        for symbol, market in self._exchange.markets.items():
            if not market.get("linear"):
                continue  # Only USDT-M linear futures

            # Derive exchange-native symbol (e.g. "BTCUSDT")
            base = (market.get("base") or "").upper()
            quote = (market.get("quote") or "").upper()
            exchange_symbol = f"{base}{quote}" if base and quote else market.get("id", symbol)

            # Parse precision
            prec = market.get("precision", {})
            tick_size = 0.01
            lot_size = 0.001
            if prec.get("price") is not None:
                p = prec["price"]
                tick_size = 10 ** (-p) if isinstance(p, int) else float(p) if isinstance(p, float) and p < 1 else 0.01
            if prec.get("amount") is not None:
                p = prec["amount"]
                lot_size = 10 ** (-p) if isinstance(p, int) else float(p) if isinstance(p, float) and p < 1 else 0.001

            # Parse limits
            limits = market.get("limits", {})
            amt_limits = limits.get("amount", {})
            cost_limits = limits.get("cost", {})
            lev_limits = limits.get("leverage", {})

            min_qty = float(amt_limits.get("min") or 0)
            max_qty_raw = amt_limits.get("max")
            max_qty = float(max_qty_raw) if max_qty_raw else None

            # Also check raw Binance filters for max qty
            raw_info = market.get("info", {})
            for f in (raw_info.get("filters") or []):
                if isinstance(f, dict) and f.get("filterType") in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                    raw_max = f.get("maxQty") or f.get("maxAmount")
                    if raw_max:
                        try:
                            parsed = float(raw_max)
                            if max_qty is None or parsed < max_qty:
                                max_qty = parsed
                        except (ValueError, TypeError):
                            pass
                    break

            min_notional = float(cost_limits.get("min") or 5.0)
            max_leverage = int(lev_limits.get("max") or 20)

            self._instruments[symbol] = InstrumentMeta(
                symbol=symbol,
                exchange_symbol=exchange_symbol,
                asset_class="crypto",
                exchange_id="binanceusdm",
                tick_size=tick_size,
                lot_size=lot_size,
                min_qty=min_qty,
                max_qty=max_qty,
                min_notional=min_notional,
                max_leverage=max_leverage,
                default_leverage=20,
                trades_24_7=True,
                spread_typical_pct=0.0001,
                commission_pct=0.0004,
                raw_info=raw_info,
            )

        self._markets_loaded = True
        logger.info("BinanceAdapter loaded %d instruments", len(self._instruments))
        return self._instruments

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list[float]]:
        return await self._exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    async def fetch_ohlcv_sync(
        self,
        symbol: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 500,
    ) -> list[list[float]]:
        """Use sync ccxt for startup history loading (avoids event loop issues)."""
        if self._sync_exchange is None:
            self._sync_exchange = ccxt_sync.binanceusdm({"enableRateLimit": True})
        return self._sync_exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    async def watch_ohlcv(self, symbol: str, timeframe: str = "5m") -> list[list]:
        return await self._exchange.watch_ohlcv(symbol, timeframe)

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        return await self._exchange.watch_ticker(symbol)

    # ── Instrument Info ─────────────────────────────────────────────

    def get_instrument(self, symbol: str) -> InstrumentMeta | None:
        return self._instruments.get(symbol)

    def normalize_symbol(self, symbol: str) -> str:
        """Convert unified to Binance REST symbol: 'BTC/USDT:USDT' → 'BTCUSDT'."""
        base_quote = symbol.split(":")[0] if ":" in symbol else symbol
        return base_quote.replace("/", "")

    # ── Precision Helpers ───────────────────────────────────────────

    def price_to_precision(self, symbol: str, price: float) -> float:
        try:
            return float(self._exchange.price_to_precision(symbol, price))
        except Exception:
            meta = self.get_instrument(symbol)
            if meta and meta.tick_size > 0:
                return round(price / meta.tick_size) * meta.tick_size
            return price

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        try:
            return float(self._exchange.amount_to_precision(symbol, amount))
        except Exception:
            meta = self.get_instrument(symbol)
            if meta and meta.lot_size > 0:
                return float(int(amount / meta.lot_size) * meta.lot_size)
            return amount

    # ── Trading ─────────────────────────────────────────────────────

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        result = await self._exchange.create_order(
            symbol, "market", side, qty, params=params or {},
        )
        return OrderResult(
            order_id=result.get("id"),
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            price=float(result.get("average") or result.get("price") or 0),
            status=result.get("status", "unknown"),
            raw=result,
        )

    async def create_stop_loss(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        # closePosition: Binance closes the entire position when triggered,
        # ignoring qty. Eliminates qty-mismatch bugs from partial fills.
        # NOTE: if multiple trades share one Binance position (multi-style),
        # closePosition closes the aggregate — acceptable since per-bot limit is 1.
        # NOTE: reduceOnly removed — Binance testnet rejects it (-1106) and
        # closePosition already implies reduce-only behavior.
        merged = {"stopPrice": stop_price, "closePosition": True}
        if params:
            merged.update(params)
        result = await self._exchange.create_order(
            symbol, "STOP_MARKET", side, qty, params=merged,
        )
        return OrderResult(
            order_id=result.get("id"),
            symbol=symbol,
            side=side,
            order_type="stop_market",
            qty=qty,
            price=stop_price,
            status=result.get("status", "unknown"),
            raw=result,
        )

    async def create_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        params: dict[str, Any] | None = None,
    ) -> OrderResult:
        # closePosition: see create_stop_loss comment (reduceOnly removed for testnet compat)
        merged = {"stopPrice": stop_price, "closePosition": True}
        if params:
            merged.update(params)
        result = await self._exchange.create_order(
            symbol, "TAKE_PROFIT_MARKET", side, qty, params=merged,
        )
        return OrderResult(
            order_id=result.get("id"),
            symbol=symbol,
            side=side,
            order_type="take_profit_market",
            qty=qty,
            price=stop_price,
            status=result.get("status", "unknown"),
            raw=result,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            logger.warning("cancel_order failed %s %s: %s", order_id, symbol, exc)
            return False

    async def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return await self._exchange.fetch_open_orders(symbol)

    # ── Account ─────────────────────────────────────────────────────

    async def fetch_balance(self) -> BalanceInfo:
        bal = await self._exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        return BalanceInfo(
            currency="USDT",
            total=float(usdt.get("total", 0)),
            free=float(usdt.get("free", 0)),
            used=float(usdt.get("used", 0)),
            raw=bal,
        )

    async def fetch_positions(self) -> list[PositionInfo]:
        raw_positions = await self._exchange.fetch_positions()
        result: list[PositionInfo] = []
        for p in raw_positions:
            contracts = abs(float(p.get("contracts", 0) or 0))
            if contracts <= 0:
                continue
            result.append(PositionInfo(
                symbol=p.get("symbol", ""),
                side=p.get("side", ""),
                qty=contracts,
                entry_price=float(p.get("entryPrice", 0) or 0),
                unrealized_pnl=float(p.get("unrealizedPnl", 0) or 0),
                leverage=int(p.get("leverage", 1) or 1),
                margin_mode=p.get("marginMode", "cross"),
                raw=p,
            ))
        return result

    async def fetch_my_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._exchange.fetch_my_trades(symbol, since=since, limit=limit)

    # ── Leverage & Margin ───────────────────────────────────────────

    async def set_leverage(self, leverage: int, symbol: str) -> None:
        await self._exchange.set_leverage(leverage, symbol)

    async def set_margin_mode(self, mode: str, symbol: str) -> None:
        try:
            await self._exchange.set_margin_mode(mode, symbol)
        except Exception:
            pass  # Already set or position open — both fine

    async def fetch_max_leverage(self, symbol: str) -> int:
        """
        Fetch max leverage from Binance bracket data.

        Tries multiple methods: ccxt unified, Binance private API, market limits.
        """
        max_lev = 20  # Conservative default
        exchange_symbol = self.normalize_symbol(symbol)

        # Method 1: ccxt unified
        for method_name in ("fetch_leverage_tiers", "fetch_leverage_bracket"):
            method_fn = getattr(self._exchange, method_name, None)
            if method_fn is None:
                continue
            try:
                tiers = await method_fn([symbol])
                if isinstance(tiers, dict):
                    for key, val in tiers.items():
                        if isinstance(val, list):
                            for tier in val:
                                if isinstance(tier, dict):
                                    lev = tier.get("maxLeverage") or tier.get("initialLeverage")
                                    if lev:
                                        max_lev = max(max_lev, int(lev))
                return max_lev
            except Exception:
                continue

        # Method 2: Binance private API
        try:
            raw = await self._exchange.fapiPrivateGetLeverageBracket(
                {"symbol": exchange_symbol}
            )
            if isinstance(raw, list):
                for item in raw:
                    for bracket in (item.get("brackets") or []):
                        lev = bracket.get("initialLeverage")
                        if lev:
                            max_lev = max(max_lev, int(lev))
        except Exception:
            pass

        # Method 3: From cached instrument
        meta = self.get_instrument(symbol)
        if meta:
            max_lev = max(max_lev, meta.max_leverage)

        return max_lev
