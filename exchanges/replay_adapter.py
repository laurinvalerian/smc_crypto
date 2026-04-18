"""
Replay Exchange Adapter
=======================
Virtual exchange that simulates order fills against historical candle data.
Implements the same ExchangeAdapter interface as Binance/OANDA/Alpaca
so the live bot code runs identically in replay mode.
"""
from __future__ import annotations

import logging
import uuid
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

# Phase 2.1 SSOT (2026-04-18): value imported from core.constants.
# Crypto-only after Phase 1 strip — dict form retained for caller compatibility.
from core.constants import COMMISSION
DEFAULT_COMMISSION = {"crypto": COMMISSION}


class _VirtualOrder:
    """Tracks a simulated bracket order (entry + SL + TP)."""
    __slots__ = (
        "order_id", "symbol", "direction", "qty", "entry_price",
        "sl_price", "tp_price", "sl_order_id", "tp_order_id",
        "entry_time", "filled",
    )

    def __init__(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        qty: float,
        entry_price: float,
        entry_time: datetime,
    ):
        self.order_id = order_id
        self.symbol = symbol
        self.direction = direction  # "long" or "short"
        self.qty = qty
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.sl_price: float = 0.0
        self.tp_price: float = 0.0
        self.sl_order_id: str = ""
        self.tp_order_id: str = ""
        self.filled = True  # market orders fill immediately


class ReplayAdapter(ExchangeAdapter):
    """
    Virtual exchange for backtesting / market replay.

    Tracks virtual positions and simulates SL/TP fills against candle OHLC.
    No real exchange connection — all state is in-memory.
    """

    def __init__(
        self,
        asset_class: str = "crypto",
        initial_balance: float = 100_000.0,
        commission_rate: float | None = None,
        leverage: int = 1,
    ):
        self._asset_class = asset_class
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._realized_pnl = 0.0
        self._commission_rate = commission_rate or DEFAULT_COMMISSION.get(asset_class, 0.0004)
        self._leverage = leverage

        # Active orders (symbol → _VirtualOrder)
        self._orders: dict[str, _VirtualOrder] = {}
        # Instrument cache
        self._instruments: dict[str, InstrumentMeta] = {}
        # Current prices (updated on each candle)
        self._current_prices: dict[str, float] = {}
        # Current timestamp
        self._current_time: datetime = datetime(2023, 1, 1, tzinfo=timezone.utc)

    # ── Identity ────────────────────────────────────────────────────

    @property
    def exchange_id(self) -> str:
        return "replay"

    @property
    def asset_class(self) -> str:
        return self._asset_class

    @property
    def supports_attached_sl_tp(self) -> bool:
        return False

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        logger.info("ReplayAdapter connected: asset_class=%s, balance=%.2f", self._asset_class, self._balance)

    async def close(self) -> None:
        pass

    # ── Market Data ─────────────────────────────────────────────────

    async def load_markets(self) -> dict[str, InstrumentMeta]:
        return self._instruments

    def _ensure_instrument(self, symbol: str) -> InstrumentMeta:
        """Create a default InstrumentMeta if not cached."""
        if symbol not in self._instruments:
            is_crypto = self._asset_class == "crypto"
            is_forex = self._asset_class in ("forex", "commodities")
            is_stocks = self._asset_class == "stocks"

            trading_hours = None
            if is_stocks:
                trading_hours = [(13, 20)]  # NYSE 13:30-20:00 UTC (approximated)
            elif is_forex:
                trading_hours = [(0, 24)]  # 24h Mon-Fri

            self._instruments[symbol] = InstrumentMeta(
                symbol=symbol,
                exchange_symbol=symbol,
                asset_class=self._asset_class,
                exchange_id="replay",
                tick_size=0.00001 if is_forex else 0.01,
                lot_size=1.0 if is_stocks else 0.001,
                min_qty=1.0 if is_stocks else 0.001,
                max_leverage=self._leverage,
                default_leverage=self._leverage,
                trades_24_7=is_crypto,
                trading_hours=trading_hours,
                commission_pct=self._commission_rate,
            )
        return self._instruments[symbol]

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "5m",
                          since: int | None = None, limit: int = 500) -> list[list[float]]:
        return []  # Not used in replay — data comes from runner

    async def watch_ohlcv(self, symbol: str, timeframe: str = "5m") -> list[list]:
        return []  # Not used in replay

    async def watch_ticker(self, symbol: str) -> dict[str, Any]:
        price = self._current_prices.get(symbol, 0.0)
        return {"last": price, "bid": price, "ask": price}

    # ── Instrument Info ─────────────────────────────────────────────

    def get_instrument(self, symbol: str) -> InstrumentMeta | None:
        return self._ensure_instrument(symbol)

    def price_to_precision(self, symbol: str, price: float) -> float:
        return price  # No rounding needed in replay

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return amount

    # ── Trading ─────────────────────────────────────────────────────

    async def create_market_order(self, symbol: str, side: str, qty: float,
                                  params: dict[str, Any] | None = None) -> OrderResult:
        """Simulate a market order fill at current price."""
        order_id = str(uuid.uuid4())[:8]
        price = self._current_prices.get(symbol, 0.0)
        direction = "long" if side == "buy" else "short"

        order = _VirtualOrder(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            qty=qty,
            entry_price=price,
            entry_time=self._current_time,
        )
        self._orders[symbol] = order

        # Deduct commission
        commission = qty * price * self._commission_rate
        self._balance -= commission

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            price=price,
            status="filled",
        )

    async def create_stop_loss(self, symbol: str, side: str, qty: float,
                               stop_price: float,
                               params: dict[str, Any] | None = None) -> OrderResult:
        sl_id = f"sl-{uuid.uuid4().hex[:6]}"
        if symbol in self._orders:
            self._orders[symbol].sl_price = stop_price
            self._orders[symbol].sl_order_id = sl_id
        return OrderResult(order_id=sl_id, symbol=symbol, side=side,
                           order_type="stop_market", qty=qty, price=stop_price, status="open")

    async def create_take_profit(self, symbol: str, side: str, qty: float,
                                 stop_price: float,
                                 params: dict[str, Any] | None = None) -> OrderResult:
        tp_id = f"tp-{uuid.uuid4().hex[:6]}"
        if symbol in self._orders:
            self._orders[symbol].tp_price = stop_price
            self._orders[symbol].tp_order_id = tp_id
        return OrderResult(order_id=tp_id, symbol=symbol, side=side,
                           order_type="take_profit_market", qty=qty, price=stop_price, status="open")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        # Cancel SL or TP — find by order_id
        if symbol in self._orders:
            o = self._orders[symbol]
            if o.sl_order_id == order_id:
                o.sl_order_id = ""
                o.sl_price = 0.0
                return True
            if o.tp_order_id == order_id:
                o.tp_order_id = ""
                o.tp_price = 0.0
                return True
        return False

    async def modify_stop_loss(self, old_order_id: str, symbol: str, side: str,
                               qty: float, new_stop_price: float, *,
                               trade_id: str | None = None) -> OrderResult | None:
        """Update SL price directly (no cancel+replace needed in replay)."""
        if symbol in self._orders:
            o = self._orders[symbol]
            new_sl_id = f"sl-{uuid.uuid4().hex[:6]}"
            o.sl_price = new_stop_price
            o.sl_order_id = new_sl_id
            return OrderResult(order_id=new_sl_id, symbol=symbol, side=side,
                               order_type="stop_market", qty=qty, price=new_stop_price, status="open")
        return None

    async def close_position(self, symbol: str, qty: float | None = None,
                             side: str | None = None) -> OrderResult | None:
        """Force-close a position at current price."""
        if symbol not in self._orders:
            return None
        o = self._orders[symbol]
        exit_price = self._current_prices.get(symbol, o.entry_price)
        pnl = self._compute_pnl(o, exit_price)
        self._realized_pnl += pnl
        self._balance += pnl - (o.qty * exit_price * self._commission_rate)  # exit commission
        del self._orders[symbol]
        return OrderResult(order_id=f"close-{uuid.uuid4().hex[:6]}", symbol=symbol,
                           side="sell" if o.direction == "long" else "buy",
                           order_type="market", qty=o.qty, price=exit_price, status="filled")

    async def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        if symbol in self._orders:
            o = self._orders[symbol]
            orders = []
            if o.sl_order_id:
                orders.append({"id": o.sl_order_id, "symbol": symbol, "type": "stop_market",
                               "price": o.sl_price, "status": "open"})
            if o.tp_order_id:
                orders.append({"id": o.tp_order_id, "symbol": symbol, "type": "take_profit_market",
                               "price": o.tp_price, "status": "open"})
            return orders
        return []

    # ── Account ─────────────────────────────────────────────────────

    async def fetch_balance(self) -> BalanceInfo:
        # Include unrealized PnL in total
        unrealized = sum(
            self._compute_pnl(o, self._current_prices.get(o.symbol, o.entry_price))
            for o in self._orders.values()
        )
        return BalanceInfo(
            currency="USD",
            total=self._balance + unrealized,
            free=self._balance,
            used=sum(o.qty * o.entry_price / self._leverage for o in self._orders.values()),
        )

    async def fetch_positions(self) -> list[PositionInfo]:
        positions = []
        for o in self._orders.values():
            price = self._current_prices.get(o.symbol, o.entry_price)
            positions.append(PositionInfo(
                symbol=o.symbol,
                side=o.direction,
                qty=o.qty,
                entry_price=o.entry_price,
                unrealized_pnl=self._compute_pnl(o, price),
                leverage=self._leverage,
            ))
        return positions

    async def fetch_my_trades(self, symbol: str, since: int | None = None,
                              limit: int = 50) -> list[dict[str, Any]]:
        return []  # Trade history managed by journal

    # ── Leverage & Margin ───────────────────────────────────────────

    async def set_leverage(self, leverage: int, symbol: str) -> None:
        self._leverage = leverage

    async def set_margin_mode(self, mode: str, symbol: str) -> None:
        pass  # No margin modes in replay

    # ── Replay-Specific Methods ─────────────────────────────────────

    def update_price(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Called by replay runner on each candle to update current price."""
        self._current_prices[symbol] = price
        self._current_time = timestamp
        self._ensure_instrument(symbol)

    def check_and_fill_orders(
        self, symbol: str, candle: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Check if any active order's SL/TP was hit by this candle.

        Returns list of fill events: [{"symbol", "exit_price", "exit_reason", "order"}]
        Conservative: if both SL and TP hit in same candle, SL wins.
        """
        if symbol not in self._orders:
            return []

        o = self._orders[symbol]
        high = candle.get("high", 0.0)
        low = candle.get("low", 0.0)

        sl_hit = False
        tp_hit = False

        if o.direction == "long":
            if o.sl_price > 0 and low <= o.sl_price:
                sl_hit = True
            if o.tp_price > 0 and high >= o.tp_price:
                tp_hit = True
        else:  # short
            if o.sl_price > 0 and high >= o.sl_price:
                sl_hit = True
            if o.tp_price > 0 and low <= o.tp_price:
                tp_hit = True

        if not sl_hit and not tp_hit:
            return []

        # Conservative: SL priority if both hit
        if sl_hit:
            exit_price = o.sl_price
            exit_reason = "sl_hit"
        else:
            exit_price = o.tp_price
            exit_reason = "tp_hit"

        # Compute PnL and update balance
        pnl = self._compute_pnl(o, exit_price)
        exit_commission = o.qty * exit_price * self._commission_rate
        self._realized_pnl += pnl
        self._balance += pnl - exit_commission

        fill = {
            "symbol": symbol,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl - exit_commission,
            "order": o,
            "direction": o.direction,
            "entry_price": o.entry_price,
            "qty": o.qty,
            "order_id": o.order_id,
        }
        del self._orders[symbol]
        return [fill]

    def has_position(self, symbol: str) -> bool:
        return symbol in self._orders

    def get_order(self, symbol: str) -> _VirtualOrder | None:
        return self._orders.get(symbol)

    def reset(self, initial_balance: float | None = None) -> None:
        """Reset all state for a new variant run."""
        self._balance = initial_balance or self._initial_balance
        self._realized_pnl = 0.0
        self._orders.clear()
        self._current_prices.clear()

    # ── Internal ────────────────────────────────────────────────────

    @staticmethod
    def _compute_pnl(order: _VirtualOrder, exit_price: float) -> float:
        if order.direction == "long":
            return (exit_price - order.entry_price) * order.qty
        else:
            return (order.entry_price - exit_price) * order.qty
