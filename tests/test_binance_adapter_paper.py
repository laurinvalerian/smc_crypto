"""
tests/test_binance_adapter_paper.py — paper_only mode regression tests.

The paper_only mode was added 2026-04-20 so the paper-validation phase can
run against real Mainnet public market data while executing orders locally
(no real exchange submission). These tests lock in that:

  1. Every order-submission method returns a synthetic OrderResult with a
     `pap-XXXXXXXX` ID and never touches `self._exchange.create_order` etc.
  2. Account-read methods return synthetic values (balance = paper_balance,
     positions = [], trades = [], open_orders = []).
  3. Margin / leverage setters are no-ops.
  4. cancel_order returns True without touching the exchange.

If any of these regress, a paper_only deployment could silently submit real
orders on mainnet — which is the exact failure these tests exist to prevent.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from exchanges.binance_adapter import BinanceAdapter


# ════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════

def _make_paper_adapter(paper_balance: float = 5000.0) -> BinanceAdapter:
    """A paper_only adapter with a MagicMock exchange that MUST NOT be touched."""
    a = BinanceAdapter(
        api_key="test",
        api_secret="test",
        testnet=False,
        paper_only=True,
        paper_balance=paper_balance,
    )
    # Attach a mock so any accidental call would throw in MagicMock side_effect.
    a._exchange = MagicMock()
    # Make sure any awaited call on the mock raises — sentinel for regressions.
    a._exchange.create_order.side_effect = AssertionError(
        "paper_only adapter must not call create_order"
    )
    a._exchange.fetch_balance.side_effect = AssertionError(
        "paper_only adapter must not call fetch_balance"
    )
    a._exchange.fetch_positions.side_effect = AssertionError(
        "paper_only adapter must not call fetch_positions"
    )
    a._exchange.fetch_my_trades.side_effect = AssertionError(
        "paper_only adapter must not call fetch_my_trades"
    )
    a._exchange.fetch_open_orders.side_effect = AssertionError(
        "paper_only adapter must not call fetch_open_orders"
    )
    a._exchange.cancel_order.side_effect = AssertionError(
        "paper_only adapter must not call cancel_order"
    )
    a._exchange.set_leverage.side_effect = AssertionError(
        "paper_only adapter must not call set_leverage"
    )
    a._exchange.set_margin_mode.side_effect = AssertionError(
        "paper_only adapter must not call set_margin_mode"
    )
    return a


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ════════════════════════════════════════════════════════════════════
#  __init__ flag semantics
# ════════════════════════════════════════════════════════════════════

def test_paper_only_default_is_false():
    """Default paper_only must be False — opt-in, not opt-out."""
    a = BinanceAdapter(api_key="k", api_secret="s", testnet=True)
    assert a._paper_only is False


def test_paper_only_flag_stored():
    a = BinanceAdapter(api_key="k", api_secret="s", testnet=False, paper_only=True)
    assert a._paper_only is True
    assert a._testnet is False


def test_paper_balance_stored_as_float():
    a = BinanceAdapter(paper_only=True, paper_balance=123)
    assert isinstance(a._paper_balance, float)
    assert a._paper_balance == 123.0


# ════════════════════════════════════════════════════════════════════
#  Synthetic order helper
# ════════════════════════════════════════════════════════════════════

def test_synthetic_order_has_pap_prefix():
    a = _make_paper_adapter()
    r = a._synthetic_order("BTC/USDT:USDT", "buy", 0.001, "market", 50000.0)
    assert r.order_id.startswith("pap-")


def test_synthetic_order_ids_are_unique_and_sequential():
    a = _make_paper_adapter()
    ids = [
        a._synthetic_order("BTC/USDT:USDT", "buy", 0.001, "market", 50000.0).order_id
        for _ in range(3)
    ]
    assert len(set(ids)) == 3
    # Sequential counter → ids sort lexicographically in creation order
    assert ids == sorted(ids)


def test_synthetic_order_market_status_is_filled():
    a = _make_paper_adapter()
    r = a._synthetic_order("BTC/USDT:USDT", "buy", 0.001, "market", 50000.0)
    assert r.status == "filled"


def test_synthetic_order_non_market_status_is_open():
    a = _make_paper_adapter()
    r = a._synthetic_order("BTC/USDT:USDT", "sell", 0.001, "stop_market", 49000.0)
    assert r.status == "open"


def test_synthetic_order_raw_marks_paper_only():
    a = _make_paper_adapter()
    r = a._synthetic_order("BTC/USDT:USDT", "buy", 0.001, "market", 50000.0)
    assert r.raw["paper_only"] is True
    assert r.raw["synthetic_id"] == r.order_id


# ════════════════════════════════════════════════════════════════════
#  Order-submission methods never touch the exchange in paper_only
# ════════════════════════════════════════════════════════════════════

def test_create_market_order_is_synthetic_in_paper_mode():
    a = _make_paper_adapter()
    # Mock tickers dict since create_market_order probes it for fill price.
    a._exchange.tickers = {"BTC/USDT:USDT": {"last": 50000.0}}
    r = _run(a.create_market_order("BTC/USDT:USDT", "buy", 0.001))
    assert r.order_id.startswith("pap-")
    assert r.order_type == "market"
    assert r.status == "filled"
    assert r.qty == 0.001


def test_create_stop_loss_is_synthetic_in_paper_mode():
    a = _make_paper_adapter()
    r = _run(a.create_stop_loss("BTC/USDT:USDT", "sell", 0.001, 49000.0))
    assert r.order_id.startswith("pap-")
    assert r.order_type == "stop_market"
    assert r.price == 49000.0


def test_create_take_profit_is_synthetic_in_paper_mode():
    a = _make_paper_adapter()
    r = _run(a.create_take_profit("BTC/USDT:USDT", "sell", 0.001, 51000.0))
    assert r.order_id.startswith("pap-")
    assert r.order_type == "take_profit_market"
    assert r.price == 51000.0


def test_cancel_order_returns_true_without_exchange_call():
    a = _make_paper_adapter()
    assert _run(a.cancel_order("pap-00000001", "BTC/USDT:USDT")) is True


# ════════════════════════════════════════════════════════════════════
#  Account-read methods return synthetic values
# ════════════════════════════════════════════════════════════════════

def test_fetch_balance_returns_paper_balance():
    a = _make_paper_adapter(paper_balance=2500.0)
    bal = _run(a.fetch_balance())
    assert bal.currency == "USDT"
    assert bal.total == 2500.0
    assert bal.free == 2500.0
    assert bal.used == 0.0
    assert bal.raw["paper_only"] is True


def test_fetch_positions_returns_empty_list():
    a = _make_paper_adapter()
    assert _run(a.fetch_positions()) == []


def test_fetch_my_trades_returns_empty_list():
    a = _make_paper_adapter()
    assert _run(a.fetch_my_trades("BTC/USDT:USDT")) == []


def test_fetch_open_orders_returns_empty_list():
    a = _make_paper_adapter()
    assert _run(a.fetch_open_orders("BTC/USDT:USDT")) == []


# ════════════════════════════════════════════════════════════════════
#  Margin / leverage setters are no-ops
# ════════════════════════════════════════════════════════════════════

def test_set_leverage_is_noop_in_paper_mode():
    a = _make_paper_adapter()
    # Should not raise, not touch exchange
    _run(a.set_leverage(5, "BTC/USDT:USDT"))


def test_set_margin_mode_is_noop_in_paper_mode():
    a = _make_paper_adapter()
    _run(a.set_margin_mode("cross", "BTC/USDT:USDT"))


def test_fetch_max_leverage_uses_meta_not_exchange_in_paper():
    """fetch_max_leverage must fall back to cached meta in paper mode."""
    from exchanges.models import InstrumentMeta

    a = _make_paper_adapter()
    a._instruments["BTC/USDT:USDT"] = InstrumentMeta(
        symbol="BTC/USDT:USDT",
        exchange_symbol="BTCUSDT",
        asset_class="crypto",
        exchange_id="binanceusdm",
        tick_size=0.01,
        lot_size=0.001,
        min_qty=0.001,
        max_qty=1000.0,
        min_notional=5.0,
        max_leverage=50,
        default_leverage=20,
        trades_24_7=True,
        spread_typical_pct=0.0001,
        commission_pct=0.0004,
        raw_info={},
    )
    lev = _run(a.fetch_max_leverage("BTC/USDT:USDT"))
    assert lev >= 20  # default floor
    assert lev == 50  # from meta


# ════════════════════════════════════════════════════════════════════
#  Negative tests: non-paper mode should still call exchange
# ════════════════════════════════════════════════════════════════════

def test_non_paper_mode_still_calls_exchange_for_orders():
    """Guard against accidentally wiring paper_only=True for everyone."""
    import asyncio as _asyncio
    a = BinanceAdapter(api_key="k", api_secret="s", testnet=True, paper_only=False)
    a._exchange = MagicMock()
    # Mock create_order to return a dict like the real exchange would
    fut = _asyncio.Future()
    fut.set_result({"id": "real-123", "status": "filled", "average": 50000})
    a._exchange.create_order.return_value = fut
    r = _run(a.create_market_order("BTC/USDT:USDT", "buy", 0.001))
    a._exchange.create_order.assert_called_once()
    assert r.order_id == "real-123"
