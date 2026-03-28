"""
Unit tests for trade_journal.py — uses :memory: SQLite, no exchange needed.

Run:  python3 -m pytest test_trade_journal.py -v
  or: python3 test_trade_journal.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from bot/ root
sys.path.insert(0, str(Path(__file__).parent))

from trade_journal import TradeJournal


def _make_journal() -> TradeJournal:
    return TradeJournal(":memory:")


def _entry_time() -> datetime:
    return datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def _bar_time(minutes: int) -> datetime:
    from datetime import timedelta
    return _entry_time() + timedelta(minutes=minutes * 5)


def test_open_trade_creates_row() -> None:
    j = _make_journal()
    j.open_trade(
        trade_id="t1",
        symbol="BTCUSDT",
        asset_class="crypto",
        direction="long",
        style="day",
        tier="AAA++",
        entry_time=_entry_time(),
        entry_price=50000.0,
        sl_original=49000.0,
        tp=53000.0,
        score=0.91,
        rr_target=3.0,
        leverage=5,
        risk_pct=0.01,
        entry_features={"rsi_5m": 0.45, "adx_1h": 0.6},
    )
    row = j.get_trade("t1")
    assert row is not None
    assert row["symbol"] == "BTCUSDT"
    assert row["direction"] == "long"
    assert row["tier"] == "AAA++"
    assert row["exit_time"] is None  # not yet closed
    print("PASS: test_open_trade_creates_row")


def test_record_bar_exact_count() -> None:
    j = _make_journal()
    j.open_trade(
        trade_id="t1", symbol="BTCUSDT", asset_class="crypto",
        direction="long", style="day", tier="AAA++",
        entry_time=_entry_time(), entry_price=50000.0,
        sl_original=49000.0, tp=53000.0, score=0.91,
        rr_target=3.0, leverage=5, risk_pct=0.01,
    )
    for i in range(5):
        j.record_bar(
            trade_id="t1",
            bar_index=i,
            timestamp=_bar_time(i),
            close=50100.0 + i * 50,
            high=50200.0 + i * 50,
            low=50050.0 + i * 50,
            volume=100.0,
            unrealized_pnl_pct=0.002 * (i + 1),
            sl_distance_pct=0.02,
            rsi_5m=0.55,
            adx_1h=0.4,
        )
    bars = j.get_trade_bars("t1")
    assert len(bars) == 5, f"Expected 5 bars, got {len(bars)}"
    print("PASS: test_record_bar_exact_count")


def test_record_bar_no_duplicates() -> None:
    j = _make_journal()
    j.open_trade(
        trade_id="t1", symbol="BTCUSDT", asset_class="crypto",
        direction="long", style="day", tier="AAA++",
        entry_time=_entry_time(), entry_price=50000.0,
        sl_original=49000.0, tp=53000.0, score=0.91,
        rr_target=3.0, leverage=5, risk_pct=0.01,
    )
    # Insert bar 0 twice (simulates poll loop double-fire)
    for _ in range(3):
        j.record_bar("t1", 0, _bar_time(0), 50100.0, 50200.0, 50050.0, 100.0,
                     0.002, 0.02)
    bars = j.get_trade_bars("t1")
    timestamps = [b["timestamp"] for b in bars]
    assert len(timestamps) == len(set(timestamps)), "Duplicate bar indexes"
    assert len(bars) == 1, f"Expected 1 bar (deduped), got {len(bars)}"
    print("PASS: test_record_bar_no_duplicates")


def test_close_trade_sets_exit_time() -> None:
    j = _make_journal()
    j.open_trade(
        trade_id="t1", symbol="BTCUSDT", asset_class="crypto",
        direction="long", style="day", tier="AAA++",
        entry_time=_entry_time(), entry_price=50000.0,
        sl_original=49000.0, tp=53000.0, score=0.91,
        rr_target=3.0, leverage=5, risk_pct=0.01,
    )
    for i in range(3):
        j.record_bar("t1", i, _bar_time(i), 50100.0, 50200.0, 50050.0,
                     100.0, 0.002, 0.02)

    exit_time = _bar_time(3)
    j.close_trade(
        trade_id="t1",
        exit_time=exit_time,
        exit_price=53000.0,
        outcome="win",
        exit_reason="tp_hit",
        bars_held=3,
        pnl_pct=0.06,
        rr_actual=3.0,
        max_favorable_pct=0.063,
        max_adverse_pct=-0.001,
        be_triggered=False,
    )
    row = j.get_trade("t1")
    assert row["exit_time"] is not None, "exit_time should be set"
    assert row["outcome"] == "win"
    assert row["exit_reason"] == "tp_hit"
    assert row["bars_held"] == 3
    print("PASS: test_close_trade_sets_exit_time")


def test_post_trade_bars_cumulative() -> None:
    j = _make_journal()
    j.open_trade(
        trade_id="t1", symbol="BTCUSDT", asset_class="crypto",
        direction="long", style="day", tier="AAA++",
        entry_time=_entry_time(), entry_price=50000.0,
        sl_original=49000.0, tp=53000.0, score=0.91,
        rr_target=3.0, leverage=5, risk_pct=0.01,
    )
    j.close_trade("t1", _bar_time(5), 53000.0, "win", "tp_hit", 5,
                  0.06, 3.0, 0.063, -0.001)

    post_bars = [
        {"timestamp": _bar_time(5 + i), "high": 53100.0, "low": 52900.0,
         "close": 53000.0 + i * 100}
        for i in range(10)
    ]
    j.record_post_trade_bars("t1", post_bars, exit_price=53000.0, direction="long")
    row = j.get_trade("t1")
    assert row["post_missed_pct"] is not None
    assert row["post_missed_pct"] >= 0.0
    print("PASS: test_post_trade_bars_cumulative")


def test_count_closed_trades() -> None:
    j = _make_journal()
    for i in range(3):
        tid = f"t{i}"
        j.open_trade(
            trade_id=tid, symbol="BTCUSDT", asset_class="crypto",
            direction="long", style="day", tier="AAA++",
            entry_time=_entry_time(), entry_price=50000.0,
            sl_original=49000.0, tp=53000.0, score=0.91,
            rr_target=3.0, leverage=5, risk_pct=0.01,
        )
        if i < 2:  # only close 2 of 3
            j.close_trade(tid, _bar_time(5), 53000.0, "win", "tp_hit", 5,
                          0.06, 3.0, 0.063, -0.001)

    assert j.count_closed_trades() == 2
    print("PASS: test_count_closed_trades")


def test_max_favorable_tracking() -> None:
    """Verify that max_favorable_seen increases monotonically per trade."""
    j = _make_journal()
    j.open_trade(
        trade_id="t1", symbol="BTCUSDT", asset_class="crypto",
        direction="long", style="day", tier="AAA++",
        entry_time=_entry_time(), entry_price=50000.0,
        sl_original=49000.0, tp=53000.0, score=0.91,
        rr_target=3.0, leverage=5, risk_pct=0.01,
    )
    # pnl increases then drops
    pnl_sequence = [0.01, 0.03, 0.05, 0.02, 0.01]
    for i, pnl in enumerate(pnl_sequence):
        j.record_bar("t1", i, _bar_time(i), 50000.0, 50000.0, 50000.0,
                     100.0, pnl, 0.02)

    bars = j.get_trade_bars("t1")
    max_favs = [b["max_favorable_seen"] for b in bars]
    # Should be monotonically non-decreasing
    for prev, cur in zip(max_favs, max_favs[1:]):
        assert cur >= prev, f"max_favorable dropped: {prev} -> {cur}"
    assert max_favs[-1] == 0.05, f"Expected 0.05, got {max_favs[-1]}"
    print("PASS: test_max_favorable_tracking")


def test_export_to_parquet_produces_label() -> None:
    """Test that export generates label_hold_better column."""
    try:
        import pandas as pd
    except ImportError:
        print("SKIP: test_export_to_parquet (pandas not available)")
        return

    import tempfile, os
    j = _make_journal()
    j.open_trade(
        trade_id="t1", symbol="BTCUSDT", asset_class="crypto",
        direction="long", style="day", tier="AAA++",
        entry_time=_entry_time(), entry_price=50000.0,
        sl_original=49000.0, tp=53000.0, score=0.91,
        rr_target=3.0, leverage=5, risk_pct=0.01,
    )
    for i in range(5):
        j.record_bar("t1", i, _bar_time(i), 50100.0 + i * 50,
                     50200.0 + i * 50, 50050.0, 100.0, 0.01 * (i + 1), 0.02)
    j.close_trade("t1", _bar_time(5), 53000.0, "win", "tp_hit", 5,
                  0.06, 3.0, 0.063, -0.001)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = j.export_to_parquet(tmpdir)
        assert out_path.exists(), f"Parquet not created at {out_path}"
        df = pd.read_parquet(out_path)
        assert "label_hold_better" in df.columns, "Missing label_hold_better"
        assert len(df) == 5
        assert df["label_hold_better"].isin([0, 1]).all()
    print("PASS: test_export_to_parquet_produces_label")


if __name__ == "__main__":
    tests = [
        test_open_trade_creates_row,
        test_record_bar_exact_count,
        test_record_bar_no_duplicates,
        test_close_trade_sets_exit_time,
        test_post_trade_bars_cumulative,
        test_count_closed_trades,
        test_max_favorable_tracking,
        test_export_to_parquet_produces_label,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)}/{len(tests)} tests FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed.")
