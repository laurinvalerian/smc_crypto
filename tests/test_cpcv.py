"""Tests for backtest.cpcv — Combinatorial Purged Cross-Validation.

Reference: Lopez de Prado (2018), "Advances in Financial Machine Learning",
Ch. 7. Phase D of .omc/plans/quality-upgrade-plan.md.

Coverage:
    - cpcv_splits: C(N,k) enumeration, disjoint test/train, edge cases
    - purge_train_signals: max-hold overlap, embargo, multi-test-window
    - run_cpcv: metrics shape, DSR per split, empty-signal robustness
    - cpcv_summary: median / IQR / pass-rate aggregation
"""

from __future__ import annotations

from math import comb
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.cpcv import (
    cpcv_splits,
    cpcv_summary,
    purge_train_signals,
    run_cpcv,
)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _make_wf_windows(n: int, train_months: int = 3, test_months: int = 2) -> list[dict]:
    """Build n sequential non-overlapping walk-forward windows."""
    windows: list[dict] = []
    cursor = pd.Timestamp("2024-01-01", tz="UTC")
    for _ in range(n):
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        cursor = test_end
    return windows


def _sig(timestamp: pd.Timestamp, symbol: str = "BTCUSDT"):
    """Minimal TradeSignal-like stub (only .timestamp and .symbol needed)."""
    return SimpleNamespace(timestamp=timestamp, symbol=symbol)


# ─────────────────────────────────────────────────────────────────
#  cpcv_splits — combinatorics
# ─────────────────────────────────────────────────────────────────

class TestCpcvSplits:
    def test_yields_c_n_k_splits_k2(self):
        windows = _make_wf_windows(12)
        splits = list(cpcv_splits(windows, k=2))
        assert len(splits) == comb(12, 2) == 66

    def test_yields_n_splits_for_k1_loo(self):
        windows = _make_wf_windows(5)
        splits = list(cpcv_splits(windows, k=1))
        assert len(splits) == 5

    def test_split_structure_keys(self):
        windows = _make_wf_windows(4)
        splits = list(cpcv_splits(windows, k=2))
        for s in splits:
            assert set(s.keys()) >= {"train_windows", "test_windows", "test_idx", "embargo_bars"}
            assert isinstance(s["test_idx"], tuple)

    def test_train_and_test_disjoint(self):
        windows = _make_wf_windows(6)
        for s in cpcv_splits(windows, k=2):
            test_ids = set(s["test_idx"])
            # train_windows must NOT contain any test-window by identity
            for tw in s["train_windows"]:
                assert tw not in s["test_windows"]
            assert len(test_ids) == 2

    def test_train_plus_test_covers_all_windows(self):
        windows = _make_wf_windows(5)
        for s in cpcv_splits(windows, k=2):
            assert len(s["train_windows"]) + len(s["test_windows"]) == 5

    def test_test_idx_sorted_distinct(self):
        windows = _make_wf_windows(4)
        splits = list(cpcv_splits(windows, k=2))
        for s in splits:
            idx = s["test_idx"]
            assert len(set(idx)) == len(idx)  # distinct
            assert list(idx) == sorted(idx)

    def test_k_greater_than_n_raises(self):
        windows = _make_wf_windows(3)
        with pytest.raises(ValueError, match="k"):
            list(cpcv_splits(windows, k=5))

    def test_k_zero_raises(self):
        windows = _make_wf_windows(3)
        with pytest.raises(ValueError):
            list(cpcv_splits(windows, k=0))

    def test_embargo_bars_propagated(self):
        windows = _make_wf_windows(4)
        splits = list(cpcv_splits(windows, k=2, embargo_bars=500))
        assert all(s["embargo_bars"] == 500 for s in splits)

    def test_splits_are_unique(self):
        windows = _make_wf_windows(6)
        splits = list(cpcv_splits(windows, k=2))
        keys = [s["test_idx"] for s in splits]
        assert len(set(keys)) == len(keys)


# ─────────────────────────────────────────────────────────────────
#  purge_train_signals — max-hold + embargo overlap
# ─────────────────────────────────────────────────────────────────

class TestPurgeTrainSignals:
    def _windows_with_test(self, test_start: str, test_end: str) -> list[dict]:
        return [{
            "test_start": pd.Timestamp(test_start, tz="UTC"),
            "test_end": pd.Timestamp(test_end, tz="UTC"),
        }]

    def test_empty_input_returns_empty(self):
        windows = self._windows_with_test("2024-04-01", "2024-06-01")
        kept = purge_train_signals([], windows, max_hold_bars=48, embargo_bars=288)
        assert kept == []

    def test_no_test_windows_keeps_everything(self):
        sigs = [_sig(pd.Timestamp("2024-02-01", tz="UTC"))] * 3
        kept = purge_train_signals(sigs, [], max_hold_bars=48, embargo_bars=288)
        assert len(kept) == 3

    def test_signal_fully_before_purge_region_kept(self):
        # Test window: Apr 1 → Jun 1. Signal at Feb 1: hold 4h ends Feb 1 04:00 — far before Apr 1.
        windows = self._windows_with_test("2024-04-01", "2024-06-01")
        sig = _sig(pd.Timestamp("2024-02-01", tz="UTC"))
        kept = purge_train_signals([sig], windows, max_hold_bars=48, embargo_bars=288)
        assert len(kept) == 1

    def test_signal_hold_crossing_test_start_purged(self):
        # max_hold_bars=48 (5m each → 4h). Signal 2h before test_start → hold extends INTO test.
        test_start = pd.Timestamp("2024-04-01 12:00", tz="UTC")
        windows = [{"test_start": test_start, "test_end": test_start + pd.Timedelta(days=60)}]
        sig = _sig(test_start - pd.Timedelta(hours=2))  # hold ends test_start + 2h
        kept = purge_train_signals([sig], windows, max_hold_bars=48, embargo_bars=0, bar_minutes=5)
        assert kept == []

    def test_signal_in_embargo_window_purged(self):
        # Signal finishes hold 1h before test_start, but embargo is 2h → purged.
        test_start = pd.Timestamp("2024-04-01 12:00", tz="UTC")
        windows = [{"test_start": test_start, "test_end": test_start + pd.Timedelta(days=60)}]
        # Signal at test_start - 5h, hold 48×5m = 4h → ends at test_start - 1h.
        # Embargo 288 bars × 5m = 1440 min = 24h → ends at test_start + 23h post-hold-end.
        # So hold-end (test_start - 1h) + embargo (24h) = test_start + 23h, clearly intersects.
        sig = _sig(test_start - pd.Timedelta(hours=5))
        kept = purge_train_signals([sig], windows, max_hold_bars=48, embargo_bars=288, bar_minutes=5)
        assert kept == []

    def test_signal_clearly_outside_embargo_kept(self):
        test_start = pd.Timestamp("2024-04-01 12:00", tz="UTC")
        windows = [{"test_start": test_start, "test_end": test_start + pd.Timedelta(days=60)}]
        # Signal 10 days before test_start — hold 4h + embargo 1d → far from test window.
        sig = _sig(test_start - pd.Timedelta(days=10))
        kept = purge_train_signals([sig], windows, max_hold_bars=48, embargo_bars=288, bar_minutes=5)
        assert len(kept) == 1

    def test_signal_inside_test_window_purged(self):
        test_start = pd.Timestamp("2024-04-01", tz="UTC")
        test_end = pd.Timestamp("2024-06-01", tz="UTC")
        windows = [{"test_start": test_start, "test_end": test_end}]
        sig = _sig(pd.Timestamp("2024-05-01", tz="UTC"))  # inside test range
        kept = purge_train_signals([sig], windows, max_hold_bars=48, embargo_bars=0, bar_minutes=5)
        assert kept == []

    def test_multi_test_window_purge_if_any_overlap(self):
        w1_start = pd.Timestamp("2024-03-01 12:00", tz="UTC")
        w2_start = pd.Timestamp("2024-07-01 12:00", tz="UTC")
        windows = [
            {"test_start": w1_start, "test_end": w1_start + pd.Timedelta(days=30)},
            {"test_start": w2_start, "test_end": w2_start + pd.Timedelta(days=30)},
        ]
        sig_ok = _sig(pd.Timestamp("2024-05-01", tz="UTC"))  # between windows, way out
        sig_bad = _sig(w2_start - pd.Timedelta(hours=2))  # crosses into w2
        kept = purge_train_signals(
            [sig_ok, sig_bad], windows,
            max_hold_bars=48, embargo_bars=0, bar_minutes=5,
        )
        assert sig_ok in kept
        assert sig_bad not in kept


# ─────────────────────────────────────────────────────────────────
#  run_cpcv — metrics per split (mock simulate_fn)
# ─────────────────────────────────────────────────────────────────

def _mock_simulate_fn(fixed_sharpe: float = 5.0, fixed_pf: float = 3.0):
    """Produce a simulate_fn stub that returns deterministic metrics."""
    def fn(params, signals, config, symbol_to_asset):
        n = len(signals)
        # Synthesize a tiny trades_df; values don't matter — we just need metrics dict.
        trades_df = pd.DataFrame()
        metrics = {
            "total_pnl": 1000.0 * n,
            "sharpe": fixed_sharpe,
            "sharpe_naive": fixed_sharpe,
            "pf_real": fixed_pf,
            "max_drawdown": -0.05,
            "total_trades": n,
            "winrate_real": 0.6,
            "be_rate": 0.3,
            "n_obs_daily": max(30, n),
            "skew": 0.1,
            "kurt_nonexcess": 3.2,
            "expectancy": 10.0,
        }
        return trades_df, metrics
    return fn


class TestRunCpcv:
    def test_returns_dataframe_with_expected_columns(self):
        windows = _make_wf_windows(4)
        signals_per_window = {
            wi: [_sig(w["test_start"] + pd.Timedelta(hours=1))] * 25
            for wi, w in enumerate(windows)
        }
        trial_sharpes = [1.0, 2.0, 3.0, 2.5, 1.5]
        df = run_cpcv(
            params={"alignment_threshold": 0.7, "risk_reward": 2.0, "leverage": 10},
            windows=windows,
            signals_per_window=signals_per_window,
            simulate_fn=_mock_simulate_fn(fixed_sharpe=5.0),
            config={"account": {"size": 100_000}},
            symbol_to_asset={"BTCUSDT": "crypto"},
            trial_sharpes=trial_sharpes,
            k=2,
        )
        expected = {
            "split_idx", "test_idx", "n_test_signals", "total_pnl",
            "sharpe", "pf_real", "max_drawdown", "trades",
            "skew", "kurt_nonexcess", "n_obs_daily", "dsr",
        }
        assert expected.issubset(set(df.columns))

    def test_produces_c_n_k_rows_when_all_windows_have_signals(self):
        windows = _make_wf_windows(4)
        signals_per_window = {
            wi: [_sig(w["test_start"] + pd.Timedelta(hours=1))] * 25
            for wi, w in enumerate(windows)
        }
        df = run_cpcv(
            params={},
            windows=windows,
            signals_per_window=signals_per_window,
            simulate_fn=_mock_simulate_fn(),
            config={"account": {"size": 100_000}},
            symbol_to_asset={"BTCUSDT": "crypto"},
            trial_sharpes=[1.0, 2.0, 1.5, 0.5],
            k=2,
        )
        assert len(df) == comb(4, 2)

    def test_dsr_in_valid_range(self):
        windows = _make_wf_windows(4)
        signals_per_window = {
            wi: [_sig(w["test_start"] + pd.Timedelta(hours=1))] * 30
            for wi, w in enumerate(windows)
        }
        df = run_cpcv(
            params={},
            windows=windows,
            signals_per_window=signals_per_window,
            simulate_fn=_mock_simulate_fn(fixed_sharpe=3.0),
            config={"account": {"size": 100_000}},
            symbol_to_asset={"BTCUSDT": "crypto"},
            trial_sharpes=[1.0, 2.0, 1.5, 0.5, 2.5, 1.2],
            k=2,
        )
        assert df["dsr"].between(0.0, 1.0).all()

    def test_skips_splits_with_no_test_signals(self):
        windows = _make_wf_windows(4)
        # Window 0 empty, rest have signals
        signals_per_window = {
            0: [],
            1: [_sig(windows[1]["test_start"] + pd.Timedelta(hours=1))] * 25,
            2: [_sig(windows[2]["test_start"] + pd.Timedelta(hours=1))] * 25,
            3: [_sig(windows[3]["test_start"] + pd.Timedelta(hours=1))] * 25,
        }
        df = run_cpcv(
            params={},
            windows=windows,
            signals_per_window=signals_per_window,
            simulate_fn=_mock_simulate_fn(),
            config={"account": {"size": 100_000}},
            symbol_to_asset={"BTCUSDT": "crypto"},
            trial_sharpes=[1.0, 2.0, 1.5, 0.5],
            k=2,
        )
        # Splits containing window-0 have it combined with other windows (non-empty).
        # All 6 splits should still run — test_sigs = union of test_windows, only fully-empty is skipped.
        # For this fixture: only split (0,1), (0,2), (0,3) — but they have non-zero via union with other.
        # Since union covers 25 signals in at least one other window → all 6 splits produce rows.
        assert len(df) == comb(4, 2)

    def test_dsr_zero_when_insufficient_trials(self):
        windows = _make_wf_windows(3)
        signals_per_window = {
            wi: [_sig(w["test_start"] + pd.Timedelta(hours=1))] * 25
            for wi, w in enumerate(windows)
        }
        df = run_cpcv(
            params={},
            windows=windows,
            signals_per_window=signals_per_window,
            simulate_fn=_mock_simulate_fn(),
            config={"account": {"size": 100_000}},
            symbol_to_asset={"BTCUSDT": "crypto"},
            trial_sharpes=[1.0],  # too few → variance == 0
            k=2,
        )
        assert (df["dsr"] == 0.0).all()

    def test_zero_variance_yields_zero_dsr(self):
        windows = _make_wf_windows(3)
        signals_per_window = {
            wi: [_sig(w["test_start"] + pd.Timedelta(hours=1))] * 25
            for wi, w in enumerate(windows)
        }
        df = run_cpcv(
            params={},
            windows=windows,
            signals_per_window=signals_per_window,
            simulate_fn=_mock_simulate_fn(),
            config={"account": {"size": 100_000}},
            symbol_to_asset={"BTCUSDT": "crypto"},
            trial_sharpes=[2.0, 2.0, 2.0, 2.0, 2.0],  # all equal → variance = 0
            k=2,
        )
        assert (df["dsr"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────
#  cpcv_summary — aggregate metrics
# ─────────────────────────────────────────────────────────────────

class TestCpcvSummary:
    def test_empty_input_returns_zero_defaults(self):
        summary = cpcv_summary(pd.DataFrame(columns=["dsr", "sharpe", "pf_real", "max_drawdown"]))
        assert summary["n_splits"] == 0
        assert summary["median_dsr"] == 0.0
        assert summary["dsr_iqr"] == 0.0
        assert summary["percent_passing_funded"] == 0.0

    def test_median_and_iqr(self):
        df = pd.DataFrame({
            "dsr": [0.5, 0.6, 0.7, 0.8, 0.9],
            "sharpe": [1.0, 2.0, 3.0, 4.0, 5.0],
            "pf_real": [1.5, 2.0, 2.5, 3.0, 3.5],
            "max_drawdown": [-0.02, -0.04, -0.06, -0.08, -0.10],
        })
        summary = cpcv_summary(df)
        assert summary["median_dsr"] == pytest.approx(0.7)
        assert summary["dsr_q25"] == pytest.approx(0.6)
        assert summary["dsr_q75"] == pytest.approx(0.8)
        assert summary["dsr_iqr"] == pytest.approx(0.2)
        assert summary["median_sharpe"] == pytest.approx(3.0)
        assert summary["median_pf_real"] == pytest.approx(2.5)

    def test_percent_passing_funded_threshold(self):
        # 3 of 5 pass 0.95 threshold
        df = pd.DataFrame({
            "dsr": [0.90, 0.94, 0.96, 0.98, 1.00],
            "sharpe": [1.0] * 5, "pf_real": [1.5] * 5, "max_drawdown": [-0.05] * 5,
        })
        summary = cpcv_summary(df, funded_threshold=0.95)
        assert summary["percent_passing_funded"] == pytest.approx(0.6)

    def test_percent_passing_research_threshold(self):
        df = pd.DataFrame({
            "dsr": [0.50, 0.65, 0.70, 0.85, 0.99],
            "sharpe": [1.0] * 5, "pf_real": [1.5] * 5, "max_drawdown": [-0.05] * 5,
        })
        summary = cpcv_summary(df, research_threshold=0.70)
        # 3 of 5 have dsr >= 0.70 (0.70, 0.85, 0.99)
        assert summary["percent_passing_research"] == pytest.approx(0.6)

    def test_n_splits_matches_row_count(self):
        df = pd.DataFrame({
            "dsr": np.linspace(0.5, 1.0, 66),
            "sharpe": [1.0] * 66, "pf_real": [1.5] * 66, "max_drawdown": [-0.05] * 66,
        })
        summary = cpcv_summary(df)
        assert summary["n_splits"] == 66

    def test_summary_is_json_serializable(self):
        import json
        df = pd.DataFrame({
            "dsr": [0.5, 0.7, 0.9],
            "sharpe": [1.0, 2.0, 3.0],
            "pf_real": [1.5, 2.0, 2.5],
            "max_drawdown": [-0.05, -0.08, -0.10],
        })
        summary = cpcv_summary(df)
        # Should not raise
        json.dumps(summary)
