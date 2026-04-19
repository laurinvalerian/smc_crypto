"""Tests for drift_monitor harden-ups that support the cron safety net.

Rationale
---------
The pre-funded paper phase launches against a fresh journal DB. The
``rejected_signals`` table does not exist until the bot writes its first
near-miss — a window of 0–60 min after startup. The standalone cron
runner must not crash in that window (false-positive alerts feed an
on-call page).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from drift_monitor import _load_live_features, run_drift_check_once


def _make_empty_db(path: Path) -> Path:
    """Create a brand-new SQLite DB with no tables (sim. pre-bot boot)."""
    conn = sqlite3.connect(path)
    conn.close()
    return path


def _make_db_with_other_tables(path: Path) -> Path:
    """Simulate a bot that wrote `trades` but never wrote a near-miss yet."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (id INTEGER)")
    conn.close()
    return path


def _make_db_with_rejected_signals(path: Path) -> Path:
    """Create the full drift-relevant schema with one row."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE rejected_signals (
            symbol TEXT,
            asset_class TEXT,
            timestamp TEXT,
            entry_features TEXT
        )
    """)
    conn.execute("""
        INSERT INTO rejected_signals VALUES (
            'BTCUSDT', 'crypto', '2026-04-19T09:00:00',
            '{"alignment_score": 0.75, "adx_1h": 25.0}'
        )
    """)
    conn.commit()
    conn.close()
    return path


class TestLoadLiveFeaturesSafety:
    def test_nonexistent_db_returns_empty_df(self, tmp_path):
        assert _load_live_features(str(tmp_path / "does_not_exist.db")).empty

    def test_fresh_db_with_no_tables_returns_empty(self, tmp_path):
        """Pre-bot boot: DB file exists but no schema yet."""
        db = _make_empty_db(tmp_path / "journal.db")
        out = _load_live_features(str(db))
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_db_without_rejected_signals_table_returns_empty(self, tmp_path):
        """Bot created `trades` but never emitted a near-miss — no table."""
        db = _make_db_with_other_tables(tmp_path / "journal.db")
        out = _load_live_features(str(db))
        assert out.empty

    def test_db_with_populated_table_returns_row(self, tmp_path):
        db = _make_db_with_rejected_signals(tmp_path / "journal.db")
        out = _load_live_features(str(db))
        assert len(out) == 1
        assert out.iloc[0]["symbol"] == "BTCUSDT"
        assert out.iloc[0]["alignment_score"] == 0.75


class TestCronRunnerReturnsOkOnEmptyDb:
    def test_run_drift_check_once_no_crash_on_missing_table(self, tmp_path):
        """Regression: the cron runner must handle a brand-new DB without
        crashing on `sqlite3.OperationalError: no such table`."""
        db = _make_empty_db(tmp_path / "journal.db")
        state_file = tmp_path / "drift_state.json"
        result = run_drift_check_once(
            str(db), state_file=str(state_file),
        )
        assert result["ok"] is False
        assert result["reason"] == "no_live_data"
