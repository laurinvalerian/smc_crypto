"""
Shared pytest fixtures for the Crypto-Only Bot test suite.

Phase 3.5 of the Crypto-Only Refocus seeds the test infrastructure.
Prior to this commit the repo had no pytest tests — only ad-hoc scripts
in backtest/_test_*.py (archived). The suite is intentionally small at
first; it grows as more of live_multi_bot.py gets extracted into bot/
modules with clear interfaces.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make repo root importable when pytest is invoked from any subdirectory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def utc_now_fixed() -> datetime:
    """Deterministic 'now' for time-dependent tests. 2026-04-18 Saturday 10:30 UTC."""
    return datetime(2026, 4, 18, 10, 30, 45, tzinfo=timezone.utc)


@pytest.fixture
def midnight_utc() -> datetime:
    """Exact midnight UTC for boundary-condition tests."""
    return datetime(2026, 4, 18, 0, 0, 0, tzinfo=timezone.utc)
