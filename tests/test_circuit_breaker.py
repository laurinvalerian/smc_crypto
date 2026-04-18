"""
Circuit Breaker tests — Phase 4.4 calendar-day period mode.

Protects the funded-account-compliance fix from regressions. The bot's
daily-loss window MUST reset at 00:00 UTC when period_mode="calendar_day"
(the default after Phase 4.4), matching funded-provider reset semantics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from risk.circuit_breaker import CircuitBreaker


class TestCalendarDayMode:
    """Default mode — funded-account compatible."""

    def test_day_start_returns_midnight_utc_same_day(self, utc_now_fixed):
        cb = CircuitBreaker()
        expected = datetime(2026, 4, 18, 0, 0, 0, tzinfo=timezone.utc)
        assert cb._day_start(utc_now_fixed) == expected

    def test_week_start_returns_monday_00_utc(self, utc_now_fixed):
        """2026-04-18 is Saturday. ISO week starts Monday 2026-04-13."""
        cb = CircuitBreaker()
        expected = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
        assert cb._week_start(utc_now_fixed) == expected

    def test_midnight_boundary_returns_same_midnight(self, midnight_utc):
        cb = CircuitBreaker()
        assert cb._day_start(midnight_utc) == midnight_utc

    def test_monday_week_start_returns_same_monday(self):
        cb = CircuitBreaker()
        monday = datetime(2026, 4, 13, 14, 30, 0, tzinfo=timezone.utc)
        expected = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
        assert cb._week_start(monday) == expected

    def test_sunday_week_start_returns_previous_monday(self):
        cb = CircuitBreaker()
        sunday = datetime(2026, 4, 19, 23, 59, 0, tzinfo=timezone.utc)
        # Sunday 2026-04-19 is weekday=6 → subtract 6 days → Monday 2026-04-13
        expected = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc)
        assert cb._week_start(sunday) == expected


class TestRolling24hMode:
    """Legacy mode — explicitly opt-in."""

    def test_day_start_is_minus_24h(self, utc_now_fixed):
        cb = CircuitBreaker(period_mode="rolling_24h")
        expected = utc_now_fixed - timedelta(hours=24)
        assert cb._day_start(utc_now_fixed) == expected

    def test_week_start_is_minus_7d(self, utc_now_fixed):
        cb = CircuitBreaker(period_mode="rolling_24h")
        expected = utc_now_fixed - timedelta(days=7)
        assert cb._week_start(utc_now_fixed) == expected


class TestModeValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="period_mode"):
            CircuitBreaker(period_mode="invalid")

    def test_default_mode_is_calendar_day(self):
        cb = CircuitBreaker()
        assert cb._period_mode == "calendar_day"


class TestFundedAccountScenario:
    """
    The scenario that motivated Phase 4.4.

    23:50 UTC loss should NOT extend the pause past the next 00:00 UTC
    funded reset — rolling_24h would, calendar_day doesn't.
    """

    def test_calendar_day_resets_at_midnight(self):
        cb = CircuitBreaker()
        late_night = datetime(2026, 4, 18, 23, 50, 0, tzinfo=timezone.utc)
        ds_today = cb._day_start(late_night)
        assert ds_today == datetime(2026, 4, 18, 0, 0, 0, tzinfo=timezone.utc)

        just_past_midnight = datetime(2026, 4, 19, 0, 5, 0, tzinfo=timezone.utc)
        ds_tomorrow = cb._day_start(just_past_midnight)
        assert ds_tomorrow == datetime(2026, 4, 19, 0, 0, 0, tzinfo=timezone.utc)
        assert ds_today != ds_tomorrow, "day_start must advance at midnight"

    def test_rolling_24h_does_not_reset_at_midnight(self):
        cb = CircuitBreaker(period_mode="rolling_24h")
        late_night = datetime(2026, 4, 18, 23, 50, 0, tzinfo=timezone.utc)
        just_past_midnight = datetime(2026, 4, 19, 0, 5, 0, tzinfo=timezone.utc)
        ds_late = cb._day_start(late_night)
        ds_after = cb._day_start(just_past_midnight)
        assert (ds_after - ds_late).total_seconds() == 15 * 60  # just 15 minutes
