"""
Circuit Breaker
===============
Portfolio-level risk management that halts or reduces trading
when drawdown limits are breached.

Rules:
- Daily loss >= 3%  → Stop ALL trading for 24h
- Weekly loss >= 5% → Halve all position sizes
- Asset-class drawdown >= 2% → Pause that asset class for 12h
- Portfolio heat > 6% → No new positions until heat decreases
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────

DAILY_LOSS_LIMIT_PCT = 0.03       # -3% daily → full stop 24h (funded: -5% limit, 2% buffer)
WEEKLY_LOSS_LIMIT_PCT = 0.05      # -5% weekly → halve sizes
ASSET_CLASS_DD_LIMIT_PCT = 0.02   # -2% per asset class → pause 12h
ALLTIME_DD_LIMIT_PCT = 0.08       # -8% all-time DD → STOP ALL (funded: -10% limit, 2% buffer)
MAX_PORTFOLIO_HEAT_PCT = 1.00     # Effectively disabled — dynamic risk budget handles this now
DAILY_PAUSE_HOURS = 24
ASSET_CLASS_PAUSE_HOURS = 12


@dataclass
class PnLRecord:
    """Single PnL entry for tracking."""
    timestamp: datetime
    pnl_pct: float
    asset_class: str = ""
    symbol: str = ""


@dataclass
class CircuitBreakerState:
    """Current state of all circuit breakers."""

    # Trading pauses
    all_trading_paused_until: datetime | None = None
    asset_class_paused_until: dict[str, datetime] = field(default_factory=dict)

    # Size reduction
    size_reduction_factor: float = 1.0  # 1.0 = normal, 0.5 = halved

    # Metrics
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    asset_class_pnl_pct: dict[str, float] = field(default_factory=dict)
    portfolio_heat_pct: float = 0.0

    # All-time drawdown tracking
    alltime_pnl_pct: float = 0.0
    alltime_peak_pnl_pct: float = 0.0
    alltime_dd_pct: float = 0.0  # Current drawdown from peak (negative)
    alltime_breaker_active: bool = False

    # Active breaker flags
    daily_breaker_active: bool = False
    weekly_breaker_active: bool = False
    asset_class_breakers: dict[str, bool] = field(default_factory=dict)
    heat_breaker_active: bool = False


class CircuitBreaker:
    """
    Portfolio-level circuit breaker system.

    Tracks realized PnL and open risk, triggers protective actions
    when limits are breached.
    """

    def __init__(
        self,
        daily_loss_limit: float = DAILY_LOSS_LIMIT_PCT,
        weekly_loss_limit: float = WEEKLY_LOSS_LIMIT_PCT,
        asset_class_dd_limit: float = ASSET_CLASS_DD_LIMIT_PCT,
        alltime_dd_limit: float = ALLTIME_DD_LIMIT_PCT,
        max_portfolio_heat: float = MAX_PORTFOLIO_HEAT_PCT,
    ) -> None:
        self._daily_limit = daily_loss_limit
        self._weekly_limit = weekly_loss_limit
        self._asset_dd_limit = asset_class_dd_limit
        self._alltime_dd_limit = alltime_dd_limit
        self._max_heat = max_portfolio_heat

        # PnL history (rolling)
        self._pnl_history: list[PnLRecord] = []
        self._max_history = 10_000

        # State
        self._state = CircuitBreakerState()

        # Dedup log state — only log on state CHANGES, not every check()
        self._last_log_state: dict[str, Any] = {}

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    def record_trade_pnl(
        self,
        pnl_pct: float,
        asset_class: str = "crypto",
        symbol: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        """Record a closed trade's PnL for circuit breaker tracking."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self._pnl_history.append(PnLRecord(
            timestamp=timestamp,
            pnl_pct=pnl_pct,
            asset_class=asset_class,
            symbol=symbol,
        ))

        # Track all-time PnL and drawdown from peak
        self._state.alltime_pnl_pct += pnl_pct
        if self._state.alltime_pnl_pct > self._state.alltime_peak_pnl_pct:
            self._state.alltime_peak_pnl_pct = self._state.alltime_pnl_pct
        self._state.alltime_dd_pct = (
            self._state.alltime_pnl_pct - self._state.alltime_peak_pnl_pct
        )

        # Trim history
        if len(self._pnl_history) > self._max_history:
            self._pnl_history = self._pnl_history[-self._max_history:]

    def update_portfolio_heat(self, heat_pct: float) -> None:
        """Update current portfolio heat (sum of all position risk %)."""
        self._state.portfolio_heat_pct = heat_pct

    def _compute_period_pnl(
        self,
        since: datetime,
        asset_class: str | None = None,
    ) -> float:
        """Sum PnL % for trades since a given time, optionally filtered by asset class."""
        total = 0.0
        for record in self._pnl_history:
            if record.timestamp < since:
                continue
            if asset_class and record.asset_class != asset_class:
                continue
            total += record.pnl_pct
        return total

    def check(self, utc_now: datetime | None = None) -> CircuitBreakerState:
        """
        Evaluate all circuit breaker conditions.

        Call this before each potential trade entry. Returns the
        current CircuitBreakerState with all flags and limits.
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        # ── Daily PnL ──────────────────────────────────────────────
        day_start = utc_now - timedelta(hours=24)
        self._state.daily_pnl_pct = self._compute_period_pnl(day_start)

        # If already paused, only check expiry — don't re-trigger
        if self._state.all_trading_paused_until is not None:
            if utc_now >= self._state.all_trading_paused_until:
                self._state.daily_breaker_active = False
                self._state.all_trading_paused_until = None
                if self._last_log_state.get("daily") != "resumed":
                    logger.info("CIRCUIT BREAKER: Daily pause expired. Trading resumed.")
                    self._last_log_state["daily"] = "resumed"
        elif self._state.daily_pnl_pct <= -self._daily_limit:
            if self._last_log_state.get("daily") != "paused":
                logger.warning(
                    "CIRCUIT BREAKER: Daily loss %.2f%% exceeds -%.1f%% limit. "
                    "Pausing ALL trading for %dh.",
                    self._state.daily_pnl_pct * 100,
                    self._daily_limit * 100,
                    DAILY_PAUSE_HOURS,
                )
                self._last_log_state["daily"] = "paused"
            self._state.daily_breaker_active = True
            self._state.all_trading_paused_until = utc_now + timedelta(
                hours=DAILY_PAUSE_HOURS
            )
        else:
            # Normal state — clear log dedup so next trigger logs again
            self._last_log_state.pop("daily", None)

        # ── Weekly PnL ─────────────────────────────────────────────
        week_start = utc_now - timedelta(days=7)
        self._state.weekly_pnl_pct = self._compute_period_pnl(week_start)

        if self._state.weekly_pnl_pct <= -self._weekly_limit:
            if not self._state.weekly_breaker_active:
                logger.warning(
                    "CIRCUIT BREAKER: Weekly loss %.2f%% exceeds -%.1f%% limit. "
                    "Halving position sizes.",
                    self._state.weekly_pnl_pct * 100,
                    self._weekly_limit * 100,
                )
                self._state.weekly_breaker_active = True
                self._state.size_reduction_factor = 0.5
        else:
            if self._state.weekly_breaker_active:
                self._state.weekly_breaker_active = False
                self._state.size_reduction_factor = 1.0
                logger.info("CIRCUIT BREAKER: Weekly loss recovered. Normal sizing.")

        # ── Per-Asset-Class Drawdown ───────────────────────────────
        for asset_class in ("crypto", "forex", "stocks", "commodities"):
            class_pnl = self._compute_period_pnl(day_start, asset_class)
            self._state.asset_class_pnl_pct[asset_class] = class_pnl

            paused_until = self._state.asset_class_paused_until.get(asset_class)
            log_key = f"class_{asset_class}"

            # If already paused, only check expiry
            if paused_until is not None:
                if utc_now >= paused_until:
                    self._state.asset_class_breakers[asset_class] = False
                    del self._state.asset_class_paused_until[asset_class]
                    if self._last_log_state.get(log_key) != "resumed":
                        logger.info(
                            "CIRCUIT BREAKER: %s pause expired. Trading resumed.",
                            asset_class,
                        )
                        self._last_log_state[log_key] = "resumed"
            elif class_pnl <= -self._asset_dd_limit:
                if self._last_log_state.get(log_key) != "paused":
                    logger.warning(
                        "CIRCUIT BREAKER: %s loss %.2f%% exceeds -%.1f%%. "
                        "Pausing %s for %dh.",
                        asset_class, class_pnl * 100,
                        self._asset_dd_limit * 100,
                        asset_class, ASSET_CLASS_PAUSE_HOURS,
                    )
                    self._last_log_state[log_key] = "paused"
                self._state.asset_class_breakers[asset_class] = True
                self._state.asset_class_paused_until[asset_class] = (
                    utc_now + timedelta(hours=ASSET_CLASS_PAUSE_HOURS)
                )
            else:
                # Normal state — clear log dedup so next trigger logs again
                self._last_log_state.pop(log_key, None)

        # ── All-Time Drawdown (Funded Account Protection) ─────────
        if self._state.alltime_dd_pct <= -self._alltime_dd_limit:
            if not self._state.alltime_breaker_active:
                logger.critical(
                    "CIRCUIT BREAKER: All-time DD %.2f%% exceeds -%.1f%% limit. "
                    "STOPPING ALL TRADING PERMANENTLY until manual reset.",
                    self._state.alltime_dd_pct * 100,
                    self._alltime_dd_limit * 100,
                )
            self._state.alltime_breaker_active = True

        # ── Portfolio Heat ─────────────────────────────────────────
        self._state.heat_breaker_active = (
            self._state.portfolio_heat_pct >= self._max_heat
        )

        # ── Risk Budget Warning (throttled: max once per 5 min) ────
        budget = self.remaining_risk_budget()
        if budget < 0.01:  # < 1% remaining
            if self._last_log_state.get("_budget_warn_time") is None or \
               (utc_now - self._last_log_state["_budget_warn_time"]).total_seconds() > 300:
                logger.warning(
                    "RISK BUDGET LOW: %.2f%% remaining (daily=%.2f%% weekly=%.2f%% "
                    "alltime=%.2f%% heat=%.2f%%)",
                    budget * 100,
                    self._state.daily_pnl_pct * 100,
                    self._state.weekly_pnl_pct * 100,
                    self._state.alltime_dd_pct * 100,
                    self._state.portfolio_heat_pct * 100,
                )
                self._last_log_state["_budget_warn_time"] = utc_now

        return self._state

    def remaining_risk_budget(self) -> float:
        """Return remaining risk budget as a fraction (0.0 to 1.0).

        Dynamically computed from proximity to DD limits:
        - Daily DD: limit minus abs(daily_loss) minus current_heat
        - Weekly DD: limit minus abs(weekly_loss) minus current_heat
        - Alltime DD: limit minus abs(alltime_dd) minus current_heat
        - Returns the MINIMUM of all budgets (most restrictive wins)
        - 0.0 means no more risk allowed
        """
        # Daily budget: how much more can we lose today?
        daily_used = abs(min(self._state.daily_pnl_pct, 0.0))
        daily_remaining = max(
            0.0, self._daily_limit - daily_used - self._state.portfolio_heat_pct
        )

        # Weekly budget
        weekly_used = abs(min(self._state.weekly_pnl_pct, 0.0))
        weekly_remaining = max(
            0.0, self._weekly_limit - weekly_used - self._state.portfolio_heat_pct
        )

        # Alltime budget
        alltime_used = abs(min(self._state.alltime_dd_pct, 0.0))
        alltime_remaining = max(
            0.0, self._alltime_dd_limit - alltime_used - self._state.portfolio_heat_pct
        )

        # Return the most restrictive
        return min(daily_remaining, weekly_remaining, alltime_remaining)

    def risk_budget_allows(self, new_risk_pct: float) -> tuple[bool, str]:
        """Check if opening a trade with given risk % is within budget.

        Returns (allowed, reason).
        """
        budget = self.remaining_risk_budget()

        if budget <= 0.001:  # < 0.1% remaining = emergency stop
            return False, f"RISK BUDGET EXHAUSTED: budget={budget * 100:.2f}%"

        if new_risk_pct > budget:
            return False, (
                f"RISK BUDGET: trade needs {new_risk_pct * 100:.2f}% "
                f"but only {budget * 100:.2f}% remaining"
            )

        return True, ""

    def can_trade(
        self,
        asset_class: str = "crypto",
        utc_now: datetime | None = None,
    ) -> tuple[bool, str]:
        """
        Quick check: is trading allowed right now for this asset class?

        Returns (allowed, reason).
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        state = self.check(utc_now)

        # All-time DD breaker — permanent stop
        if state.alltime_breaker_active:
            return False, (
                f"ALL-TIME DD BREAKER: DD {state.alltime_dd_pct:.2%} exceeds "
                f"-{self._alltime_dd_limit:.0%} limit. Trading permanently stopped."
            )

        # Full trading pause
        if state.all_trading_paused_until and utc_now < state.all_trading_paused_until:
            remaining = state.all_trading_paused_until - utc_now
            return False, (
                f"All trading paused (daily loss {state.daily_pnl_pct:.2%}). "
                f"Resumes in {remaining.total_seconds() / 3600:.1f}h"
            )

        # Asset class pause
        class_paused = state.asset_class_paused_until.get(asset_class)
        if class_paused and utc_now < class_paused:
            remaining = class_paused - utc_now
            return False, (
                f"{asset_class} paused (class loss "
                f"{state.asset_class_pnl_pct.get(asset_class, 0):.2%}). "
                f"Resumes in {remaining.total_seconds() / 3600:.1f}h"
            )

        # Portfolio heat
        if state.heat_breaker_active:
            return False, (
                f"Portfolio heat {state.portfolio_heat_pct:.2%} exceeds "
                f"limit {self._max_heat:.2%}"
            )

        return True, ""

    def get_size_factor(self) -> float:
        """
        Returns the position size multiplier.

        1.0 = normal, 0.5 = halved (weekly breaker active).
        """
        return self._state.size_reduction_factor
