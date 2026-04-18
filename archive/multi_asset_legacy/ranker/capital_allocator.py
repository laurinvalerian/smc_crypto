"""
Capital Allocator
=================
Manages position sizing, portfolio limits, and correlation-based
filtering for the multi-asset trading system.

Key Rules:
- Max 5 simultaneous positions across all asset classes
- Max 3 positions per single asset class
- No two positions with correlation > 0.7
- Risk 1-2% per trade (tier-dependent)
- Max portfolio heat: 6% (sum of all position risks)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ranker.opportunity_ranker import RankedOpportunity

logger = logging.getLogger(__name__)

# ── Portfolio Limits ────────────────────────────────────────────────

MAX_TOTAL_POSITIONS = 9  # raised from 5 to match per-style limit (3×3=9); allocator not wired yet
MAX_PER_ASSET_CLASS = 3
MAX_PORTFOLIO_HEAT_PCT = 0.06     # 6% total risk
MAX_CORRELATION = 0.70             # No two positions with r > 0.7
CORRELATION_LOOKBACK_BARS = 100    # 20-day rolling on 1H (20*5 bars)

# Leverage limits per asset class
LEVERAGE_LIMITS: dict[str, int] = {
    "crypto": 20,
    "forex": 30,
    "stocks": 4,
    "commodities": 20,
}

# Risk per trade (base) per tier
TIER_RISK_PCT: dict[str, dict[str, float]] = {
    "AAA++": {"base": 0.010, "max": 0.015},   # 1.0% - 1.5%
    "AAA+":  {"base": 0.005, "max": 0.010},   # 0.5% - 1.0%
}


@dataclass
class AllocationDecision:
    """Decision about whether and how to allocate capital to an opportunity."""

    symbol: str
    asset_class: str = ""
    approved: bool = False
    reject_reason: str = ""

    # Position sizing
    risk_pct: float = 0.0          # % of account to risk
    leverage: int = 1
    position_size_usd: float = 0.0
    qty: float = 0.0

    # Context
    opportunity_score: float = 0.0
    tier: str = ""
    current_portfolio_heat: float = 0.0


@dataclass
class PortfolioState:
    """Current state of the portfolio for allocation decisions."""

    # Active positions (symbol → {direction, risk_pct, entry_price, ...})
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Account balance
    equity: float = 100_000.0

    # Return series for correlation (symbol → list of returns)
    return_series: dict[str, list[float]] = field(default_factory=dict)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def positions_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for info in self.positions.values():
            ac = info.get("asset_class", "crypto")
            counts[ac] = counts.get(ac, 0) + 1
        return counts

    @property
    def portfolio_heat(self) -> float:
        """Sum of all position risk percentages."""
        return sum(
            info.get("risk_pct", 0.0) for info in self.positions.values()
        )


class CapitalAllocator:
    """
    Decides which opportunities get capital and how much.

    Process:
    1. Check portfolio-level constraints (max positions, heat)
    2. Check asset-class constraints (max per class)
    3. Check correlation with existing positions
    4. Compute position size based on tier and risk budget
    """

    def __init__(
        self,
        max_positions: int = MAX_TOTAL_POSITIONS,
        max_per_class: int = MAX_PER_ASSET_CLASS,
        max_heat: float = MAX_PORTFOLIO_HEAT_PCT,
        max_correlation: float = MAX_CORRELATION,
    ) -> None:
        self._max_positions = max_positions
        self._max_per_class = max_per_class
        self._max_heat = max_heat
        self._max_correlation = max_correlation
        self._portfolio = PortfolioState()

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    def update_portfolio(
        self,
        positions: dict[str, dict[str, Any]],
        equity: float,
        return_series: dict[str, list[float]] | None = None,
    ) -> None:
        """Update the portfolio state with current positions and equity."""
        self._portfolio.positions = positions
        self._portfolio.equity = equity
        if return_series:
            self._portfolio.return_series = return_series

    def _compute_correlation(
        self,
        returns_a: list[float],
        returns_b: list[float],
    ) -> float:
        """Compute Pearson correlation between two return series."""
        min_len = min(len(returns_a), len(returns_b))
        if min_len < 20:
            return 0.0  # Not enough data

        a = np.array(returns_a[-min_len:])
        b = np.array(returns_b[-min_len:])

        std_a = np.std(a)
        std_b = np.std(b)
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0

        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _check_correlation(
        self,
        symbol: str,
        candidate_returns: list[float] | None,
    ) -> tuple[bool, str]:
        """Check if the candidate is too correlated with existing positions."""
        if candidate_returns is None or len(candidate_returns) < 20:
            return True, ""  # Not enough data — allow

        for pos_symbol, pos_info in self._portfolio.positions.items():
            pos_returns = self._portfolio.return_series.get(pos_symbol)
            if pos_returns is None or len(pos_returns) < 20:
                continue

            corr = self._compute_correlation(candidate_returns, pos_returns)
            if abs(corr) > self._max_correlation:
                return False, (
                    f"Correlation {corr:.2f} with {pos_symbol} exceeds "
                    f"limit {self._max_correlation}"
                )

        return True, ""

    def _compute_risk_and_size(
        self,
        opportunity: RankedOpportunity,
        remaining_heat: float,
    ) -> tuple[float, int]:
        """Compute risk % and leverage for this opportunity."""
        tier = opportunity.scan.tier or "AAA+"
        tier_risk = TIER_RISK_PCT.get(tier, TIER_RISK_PCT["AAA+"])

        base_risk = tier_risk["base"]
        max_risk = tier_risk["max"]

        # Scale risk by opportunity score (higher score → closer to max_risk)
        score = opportunity.opportunity_score
        risk_pct = base_risk + (max_risk - base_risk) * min(score, 1.0)

        # Cap by remaining portfolio heat budget
        risk_pct = min(risk_pct, remaining_heat)

        # Leverage based on asset class
        asset_class = opportunity.scan.asset_class
        max_lev = LEVERAGE_LIMITS.get(asset_class, 1)

        return risk_pct, max_lev

    def allocate(
        self,
        opportunities: list[RankedOpportunity],
    ) -> list[AllocationDecision]:
        """
        Decide which opportunities to trade and with how much capital.

        Takes a ranked list of opportunities and returns allocation
        decisions, respecting all portfolio constraints.

        Arguments:
            opportunities: Sorted list (best first) from OpportunityRanker.

        Returns:
            List of AllocationDecision (approved ones first, then rejected).
        """
        decisions: list[AllocationDecision] = []
        approved_count = 0
        approved_by_class: dict[str, int] = dict(self._portfolio.positions_by_class)
        current_heat = self._portfolio.portfolio_heat

        for opp in opportunities:
            symbol = opp.scan.symbol
            asset_class = opp.scan.asset_class

            decision = AllocationDecision(
                symbol=symbol,
                asset_class=asset_class,
                opportunity_score=opp.opportunity_score,
                tier=opp.scan.tier,
                current_portfolio_heat=current_heat,
            )

            # ── Check 1: Already in position ────────────────────────
            if symbol in self._portfolio.positions:
                decision.reject_reason = "Already in position"
                decisions.append(decision)
                continue

            # ── Check 2: Max total positions ────────────────────────
            total_positions = self._portfolio.position_count + approved_count
            if total_positions >= self._max_positions:
                decision.reject_reason = (
                    f"Max positions reached ({self._max_positions})"
                )
                decisions.append(decision)
                continue

            # ── Check 3: Max per asset class ────────────────────────
            class_count = approved_by_class.get(asset_class, 0)
            if class_count >= self._max_per_class:
                decision.reject_reason = (
                    f"Max {asset_class} positions reached ({self._max_per_class})"
                )
                decisions.append(decision)
                continue

            # ── Check 4: Portfolio heat budget ──────────────────────
            remaining_heat = self._max_heat - current_heat
            if remaining_heat <= 0.001:
                decision.reject_reason = (
                    f"Portfolio heat limit reached ({current_heat:.1%} / "
                    f"{self._max_heat:.1%})"
                )
                decisions.append(decision)
                continue

            # ── Check 5: Correlation with existing positions ────────
            candidate_returns = self._portfolio.return_series.get(symbol)
            corr_ok, corr_reason = self._check_correlation(
                symbol, candidate_returns,
            )
            if not corr_ok:
                decision.reject_reason = corr_reason
                decisions.append(decision)
                continue

            # ── All checks passed — compute allocation ──────────────
            risk_pct, leverage = self._compute_risk_and_size(
                opp, remaining_heat,
            )

            if risk_pct < 0.001:
                decision.reject_reason = "Risk too small after constraints"
                decisions.append(decision)
                continue

            # Position size in USD
            position_size_usd = self._portfolio.equity * risk_pct * leverage

            decision.approved = True
            decision.risk_pct = risk_pct
            decision.leverage = leverage
            decision.position_size_usd = position_size_usd

            # Update running counters
            approved_count += 1
            approved_by_class[asset_class] = class_count + 1
            current_heat += risk_pct

            decisions.append(decision)

            logger.info(
                "APPROVED: %s (%s) | score=%.3f tier=%s risk=%.2f%% lev=%dx size=$%.0f heat=%.2f%%",
                symbol, asset_class, opp.opportunity_score,
                opp.scan.tier, risk_pct * 100, leverage,
                position_size_usd, current_heat * 100,
            )

        # Summary
        approved = [d for d in decisions if d.approved]
        rejected = [d for d in decisions if not d.approved]
        logger.info(
            "Allocation: %d approved, %d rejected | heat: %.2f%% / %.2f%%",
            len(approved), len(rejected),
            current_heat * 100, self._max_heat * 100,
        )

        return decisions
