"""
Opportunity Ranker
==================
Ranks all scanned instruments by composite opportunity score.
Uses Z-Score normalisation per asset class so that crypto, forex,
stocks, and commodities compete on equal footing.

The ranker selects the top-N opportunities and passes them to
the Capital Allocator for position sizing and execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ranker.universe_scanner import ScanResult, UniverseState

logger = logging.getLogger(__name__)

# ── Scoring Weights ─────────────────────────────────────────────────
# These weights determine how each component contributes to the
# final opportunity score. They should sum to 1.0.

DEFAULT_WEIGHTS: dict[str, float] = {
    "alignment": 0.35,
    "volume": 0.20,
    "trend_strength": 0.15,
    "session": 0.10,
    "zone_quality": 0.10,
    "rr_score": 0.10,
}


@dataclass
class RankedOpportunity:
    """A scan result enriched with ranking information."""

    scan: ScanResult
    opportunity_score: float = 0.0
    z_scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0  # 1 = best


class OpportunityRanker:
    """
    Ranks instruments by composite opportunity score.

    Process:
    1. Group scan results by asset class
    2. Z-Score normalise each component within its asset class
    3. Compute weighted composite score
    4. Rank across all asset classes
    5. Return top-N opportunities that meet minimum thresholds
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_opportunity_score: float = 0.5,
        max_opportunities: int = 10,
    ) -> None:
        self._weights = weights or DEFAULT_WEIGHTS.copy()
        self._min_score = min_opportunity_score
        self._max_opportunities = max_opportunities

        # Historical scores for rolling Z-Score (optional, for stability)
        self._score_history: dict[str, list[float]] = {}
        self._history_max_len = 100

    def _z_score_normalize(
        self,
        values: list[float],
    ) -> list[float]:
        """
        Z-Score normalise a list of values, then clip to [0, 1].

        If all values are identical (std=0), return 0.5 for all.
        """
        if not values:
            return []

        arr = np.array(values, dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr)

        if std < 1e-10:
            return [0.5] * len(values)

        z = (arr - mean) / std
        # Convert Z-scores to 0-1 range using sigmoid-like mapping
        # Z of -2 → ~0.05, Z of 0 → 0.5, Z of +2 → ~0.95
        normalized = 1.0 / (1.0 + np.exp(-z))
        return normalized.tolist()

    def _compute_raw_scores(self, scan: ScanResult) -> dict[str, float]:
        """Extract raw component scores from a scan result."""
        # RR score: normalise RR ratio (5.0 = max target)
        rr_score = min(scan.rr_ratio / 5.0, 1.0) if scan.rr_ratio > 0 else 0.0

        return {
            "alignment": scan.alignment_score,
            "volume": scan.volume_score,
            "trend_strength": scan.trend_strength_score,
            "session": scan.session_score,
            "zone_quality": scan.zone_quality_score,
            "rr_score": rr_score,
        }

    def rank(self, state: UniverseState) -> list[RankedOpportunity]:
        """
        Rank all instruments from the latest scan cycle.

        Returns sorted list of RankedOpportunity (best first),
        filtered to only include those above min_opportunity_score.
        """
        if not state.results:
            return []

        # Filter to only market-open instruments with some activity
        active = [
            r for r in state.results
            if r.market_open and (
                r.volume_score > 0
                or r.trend_strength_score > 0
                or r.session_score > 0
            )
        ]

        if not active:
            return []

        # ── Step 1: Group by asset class ────────────────────────────
        by_class: dict[str, list[ScanResult]] = {}
        for r in active:
            by_class.setdefault(r.asset_class, []).append(r)

        # ── Step 2: Z-Score normalise per asset class ───────────────
        # Store normalised scores keyed by (symbol, component)
        norm_scores: dict[str, dict[str, float]] = {}

        for asset_class, scans in by_class.items():
            if len(scans) < 2:
                # Not enough data for Z-Score — use raw scores
                for s in scans:
                    norm_scores[s.symbol] = self._compute_raw_scores(s)
                continue

            # Collect raw scores per component
            raw_by_component: dict[str, list[float]] = {}
            for component in self._weights:
                raw_by_component[component] = []

            for s in scans:
                raw = self._compute_raw_scores(s)
                for component in self._weights:
                    raw_by_component[component].append(raw.get(component, 0.0))

            # Normalise each component
            norm_by_component: dict[str, list[float]] = {}
            for component, values in raw_by_component.items():
                norm_by_component[component] = self._z_score_normalize(values)

            # Map back to symbols
            for idx, s in enumerate(scans):
                norm_scores[s.symbol] = {
                    component: norm_by_component[component][idx]
                    for component in self._weights
                }

        # ── Step 3: Compute weighted composite score ────────────────
        ranked: list[RankedOpportunity] = []

        for scan in active:
            scores = norm_scores.get(scan.symbol, {})
            composite = sum(
                scores.get(comp, 0.0) * weight
                for comp, weight in self._weights.items()
            )

            # Bonus for instruments with active trade signals
            if scan.has_signal:
                composite *= 1.2  # 20% bonus for having a signal

            composite = min(composite, 1.0)

            ranked.append(RankedOpportunity(
                scan=scan,
                opportunity_score=composite,
                z_scores=scores,
            ))

        # ── Step 4: Sort by composite score (descending) ────────────
        ranked.sort(key=lambda r: r.opportunity_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(ranked):
            r.rank = i + 1
            r.scan.opportunity_score = r.opportunity_score

        # ── Step 5: Filter to minimum score ─────────────────────────
        qualified = [
            r for r in ranked
            if r.opportunity_score >= self._min_score
        ]

        # Limit to max opportunities
        top = qualified[: self._max_opportunities]

        logger.info(
            "Ranked %d instruments: %d qualified (>= %.2f), returning top %d",
            len(ranked), len(qualified), self._min_score, len(top),
        )

        if top:
            best = top[0]
            logger.info(
                "Best opportunity: %s (%s) score=%.3f | vol=%.2f trend=%.2f sess=%.2f",
                best.scan.symbol,
                best.scan.asset_class,
                best.opportunity_score,
                best.z_scores.get("volume", 0),
                best.z_scores.get("trend_strength", 0),
                best.z_scores.get("session", 0),
            )

        return top
