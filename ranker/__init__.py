"""
Ranker Package
==============
Cross-asset opportunity ranking and capital allocation.

Usage:
    from ranker import UniverseScanner, OpportunityRanker, CapitalAllocator
"""
from ranker.universe_scanner import UniverseScanner, ScanResult, UniverseState
from ranker.opportunity_ranker import OpportunityRanker, RankedOpportunity
from ranker.capital_allocator import CapitalAllocator, AllocationDecision, PortfolioState

__all__ = [
    "UniverseScanner",
    "ScanResult",
    "UniverseState",
    "OpportunityRanker",
    "RankedOpportunity",
    "CapitalAllocator",
    "AllocationDecision",
    "PortfolioState",
]
