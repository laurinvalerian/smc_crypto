"""
Universe Scanner
================
Scans all instruments across all connected exchanges, computes
alignment scores, and feeds results to the Opportunity Ranker.

The scanner replaces the fixed 100-bot model with a lightweight
scanning pool that evaluates ~200 instruments per cycle.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from exchanges.base import ExchangeAdapter
from exchanges.models import InstrumentMeta

logger = logging.getLogger(__name__)

# ── Scan Result ─────────────────────────────────────────────────────


@dataclass
class ScanResult:
    """Result of scanning a single instrument for trade opportunities."""

    symbol: str
    asset_class: str = "crypto"
    exchange_id: str = ""

    # Scores (all 0.0-1.0)
    alignment_score: float = 0.0
    volume_score: float = 0.0
    trend_strength_score: float = 0.0
    session_score: float = 0.0
    zone_quality_score: float = 0.0
    momentum_score: float = 0.0
    rr_ratio: float = 0.0

    # Composite opportunity score (computed by ranker)
    opportunity_score: float = 0.0

    # Signal details
    direction: str = ""              # "long" | "short" | "" (no signal)
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    tier: str = ""                   # "AAA++" | "AAA+" | ""

    # Component flags (for tier classification)
    components: dict[str, bool] = field(default_factory=dict)

    # Metadata
    scan_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    market_open: bool = True

    @property
    def has_signal(self) -> bool:
        return self.direction != "" and self.alignment_score > 0


@dataclass
class UniverseState:
    """Aggregated state of all scanned instruments."""

    results: list[ScanResult] = field(default_factory=list)
    scan_cycle: int = 0
    scan_duration_sec: float = 0.0
    last_scan_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def signals(self) -> list[ScanResult]:
        """Return only results with active trade signals."""
        return [r for r in self.results if r.has_signal]

    @property
    def by_asset_class(self) -> dict[str, list[ScanResult]]:
        """Group results by asset class."""
        groups: dict[str, list[ScanResult]] = {}
        for r in self.results:
            groups.setdefault(r.asset_class, []).append(r)
        return groups


class UniverseScanner:
    """
    Scans all instruments across multiple exchange adapters.

    Architecture:
    - Holds references to all connected ExchangeAdapters
    - Each scan cycle iterates through all instruments
    - Computes lightweight pre-filter scores (ATR, volume, session)
    - Full alignment scoring only for instruments passing pre-filter
    - Results are collected into UniverseState for the Ranker
    """

    def __init__(
        self,
        adapters: list[ExchangeAdapter],
        scan_interval_sec: float = 30.0,
        pre_filter_atr_pct: float = 0.004,  # Min daily ATR to consider
    ) -> None:
        self._adapters = adapters
        self._scan_interval = scan_interval_sec
        self._pre_filter_atr_pct = pre_filter_atr_pct

        # All known instruments (populated after load_all_markets)
        self._instruments: dict[str, tuple[InstrumentMeta, ExchangeAdapter]] = {}

        # Cached OHLCV data per instrument (symbol → {tf: DataFrame})
        self._ohlcv_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._cache_expiry: dict[str, float] = {}  # symbol → timestamp
        self._cache_ttl_sec = 300.0  # 5 min cache TTL

        self._state = UniverseState()
        self._scan_cycle = 0

    @property
    def state(self) -> UniverseState:
        return self._state

    @property
    def instrument_count(self) -> int:
        return len(self._instruments)

    async def load_all_markets(self) -> int:
        """Load markets from all adapters and build the unified instrument universe."""
        total = 0
        for adapter in self._adapters:
            try:
                instruments = await adapter.load_markets()
                for symbol, meta in instruments.items():
                    self._instruments[symbol] = (meta, adapter)
                total += len(instruments)
                logger.info(
                    "Loaded %d instruments from %s (%s)",
                    len(instruments), adapter.exchange_id, adapter.asset_class,
                )
            except Exception as exc:
                logger.error(
                    "Failed to load markets from %s: %s",
                    adapter.exchange_id, exc,
                )
        logger.info("Universe scanner: %d total instruments loaded", total)
        return total

    async def _fetch_ohlcv_cached(
        self,
        symbol: str,
        adapter: ExchangeAdapter,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV with caching to avoid hammering APIs."""
        cache_key = f"{symbol}:{timeframe}"
        now = time.time()

        if (
            cache_key in self._ohlcv_cache
            and self._cache_expiry.get(cache_key, 0) > now
        ):
            cached = self._ohlcv_cache.get(cache_key)
            if cached is not None:
                return cached.get(timeframe)

        try:
            raw = await adapter.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw:
                return None

            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            if symbol not in self._ohlcv_cache:
                self._ohlcv_cache[symbol] = {}
            self._ohlcv_cache[symbol][timeframe] = df
            self._cache_expiry[cache_key] = now + self._cache_ttl_sec

            return df
        except Exception as exc:
            logger.debug("OHLCV fetch failed %s %s: %s", symbol, timeframe, exc)
            return None

    def _compute_atr_pct(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR as percentage of price."""
        if df is None or len(df) < period + 1:
            return 0.0

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        if len(tr) < period:
            return 0.0

        atr = np.mean(tr[-period:])
        current_price = closes[-1]
        if current_price <= 0:
            return 0.0

        return atr / current_price

    async def _scan_instrument(
        self,
        symbol: str,
        meta: InstrumentMeta,
        adapter: ExchangeAdapter,
    ) -> ScanResult:
        """
        Scan a single instrument and compute scores.

        This is a lightweight scan — full alignment scoring happens
        only if the pre-filter passes.
        """
        result = ScanResult(
            symbol=symbol,
            asset_class=meta.asset_class,
            exchange_id=meta.exchange_id,
        )

        # Check market hours
        utc_now = datetime.now(timezone.utc)
        result.market_open = adapter.is_market_open(symbol, utc_now)
        if not result.market_open:
            return result

        # Fetch 5m data for pre-filter
        df_5m = await self._fetch_ohlcv_cached(symbol, adapter, "5m", 200)
        if df_5m is None or len(df_5m) < 30:
            return result

        # Pre-filter: ATR check
        atr_pct = self._compute_atr_pct(df_5m)
        if atr_pct < self._pre_filter_atr_pct:
            return result

        # Pre-filter: Volume check (current vs 20-bar avg)
        volumes = df_5m["volume"].values
        if len(volumes) >= 20:
            avg_vol = np.mean(volumes[-20:])
            current_vol = volumes[-1]
            if avg_vol > 0 and current_vol / avg_vol < 0.5:
                return result  # Volume too thin

        # ── Compute scores ──────────────────────────────────────────
        # These are lightweight approximations for ranking.
        # Full alignment scoring happens in PaperBot._multi_tf_alignment_score()
        # when the Ranker selects this instrument for trading.

        closes = df_5m["close"].values
        highs = df_5m["high"].values
        lows = df_5m["low"].values

        # Volume score (relative volume)
        if len(volumes) >= 100:
            avg_100 = np.mean(volumes[-100:])
            result.volume_score = min(volumes[-1] / max(avg_100, 1e-10), 2.0) / 2.0
        elif len(volumes) >= 20:
            avg_20 = np.mean(volumes[-20:])
            result.volume_score = min(volumes[-1] / max(avg_20, 1e-10), 2.0) / 2.0

        # Trend strength (simplified: EMA20 vs EMA50 slope)
        if len(closes) >= 50:
            ema20 = pd.Series(closes).ewm(span=20).mean().values
            ema50 = pd.Series(closes).ewm(span=50).mean().values
            trend_aligned = ema20[-1] > ema50[-1]
            ema_spread = abs(ema20[-1] - ema50[-1]) / closes[-1]
            result.trend_strength_score = min(ema_spread * 100, 1.0) if trend_aligned else 0.0

        # Session score
        from filters.session_filter import compute_session_score
        result.session_score = compute_session_score(meta.asset_class, utc_now)

        # Momentum (simplified RSI check)
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0

            # Momentum score: best when RSI is 40-60 (room to move)
            if 40 <= rsi <= 60:
                result.momentum_score = 1.0
            elif 30 <= rsi <= 70:
                result.momentum_score = 0.6
            else:
                result.momentum_score = 0.3

        # ATR-based rough RR estimate
        atr_val = atr_pct * closes[-1]
        if atr_val > 0:
            result.rr_ratio = min((3.0 * atr_val) / (1.5 * atr_val), 5.0)

        return result

    async def scan_all(self) -> UniverseState:
        """
        Run one full scan cycle across all instruments.

        Returns the updated UniverseState with all scan results.
        """
        start = time.time()
        self._scan_cycle += 1

        # Scan in batches per adapter to respect rate limits
        all_results: list[ScanResult] = []

        for adapter in self._adapters:
            # Get instruments for this adapter
            adapter_instruments = [
                (sym, meta)
                for sym, (meta, adp) in self._instruments.items()
                if adp is adapter
            ]

            # Scan in concurrent batches of 10
            batch_size = 10
            for i in range(0, len(adapter_instruments), batch_size):
                batch = adapter_instruments[i : i + batch_size]
                tasks = [
                    self._scan_instrument(sym, meta, adapter)
                    for sym, meta in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if isinstance(r, ScanResult):
                        all_results.append(r)
                    elif isinstance(r, Exception):
                        logger.debug("Scan error: %s", r)

                # Small delay between batches for rate limiting
                await asyncio.sleep(0.1)

        duration = time.time() - start

        self._state = UniverseState(
            results=all_results,
            scan_cycle=self._scan_cycle,
            scan_duration_sec=duration,
            last_scan_time=datetime.now(timezone.utc),
        )

        logger.info(
            "Scan cycle #%d: %d instruments, %d signals, %.1fs",
            self._scan_cycle,
            len(all_results),
            len(self._state.signals),
            duration,
        )

        return self._state

    async def run_continuous(self, shutdown_event: asyncio.Event) -> None:
        """Run scanning loop until shutdown."""
        while not shutdown_event.is_set():
            try:
                await self.scan_all()
            except Exception as exc:
                logger.error("Scan cycle failed: %s", exc)

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=self._scan_interval,
                )
                return
            except asyncio.TimeoutError:
                pass
