"""
═══════════════════════════════════════════════════════════════════
 live_multi_bot.py  –  Multi-Asset SMC Trading Bot
 ──────────────────────────────────────────────────────────────
 112 bots across 4 asset classes (30 Crypto + 28 Forex +
 50 Stocks + 4 Commodities), matching the backtester universe.

 Each bot is assigned to one instrument with asset-class-specific
 SMC parameters, commission rates, and leverage caps.

 Features:
   • 112 bots (30 crypto, 28 forex, 50 stocks, 4 commodities)
   • 3 exchange adapters: Binance, OANDA, Alpaca
   • Asset-class-specific SMC params, fees, and leverage
   • WebSocket for crypto, REST polling for forex/stocks/commodities
   • Central PPO RL brain shared by all instruments
   • 20-variant Paper Grid A/B testing
   • Circuit Breaker per asset class
   • Trading hours enforcement (forex 24/5, stocks regular hours)
   • Graceful degradation (missing API keys → skip that class)
   • Rich Live Dashboard with asset-class grouping

 Requirements:
   pip install 'ccxt[pro]' pandas numpy python-dotenv pyyaml rich torch
   Optional: pip install v20 (OANDA), pip install alpaca-py (Alpaca)

 Quick Start:
   1. Copy .env.example → .env and fill in broker API keys
   2. python live_multi_bot.py [--config config/default_config.yaml]
   3. Ctrl+C → graceful shutdown with final summary.
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# rl_brain.py (PPO) is superseded by rl_brain_v2.py (XGBoost).
# Keep import for extract_features (used for obs vector) but CentralRLBrain is no longer needed.
from rl_brain import extract_features
from rl_brain_v2 import RLBrainSuite
from trade_journal import TradeJournal
from utils.indicators import compute_rsi_wilders, compute_atr_wilders
from strategies.smc_multi_style import (
    compute_smc_indicators,
    _precompute_running_bias,
    _precompute_running_structure,
    _bias_from_running,
    _structure_confirms_from_running,
    _find_entry_zone_at,
    _precompute_5m_trigger_mask,
    _compute_alignment_score,
)
from filters.trend_strength import compute_adx, check_momentum_confluence, multi_tf_trend_agreement
from filters.volume_liquidity import compute_volume_score
from filters.session_filter import compute_session_score
from filters.zone_quality import compute_zone_quality
from features.feature_extractor import FeatureExtractor
from rl_dqn.dqn_inference import DQNExitManager
from exchanges import BinanceAdapter
from exchanges.base import ExchangeAdapter
from risk.circuit_breaker import CircuitBreaker
from ranker.universe_scanner import UniverseScanner, ScanResult
from ranker.opportunity_ranker import OpportunityRanker, RankedOpportunity
from ranker.capital_allocator import CapitalAllocator
from paper_grid import PaperGrid

# ── ccxt imports (only needed for BinanceAdapter backward-compat) ─
try:
    import ccxt as ccxt_sync
    import ccxt.pro as ccxtpro
except ImportError:
    ccxt_sync = None  # type: ignore[assignment]
    ccxtpro = None  # type: ignore[assignment]

# ── PyTorch (hard requirement for RL brain) ──────────────────────
try:
    import torch  # noqa: F401 – ensure torch is importable
except ImportError:
    sys.exit(
        "PyTorch is required.  Install with:  pip install torch"
    )

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("live_results")

DASHBOARD_REFRESH_SEC = 10         # Dashboard refresh interval
POSITION_POLL_SEC = 5             # Interval (seconds) for polling exchange positions
ZOMBIE_SWEEP_SEC = 60             # Interval (seconds) for periodic zombie order sweep
WS_MAX_RECONNECT = 5              # Max reconnect attempts per symbol
WS_RECONNECT_BASE_DELAY = 2       # Base delay (seconds) for exponential backoff
WS_GROUP_SIZE = 10                 # Symbols per WebSocket watcher group
WS_STAGGER_SEC = 1.0              # Delay between crypto WebSocket subscriptions at startup
HEARTBEAT_SEC = 300               # Heartbeat logging interval (5 minutes)
WATCHDOG_SEC = 600                # Watchdog check interval (10 minutes)
WATCHDOG_RESTART_LIMIT = 3        # Max per-symbol restarts before REST fallback

# ── Top 30 Crypto (ranked by liquidity, same as backtester) ──────
TOP_30_CRYPTO: list[str] = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "SOL/USDT:USDT",
    "BNB/USDT:USDT",  "XRP/USDT:USDT",  "DOGE/USDT:USDT",
    "TON/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT",
    "1000SHIB/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
    "TRX/USDT:USDT",  "BCH/USDT:USDT",  "NEAR/USDT:USDT",
    "LTC/USDT:USDT",  "1000PEPE/USDT:USDT", "SUI/USDT:USDT",
    "UNI/USDT:USDT",  "HBAR/USDT:USDT", "APT/USDT:USDT",
    "ARB/USDT:USDT",  "OP/USDT:USDT",   "POL/USDT:USDT",
    "FIL/USDT:USDT",  "INJ/USDT:USDT",  "RENDER/USDT:USDT",
    "TIA/USDT:USDT",  "SEI/USDT:USDT",  "WLD/USDT:USDT",
]

# ── 28 Forex Pairs (7 Majors + 21 Crosses) — OANDA format ───────
FOREX_28: list[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "NZD_USD", "USD_CAD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_NZD", "AUD_CAD", "AUD_CHF",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
]

# ── Top 50 US Stocks (by market cap) — Alpaca format ────────────
STOCKS_50: list[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMGN", "AMZN", "AVGO",
    "BAC", "BRK.B", "CMCSA", "COST", "CRM", "CSCO", "CVX",
    "DHR", "DIS", "GE", "GOOGL", "HD", "IBM", "INTC", "INTU",
    "JNJ", "JPM", "KO", "LIN", "LLY", "MA", "MCD", "META", "MRK", "MSFT",
    "NEE", "NFLX", "NVDA", "ORCL", "PEP", "PG", "PM", "QCOM",
    "TSLA", "TMO", "TXN", "UNH", "V", "VZ", "WMT", "XOM",
]

# ── 4 Commodities — OANDA format ────────────────────────────────
COMMODITIES_4: list[str] = ["XAU_USD", "XAG_USD", "WTICO_USD", "BCO_USD"]

# ── Combined instrument universe (112 total, matching backtester) ─
ALL_INSTRUMENTS: dict[str, list[str]] = {
    "crypto": TOP_30_CRYPTO,
    "forex": FOREX_28,
    "stocks": STOCKS_50,
    "commodities": COMMODITIES_4,
}
NUM_BOTS = sum(len(v) for v in ALL_INSTRUMENTS.values())  # 112

# Backward-compat alias
TOP_100_COINS = TOP_30_CRYPTO

# ── Asset-Class-Specific Commission Rates ────────────────────────
ASSET_COMMISSION: dict[str, float] = {
    "crypto": 0.0004,       # 0.04% taker (Binance Futures)
    "forex": 0.00005,       # ~0.5 pip spread equivalent
    "stocks": 0.0,          # commission-free (Alpaca)
    "commodities": 0.0001,  # ~1 pip spread equivalent
}

# ── Training-matched cost constants for RL feature computation ───
# These MUST match generate_rl_data.py ASSET_COMMISSION/ASSET_SLIPPAGE
# (different from live ASSET_COMMISSION which represents actual broker fees)
_TRAIN_COMMISSION: dict[str, float] = {
    "crypto": 0.0004, "forex": 0.0003, "stocks": 0.0001, "commodities": 0.0003,
}
_TRAIN_SLIPPAGE: dict[str, float] = {
    "crypto": 0.0002, "forex": 0.0001, "stocks": 0.0001, "commodities": 0.0002,
}

# ── Asset-Class-Specific SMC Parameters (from config smc_profiles) ─
ASSET_SMC_PARAMS: dict[str, dict[str, Any]] = {
    "crypto": {
        "swing_length": 8, "fvg_threshold": 0.0006,
        "order_block_lookback": 20, "liquidity_range_percent": 0.01,
        "alignment_threshold": 0.50, "weight_day": 1.25, "bos_choch_filter": "medium",
        "min_daily_atr_pct": 0.005, "min_5m_atr_pct": 0.0010,
    },
    "forex": {
        "swing_length": 20, "fvg_threshold": 0.001,
        "order_block_lookback": 30, "liquidity_range_percent": 0.008,
        "alignment_threshold": 0.50, "weight_day": 1.25, "bos_choch_filter": "medium",
        "min_daily_atr_pct": 0.003, "min_5m_atr_pct": 0.0003,
    },
    "stocks": {
        "swing_length": 10, "fvg_threshold": 0.0003,
        "order_block_lookback": 20, "liquidity_range_percent": 0.005,
        "alignment_threshold": 0.50, "weight_day": 1.25, "bos_choch_filter": "medium",
        "min_daily_atr_pct": 0.007, "min_5m_atr_pct": 0.0007,
    },
    "commodities": {
        "swing_length": 10, "fvg_threshold": 0.0004,
        "order_block_lookback": 20, "liquidity_range_percent": 0.005,
        "alignment_threshold": 0.50, "weight_day": 1.25, "bos_choch_filter": "medium",
        "min_daily_atr_pct": 0.006, "min_5m_atr_pct": 0.0008,
    },
}

# Legacy alias (used by some methods that haven't been migrated yet)
FIXED_SMC_PARAMS: dict[str, Any] = ASSET_SMC_PARAMS["crypto"]


_smc_log = logging.getLogger(__name__)

def _load_optimized_smc_params() -> dict[str, dict]:
    """Load per-cluster optimized SMC params. Falls back to ASSET_SMC_PARAMS if not available."""
    clusters_path = Path("config/instrument_clusters.json")
    results_dir = Path("backtest/results/smc_optimization")

    if not clusters_path.exists() or not results_dir.exists():
        return {}  # No optimized params available

    try:
        with open(clusters_path) as f:
            clusters = json.load(f)

        symbol_params: dict[str, dict] = {}
        for cid, info in clusters.get("clusters", {}).items():
            params_file = results_dir / f"cluster_{cid}_params.json"
            if not params_file.exists():
                continue
            with open(params_file) as f:
                opt = json.load(f)
            optimized = opt.get("optimized_params", {})
            if not optimized:
                continue
            # Map optimized params to all instruments in this cluster
            # Store under MULTIPLE key formats for matching:
            #   "BTCUSDT" (raw), "BTC/USDT:USDT" (ccxt), "EUR_USD" (oanda)
            for sym in info.get("instruments", []):
                symbol_params[sym] = optimized
                # Crypto: BTCUSDT -> BTC/USDT:USDT (ccxt futures format)
                if sym.endswith("USDT") and "_" not in sym:
                    base = sym[:-4]
                    symbol_params[f"{base}/USDT:USDT"] = optimized
                # Forex/commodities already use OANDA format (EUR_USD)

        if symbol_params:
            _smc_log.info("Loaded optimized SMC params for %d symbol keys", len(symbol_params))
        return symbol_params
    except Exception as e:
        _smc_log.warning("Failed loading optimized SMC params: %s", e)
        return {}


# Module-level cache
_OPTIMIZED_SMC_PARAMS: dict[str, dict] = _load_optimized_smc_params()

# ── Asset-Class Leverage Caps ────────────────────────────────────
ASSET_MAX_LEVERAGE: dict[str, int] = {
    "crypto": 10, "forex": 15, "stocks": 4, "commodities": 10,
}

# ── Per-Class Tier Flag Configuration ────────────────────────────
# Flags to SKIP for tier gate checks (still contribute to alignment score).
# Tick-volume classes skip volume_ok since it's unreliable without real volume.
TIER_SKIP_FLAGS: dict[str, list[str]] = {
    "crypto": [],
    "forex": ["volume_ok"],
    "stocks": [],
    "commodities": ["volume_ok"],
}

# Max new signals per SYMBOL per 4-hour window, by asset class (safety throttle).
# Per-symbol, not per-class: 28 forex pairs × 3 = 84 theoretical max per class.
# Portfolio-level protection comes from circuit breakers (daily -3%, weekly -5%).
MAX_SIGNALS_PER_SYMBOL_4H: dict[str, int] = {
    "crypto": 5, "forex": 3, "stocks": 5, "commodities": 2,
}

# ── Asset-Class IDs for RL Brain ─────────────────────────────────
ASSET_CLASS_ID: dict[str, float] = {
    "crypto": 0.0, "forex": 0.25, "stocks": 0.5, "commodities": 0.75,
}

# ── REST Polling interval for OANDA/Alpaca (no WebSocket) ────────
REST_POLL_INTERVAL_SEC = 10             # Forex/commodities: 10s (faster candle detection)
REST_POLL_INTERVAL_STOCKS_CANDLE = 60   # Stocks: 60s (Alpaca rate limit: 200 req/min for 50 symbols)
REST_POLL_INTERVAL_STOCKS_TICKER = 30
REST_STAGGER_SEC = 2.0                  # Stagger between REST bots (was 0.5, caused OANDA timeouts)


def symbol_to_asset_class(symbol: str) -> str:
    """Determine asset class from symbol format."""
    if symbol in COMMODITIES_4:
        return "commodities"
    if "_" in symbol and "/" not in symbol:
        return "forex"
    if "/" in symbol:
        return "crypto"
    return "stocks"

# ── Fixed Money Management ────────────────────────────────────────
FIXED_RISK_PCT = 0.0025     # 0.25 % risk per trade
FIXED_RR_MIN = 2.5          # minimum 1:2.5 reward-to-risk
FIXED_ATR_PERIOD = 14
FIXED_EMA_FAST = 20
FIXED_EMA_SLOW = 50
FIXED_MIN_VOL_MULT = 1.0    # min volume = 1.0× average
EXIT_QTY_MATCH_TOLERANCE = 0.05
# Warm-up: trades executed without RL gating before learning starts
WARMUP_TRADES = 100
# Commission assumptions (Binance USDT-M taker, both sides)
COMMISSION_RATE = 0.0004
COMMISSION_MULTIPLIER = 2
MAX_RISK_REDUCTION_STEP = 9  # 0.9% risk when stepping in 0.1% increments
MIN_DYNAMIC_RISK_PCT = 0.0025  # 0.25 % floor for dynamic sizing
MAX_DYNAMIC_RISK_PCT = 0.015   # 1.5 % cap for dynamic sizing
RR_DIVISOR = 3.0               # RR contribution scaled down (RR / 3) to avoid aggressive sizing
RR_CONTRIBUTION_CAP = 2.0      # RR contribution capped at +2.0 to bound boost from extreme RR setups
EPSILON_SL_DIST = 1e-6         # Minimum SL distance tolerance to avoid divide-by-zero
PRICE_TOLERANCE_FACTOR = 0.001
MIN_PRICE_TOLERANCE = 1e-6

# ── Volatility Filter ────────────────────────────────────────────
# Coins with daily ATR below this % are too quiet for SMC
# (noise dominates, structure is unreliable, stop-hunts are random)
MIN_DAILY_ATR_PCT = 0.008    # 0.8% daily ATR minimum
# 5m ATR floor – prevents entries where the per-bar range is too small
# for SL/TP to be meaningful (spreads & slippage eat the edge)
MIN_5M_ATR_PCT = 0.0015      # 0.15% per 5m bar minimum
# Minimum absolute SL distance as multiple of ATR(14) on 5m
# Ensures SL is placed beyond noise, not inside it
# 2.5× ATR means the SL sits well outside random wicks
MIN_SL_ATR_MULT = 2.5
# Minimum SL expressed as number of "ticks" (estimated from price magnitude)
# Prevents 1-2 tick SLs that are pure noise on low-priced coins
MIN_SL_TICKS = 5

# ── Trade Style Configuration ────────────────────────────────────
# Styles are STRICTLY separated – no mixing of entry/exit tactics
STYLE_SCALP = "scalp"
STYLE_DAY = "day"
STYLE_SWING = "swing"

STYLE_CONFIG: dict[str, dict[str, Any]] = {
    STYLE_SCALP: {
        "min_sl_pct": 0.002,    # 0.2% min SL (lowered — brain validates)
        "max_sl_pct": 0.008,    # 0.8% max SL
        "min_tp_pct": 0.002,    # 0.2% min TP (lowered from 0.6% — was blocking forex)
        "max_tp_pct": 0.015,    # 1.5% max TP
        "min_rr": 1.5,          # lowered from 2.0 — brain filters bad RR
        "cooldown_minutes": 20,
    },
    STYLE_DAY: {
        "min_sl_pct": 0.002,    # 0.2% min SL (lowered — brain validates)
        "max_sl_pct": 0.025,    # 2.5% max SL
        "min_tp_pct": 0.004,    # 0.4% min TP (lowered from 0.8%)
        "max_tp_pct": 0.06,     # 6% max TP
        "min_rr": 1.5,          # lowered from 2.5 — brain filters bad RR
        "cooldown_minutes": 60,
    },
    STYLE_SWING: {
        "min_sl_pct": 0.008,    # 0.8% min SL
        "max_sl_pct": 0.05,     # 5% max SL
        "min_tp_pct": 0.02,     # 2% min TP
        "max_tp_pct": 0.15,     # 15% max TP
        "min_rr": 2.0,          # lowered from 3.0 — brain filters bad RR
        "cooldown_minutes": 240,
    },
}

# ── Setup Quality Tiers ─────────────────────────────────────────
# AAA++ = sniper-only (highest probability, all components aligned)
# AAA+  = strong fallback (still high quality)
# No A or SPEC tiers – only the best trades
TIER_AAA_PLUS_PLUS = "AAA++"
TIER_AAA_PLUS = "AAA+"

TIER_THRESHOLDS: dict[str, dict[str, float]] = {
    TIER_AAA_PLUS_PLUS: {"min_score": 0.75, "min_rr": 2.5},  # lowered from 0.88/3.0 — XGBoost is primary filter
    TIER_AAA_PLUS:      {"min_score": 0.60, "min_rr": 2.0},  # lowered from 0.78/2.0 — let brain evaluate more signals
}

TIER_RISK: dict[str, dict[str, float]] = {
    TIER_AAA_PLUS_PLUS: {"base_risk": 0.010, "max_risk": 0.015},  # 1.0%–1.5%
    TIER_AAA_PLUS:      {"base_risk": 0.005, "max_risk": 0.010},  # 0.5%–1.0%
}

# Maximum leverage per tier – both tiers get precise entry leverage
TIER_MAX_LEVERAGE: dict[str, int] = {
    TIER_AAA_PLUS_PLUS: 50,
    TIER_AAA_PLUS: 30,
}

# (removed: _history_exchange / _get_history_exchange — history now loaded via adapter)


# ═══════════════════════════════════════════════════════════════════
#  Logging helpers
# ═══════════════════════════════════════════════════════════════════

_console_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                 datefmt="%H:%M:%S")


def _make_logger(name: str, log_path: Path) -> logging.Logger:
    """Create a logger that writes to *log_path* (file-only, no stderr)."""
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    lgr.addHandler(fh)

    return lgr


# Root logger → file only (Rich owns the console)
root_logger = logging.getLogger("live_multi")
root_logger.setLevel(logging.INFO)

# ── Centralised log file for all INFO/WARNING/ERROR messages ─────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_central_fh = logging.FileHandler(
    OUTPUT_DIR / "live_multi.log", mode="a", encoding="utf-8"
)
_central_fh.setLevel(logging.INFO)
_central_fh.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
root_logger.addHandler(_central_fh)

logger = root_logger

# ═══════════════════════════════════════════════════════════════════
#  Paper-Trading Bot  (coin-specialised, fixed params, RL brain)
# ═══════════════════════════════════════════════════════════════════

class PaperBot:
    """
    A single trading bot specialised on one coin.

    Uses fixed SMC parameters and money management.  An RL brain
    (PPO) gates each potential entry (yes / no).  Reward for the
    brain = pure PnL change in % (no shaping, no R:R bonus).

    Strategy logic:
      – Monitors 5 m candles for analysis (multi-TF alignment)
      – Real-time entry via ``watch_ticker``: when 5 m signal + zone
        are valid and the live price touches the entry zone, enter
        immediately (no waiting for a closed candle).
      – Places real bracket orders on Binance Testnet with SL + TP.
      – Risk per trade = 1 % of the *real* account balance
        (``fetch_balance`` before every trade).
      – Full SL or full TP (no partial exits)
      – Volume filter: skip bar if volume < 1.0× average(20)
      – Exit detection via exchange position polling (no candle-based exits)
    """

    def __init__(
        self,
        bot_id: int,
        symbol: str,
        config: dict[str, Any],
        output_dir: Path,
        asset_class: str = "crypto",
        adapter: ExchangeAdapter | None = None,
        central_brain: Any | None = None,  # Legacy PPO brain, no longer used
        rl_suite: RLBrainSuite | None = None,
    ) -> None:
        self.bot_id = bot_id
        self.tag = f"bot_{bot_id:03d}"
        self.symbol = symbol
        self.asset_class = asset_class
        self.adapter = adapter
        # Backward-compat: raw ccxt.pro exchange (only BinanceAdapter has .raw)
        # Legacy self.exchange removed — all calls go through self.adapter now

        # Use per-symbol optimized params merged on top of per-class defaults
        smc = ASSET_SMC_PARAMS.get(asset_class, ASSET_SMC_PARAMS["crypto"]).copy()
        _opt = _OPTIMIZED_SMC_PARAMS.get(self.symbol)
        if _opt:
            smc.update(_opt)
        self.swing_length: int = smc["swing_length"]
        self.fvg_threshold: float = smc["fvg_threshold"]
        self.ob_lookback: int = smc.get("order_block_lookback", 15)
        self.liq_range: float = smc.get("liquidity_range_percent", 0.005)
        self.min_daily_atr_pct: float = smc["min_daily_atr_pct"]
        self.min_5m_atr_pct: float = smc["min_5m_atr_pct"]
        self.alignment_threshold: float = smc.get("alignment_threshold", 0.65)

        # Asset-class-specific commission & leverage
        self.commission_rate: float = ASSET_COMMISSION.get(asset_class, 0.0004)
        self.max_asset_leverage: int = ASSET_MAX_LEVERAGE.get(asset_class, 10)

        self.risk_pct: float = FIXED_RISK_PCT
        self.rr_ratio: float = FIXED_RR_MIN
        self.leverage: int = min(10, self.max_asset_leverage)  # default, capped by asset class

        # Account tracking (for dashboard / RL brain)
        self.equity: float = 0.0  # set later by Runner with real balance
        self.peak_equity: float = 0.0  # managed by Runner
        self._account_equity: float = 0.0  # real account equity, set by Runner

        # Tracking
        self.total_pnl: float = 0.0
        self.trades: int = 0
        self.wins: int = 0

        # Active trades on exchange (max 3, cleared by position poller)
        self._active_trades: list[dict[str, Any]] = []
        self._processed_exit_ids: set[str] = set()

        # Candle history  {symbol: list[dict]}
        self._candle_buf: dict[str, list[dict[str, Any]]] = {}

        # Pending signal for real-time entry (set by on_candle, consumed by on_tick)
        self._pending_signal: dict[str, Any] | None = None

        # Per-class tier flag skip list and signal rate limiting
        # On testnet, skip volume_ok for crypto (unrealistic testnet volumes)
        _skip = list(TIER_SKIP_FLAGS.get(asset_class, []))
        if asset_class == "crypto" and getattr(adapter, "_testnet", False):
            if "volume_ok" not in _skip:
                _skip.append("volume_ok")
        self._tier_skip_flags: set[str] = set(_skip)
        self._max_signals_per_4h: int = MAX_SIGNALS_PER_SYMBOL_4H.get(asset_class, 5)
        self._recent_signal_times: list[datetime] = []

        # Cooldown: dynamic per trade style
        self._last_entry_time: datetime | None = None
        self._entry_cooldown = timedelta(hours=1)
        self._last_trade_style: str = STYLE_DAY

        # Volatility cache (refreshed every candle)
        self._daily_atr_pct: float = 0.0
        self._5m_atr_pct: float = 0.0

        # Active exchange order ID (for bracket order tracking)
        self._active_order_id: str | None = None

        # Circuit breaker (shared across all bots, set by Runner)
        self.circuit_breaker: CircuitBreaker | None = None

        # Paper Grid (multi-variant A/B testing, set by Runner)
        self.paper_grid: PaperGrid | None = None

        # Legacy PPO brain (superseded by XGBoost rl_brain_v2, kept for compat)
        self.brain = central_brain
        self.rl_suite = rl_suite

        # DQN exit manager (shadow mode -- logs alongside XGBoost for comparison)
        self._dqn_exit = None
        self._dqn_cfg = config.get("dqn_exit_manager", {})

        # RL monitoring stats
        self._rl_accepted: int = 0
        self._rl_rejected: int = 0
        # Store last obs so we can record reward when trade closes
        self._pending_obs: np.ndarray | None = None

        # Trade lifecycle journal (set by Runner; logs every bar for ML training)
        self.journal: TradeJournal | None = None

        # Output
        output_dir.mkdir(parents=True, exist_ok=True)
        # === PERSISTENCE ===
        self._state_path = output_dir / f"{self.tag}_state.json"
        self._equity_path = output_dir / f"{self.tag}_equity.csv"
        self._init_equity_csv()
        self.logger = _make_logger(
            f"live_multi.{self.tag}",
            output_dir / f"{self.tag}.log",
        )
        self.logger.info(
            "Initialised %s | symbol=%s | class=%s | equity=%.2f",
            self.tag, symbol, self.asset_class, self.equity,
        )
        if self.symbol in _OPTIMIZED_SMC_PARAMS:
            self.logger.info("Using OPTIMIZED SMC params for %s (cluster-tuned)", self.symbol)
        else:
            self.logger.info("Using DEFAULT SMC params for %s (%s class)", self.symbol, asset_class)

        # DQN exit manager init (deferred until logger is ready)
        if self._dqn_cfg.get("shadow_log", False):
            _dqn_path = self._dqn_cfg.get("model_path", "models/dqn_exit_manager.zip")
            self._dqn_exit = DQNExitManager(_dqn_path)
            if self._dqn_exit.is_available():
                self.logger.info("DQN exit manager loaded (shadow mode)")
            else:
                self.logger.info("DQN exit manager not available (model missing or SB3 not installed)")
                self._dqn_exit = None

        # Near-miss diagnostic state (rate-limiting)
        self._last_neutral_bias_log_hour = -1

        # Multi-TF OHLCV buffers (loaded async via load_history())
        self.buffer_1d: pd.DataFrame = pd.DataFrame()
        self.buffer_4h: pd.DataFrame = pd.DataFrame()
        self.buffer_1h: pd.DataFrame = pd.DataFrame()
        self.buffer_15m: pd.DataFrame = pd.DataFrame()
        self.buffer_5m: pd.DataFrame = pd.DataFrame()
        # History is loaded async after construction: await bot.load_history()
        self._load_state()

    # ── History loading (multi-TF buffers, async via adapter) ────────

    async def load_history(self) -> None:
        """Load multi-TF OHLCV history via adapter (250+ bars for EMA200 warmup)."""
        if self.adapter is None:
            self.logger.warning("No adapter — skipping history load for %s", self.symbol)
            return
        # 250 daily bars for EMA200, proportional for lower TFs
        tf_limits = {
            "1d": 250,      # 250 days for EMA200 warmup
            "4h": 500,      # ~83 days
            "1h": 1000,     # ~42 days
            "15m": 1500,    # ~5 days
            "5m": 1500,     # ~5 days
        }
        for tf, limit in tf_limits.items():
            try:
                raw = await self.adapter.fetch_ohlcv(self.symbol, tf, limit=limit)
                if raw:
                    df = pd.DataFrame(
                        raw,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    setattr(self, f"buffer_{tf}", df)
                    self.logger.info(
                        "Loaded %d %s candles for %s", len(df), tf, self.symbol,
                    )
            except Exception as exc:
                self.logger.warning(
                    "History load failed %s/%s: %s", self.symbol, tf, exc,
                )

        # Pre-fill _candle_buf from loaded 5m history so _prepare_signal
        # can run immediately instead of waiting 65-125 min for live candles
        if not self.buffer_5m.empty:
            for _, row in self.buffer_5m.tail(300).iterrows():
                candle = {
                    "timestamp": row["timestamp"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                if self.symbol not in self._candle_buf:
                    self._candle_buf[self.symbol] = []
                self._candle_buf[self.symbol].append(candle)
            self.logger.info(
                "Pre-filled candle buffer with %d 5m bars for %s",
                len(self._candle_buf.get(self.symbol, [])), self.symbol,
            )

    # === PERSISTENCE ===
    def _save_state(self) -> None:
        """Persist core bot state to JSON (atomic write)."""
        try:
            def _ser(trade: dict[str, Any]) -> dict[str, Any]:
                d = dict(trade)
                et = d.get("entry_time")
                if isinstance(et, datetime):
                    d["entry_time"] = et.isoformat()
                return d

            payload = {
                "trades": self.trades,
                "wins": self.wins,
                "total_pnl": self.total_pnl,
                "equity": self.equity,
                "peak_equity": self.peak_equity,
                "active_trades": [_ser(t) for t in self._active_trades],
            }
            tmp_path = self._state_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(str(tmp_path), str(self._state_path))  # atomic on POSIX
        except Exception as exc:
            self.logger.warning("State save failed: %s", exc)

    def _load_state(self) -> None:
        """Load persisted bot state if present."""
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.trades = int(data.get("trades", self.trades))
            self.wins = int(data.get("wins", self.wins))
            self.total_pnl = float(data.get("total_pnl", self.total_pnl))
            self.equity = float(data.get("equity", self.equity))
            self.peak_equity = float(data.get("peak_equity", self.peak_equity))

            active: list[dict[str, Any]] = []
            for raw in data.get("active_trades", []):
                if not isinstance(raw, dict):
                    continue
                t = dict(raw)
                et = t.get("entry_time")
                if isinstance(et, str):
                    try:
                        parsed = datetime.fromisoformat(et)
                        if parsed.tzinfo is None:
                            # Persisted timestamps are written in UTC; assume UTC if missing tzinfo.
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        if parsed > now + timedelta(days=1) or parsed < now - timedelta(days=3650):
                            parsed = now
                        t["entry_time"] = parsed
                    except Exception:
                        t["entry_time"] = datetime.now(timezone.utc)
                active.append(t)
            self._active_trades = active
            self.logger.info(
                "Restored state: trades=%d wins=%d pnl=%.2f equity=%.2f active=%d",
                self.trades,
                self.wins,
                self.total_pnl,
                self.equity,
                len(self._active_trades),
            )
        except Exception as exc:
            self.logger.warning("State load failed: %s", exc)

    # ── Multi-TF alignment score ──────────────────────────────────

    def _multi_tf_alignment_score(
        self, current_candle: dict[str, Any],
    ) -> tuple[float, str, dict[str, Any]]:
        """
        AAA++ Granular multi-timeframe SMC alignment score.

        13-component scoring system (max 1.0):
          1D  → Daily bias (0.10)
          4H  → Structure + POI (0.08 + 0.08)
          1H  → Structure + CHoCH (0.08 + 0.06)
          15m → Entry zone quality-weighted (0.12)
          5m  → Precision trigger (0.10)
          Vol → 3-layer volume score (0.08)
          ADX → Trend strength on 1H (0.08)
          Ses → Session optimality (0.06)
          Mom → Momentum confluence (0.06)
          TFA → Multi-TF trend agreement (0.05)
          ZFr → Zone freshness decay (0.05)

        Returns (score 0–1, direction, components_dict).
        """
        swing_len = self.swing_length
        fvg_thresh = self.fvg_threshold
        ob_lookback = self.ob_lookback
        liq_range = self.liq_range

        # Components tracked for tier classification (expanded for AAA++)
        comp: dict[str, Any] = {
            "bias": False, "bias_strong": False,
            "h4_confirms": False, "h4_poi": False,
            "h1_confirms": False, "h1_choch": False,
            "entry_zone": None, "zone_fresh": False,
            "precision_trigger": False, "volume_ok": False,
            # New AAA++ components
            "adx_strong": False, "adx_value": 0.0,
            "session_optimal": False, "session_score": 0.0,
            "momentum_confluent": False, "momentum_score": 0.0,
            "tf_agreement": 0, "tf_agreement_score": 0.0,
            "zone_quality": 0.0, "zone_quality_ok": False,
            "volume_score": 0.0, "volume_details": None,
        }

        daily_bias = "neutral"
        score = 0.0

        # Forex-specific scoring weights (redistribute from entry_zone/trigger to HTF)
        _fx = self.asset_class == "forex"
        _w_bias      = 0.10               # same for all
        _w_bias_half = 0.05               # same for all
        _w_h4        = 0.08               # same for all
        _w_h4_poi    = 0.08               # same for all
        _w_h1        = 0.08 if not _fx else 0.10
        _w_h1_choch  = 0.06               # same for all
        _w_zone      = 0.12 if not _fx else 0.06   # forex: zone unreliable w/ tick vol
        _w_trigger   = 0.10 if not _fx else 0.06   # forex: trigger unreliable
        _w_volume    = 0.08 if not _fx else 0.12   # forex: compensate
        _w_adx       = 0.08 if not _fx else 0.10   # forex: compensate
        _w_session   = 0.06               # same for all
        _w_momentum  = 0.06               # same for all
        _w_tf_agree  = 0.05               # same for all
        _w_freshness = 0.05               # same for all

        # ═══ STEP 1: Daily Bias (1D) – HTF direction (0.10) ═════
        if len(self.buffer_1d) >= swing_len * 2:
            try:
                ind_1d = compute_smc_indicators(
                    self.buffer_1d, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                comp["_ind_1d"] = ind_1d
                running_bias = _precompute_running_bias(ind_1d, self.buffer_1d)
                daily_bias = _bias_from_running(running_bias, len(self.buffer_1d))

                if daily_bias != "neutral":
                    comp["bias"] = True
                    # Only count if from BOS/CHoCH (strong bias required for AAA++)
                    pure_struct = _precompute_running_structure(ind_1d)
                    pure_bias = _bias_from_running(pure_struct, len(self.buffer_1d))
                    if pure_bias != "neutral" and pure_bias == daily_bias:
                        score += _w_bias
                        comp["bias_strong"] = True
                    else:
                        score += _w_bias_half  # EMA fallback only = half credit
            except Exception as exc:
                self.logger.debug("1D bias computation failed: %s", exc)

        if daily_bias == "neutral":
            _cur_hour = datetime.now(timezone.utc).hour
            if self.asset_class == "forex" and _cur_hour != self._last_neutral_bias_log_hour:
                self._last_neutral_bias_log_hour = _cur_hour
                self.logger.info(
                    "NEAR-MISS NEUTRAL-BIAS %s | class=%s | no_daily_direction",
                    self.symbol, self.asset_class,
                )
            direction = "long"
            return 0.0, direction, comp

        direction = "long" if daily_bias == "bullish" else "short"

        # ═══ STEP 2: 4H – Structure + POI (0.08 + 0.08) ═════════
        htf_zones = []
        ind_4h = None
        if len(self.buffer_4h) >= swing_len * 2:
            try:
                ind_4h = compute_smc_indicators(
                    self.buffer_4h, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                comp["_ind_4h"] = ind_4h
                running_4h = _precompute_running_structure(ind_4h)
                if _structure_confirms_from_running(running_4h, daily_bias, len(self.buffer_4h)):
                    score += _w_h4
                    comp["h4_confirms"] = True

                price = float(self.buffer_4h["close"].iloc[-1])
                h4_poi = self._find_poi_from_indicators(
                    ind_4h, price, daily_bias, lookback_bars=10,
                )
                if h4_poi is not None:
                    score += _w_h4_poi
                    comp["h4_poi"] = True
                    comp["h4_poi_data"] = h4_poi
                    htf_zones.append(h4_poi)
            except Exception as exc:
                self.logger.debug("4H computation failed: %s", exc)

        # ═══ STEP 2b: Discount/Premium Filter (4H swing range) ═══
        if ind_4h is not None:
            try:
                swing_hl = ind_4h.get("swing_highs_lows")
                if swing_hl is not None and not swing_hl.empty:
                    _sh, _sl = None, None
                    for j in range(len(swing_hl)):
                        hl = swing_hl["HighLow"].iat[j]
                        lvl = swing_hl["Level"].iat[j]
                        if pd.notna(hl) and pd.notna(lvl):
                            if hl > 0:
                                _sh = float(lvl)
                            elif hl < 0:
                                _sl = float(lvl)
                    if _sh is not None and _sl is not None and _sh > _sl:
                        _mid = (_sh + _sl) / 2.0
                        _cur = float(self.buffer_5m["close"].iloc[-1])
                        comp["_premium_discount"] = 1.0 if _cur > _mid else -1.0
                        # D/P is now a soft feature — brain learns when it matters.
                        # Previously this was a hard return (blocked 97% of signals).
                        # Wrong-zone signals get a score penalty instead of instant rejection.
                        if (daily_bias == "bullish" and _cur > _mid) or \
                           (daily_bias == "bearish" and _cur < _mid):
                            comp["_dp_wrong_zone"] = True  # brain feature
                            score -= 0.10  # penalty, not rejection
            except Exception as exc:
                self.logger.debug("D/P filter failed: %s", exc)

        # ═══ STEP 3: 1H – Structure + CHoCH (0.08 + 0.06) ═══════
        if len(self.buffer_1h) >= swing_len * 2:
            try:
                ind_1h = compute_smc_indicators(
                    self.buffer_1h, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                comp["_ind_1h"] = ind_1h
                running_1h = _precompute_running_structure(ind_1h)
                if _structure_confirms_from_running(running_1h, daily_bias, len(self.buffer_1h)):
                    score += _w_h1
                    comp["h1_confirms"] = True

                bos_choch_1h = ind_1h.get("bos_choch")
                if bos_choch_1h is not None and not bos_choch_1h.empty:
                    for i in range(len(bos_choch_1h) - 1, max(0, len(bos_choch_1h) - 4), -1):
                        choch_val = bos_choch_1h["CHOCH"].iat[i]
                        if pd.notna(choch_val) and choch_val != 0:
                            choch_dir = "bullish" if choch_val > 0 else "bearish"
                            if choch_dir == daily_bias:
                                score += _w_h1_choch
                                comp["h1_choch"] = True
                            break
            except Exception as exc:
                self.logger.debug("1H structure computation failed: %s", exc)

        # ═══ STEP 4: 15m – Entry zone quality-weighted (0.12) ════
        entry_zone = None
        zone_quality_result = {}
        if len(self.buffer_15m) >= swing_len * 2:
            try:
                ind_15m = compute_smc_indicators(
                    self.buffer_15m, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                comp["_ind_15m"] = ind_15m
                _zone_bars = 12 if _fx else 6
                entry_zone = _find_entry_zone_at(
                    ind_15m, self.buffer_15m, daily_bias,
                    fvg_thresh, len(self.buffer_15m),
                    max_zone_bars=_zone_bars,
                )
                if entry_zone is not None:
                    comp["entry_zone"] = entry_zone

                # Compute zone quality on the NEAREST matching zone (even without penetration).
                # entry_zone requires 30% penetration which rarely happens bar-by-bar in live.
                # Zone quality should reflect whether good zones EXIST nearby.
                zone_for_quality = entry_zone
                if zone_for_quality is None:
                    # Find nearest zone matching bias WITHOUT penetration requirement
                    zone_for_quality = self._find_nearest_zone(
                        ind_15m, self.buffer_15m, daily_bias, fvg_thresh, _zone_bars * 3,
                    )

                if zone_for_quality is not None:
                    closes_15m = self.buffer_15m["close"].values.astype(np.float64)
                    zone_bar_idx = max(0, len(self.buffer_15m) - 4)
                    # ATR on 15m
                    atr_15m = 0.0
                    if len(self.buffer_15m) >= 15:
                        h15 = self.buffer_15m["high"].values[-15:].astype(np.float64)
                        l15 = self.buffer_15m["low"].values[-15:].astype(np.float64)
                        c15 = self.buffer_15m["close"].values[-15:].astype(np.float64)
                        trs = []
                        for i in range(1, len(h15)):
                            trs.append(max(h15[i] - l15[i], abs(h15[i] - c15[i-1]), abs(l15[i] - c15[i-1])))
                        atr_15m = float(np.mean(trs)) if trs else 0.0

                    zone_quality_result = compute_zone_quality(
                        zone_data=zone_for_quality,
                        zone_bar_idx=zone_bar_idx,
                        current_bar_idx=len(self.buffer_15m) - 1,
                        closes_15m=closes_15m,
                        df_15m=self.buffer_15m,
                        atr_15m=atr_15m,
                        htf_zones=htf_zones if htf_zones else None,
                    )
                    zq = zone_quality_result.get("zone_quality", 0.0)
                    comp["zone_quality"] = zq
                    comp["zone_quality_ok"] = zone_quality_result.get("zone_quality_ok", False)

                    # Score weighted by zone quality
                    score += _w_zone * zq
                    comp["zone_fresh"] = zone_quality_result.get("decay_factor", 0.0) > 0.5
            except Exception as exc:
                self.logger.warning("15m entry zone computation failed: %s", exc)

        # ═══ STEP 5: 5m – Precision trigger (0.10) ═══════════════
        if len(self.buffer_5m) >= swing_len * 2:
            try:
                ind_5m = compute_smc_indicators(
                    self.buffer_5m, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                comp["_ind_5m"] = ind_5m
                _trig_lb = 3 if _fx else 1
                bull_mask, bear_mask = _precompute_5m_trigger_mask(ind_5m, lookback_bars=_trig_lb)
                if len(bull_mask) > 0:
                    if daily_bias == "bullish" and bull_mask[-1]:
                        score += _w_trigger
                        comp["precision_trigger"] = True
                    elif daily_bias == "bearish" and bear_mask[-1]:
                        score += _w_trigger
                        comp["precision_trigger"] = True
            except Exception as exc:
                self.logger.debug("5m trigger computation failed: %s", exc)

        # ═══ STEP 6: Volume – 3-layer scoring (0.08) ═════════════
        if not self.buffer_5m.empty and len(self.buffer_5m) >= 21:
            try:
                volumes_5m = self.buffer_5m["volume"].values.astype(np.float64)
                price = float(self.buffer_5m["close"].iloc[-1])
                current_vol = float(volumes_5m[-1])

                # Optional 1H data for volume profile
                h1h = l1h = c1h = v1h = None
                if not self.buffer_1h.empty and len(self.buffer_1h) >= 20:
                    h1h = self.buffer_1h["high"].values.astype(np.float64)
                    l1h = self.buffer_1h["low"].values.astype(np.float64)
                    c1h = self.buffer_1h["close"].values.astype(np.float64)
                    v1h = self.buffer_1h["volume"].values.astype(np.float64)

                vol_result = compute_volume_score(
                    volumes_5m=volumes_5m,
                    price=price,
                    current_volume=current_vol,
                    asset_class=self.asset_class,
                    highs_1h=h1h, lows_1h=l1h, closes_1h=c1h, volumes_1h=v1h,
                    direction=direction,
                )
                vol_score = vol_result.get("volume_score", 0.0)
                comp["volume_ok"] = vol_result.get("volume_ok", False)
                comp["volume_score"] = vol_score
                comp["volume_details"] = vol_result
                score += _w_volume * vol_score
            except Exception as exc:
                self.logger.debug("Volume scoring failed: %s", exc)

        # ═══ STEP 7: ADX Trend Strength on 1H (0.08) ═════════════
        if not self.buffer_1h.empty and len(self.buffer_1h) >= 30:
            try:
                h1 = self.buffer_1h["high"].values.astype(np.float64)
                l1 = self.buffer_1h["low"].values.astype(np.float64)
                c1 = self.buffer_1h["close"].values.astype(np.float64)
                adx, plus_di, minus_di = compute_adx(h1, l1, c1, period=14)
                comp["adx_value"] = adx
                comp["adx_strong"] = adx > 25.0
                # Score: ADX 25+ = starts contributing, 50+ = full score
                adx_score = min(max(adx - 15.0, 0.0) / 35.0, 1.0)
                score += _w_adx * adx_score
            except Exception as exc:
                self.logger.debug("ADX computation failed: %s", exc)

        # ═══ STEP 8: Session Optimality (0.06) ════════════════════
        try:
            session_sc = compute_session_score(asset_class=self.asset_class)
            comp["session_score"] = session_sc
            comp["session_optimal"] = session_sc >= 0.8
            score += _w_session * session_sc
        except Exception as exc:
            self.logger.debug("Session scoring failed: %s", exc)

        # ═══ STEP 9: Momentum Confluence on 1H (0.06) ════════════
        if not self.buffer_1h.empty and len(self.buffer_1h) >= 35:
            try:
                c1h = self.buffer_1h["close"].values.astype(np.float64)
                mom_ok, mom_score = check_momentum_confluence(c1h, direction)
                comp["momentum_confluent"] = mom_ok
                comp["momentum_score"] = mom_score
                score += _w_momentum * mom_score
            except Exception as exc:
                self.logger.debug("Momentum confluence failed: %s", exc)

        # ═══ STEP 10: Multi-TF Trend Agreement (0.05) ════════════
        try:
            c_1d = self.buffer_1d["close"].values.astype(np.float64) if not self.buffer_1d.empty and len(self.buffer_1d) >= 50 else None
            c_4h = self.buffer_4h["close"].values.astype(np.float64) if not self.buffer_4h.empty and len(self.buffer_4h) >= 50 else None
            c_1h = self.buffer_1h["close"].values.astype(np.float64) if not self.buffer_1h.empty and len(self.buffer_1h) >= 50 else None
            c_15m = self.buffer_15m["close"].values.astype(np.float64) if not self.buffer_15m.empty and len(self.buffer_15m) >= 50 else None

            tf_count, tf_score = multi_tf_trend_agreement(c_1d, c_4h, c_1h, c_15m, direction)
            comp["tf_agreement"] = tf_count
            comp["tf_agreement_score"] = tf_score
            score += _w_tf_agree * tf_score
        except Exception as exc:
            self.logger.debug("Multi-TF trend agreement failed: %s", exc)

        # ═══ STEP 11: Zone Freshness Decay Bonus (0.05) ══════════
        if zone_quality_result:
            decay_factor = zone_quality_result.get("decay_factor", 0.0)
            score += _w_freshness * decay_factor

        # Clamp final score
        score = float(np.clip(score, 0.0, 1.0))

        return score, direction, comp

    # ── Equity CSV ────────────────────────────────────────────────

    def _init_equity_csv(self) -> None:
        with open(self._equity_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity", "pnl", "trades", "drawdown_pct"])

    def _append_equity(self) -> None:
        dd = (
            (self.peak_equity - self.equity) / self.peak_equity * 100
            if self.peak_equity > 0 else 0.0
        )
        with open(self._equity_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                f"{self.equity:.2f}",
                f"{self.total_pnl:.2f}",
                self.trades,
                f"{dd:.2f}",
            ])

    # ── Price processing ──────────────────────────────────────────

    def on_candle(self, symbol: str, candle: dict[str, Any]) -> None:
        """
        Called whenever a new 5 m OHLCV candle arrives for *symbol*.

        Analysis runs here (multi-TF alignment, SMC SL/TP).
        If valid, a pending signal is stored.  Actual entry happens in
        ``on_tick`` when the live price touches the entry zone.

        candle keys: timestamp, open, high, low, close, volume
        """
        if symbol not in self._candle_buf:
            self._candle_buf[symbol] = []
        buf = self._candle_buf[symbol]
        buf.append(candle)
        if len(buf) > 300:
            buf.pop(0)

        # Keep multi-TF 5m buffer up to date
        if not self.buffer_5m.empty:
            idx = len(self.buffer_5m)
            self.buffer_5m.loc[idx] = {
                "timestamp": candle["timestamp"],
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
            }
            if len(self.buffer_5m) > 1500:
                self.buffer_5m = self.buffer_5m.iloc[-1500:].reset_index(drop=True)

        if len(buf) >= self.swing_length + 5:
            self._prepare_signal(symbol, buf, candle)

        # ── Trade Journal: record bar for each active trade ──────────
        # Fires only on confirmed closed 5m candles (not poll ticks).
        if self._active_trades and self.journal is not None:
            candle_ts = candle.get("timestamp")
            if isinstance(candle_ts, (int, float)):
                # Millisecond timestamp → datetime
                from datetime import timezone
                candle_dt = datetime.fromtimestamp(candle_ts / 1000, tz=timezone.utc)
            elif isinstance(candle_ts, datetime):
                candle_dt = candle_ts
            else:
                candle_dt = datetime.now(timezone.utc)

            bar_close = float(candle.get("close", 0.0))
            bar_high = float(candle.get("high", 0.0))
            bar_low = float(candle.get("low", 0.0))
            bar_vol = float(candle.get("volume", 0.0))

            # Compute RSI from recent 5m buffer (causal: only closed bars)
            rsi_val = 0.0
            if not self.buffer_5m.empty and len(self.buffer_5m) >= 15:
                try:
                    rsi_arr = compute_rsi_wilders(
                        self.buffer_5m["close"].values.astype(np.float64), period=14
                    )
                    rsi_val = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 0.0
                except Exception:
                    pass

            for trade in self._active_trades:
                trade_id = trade.get("rl_trade_id") or trade.get("order_id", "")
                if not trade_id:
                    continue
                entry_price = float(trade.get("entry", bar_close))
                entry_time = trade.get("entry_time")
                if entry_time is None:
                    continue

                bars_held = max(0, int(
                    (candle_dt - entry_time).total_seconds() // 300
                ))
                direction = trade.get("direction", "long")
                current_sl = float(trade.get("sl", 0.0))
                sl_dist_pct = 0.0
                unrealized_pnl_pct = 0.0
                if entry_price > 0 and bar_close > 0:
                    if direction == "long":
                        unrealized_pnl_pct = (bar_close - entry_price) / entry_price
                        sl_dist_pct = max((bar_close - current_sl) / bar_close, 0.0)
                    else:
                        unrealized_pnl_pct = (entry_price - bar_close) / entry_price
                        sl_dist_pct = max((current_sl - bar_close) / bar_close, 0.0)

                # Track bar-level stats for exit feature extraction
                if "_prev_unrealized_pnl_pct" not in trade:
                    trade["_prev_unrealized_pnl_pct"] = 0.0
                    trade["_bars_in_profit"] = 0
                    trade["_struct_breaks_against"] = 0
                if unrealized_pnl_pct > 0:
                    trade["_bars_in_profit"] = trade.get("_bars_in_profit", 0) + 1
                trade["_prev_unrealized_pnl_pct"] = unrealized_pnl_pct

                # Compute ADX for journal recording (matching generate_rl_data logic)
                _adx_1h_val = 0.0
                if not self.buffer_1h.empty and len(self.buffer_1h) >= 20:
                    try:
                        _h1_highs = self.buffer_1h["high"].values.astype(np.float64)
                        _h1_lows = self.buffer_1h["low"].values.astype(np.float64)
                        _h1_closes = self.buffer_1h["close"].values.astype(np.float64)
                        if len(_h1_highs) >= 14:
                            _adx_result, _, _ = compute_adx(
                                _h1_highs, _h1_lows, _h1_closes, period=14
                            )
                            _adx_1h_val = float(_adx_result) if _adx_result is not None and not np.isnan(_adx_result) else 0.0
                    except Exception:
                        pass

                try:
                    self.journal.record_bar(
                        trade_id=trade_id,
                        bar_index=bars_held,
                        timestamp=candle_dt,
                        close=bar_close,
                        high=bar_high,
                        low=bar_low,
                        volume=bar_vol,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        sl_distance_pct=sl_dist_pct,
                        rsi_5m=rsi_val / 100.0,  # normalise to [0,1]
                        adx_1h=_adx_1h_val,
                    )
                except Exception as exc:
                    self.logger.debug("journal.record_bar error: %s", exc)

                # ── ML exit prediction (acts when enabled + gates pass) ────
                if (self.rl_suite is not None and
                        hasattr(self.rl_suite, "predict_early_exit") and
                        self._check_component_enabled("exit_classifier")):
                    try:
                        _risk_pct = float(trade.get("risk_pct", 0.01))
                        _prev_pnl = float(trade.get("_prev_unrealized_pnl_pct", 0.0))
                        _bars_in_profit = int(trade.get("_bars_in_profit", 0))

                        bar_features = FeatureExtractor.extract_exit_bar_features(
                            bars_held=bars_held,
                            unrealized_pnl_pct=unrealized_pnl_pct,
                            risk_pct=_risk_pct,
                            sl_distance_pct=sl_dist_pct,
                            max_favorable_seen=self.journal._max_favorable.get(trade_id, 0.0),
                            be_triggered=bool(trade.get("be_triggered", False)),
                            asset_class=self.asset_class,
                            rsi_5m=rsi_val,  # raw [0,100], extractor normalizes
                            adx_1h=_adx_1h_val,  # raw [0,100] from compute_adx; extractor normalizes
                            atr_5m=float(trade.get("_atr_5m", 0.001)),
                            prev_unrealized_pnl_pct=_prev_pnl,
                            bars_in_profit=_bars_in_profit,
                            std_returns_50=float(trade.get("_std_ret_50", 0.01)),
                            std_returns_200=float(trade.get("_std_ret_200", 0.01)),
                            structure_breaks_against=int(trade.get("_struct_breaks_against", 0)),
                        )
                        should_exit, exit_conf = self.rl_suite.predict_early_exit(bar_features)

                        # Safety gates
                        _exit_conf_threshold = self.rl_suite.exit_threshold
                        _exit_min_bars = 6
                        _exit_min_favorable = 0.5
                        _exit_be_priority = True

                        bars_count = int(trade.get("_bars_held_count", bars_held))
                        unrealized_rr = bar_features.get("bar_unrealized_rr", 0)
                        be_trig = bool(trade.get("be_triggered", False))
                        be_ratchet_r = self.rl_suite.min_be_rr if self.rl_suite.be_enabled else 1.5

                        gate_pass = (
                            should_exit
                            and exit_conf >= _exit_conf_threshold
                            and bars_count >= _exit_min_bars
                            and unrealized_rr >= _exit_min_favorable
                        )

                        # BE ratchet priority: suppress between 0.5R and BE trigger if BE not yet active
                        if _exit_be_priority and not be_trig and _exit_min_favorable < unrealized_rr < be_ratchet_r:
                            gate_pass = False

                        if gate_pass and self.rl_suite.exit_enabled:
                            self.logger.info(
                                "ML_EXIT %s %s bar=%d conf=%.3f rr=%.2f — requesting close",
                                symbol, direction.upper(), bars_count, exit_conf, unrealized_rr,
                            )
                            trade["_ml_exit_requested"] = True
                            trade["_ml_exit_reason"] = "ml_exit"
                        elif should_exit:
                            self.logger.info(
                                "EXIT_SHADOW %s %s bar=%d conf=%.3f rr=%.2f "
                                "[gates: enabled=%s conf_ok=%s bars_ok=%s pnl_ok=%s be_ok=%s]",
                                symbol, direction.upper(), bars_count, exit_conf, unrealized_rr,
                                self.rl_suite.exit_enabled, exit_conf >= _exit_conf_threshold,
                                bars_count >= _exit_min_bars, unrealized_rr >= _exit_min_favorable,
                                not (_exit_be_priority and not be_trig and _exit_min_favorable < unrealized_rr < be_ratchet_r),
                            )
                    except Exception as exc:
                        self.logger.debug("exit prediction error: %s", exc)

                # DQN shadow prediction (alongside XGBoost)
                if self._dqn_exit is not None and self._dqn_exit.is_available():
                    try:
                        _risk_pct_dqn = float(trade.get("risk_pct", 0.01))
                        _prev_pnl_dqn = float(trade.get("_prev_unrealized_pnl_pct", 0.0))
                        _bars_in_profit_dqn = int(trade.get("_bars_in_profit", 0))
                        dqn_features = FeatureExtractor.extract_exit_bar_features(
                            bars_held=bars_held,
                            unrealized_pnl_pct=unrealized_pnl_pct,
                            risk_pct=_risk_pct_dqn,
                            sl_distance_pct=sl_dist_pct,
                            max_favorable_seen=self.journal._max_favorable.get(trade_id, 0.0) if self.journal else 0.0,
                            be_triggered=bool(trade.get("be_triggered", False)),
                            asset_class=self.asset_class,
                            rsi_5m=rsi_val,
                            adx_1h=_adx_1h_val,
                            atr_5m=float(trade.get("_atr_5m", 0.001)),
                            prev_unrealized_pnl_pct=_prev_pnl_dqn,
                            bars_in_profit=_bars_in_profit_dqn,
                            std_returns_50=float(trade.get("_std_ret_50", 0.01)),
                            std_returns_200=float(trade.get("_std_ret_200", 0.01)),
                            structure_breaks_against=int(trade.get("_struct_breaks_against", 0)),
                        )
                        dqn_action, dqn_conf = self._dqn_exit.predict(dqn_features)
                        action_names = {0: "HOLD", 1: "EXIT", 2: "MOVE_SL", 3: "PARTIAL"}
                        self.logger.info(
                            "DQN_SHADOW %s %s bar=%d action=%s conf=%.3f rr=%.2f",
                            symbol, direction.upper(), bars_held,
                            action_names.get(dqn_action, "?"), dqn_conf,
                            unrealized_pnl_pct / max(float(trade.get("risk_pct", 0.01)), 1e-6),
                        )
                    except Exception as exc:
                        self.logger.debug("DQN shadow error: %s", exc)

    # ── Simple alignment score ────────────────────────────────────

    @staticmethod
    def _alignment_score(candles: list[dict[str, Any]], swing_len: int) -> tuple[float, str]:
        """
        Compute a simplified alignment score [0..1] and suggested direction.

        Uses:
          • EMA-20 / EMA-50 trend on close prices
          • Recent swing-high/low break (momentum)
          • Normalised to [0, 1]
        """
        closes = np.array([c["close"] for c in candles], dtype=float)
        highs = np.array([c["high"] for c in candles], dtype=float)
        lows = np.array([c["low"] for c in candles], dtype=float)

        if len(closes) < 50:
            return 0.0, "neutral"

        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]

        recent_high = float(np.max(highs[-swing_len:]))
        recent_low = float(np.min(lows[-swing_len:]))
        prev_high = float(np.max(highs[-2 * swing_len: -swing_len])) if len(highs) >= 2 * swing_len else recent_high
        prev_low = float(np.min(lows[-2 * swing_len: -swing_len])) if len(lows) >= 2 * swing_len else recent_low

        price = closes[-1]

        trend_bull = 1.0 if ema20 > ema50 else 0.0
        bos_bull = 1.0 if recent_high > prev_high else 0.0
        bos_bear = 1.0 if recent_low < prev_low else 0.0
        ema_pos = 1.0 if price > ema20 else 0.0

        if trend_bull > 0.5:
            score = 0.25 * trend_bull + 0.35 * bos_bull + 0.20 * ema_pos + 0.20 * (1.0 - bos_bear)
            direction = "long"
        else:
            score = 0.25 * (1.0 - trend_bull) + 0.35 * bos_bear + 0.20 * (1.0 - ema_pos) + 0.20 * (1.0 - bos_bull)
            direction = "short"

        return float(np.clip(score, 0.0, 1.0)), direction

    # ── SMC-based SL/TP ──────────────────────────────────────────

    def _find_smc_sl_tp(
        self, price: float, direction: str,
    ) -> tuple[float, float] | None:
        """
        Compute dynamic SL and TP from SMC levels.

        SL: Under last Bullish OB (long) / over last Bearish OB (short),
            fallback to nearest Liquidity level.
        TP: Nearest opposite Liquidity Zone, fallback to opposite FVG.

        Returns ``(sl, tp)`` or ``None`` when valid levels cannot be found.
        """
        swing_len = self.swing_length
        fvg_thresh = self.fvg_threshold
        ob_lookback = self.ob_lookback
        liq_range = self.liq_range

        if len(self.buffer_5m) < swing_len * 2:
            return None

        try:
            ind = compute_smc_indicators(
                self.buffer_5m, swing_len, fvg_thresh, ob_lookback, liq_range,
            )
        except Exception:
            return None

        ob_data = ind.get("order_blocks")
        liq_data = ind.get("liquidity")
        fvg_data = ind.get("fvg")

        sl: float | None = None
        tp: float | None = None

        # ── SL: Order Block (primary) → Liquidity (fallback) ─────
        if direction == "long":
            # Last Bullish OB bottom below price
            if ob_data is not None and not ob_data.empty:
                for i in range(len(ob_data) - 1, -1, -1):
                    row = ob_data.iloc[i]
                    d = row.get("OB", np.nan)
                    bot = row.get("Bottom", np.nan)
                    if pd.notna(d) and d > 0 and pd.notna(bot) and bot < price:
                        sl = float(bot)
                        break
            # Fallback: nearest Liquidity level below price
            if sl is None and liq_data is not None and not liq_data.empty:
                best: float | None = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl < price:
                        if best is None or lvl > best:
                            best = lvl
                if best is not None:
                    sl = float(best)
        else:  # short
            # Last Bearish OB top above price
            if ob_data is not None and not ob_data.empty:
                for i in range(len(ob_data) - 1, -1, -1):
                    row = ob_data.iloc[i]
                    d = row.get("OB", np.nan)
                    top = row.get("Top", np.nan)
                    if pd.notna(d) and d < 0 and pd.notna(top) and top > price:
                        sl = float(top)
                        break
            # Fallback: nearest Liquidity level above price
            if sl is None and liq_data is not None and not liq_data.empty:
                best = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl > price:
                        if best is None or lvl < best:
                            best = lvl
                if best is not None:
                    sl = float(best)

        # ── TP: opposite Liquidity (primary) → opposite FVG (fallback)
        if direction == "long":
            # Nearest Liquidity level above price
            if liq_data is not None and not liq_data.empty:
                best = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl > price:
                        if best is None or lvl < best:
                            best = lvl
                if best is not None:
                    tp = float(best)
            # Fallback: nearest Bearish FVG bottom above price
            if tp is None and fvg_data is not None and not fvg_data.empty:
                best = None
                for i in range(len(fvg_data)):
                    row = fvg_data.iloc[i]
                    d = row.get("FVG", np.nan)
                    bot = row.get("Bottom", np.nan)
                    if pd.notna(d) and d < 0 and pd.notna(bot) and bot > price:
                        if best is None or bot < best:
                            best = bot
                if best is not None:
                    tp = float(best)
        else:  # short
            # Nearest Liquidity level below price
            if liq_data is not None and not liq_data.empty:
                best = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl < price:
                        if best is None or lvl > best:
                            best = lvl
                if best is not None:
                    tp = float(best)
            # Fallback: nearest Bullish FVG top below price
            if tp is None and fvg_data is not None and not fvg_data.empty:
                best = None
                for i in range(len(fvg_data)):
                    row = fvg_data.iloc[i]
                    d = row.get("FVG", np.nan)
                    top_val = row.get("Top", np.nan)
                    if pd.notna(d) and d > 0 and pd.notna(top_val) and top_val < price:
                        if best is None or top_val > best:
                            best = top_val
                if best is not None:
                    tp = float(best)

        if sl is None or tp is None:
            return None
        return (sl, tp)

    # ── XGBoost feature extraction ────────────────────────────────────

    # XGBoost asset class mapping (must match rl_brain_v2.ASSET_CLASS_MAP)
    _XGB_AC_MAP = {"crypto": 0, "forex": 1, "stocks": 2, "commodities": 3}

    @staticmethod
    def _find_nearest_zone(
        ind_15m: dict, df_15m: pd.DataFrame, bias: str,
        fvg_thresh: float, max_bars: int,
    ) -> dict | None:
        """Find nearest FVG/OB matching bias WITHOUT penetration requirement.

        Used for zone_quality scoring when _find_entry_zone_at returns None
        (which requires 30% penetration that rarely happens in live bar-by-bar).
        """
        valid_len = len(df_15m)
        if valid_len <= 0 or bias not in ("bullish", "bearish"):
            return None
        current_price = float(df_15m["close"].iloc[-1])

        # Check FVGs (wider lookback, no penetration required)
        fvg_data = ind_15m.get("fvg")
        if fvg_data is not None and not fvg_data.empty:
            end = min(valid_len, len(fvg_data))
            scan_start = max(0, end - max_bars)
            for idx in range(end - 1, scan_start - 1, -1):
                row = fvg_data.iloc[idx]
                fvg_dir = row.get("FVG", 0)
                top_val = row.get("Top", np.nan)
                bottom_val = row.get("Bottom", np.nan)
                if pd.isna(top_val) or pd.isna(bottom_val) or pd.isna(fvg_dir) or fvg_dir == 0:
                    continue
                gap_size = abs(top_val - bottom_val) / current_price if current_price > 0 else 0
                if gap_size < fvg_thresh:
                    continue
                # Match direction to bias (no penetration check)
                if bias == "bullish" and fvg_dir > 0:
                    return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val), "direction": "bullish"}
                if bias == "bearish" and fvg_dir < 0:
                    return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val), "direction": "bearish"}

        # Fallback: Order Blocks
        ob_data = ind_15m.get("order_blocks")
        if ob_data is not None and not ob_data.empty:
            end = min(valid_len, len(ob_data))
            scan_start = max(0, end - max_bars)
            for idx in range(end - 1, scan_start - 1, -1):
                row = ob_data.iloc[idx]
                ob_dir = row.get("OB", 0)
                ob_top = row.get("Top", np.nan)
                ob_bottom = row.get("Bottom", np.nan)
                if pd.isna(ob_top) or pd.isna(ob_bottom) or pd.isna(ob_dir) or ob_dir == 0:
                    continue
                if bias == "bullish" and ob_dir > 0:
                    return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom), "direction": "bullish"}
                if bias == "bearish" and ob_dir < 0:
                    return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom), "direction": "bearish"}
        return None

    def _build_xgb_features(self, components: dict, score: float) -> dict[str, float]:
        """Build 37-feature dict matching XGBoost model expectations.

        Mirrors the feature computation in backtest/generate_rl_data.py
        but only for the current (last) bar.
        """
        feat: dict[str, float] = {}

        # ── Structure direction per TF ──────────────────────────────
        ind_1d = components.get("_ind_1d")
        if ind_1d is not None and not self.buffer_1d.empty:
            try:
                bias = _precompute_running_bias(ind_1d, self.buffer_1d)
                feat["struct_1d"] = float(bias[-1]) if len(bias) > 0 else 0.0
            except Exception:
                feat["struct_1d"] = 0.0
        else:
            feat["struct_1d"] = 0.0

        for tf_key, ind_key in [("4h", "_ind_4h"), ("1h", "_ind_1h"),
                                 ("15m", "_ind_15m"), ("5m", "_ind_5m")]:
            ind = components.get(ind_key)
            if ind is not None:
                try:
                    rs = _precompute_running_structure(ind)
                    feat[f"struct_{tf_key}"] = float(rs[-1]) if len(rs) > 0 else 0.0
                except Exception:
                    feat[f"struct_{tf_key}"] = 0.0
            else:
                feat[f"struct_{tf_key}"] = 0.0

        # ── Break decay per TF (exp(-0.05 * bars_since_last_break)) ─
        for tf_key, ind_key in [("1d", "_ind_1d"), ("4h", "_ind_4h"),
                                 ("1h", "_ind_1h"), ("15m", "_ind_15m"),
                                 ("5m", "_ind_5m")]:
            ind = components.get(ind_key)
            decay = 0.0
            if ind is not None:
                bos_choch = ind.get("bos_choch")
                if bos_choch is not None and not bos_choch.empty:
                    last_break = -100
                    for i in range(len(bos_choch)):
                        choch_v = bos_choch["CHOCH"].iat[i]
                        bos_v = bos_choch["BOS"].iat[i]
                        if (pd.notna(choch_v) and choch_v != 0) or (pd.notna(bos_v) and bos_v != 0):
                            last_break = i
                    bars = (len(bos_choch) - 1) - last_break
                    decay = float(np.exp(-0.05 * bars))
            feat[f"decay_{tf_key}"] = decay

        # ── Premium / Discount ──────────────────────────────────────
        feat["premium_discount"] = float(components.get("_premium_discount", 0.0))

        # ── Boolean component flags → float ─────────────────────────
        for flag in ("h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
                     "precision_trigger", "volume_ok"):
            feat[flag] = 1.0 if components.get(flag) else 0.0

        # has_entry_zone: always 1.0 for live entry signals — by the time we
        # reach _build_xgb_features, entry zone existence is already confirmed.
        # Entry filter model excludes this (ENTRY_QUALITY_EXCLUDE), but
        # TP/BE/sizing models use it.
        feat["has_entry_zone"] = 1.0

        # ── EMA distances (relative to price) ───────────────────────
        if not self.buffer_5m.empty and len(self.buffer_5m) >= 50:
            c5 = self.buffer_5m["close"].values.astype(np.float64)
            p = float(c5[-1])
            e20 = float(pd.Series(c5).ewm(span=20, adjust=False).mean().iloc[-1])
            e50 = float(pd.Series(c5).ewm(span=50, adjust=False).mean().iloc[-1])
            feat["ema20_dist_5m"] = (p - e20) / p if p > 0 else 0.0
            feat["ema50_dist_5m"] = (p - e50) / p if p > 0 else 0.0
        else:
            feat["ema20_dist_5m"] = 0.0
            feat["ema50_dist_5m"] = 0.0

        if not self.buffer_1h.empty and len(self.buffer_1h) >= 50:
            c1h = self.buffer_1h["close"].values.astype(np.float64)
            p1h = float(c1h[-1])
            e20h = float(pd.Series(c1h).ewm(span=20, adjust=False).mean().iloc[-1])
            e50h = float(pd.Series(c1h).ewm(span=50, adjust=False).mean().iloc[-1])
            feat["ema20_dist_1h"] = (p1h - e20h) / p1h if p1h > 0 else 0.0
            feat["ema50_dist_1h"] = (p1h - e50h) / p1h if p1h > 0 else 0.0
        else:
            feat["ema20_dist_1h"] = 0.0
            feat["ema50_dist_1h"] = 0.0

        # ── ATR normalized (ATR / price) — Wilder's smoothing ───────
        def _atr_norm(buf: pd.DataFrame) -> float:
            if buf.empty or len(buf) < 15:
                return 0.0
            h = buf["high"].values.astype(np.float64)
            lo = buf["low"].values.astype(np.float64)
            c = buf["close"].values.astype(np.float64)
            atr_arr = compute_atr_wilders(h, lo, c, 14)
            return float(atr_arr[-1]) / float(c[-1]) if c[-1] > 0 else 0.0

        feat["atr_5m_norm"] = _atr_norm(self.buffer_5m)
        feat["atr_1h_norm"] = _atr_norm(self.buffer_1h)
        feat["atr_daily_norm"] = _atr_norm(self.buffer_1d)

        # ── RSI (normalized to [0, 1]) — Wilder's smoothing ─────────
        def _rsi_last(closes: np.ndarray, period: int = 14) -> float:
            if len(closes) < period + 1:
                return 0.5
            rsi_arr = compute_rsi_wilders(closes, period)
            return float(rsi_arr[-1]) / 100.0  # normalize to [0, 1]

        feat["rsi_5m"] = _rsi_last(self.buffer_5m["close"].values.astype(np.float64)) if not self.buffer_5m.empty and len(self.buffer_5m) >= 15 else 0.5
        feat["rsi_1h"] = _rsi_last(self.buffer_1h["close"].values.astype(np.float64)) if not self.buffer_1h.empty and len(self.buffer_1h) >= 15 else 0.5

        # ── Volume ratio (current / 20-bar MA, capped at 5) ────────
        if not self.buffer_5m.empty and len(self.buffer_5m) >= 20:
            vols = self.buffer_5m["volume"].values.astype(np.float64)
            avg_v = float(np.mean(vols[-20:]))
            feat["volume_ratio"] = min(float(vols[-1]) / avg_v, 5.0) if avg_v > 0 else 1.0
        else:
            feat["volume_ratio"] = 1.0

        # ── ADX on 1H (normalized by /50, capped at 2.0) ───────────
        # Training data computes real ADX for ALL classes (default 0.5, real ~0.3-0.8)
        # Models trained Mar 27 before commit 45ebeb5 — expect real ADX for all classes
        feat["adx_1h"] = min(float(components.get("adx_value", 25.0)) / 50.0, 2.0)

        # ── Alignment score ─────────────────────────────────────────
        feat["alignment_score"] = float(score)

        # ── Hour encoding (UTC, integer hour from candle timestamp) ──
        if not self.buffer_5m.empty:
            _last_ts = self.buffer_5m["timestamp"].iloc[-1]
            _hour_int = float(pd.Timestamp(_last_ts).hour)
        else:
            _hour_int = float(datetime.now(timezone.utc).hour)
        feat["hour_sin"] = float(np.sin(2 * np.pi * _hour_int / 24))
        feat["hour_cos"] = float(np.cos(2 * np.pi * _hour_int / 24))

        # ── FVG / OB active counts on 15m (running per-bar algorithm
        #    matching generate_rl_data.py:740-778 exactly) ──────────────
        feat["fvg_bull_active"] = 0.0
        feat["fvg_bear_active"] = 0.0
        feat["ob_bull_active"] = 0.0
        feat["ob_bear_active"] = 0.0

        ind_15m = components.get("_ind_15m")
        if ind_15m is not None:
            for key, col, b_key, br_key in [
                ("fvg", "FVG", "fvg_bull_active", "fvg_bear_active"),
                ("order_blocks", "OB", "ob_bull_active", "ob_bear_active"),
            ]:
                df_z = ind_15m.get(key)
                if df_z is not None and not df_z.empty and col in df_z.columns:
                    vals = df_z[col].values
                    mit = df_z["MitigatedIndex"].values if "MitigatedIndex" in df_z.columns else np.full(len(vals), np.nan)
                    n15 = len(vals)
                    active: list[tuple[int, float]] = []
                    bull_c = 0.0
                    bear_c = 0.0
                    for j in range(n15):
                        if pd.notna(vals[j]) and vals[j] != 0:
                            active.append((j, float(vals[j])))
                        active = [(a, t) for a, t in active
                                  if np.isnan(mit[a]) or mit[a] > j]
                        bull_c = sum(1 for _, t in active if t > 0) / 5.0
                        bear_c = sum(1 for _, t in active if t < 0) / 5.0
                    feat[b_key] = min(bull_c, 1.0)
                    feat[br_key] = min(bear_c, 1.0)

        # ── Liquidity counts on 1H (running per-bar, matching training) ─
        feat["liq_above_count"] = 0.0
        feat["liq_below_count"] = 0.0

        ind_1h = components.get("_ind_1h")
        if ind_1h is not None:
            liq_df = ind_1h.get("liquidity")
            if liq_df is not None and not liq_df.empty:
                liq_vals = liq_df["Liquidity"].values
                swept = liq_df["Swept"].values if "Swept" in liq_df.columns else np.full(len(liq_vals), np.nan)
                n1h = len(liq_vals)
                active_liq: list[tuple[int, float]] = []
                above_c = 0.0
                below_c = 0.0
                for j in range(n1h):
                    if pd.notna(liq_vals[j]) and liq_vals[j] != 0:
                        active_liq.append((j, float(liq_vals[j])))
                    active_liq = [(a, t) for a, t in active_liq
                                  if np.isnan(swept[a]) or swept[a] > j]
                    above_c = sum(1 for _, t in active_liq if t > 0) / 5.0
                    below_c = sum(1 for _, t in active_liq if t < 0) / 5.0
                feat["liq_above_count"] = min(above_c, 1.0)
                feat["liq_below_count"] = min(below_c, 1.0)

        # ── Asset class ID ──────────────────────────────────────────
        feat["asset_class_id"] = float(self._XGB_AC_MAP.get(self.asset_class, 0))

        return feat

    # ── Signal preparation (called from on_candle) ──────────────────

    def _check_component_enabled(self, component: str) -> bool:
        """Check if a component is enabled via dashboard toggles."""
        toggles_path = Path("live_results/component_toggles.json")
        if not toggles_path.exists():
            return True  # Default: all enabled
        try:
            with open(toggles_path) as f:
                toggles = json.load(f)
            return toggles.get(component, True)
        except Exception:
            return True

    def _prepare_signal(
        self,
        symbol: str,
        buf: list[dict[str, Any]],
        candle: dict[str, Any],
    ) -> None:
        """
        Enhanced signal preparation with:
        1. Volatility filter (skip coins with too little price movement)
        2. Granular multi-TF alignment scoring
        3. Style-aware SL/TP (scalp/day/swing – never mixed)
        4. Setup quality tier classification (AAA+/A/SPEC)
        5. ATR-based minimum SL distance (not just fixed %)
        """
        # ── Pause flag check ──────────────────────────────────────────
        if Path("live_results/.pause_flag").exists():
            return  # Paused — don't generate new signals

        # ── Trading hours check (forex/stocks have limited hours) ────
        if self.adapter is not None and not self.adapter.is_market_open(self.symbol):
            self._pending_signal = None
            return

        # ── Duplicate zone check (replaces cooldown timer) ─────────
        # Don't prepare a new signal if there's already an active trade
        # on this symbol – prevents double-entries on the same zone
        if self._active_trades:
            self._pending_signal = None
            return

        # ── Circuit breaker check ──────────────────────────────────
        if self.circuit_breaker is not None:
            can_trade, cb_reason = self.circuit_breaker.can_trade(self.asset_class)
            if not can_trade:
                self._pending_signal = None
                self.logger.info("CIRCUIT BREAKER SKIP %s: %s", symbol, cb_reason)
                return

        # ── Volatility check (soft — training has no volatility gate) ─
        tradeable, daily_atr, fivem_atr = self._check_volatility()
        if not tradeable:
            # Soft penalty instead of hard gate (training doesn't filter on volatility)
            score -= 0.05

        # ── Volume filter (basic pre-check – detailed scoring in alignment) ─
        # Quick reject if volume is clearly dead (< 0.5x avg)
        # Skip for tick-volume classes (forex/commodities) — meaningless with tick data
        # The full 3-layer volume scoring happens in _multi_tf_alignment_score
        if self.asset_class not in ("forex", "commodities"):
            volumes = [c["volume"] for c in buf[-20:]]
            avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
            if avg_vol > 0 and candle["volume"] < 0.5 * avg_vol:
                self._pending_signal = None
                return

        # ── Multi-TF alignment score (granular) ───────────────────
        score, direction, components = self._multi_tf_alignment_score(candle)
        if score < self.alignment_threshold:
            if score >= 0.50:
                self.logger.info(
                    "NEAR-MISS ALIGNMENT %s | class=%s score=%.3f thresh=%.2f dir=%s | flags=%s",
                    symbol, self.asset_class, score, self.alignment_threshold, direction,
                    {k: v for k, v in components.items() if not k.startswith("_")},
                )
            self._pending_signal = None
            return

        price = candle["close"]
        if price <= 0:
            self._pending_signal = None
            return

        # ── Initial SL/TP from 5m SMC (to classify trade style) ──
        sl_tp = self._find_smc_sl_tp(price, direction)
        if sl_tp is None:
            self._pending_signal = None
            return

        initial_sl, initial_tp = sl_tp

        # ── Classify trade style from natural SL/TP distances ─────
        style = self._classify_trade_style(price, initial_sl, initial_tp)

        # ── Re-compute SL/TP using style-appropriate timeframes ───
        # This prevents mixing tactics (e.g. scalp SL with swing TP)
        styled_sl_tp = self._find_smc_sl_tp_for_style(price, direction, style)
        if styled_sl_tp is None:
            self._pending_signal = None
            return
        sl, tp = styled_sl_tp

        # ── RE-CLASSIFY after styled SL/TP (the style may have shifted) ─
        # e.g. initial 5m SL/TP looked like SCALP but styled TP is DAY-range
        new_style = self._classify_trade_style(price, sl, tp)
        if new_style != style:
            self.logger.debug(
                "Style reclassified %s → %s after styled SL/TP for %s",
                style, new_style, symbol,
            )
            style = new_style
            # Re-compute SL/TP for the correct style
            styled_sl_tp = self._find_smc_sl_tp_for_style(price, direction, style)
            if styled_sl_tp is None:
                self._pending_signal = None
                return
            sl, tp = styled_sl_tp

        sl_dist = abs(price - sl)
        tp_dist = abs(tp - price)

        # ── Tick-size minimum SL (prevents 1-2 tick SLs on low-price coins) ─
        # Estimate actual tick size from the buffer's smallest price changes.
        # This is more reliable than guessing from price magnitude because
        # different coins at the same price level have different precisions
        # (e.g. 1000SHIB at $0.012 has tick 0.000001, STRK at $0.037 has tick 0.0001).
        tick_min_sl = 0.0
        if not self.buffer_5m.empty and len(self.buffer_5m) >= 20:
            buf_closes = self.buffer_5m["close"].values[-20:].astype(float)
            diffs = np.abs(np.diff(buf_closes))
            non_zero = diffs[diffs > 0]
            if len(non_zero) >= 3:
                # Smallest observed price change ≈ tick size
                est_tick = float(np.sort(non_zero)[:3].mean())
                tick_min_sl = est_tick * MIN_SL_TICKS

        # ── ATR-based minimum SL distance (adaptive to volatility) ─
        # Uses MAX of: fixed % floor, ATR-multiple floor, AND tick floor
        # This ensures SL is beyond noise for THIS specific coin
        atr_min_sl = fivem_atr * price * MIN_SL_ATR_MULT if fivem_atr > 0 else 0
        style_cfg = STYLE_CONFIG[style]
        pct_min_sl = price * style_cfg["min_sl_pct"]
        min_sl_dist = max(atr_min_sl, pct_min_sl, tick_min_sl)

        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
            sl = (price - sl_dist) if direction == "long" else (price + sl_dist)

        # ── Enforce maximum SL for style (with auto-upgrade) ─────
        # If SL widening pushed beyond current style limits,
        # upgrade to next style (SCALP → DAY → SWING) instead of rejecting
        max_sl_dist = price * style_cfg["max_sl_pct"]
        if sl_dist > max_sl_dist:
            upgraded = False
            upgrade_chain = {STYLE_SCALP: STYLE_DAY, STYLE_DAY: STYLE_SWING}
            next_style = upgrade_chain.get(style)
            if next_style:
                next_cfg = STYLE_CONFIG[next_style]
                next_max = price * next_cfg["max_sl_pct"]
                if sl_dist <= next_max:
                    self.logger.info(
                        "Style upgraded %s → %s (SL %.4f%% > %s max %.4f%%) for %s",
                        style, next_style, sl_dist / price * 100,
                        style, style_cfg["max_sl_pct"] * 100, symbol,
                    )
                    style = next_style
                    style_cfg = next_cfg
                    upgraded = True
            if not upgraded:
                self._pending_signal = None
                self.logger.debug(
                    "SL TOO WIDE for %s style: %.4f%% > max %.4f%%",
                    style, sl_dist / price * 100, style_cfg["max_sl_pct"] * 100,
                )
                return

        if tp_dist <= 0:
            self._pending_signal = None
            self.logger.debug("TP_DIST<=0 %s | style=%s", symbol, style)
            return

        # ── Enforce TP constraints for style ──────────────────────
        tp_pct = tp_dist / price
        if tp_pct < style_cfg["min_tp_pct"]:
            self._pending_signal = None
            self.logger.debug("TP TOO SMALL %s | tp=%.4f%% min=%.4f%% style=%s", symbol, tp_pct * 100, style_cfg["min_tp_pct"] * 100, style)
            return
        if tp_pct > style_cfg["max_tp_pct"]:
            # Clamp TP to max for this style
            tp_dist = price * style_cfg["max_tp_pct"]
            tp = (price + tp_dist) if direction == "long" else (price - tp_dist)

        # ── RR check (global minimum, matching training pipeline) ──
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < 1.0:  # training uses global 1.0, not per-style
            self._pending_signal = None
            self.logger.debug("RR TOO LOW %s | rr=%.2f min=1.0", symbol, rr)
            return

        # ── Tier removed — XGBoost brain is the primary filter ─────
        # Training has no tier system; brain learned on raw score+features.
        # Assign tier label for logging/risk only (not as a gate).
        tier = TIER_AAA_PLUS  # default tier for risk sizing
        if score >= 0.75:
            tier = TIER_AAA_PLUS_PLUS

        # ── Signal rate limiting (per-class safety throttle) ──────
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=4)
        self._recent_signal_times = [
            t for t in self._recent_signal_times if t > cutoff
        ]
        if len(self._recent_signal_times) >= self._max_signals_per_4h:
            self.logger.info(
                "RATE-LIMITED %s | class=%s | %d signals in 4h (max=%d)",
                symbol, self.asset_class,
                len(self._recent_signal_times), self._max_signals_per_4h,
            )
            self._pending_signal = None
            return
        self._recent_signal_times.append(now)

        # ── Final style constraint validation ─────────────────────
        if not self._validate_style_constraints(style, price, sl, tp):
            self._pending_signal = None
            self.logger.debug(
                "STYLE MISMATCH %s %s | SL/TP don't conform to %s constraints",
                direction.upper(), symbol, style,
            )
            return

        # ── Compute entry zone ────────────────────────────────────
        if direction == "long":
            zone_low = sl
            zone_high = price
        else:
            zone_low = price
            zone_high = sl

        # ── Build feature vectors BEFORE stripping heavy indicator dicts ─
        _xgb_features = self._build_xgb_features(components, score)
        _ppo_obs = extract_features(
            buf, score, direction,
            setup_tier=tier, trade_style=style,
            rr_ratio=rr, daily_atr_pct=daily_atr,
            adx_normalized=min(components.get("adx_value", 0.0) / 50.0, 1.0),
            session_score=components.get("session_score", 0.5),
            zone_quality=components.get("zone_quality", 0.0),
            volume_score=components.get("volume_score", 0.0),
            momentum_score=components.get("momentum_score", 0.0),
            tf_agreement_score=components.get("tf_agreement_score", 0.0),
            spread_normalized=0.0,
            asset_class_id=ASSET_CLASS_ID.get(self.asset_class, 0.0),
        )

        # Strip heavy indicator DataFrames — only needed for feature extraction
        for _k in ("_ind_1d", "_ind_4h", "_ind_1h", "_ind_15m", "_ind_5m", "_premium_discount"):
            components.pop(_k, None)

        # ── Store pending signal with all classification data ─────
        self._pending_signal = {
            "symbol": symbol,
            "direction": direction,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "score": score,
            "style": style,
            "tier": tier,
            "components": components,
            "daily_atr_pct": daily_atr,
            "obs": _ppo_obs,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "ref_price": price,
            "features": _xgb_features,
        }
        self.logger.info(
            "PENDING %s %s %s [%s] | zone=[%.6f, %.6f] SL=%.6f TP=%.6f "
            "RR=%.1f score=%.2f daily_atr=%.3f%%",
            tier, style.upper(), direction.upper(), symbol,
            zone_low, zone_high, sl, tp, rr, score, daily_atr * 100,
        )

        # ── Route signal to Paper Grid (A/B testing) ────────────
        if self.paper_grid is not None:
            self.paper_grid.evaluate_signal(self._pending_signal, asset_class=self.asset_class)

    # ── Real-time tick handler (called from watch_ticker) ─────────

    async def on_tick(self, symbol: str, price: float) -> None:
        """
        Called on every live ticker update.

        If a pending signal exists and the live price is inside the
        entry zone, place a real bracket order on the testnet.
        """
        # ── Pause flag check ──────────────────────────────────────────
        if Path("live_results/.pause_flag").exists():
            return  # Paused — don't enter new trades

        sig = self._pending_signal
        if sig is None:
            return
        if sig["symbol"] != symbol:
            return
        # ── Duplicate style check ─────────────────────────────────
        # Allow one trade per style (scalp / day / swing) simultaneously,
        # but never two trades of the same style on the same coin.
        pending_style = sig.get("style", STYLE_DAY)
        if any(t["style"] == pending_style for t in self._active_trades):
            self._pending_signal = None
            return

        # Check if price is inside the entry zone
        if not (sig["zone_low"] <= price <= sig["zone_high"]):
            return

        direction = sig["direction"]
        sl = sig["sl"]
        tp = sig["tp"]
        score = sig["score"]
        obs = sig["obs"]
        tier = sig.get("tier", TIER_AAA_PLUS)
        style = sig.get("style", STYLE_DAY)

        # ── RL Brain gate ────────────────────────────────────────────
        # XGBoost models are pre-trained (19M rows) — no warmup needed.
        use_ppo_brain = False  # PPO gate removed (v2)
        rl_tracked = False
        rl_trade_id: str | None = None
        take_trade = True
        rl_confidence = 1.0

        # XGBoost entry filter (RLBrainSuite) — always active, no warmup
        if self.rl_suite is not None and self.rl_suite.entry_filter_enabled and self._check_component_enabled("entry_filter"):
            xgb_take, rl_confidence = self.rl_suite.predict_entry(sig.get("features", {}))
            if not xgb_take:
                self._rl_rejected += 1
                _total_rl = self._rl_accepted + self._rl_rejected
                _rate = self._rl_accepted / _total_rl * 100 if _total_rl > 0 else 0
                self.logger.info(
                    "XGB REJECT %s conf=%.3f score=%.2f | accepted=%d rejected=%d rate=%.0f%%",
                    symbol, rl_confidence, score,
                    self._rl_accepted, self._rl_rejected, _rate,
                )
                # Alert: abnormal acceptance rate after 20+ decisions
                if _total_rl >= 20 and (_rate < 10 or _rate > 95):
                    self.logger.warning(
                        "RL ALERT: acceptance rate %.0f%% (%d/%d) — check feature extraction",
                        _rate, self._rl_accepted, _total_rl,
                    )
                self._pending_signal = None
                return
            self._rl_accepted += 1
            self.logger.info(
                "XGB ACCEPT %s conf=%.3f score=%.2f | accepted=%d rejected=%d",
                symbol, rl_confidence, score,
                self._rl_accepted, self._rl_rejected,
            )

        # PPO brain gate REMOVED (v2) -- XGBoost entry filter is sole ML gate.
        # PPO with 256-buffer and 100-trade warmup learned nothing useful.
        # Module kept on disk for legacy compatibility.

        # ── Fetch real balance (1 % risk) ─────────────────────────
        balance = await self._fetch_balance()
        if balance is None or balance <= 0:
            self.logger.warning(
                "fetch_balance returned %s – falling back to tracked equity (%.2f)",
                balance, self.equity,
            )
            balance = self.equity  # fallback to tracked equity

        sl_dist = abs(price - sl)
        tp_dist = abs(tp - price)
        if sl_dist <= 0:
            return

        # ── Enforce min SL distance at live price ─────────────────
        # The pending signal's SL was validated at ref_price, but the
        # live tick price may differ, making sl_dist too tight.
        style_cfg = STYLE_CONFIG.get(style, STYLE_CONFIG[STYLE_DAY])
        pct_min_sl = price * style_cfg["min_sl_pct"]
        if sl_dist < pct_min_sl:
            sl_dist = pct_min_sl
            sl = (price - sl_dist) if direction == "long" else (price + sl_dist)

        # ── Fee profitability gate ─────────────────────────────────
        # Skip trade if hitting TP would still be net-negative after fees.
        # total_fee_pct = commission_rate * 2 (entry + exit)
        min_tp_for_profit = price * self.commission_rate * COMMISSION_MULTIPLIER
        if tp_dist <= min_tp_for_profit:
            self.logger.info(
                "FEE GATE: skipping %s %s – tp_dist=%.6f <= fee_cost=%.6f (%.4f%%)",
                direction.upper(), symbol, tp_dist, min_tp_for_profit,
                self.commission_rate * COMMISSION_MULTIPLIER * 100,
            )
            self._pending_signal = None
            return

        # ── RL TP adjustment ───────────────────────────────────────────
        rl_be_level = 0.0
        if self.rl_suite is not None:
            if self.rl_suite.tp_enabled and sl_dist > 0 and self._check_component_enabled("tp_optimizer"):
                planned_tp_rr = tp_dist / sl_dist
                adjusted_tp_rr = self.rl_suite.predict_tp_adjustment(
                    sig.get("features", {}), planned_tp_rr,
                )
                if abs(adjusted_tp_rr - planned_tp_rr) > 0.01:
                    old_tp = tp
                    tp = price + adjusted_tp_rr * sl_dist if direction == "long" else price - adjusted_tp_rr * sl_dist
                    tp_dist = abs(tp - price)
                    self.logger.info(
                        "RL TP adjusted %s: %.6f -> %.6f (RR %.1f -> %.1f)",
                        symbol, old_tp, tp, planned_tp_rr, adjusted_tp_rr,
                    )

            if self.rl_suite.be_enabled and sl_dist > 0 and self._check_component_enabled("be_manager"):
                # Match training formula: cost_rr in R-multiples
                _tc = _TRAIN_COMMISSION.get(self.asset_class, 0.0004)
                _ts = _TRAIN_SLIPPAGE.get(self.asset_class, 0.0002)
                cost_rr = (price * (_tc + _ts) * 2) / sl_dist
                rl_be_level = self.rl_suite.predict_be_level(
                    sig.get("features", {}), cost_rr,
                )

        order_id, sl_order_id, tp_order_id, qty, risk_pct, used_leverage = (
            await self._execute_bracket_order_with_risk_reduction(
                symbol=symbol,
                direction=direction,
                price=price,
                sl=sl,
                tp=tp,
                balance=balance,
                sl_dist=sl_dist,
                tp_dist=tp_dist,
                score=score,
                tier=tier,
                style=style,
            )
        )

        # Consume the pending signal
        self._pending_signal = None

        # If the real order failed, do not track as active trade
        if order_id is None:
            self.logger.warning(
                "Bracket order failed for %s %s – skipping",
                direction.upper(), symbol,
            )
            return

        self._pending_obs = obs

        # ── Track active trade (cleared by position poller on fill) ──
        trade_info = {
            "symbol": symbol,
            "direction": direction,
            "entry": price,
            "sl": sl,
            "original_sl": sl,
            "tp": tp,
            "qty": qty,
            "leverage": used_leverage,
            "risk_pct": risk_pct,
            "entry_time": datetime.now(timezone.utc),
            "score": score,
            "tier": tier,
            "style": style,
            "order_id": order_id,
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id,
            "rl_tracked": rl_tracked,
            "rl_trade_id": rl_trade_id,
            "rl_confidence": rl_confidence,
            "rl_be_level": rl_be_level,
            "be_triggered": False,
        }
        self._active_trades.append(trade_info)
        self._last_entry_time = datetime.now(timezone.utc)
        self._last_trade_style = style

        # ── Trade Journal: record trade open ─────────────────────────
        trade_id_for_journal = rl_trade_id or order_id or f"t_{len(self._active_trades)}"
        # Backfill trade_id so candle handler and close hook use same key
        self._active_trades[-1]["rl_trade_id"] = trade_id_for_journal
        if self.journal is not None:
            try:
                rr_target = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 3.0
                self.journal.open_trade(
                    trade_id=trade_id_for_journal,
                    symbol=symbol,
                    asset_class=self.asset_class,
                    direction=direction,
                    style=style,
                    tier=tier,
                    entry_time=datetime.now(timezone.utc),
                    entry_price=price,
                    sl_original=sl,
                    tp=tp,
                    score=score,
                    rr_target=rr_target,
                    leverage=used_leverage,
                    risk_pct=risk_pct,
                    entry_features=sig.get("features", {}),
                )
            except Exception as exc:
                self.logger.debug("journal.open_trade error: %s", exc)

        self.logger.info(
            "OPEN [%s|%s] %s %s @ %.6f | SL=%.6f TP=%.6f RR=%.1f | qty=%.4f "
            "risk=%.2f%% lev=%dx score=%.2f bal=%.2f order=%s",
            tier, style.upper(), direction.upper(), symbol, price, sl, tp,
            abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0,
            qty, risk_pct * 100, used_leverage, score, balance,
            order_id or "no-exchange",
        )

    async def _execute_bracket_order_with_risk_reduction(
        self,
        symbol: str,
        direction: str,
        price: float,
        sl: float,
        tp: float,
        balance: float,
        sl_dist: float,
        tp_dist: float,
        score: float,
        tier: str = TIER_AAA_PLUS,
        style: str = STYLE_DAY,
    ) -> tuple[str | None, str | None, str | None, float, float, int]:
        """
        Execute a bracket order with tier-based risk allocation.

        Risk allocation:
          AAA++ → 1.0%–1.5% (sniper trades, highest conviction)
          AAA+  → 0.5%–1.0% (strong fallback trades)

        Returns:
            tuple: (order_id, sl_order_id, tp_order_id, quantity, used_risk_pct, applied_leverage)
        """
        # === TIER-BASED DYNAMIC RISK ===
        rr = tp_dist / sl_dist if sl_dist > EPSILON_SL_DIST else 0.0

        # Get tier-specific risk bounds
        tier_cfg = TIER_RISK.get(tier, TIER_RISK[TIER_AAA_PLUS])
        base_risk = tier_cfg["base_risk"]
        max_risk = tier_cfg["max_risk"]

        # Scale risk within tier bounds based on RR and score
        # Higher RR + higher score = closer to max_risk
        rr_factor = min(rr / 6.0, 1.0)        # RR 6+ → full factor
        score_factor = min(score / 0.90, 1.0)  # Score 0.90+ → full factor
        combined_factor = (rr_factor * 0.5 + score_factor * 0.5)

        dynamic_risk = base_risk + (max_risk - base_risk) * combined_factor
        dynamic_risk = max(base_risk, min(dynamic_risk, max_risk))

        self.logger.info(
            "[TIER RISK] %s|%s | score=%.2f RR=%.1f → risk=%.3f%% "
            "(tier range %.3f%%–%.3f%%) rr_factor=%.2f score_factor=%.2f",
            tier, style.upper(), score, rr,
            dynamic_risk * 100, base_risk * 100, max_risk * 100,
            rr_factor, score_factor,
        )

        if self.adapter is None:
            return None, None, None, 0.0, dynamic_risk, self.leverage
        ORIGINAL_RISK_PCT = dynamic_risk

        # ═══════════════════════════════════════════════════════════
        #  STEP 1: Load instrument info via adapter (exchange-agnostic)
        # ═══════════════════════════════════════════════════════════
        meta = self.adapter.get_instrument(symbol)
        max_qty_limit = meta.max_qty if meta else None
        min_qty_limit = meta.min_qty if meta else None
        max_notional_limit: float | None = None  # adapter doesn't track this

        self.logger.info(
            "Instrument info for %s: max_qty=%s min_qty=%s lot_size=%s max_lev=%s",
            symbol,
            max_qty_limit,
            min_qty_limit,
            meta.lot_size if meta else None,
            meta.max_leverage if meta else None,
        )

        # ═══════════════════════════════════════════════════════════
        #  STEP 2: Fetch max leverage via adapter
        # ═══════════════════════════════════════════════════════════
        try:
            max_leverage = await self.adapter.fetch_max_leverage(symbol)
            leverage_source = "adapter"
        except Exception as exc:
            max_leverage = meta.max_leverage if meta else 20
            leverage_source = "instrument_meta"
            self.logger.warning(
                "fetch_max_leverage failed for %s: %s — using %dx from meta",
                symbol, exc, max_leverage,
            )

        self.logger.info(
            "Max leverage for %s = %dx (source: %s)",
            symbol, max_leverage, leverage_source,
        )

        planned_leverage = max(1, int(max_leverage))

        # ── Tier-based leverage cap ──────────────────────────────
        # SPEC trades get conservative leverage (max 15x),
        # A trades moderate (max 25x), AAA+ can use full exchange leverage
        tier_max_lev = TIER_MAX_LEVERAGE.get(tier, 20)
        if planned_leverage > tier_max_lev:
            self.logger.info(
                "Capping leverage %dx → %dx for tier %s on %s",
                planned_leverage, tier_max_lev, tier, symbol,
            )
            planned_leverage = tier_max_lev

        # ═══════════════════════════════════════════════════════════
        #  STEP 3: Helper functions
        # ═══════════════════════════════════════════════════════════

        def _round_qty(q: float) -> float:
            """Round qty to exchange precision, and clamp to [min_qty, max_qty]."""
            try:
                q = float(self.adapter.amount_to_precision(symbol, q))
            except Exception:
                pass
            if max_qty_limit and q > max_qty_limit:
                q = max_qty_limit
                try:
                    q = float(self.adapter.amount_to_precision(symbol, q))
                except Exception:
                    pass
            if min_qty_limit and q < min_qty_limit:
                q = 0.0
            return q

        def _calc_qty(risk_pct: float) -> tuple[float, float]:
            ra = balance * risk_pct
            q = ra / sl_dist
            q = _round_qty(q)
            return q, q * price if q > 0 else (0.0, 0.0)

        def _too_big(qty_val: float, notional_val: float) -> bool:
            if qty_val <= 0 or notional_val <= 0:
                return True
            if max_notional_limit and notional_val > max_notional_limit:
                return True
            # Note: we no longer check max_qty_limit here because _round_qty already clamps
            # Simplified pre-check using planned leverage
            if notional_val > balance * planned_leverage:
                return True
            return False

        def _extract_code_msg(exc: Exception) -> tuple[int | str | None, str]:
            code = getattr(exc, "code", None)
            msg = str(getattr(exc, "message", None) or exc)
            if code is None and hasattr(exc, "args") and exc.args:
                maybe_code = exc.args[0]
                if isinstance(maybe_code, dict):
                    code = maybe_code.get("code", code)
                    msg = maybe_code.get("msg", msg)
                elif isinstance(maybe_code, (int, str)):
                    code = maybe_code
            if hasattr(exc, "response"):
                resp = getattr(exc, "response")
                if isinstance(resp, dict):
                    code = resp.get("code", code)
                    msg = resp.get("msg", msg or resp.get("message", ""))
            return code, msg

        def _is_position_limit_error(code: int | str | None, msg: str) -> bool:
            msg_str = (msg or "").lower()
            return (
                code == -2027
                or str(code) == "-2027"
                or "-2027" in msg_str
                or code == -4005
                or str(code) == "-4005"
                or "quantity greater than max quantity" in msg_str
                or "max quantity" in msg_str
                or "exceeded the maximum allowable position at current leverage" in msg_str
                or ("position" in msg_str and "leverage" in msg_str)
            )

        def _is_insufficient_margin_error(code: int | str | None, msg: str) -> bool:
            msg_str = (msg or "").lower()
            return (
                code == -2019
                or str(code) == "-2019"
                or "insufficient" in msg_str
                or "margin" in msg_str and "not sufficient" in msg_str
            )

        def _is_rate_limit_error(code: int | str | None, msg: str) -> bool:
            msg_l = (msg or "").lower()
            return code == -1000 or "429" in msg_l or "rate limit" in msg_l

        # ═══════════════════════════════════════════════════════════
        #  STEP 4: Build risk steps (down to 0.05%)
        # ═══════════════════════════════════════════════════════════
        ABSOLUTE_MIN_RISK = 0.0005  # 0.05% - absolute floor
        risk_steps: list[float] = []
        seen_steps: set[float] = set()

        # Start from dynamic_risk, step down in 0.1% increments to 0.1%
        current = dynamic_risk
        while current >= 0.001:  # 0.1%
            rounded = round(current, 4)
            if rounded not in seen_steps and rounded >= ABSOLUTE_MIN_RISK:
                risk_steps.append(rounded)
                seen_steps.add(rounded)
            current -= 0.001  # -0.1%

        # Continue with finer 0.05% steps below 0.1%
        for fine_pct in [0.00075, 0.0005]:
            rounded = round(fine_pct, 5)
            if rounded not in seen_steps:
                risk_steps.append(rounded)
                seen_steps.add(rounded)

        # Ensure we at least have the dynamic_risk itself
        if not risk_steps:
            risk_steps = [dynamic_risk]
        risk_steps = sorted(risk_steps, reverse=True)

        total_steps = len(risk_steps)
        min_risk_pct = risk_steps[-1] if risk_steps else ABSOLUTE_MIN_RISK
        last_qty = 0.0
        last_risk = self.risk_pct
        leverage_already_set = False

        def _log_final_skip() -> None:
            self.logger.error(
                "FINAL SKIP: Even %.2f%% risk + %dx leverage not possible for %s – position limits too tight",
                min_risk_pct * 100, planned_leverage, symbol,
            )

        # ═══════════════════════════════════════════════════════════
        #  STEP 5: Order loop with robust retry
        # ═══════════════════════════════════════════════════════════
        for idx, risk_pct in enumerate(risk_steps):
            attempt_num = idx + 1
            qty, notional = _calc_qty(risk_pct)
            last_qty = qty
            last_risk = risk_pct
            expected_margin = notional / planned_leverage if planned_leverage > 0 else notional

            # Skip if qty is zero (below min or clamped to nothing)
            if qty <= 0 or notional <= 0:
                if attempt_num == total_steps:
                    _log_final_skip()
                    return None, None, None, qty, risk_pct, planned_leverage
                continue

            self.logger.info(
                "[ORDER_ATTEMPT #%d] %s %s | target risk=%.2f%% used risk=%.2f%% | "
                "leverage=%dx (%s) | notional=%.2f qty=%.6f SLdist=%.6f expected_margin=%.2f max_qty=%s",
                attempt_num, direction.upper(), symbol,
                ORIGINAL_RISK_PCT * 100, risk_pct * 100,
                planned_leverage, leverage_source,
                notional, qty, sl_dist, expected_margin,
                max_qty_limit,
            )

            if _too_big(qty, notional):
                if attempt_num == total_steps:
                    _log_final_skip()
                    return None, None, None, qty, risk_pct, planned_leverage
                self.logger.info("[ORDER_SKIP] Notional %.2f too large for leverage %dx, trying lower risk...", notional, planned_leverage)
                continue

            # Ensure cross margin mode (shared margin pool across positions)
            if not leverage_already_set:
                try:
                    await self.adapter.set_margin_mode("cross", symbol)
                except Exception:
                    pass  # already set or position open – both fine

            # Set leverage (may need to re-set if reduced during retries)
            if not leverage_already_set:
                try:
                    await self.adapter.set_leverage(planned_leverage, symbol)
                    self.leverage = planned_leverage
                    leverage_already_set = True
                except Exception as exc:
                    self.logger.warning(
                        "set_leverage(%dx) failed for %s: %s (keeping %dx)",
                        planned_leverage, symbol, exc, self.leverage,
                    )

            rate_limit_retries = 0
            leverage_retries = 0
            while rate_limit_retries <= 2 and leverage_retries <= 6:
                try:
                    order_id, sl_order_id, tp_order_id = await self._place_bracket_order(
                        symbol, direction, price, sl, tp, qty,
                    )
                    self.logger.info(
                        "[ORDER_SUCCESS] %s %s | risk_used=%.2f%% leverage=%dx (%s) "
                        "qty=%.6f notional=%.2f expected_margin=%.2f order=%s",
                        direction.upper(), symbol,
                        risk_pct * 100, planned_leverage, leverage_source,
                        qty, notional, expected_margin,
                        order_id or "no-exchange",
                    )
                    return (
                        order_id, sl_order_id, tp_order_id,
                        qty, risk_pct, planned_leverage,
                    )
                except Exception as exc:
                    code, msg = _extract_code_msg(exc)
                    self.logger.error(
                        "[ORDER_ATTEMPT #%d FAILURE] %s %s | code=%s msg=%s",
                        attempt_num, direction.upper(), symbol,
                        code if code is not None else "unknown", msg,
                    )
                    if _is_rate_limit_error(code, msg) and rate_limit_retries < 2:
                        rate_limit_retries += 1
                        await asyncio.sleep(0.5)
                        continue

                    if _is_position_limit_error(code, msg):
                        # ── Leverage step-down strategy ──────────────────
                        # Risk = qty × sl_dist → independent of leverage.
                        # Lower leverage = exchange allows larger position.
                        # Only margin changes: margin = notional / leverage.
                        if planned_leverage > 1:
                            new_leverage = max(1, planned_leverage // 2)
                            new_margin = notional / new_leverage if new_leverage > 0 else notional
                            if new_margin < balance * 0.90:
                                self.logger.info(
                                    "→ Position limit at %dx, stepping down to %dx for %s "
                                    "(margin %.2f → %.2f, qty unchanged=%.0f)",
                                    planned_leverage, new_leverage, symbol,
                                    expected_margin, new_margin, qty,
                                )
                                planned_leverage = new_leverage
                                expected_margin = new_margin
                                leverage_retries += 1
                                try:
                                    await self.adapter.set_leverage(planned_leverage, symbol)
                                    self.leverage = planned_leverage
                                    leverage_already_set = True
                                except Exception as lev_exc:
                                    self.logger.warning(
                                        "set_leverage(%dx) failed for %s: %s",
                                        planned_leverage, symbol, lev_exc,
                                    )
                                continue  # retry with lower leverage, same qty
                            else:
                                self.logger.warning(
                                    "→ Leverage %dx→%dx margin %.2f exceeds 90%% of balance %.2f for %s",
                                    planned_leverage, new_leverage, new_margin, balance, symbol,
                                )
                        # Leverage at 1x or margin too high – cannot fit position
                        self.logger.warning(
                            "→ Position limit: leverage at %dx, cannot fit qty=%.0f notional=%.2f for %s",
                            planned_leverage, qty, notional, symbol,
                        )
                        break  # go to next risk step (last resort)

                    if _is_insufficient_margin_error(code, msg):
                        if attempt_num < total_steps:
                            self.logger.warning(
                                "→ Insufficient margin, reducing risk from %.2f%% to %.2f%% for %s",
                                risk_pct * 100, risk_steps[idx + 1] * 100, symbol,
                            )
                        break  # go to next risk step

                    # Unknown error – also try next risk step
                    self.logger.warning(
                        "→ Unknown order error, trying lower risk for %s", symbol,
                    )
                    break

        _log_final_skip()
        return None, None, None, last_qty, last_risk, planned_leverage

    # ── Real testnet bracket order ────────────────────────────────

    async def _place_bracket_order(
        self,
        symbol: str,
        direction: str,
        price: float,
        sl: float,
        tp: float,
        qty: float,
    ) -> tuple[str | None, str | None, str | None]:
        """
        Place a market entry plus SL/TP orders via the exchange adapter.

        Works for all exchanges (Binance, OANDA, Alpaca) through the
        unified ExchangeAdapter interface.

        Returns (entry_id, sl_order_id, tp_order_id).
        """
        if self.adapter is None:
            return None, None, None

        side = "buy" if direction == "long" else "sell"
        exit_side = "sell" if direction == "long" else "buy"

        # Round SL/TP to exchange price precision
        try:
            sl = float(self.adapter.price_to_precision(symbol, sl))
            tp = float(self.adapter.price_to_precision(symbol, tp))
        except Exception:
            pass

        async def _close_position(reason: str) -> None:
            try:
                close = await self.adapter.create_market_order(
                    symbol, exit_side, qty, {"reduceOnly": True},
                )
                self.logger.warning(
                    "Flattened position after %s | close_order=%s",
                    reason, close.order_id,
                )
            except Exception as close_exc:
                self.logger.error(
                    "Failed to flatten position after %s: %s", reason, close_exc,
                )

        # Entry market order
        try:
            entry = await self.adapter.create_market_order(symbol, side, qty)
            entry_id = entry.order_id
            self.logger.info(
                "ENTRY %s %s qty=%.6f | id=%s",
                side.upper(), symbol, qty, entry_id,
            )
        except Exception as exc:
            self.logger.error("Entry order FAILED %s %s: %s", side.upper(), symbol, exc)
            raise

        # SL order
        sl_order_id: str | None = None
        try:
            sl_order = await self.adapter.create_stop_loss(
                symbol, exit_side, qty, sl,
            )
            sl_order_id = sl_order.order_id
            self.logger.info(
                "SL %s %s qty=%.6f @ %.6f | id=%s",
                exit_side.upper(), symbol, qty, sl, sl_order_id,
            )
        except Exception as exc:
            self.logger.error(
                "SL order FAILED %s %s: %s", exit_side.upper(), symbol, exc,
            )
            await _close_position("SL placement failure")
            raise

        # TP order
        tp_order_id: str | None = None
        try:
            tp_order = await self.adapter.create_take_profit(
                symbol, exit_side, qty, tp,
            )
            tp_order_id = tp_order.order_id
            self.logger.info(
                "TP %s %s qty=%.6f @ %.6f | id=%s",
                exit_side.upper(), symbol, qty, tp, tp_order_id,
            )
        except Exception as exc:
            self.logger.error(
                "TP order FAILED %s %s: %s", exit_side.upper(), symbol, exc,
            )
            if sl_order_id:
                try:
                    await self.adapter.cancel_order(sl_order_id, symbol)
                except Exception as cancel_exc:
                    self.logger.warning(
                        "Failed to cancel SL %s after TP failure: %s",
                        sl_order_id, cancel_exc,
                    )
            await _close_position("TP placement failure")
            raise

        return entry_id, sl_order_id, tp_order_id

    # ── Fetch real testnet balance ────────────────────────────────

    async def _fetch_balance(self) -> float | None:
        """Return the free balance from the exchange adapter."""
        if self.adapter is None:
            return None
        try:
            bal = await self.adapter.fetch_balance()
            return float(bal.free)
        except Exception as exc:
            self.logger.warning("fetch_balance failed: %s", exc)
            return None

    # ── ATR helper ────────────────────────────────────────────────

    @staticmethod
    def _simple_atr(candles: list[dict[str, Any]], period: int = 14) -> float:
        """Average True Range over last *period* candles."""
        if len(candles) < period + 1:
            return 0.0
        trs: list[float] = []
        for i in range(-period, 0):
            h = candles[i]["high"]
            l = candles[i]["low"]
            pc = candles[i - 1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return float(np.mean(trs))

    # ── Volatility check ────────────────────────────────────────────

    def _check_volatility(self) -> tuple[bool, float, float]:
        """
        Check if this coin has enough volatility for SMC trading.

        Returns (is_tradeable, daily_atr_pct, five_m_atr_pct).
        Coins with ATR% below thresholds are too noisy for reliable structure.
        """
        daily_atr_pct = 0.0
        fivem_atr_pct = 0.0

        # Daily ATR check
        if len(self.buffer_1d) >= 15:
            closes = self.buffer_1d["close"].values[-15:].astype(float)
            highs = self.buffer_1d["high"].values[-15:].astype(float)
            lows = self.buffer_1d["low"].values[-15:].astype(float)
            trs = []
            for i in range(1, len(closes)):
                h, l, pc = highs[i], lows[i], closes[i - 1]
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            if trs:
                atr = float(np.mean(trs[-14:])) if len(trs) >= 14 else float(np.mean(trs))
                price = closes[-1]
                daily_atr_pct = atr / price if price > 0 else 0.0

        # 5m ATR check
        if len(self.buffer_5m) >= 15:
            closes = self.buffer_5m["close"].values[-15:].astype(float)
            highs = self.buffer_5m["high"].values[-15:].astype(float)
            lows = self.buffer_5m["low"].values[-15:].astype(float)
            trs = []
            for i in range(1, len(closes)):
                h, l, pc = highs[i], lows[i], closes[i - 1]
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            if trs:
                atr = float(np.mean(trs[-14:])) if len(trs) >= 14 else float(np.mean(trs))
                price = closes[-1]
                fivem_atr_pct = atr / price if price > 0 else 0.0

        self._daily_atr_pct = daily_atr_pct
        self._5m_atr_pct = fivem_atr_pct

        tradeable = daily_atr_pct >= self.min_daily_atr_pct and fivem_atr_pct >= self.min_5m_atr_pct
        return tradeable, daily_atr_pct, fivem_atr_pct

    # ── Find POI (OB/FVG) from pre-computed indicators ───────────

    def _find_poi_from_indicators(
        self,
        indicators: dict[str, Any],
        price: float,
        bias: str,
        lookback_bars: int = 10,
    ) -> dict[str, Any] | None:
        """
        Find the most recent Order Block or FVG aligned with bias.
        Used for 4H primary POI detection.
        """
        # Check Order Blocks first (stronger institutional footprint)
        ob_data = indicators.get("order_blocks")
        if ob_data is not None and not ob_data.empty:
            end = len(ob_data)
            scan_start = max(0, end - lookback_bars)
            for idx in range(end - 1, scan_start - 1, -1):
                row = ob_data.iloc[idx]
                ob_dir = row.get("OB", 0)
                ob_top = row.get("Top", np.nan)
                ob_bottom = row.get("Bottom", np.nan)
                if pd.isna(ob_top) or pd.isna(ob_bottom) or pd.isna(ob_dir) or ob_dir == 0:
                    continue
                if bias == "bullish" and ob_dir > 0:
                    return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom)}
                if bias == "bearish" and ob_dir < 0:
                    return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom)}

        # Check FVGs (secondary)
        fvg_data = indicators.get("fvg")
        if fvg_data is not None and not fvg_data.empty:
            end = len(fvg_data)
            scan_start = max(0, end - lookback_bars)
            for idx in range(end - 1, scan_start - 1, -1):
                row = fvg_data.iloc[idx]
                fvg_dir = row.get("FVG", 0)
                top_val = row.get("Top", np.nan)
                bottom_val = row.get("Bottom", np.nan)
                if pd.isna(top_val) or pd.isna(bottom_val) or pd.isna(fvg_dir) or fvg_dir == 0:
                    continue
                gap_size = abs(top_val - bottom_val) / price if price > 0 else 0
                if gap_size < self.fvg_threshold:
                    continue
                if bias == "bullish" and fvg_dir > 0:
                    return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val)}
                if bias == "bearish" and fvg_dir < 0:
                    return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val)}

        return None

    # ── Trade style classification ────────────────────────────────

    def _classify_trade_style(
        self, price: float, sl: float, tp: float,
    ) -> str:
        """
        Classify trade as scalp, day, or swing based on TP distance.

        This determines which SL/TP constraints apply and prevents
        mixing of tactics (e.g. scalp SL on a swing TP target).
        """
        tp_dist_pct = abs(tp - price) / price if price > 0 else 0
        sl_dist_pct = abs(price - sl) / price if price > 0 else 0

        if tp_dist_pct < 0.005:  # < 0.5% TP
            return STYLE_SCALP
        elif tp_dist_pct < 0.03:  # < 3% TP
            return STYLE_DAY
        else:
            return STYLE_SWING

    def _validate_style_constraints(
        self, style: str, price: float, sl: float, tp: float,
    ) -> bool:
        """
        Validate that SL/TP conform to the detected trade style.
        Prevents mixing (e.g. swing TP with scalp SL).
        """
        cfg = STYLE_CONFIG.get(style)
        if cfg is None:
            return False

        sl_pct = abs(price - sl) / price if price > 0 else 0
        tp_pct = abs(tp - price) / price if price > 0 else 0
        rr = tp_pct / sl_pct if sl_pct > 0 else 0

        if sl_pct < cfg["min_sl_pct"] or sl_pct > cfg["max_sl_pct"]:
            return False
        if tp_pct < cfg["min_tp_pct"] or tp_pct > cfg["max_tp_pct"]:
            return False
        if rr < cfg["min_rr"]:
            return False

        return True

    # ── Setup quality tier classification ─────────────────────────

    def _classify_setup_tier(
        self, score: float, rr: float, components: dict[str, Any],
    ) -> str:
        """
        Classify setup quality: AAA++ (sniper) or AAA+ (strong fallback).

        AAA++ requires ALL components aligned – the absolute best setups only.
        AAA+ requires core SMC alignment + strong bias.
        Per-class skip_flags allow tick-volume classes to bypass unreliable gates.

        No A or SPEC tiers – only high-probability trades.
        """
        skip = self._tier_skip_flags

        def _flag(name: str) -> bool:
            """Check flag, returning True (pass) if flag is in skip list."""
            if name in skip:
                return True
            return bool(components.get(name, False))

        # AAA++: Sniper setup – every single component must be True
        t = TIER_THRESHOLDS[TIER_AAA_PLUS_PLUS]
        if (score >= t["min_score"]
                and rr >= t["min_rr"]
                and _flag("bias_strong")
                and _flag("h4_confirms")
                and _flag("h4_poi")
                and _flag("h1_confirms")
                and _flag("h1_choch")
                and (components.get("entry_zone") is not None or "entry_zone" in skip)
                and _flag("precision_trigger")
                and _flag("volume_ok")
                and _flag("adx_strong")
                and _flag("session_optimal")
                and _flag("zone_quality_ok")
                and _flag("momentum_confluent")
                and (components.get("tf_agreement", 0) >= 4 or "tf_agreement" in skip)):
            return TIER_AAA_PLUS_PLUS

        # AAA+: Premium setup – core SMC alignment + strong bias
        t = TIER_THRESHOLDS[TIER_AAA_PLUS]
        if (score >= t["min_score"]
                and rr >= t["min_rr"]
                and _flag("bias_strong")
                and _flag("h4_confirms")
                and _flag("h1_confirms")
                and _flag("precision_trigger")
                and _flag("volume_ok")
                and _flag("adx_strong")):
            return TIER_AAA_PLUS

        return ""  # Don't trade – quality too low

    # ── Style-aware SL/TP from multiple timeframes ───────────────

    def _find_sl_from_buffer(
        self, buffer: pd.DataFrame, price: float, direction: str,
    ) -> float | None:
        """Extract SL from a given timeframe's SMC indicators."""
        swing_len = self.swing_length
        fvg_thresh = self.fvg_threshold
        ob_lookback = self.ob_lookback
        liq_range = self.liq_range

        if len(buffer) < swing_len * 2:
            return None

        try:
            ind = compute_smc_indicators(buffer, swing_len, fvg_thresh, ob_lookback, liq_range)
        except Exception:
            return None

        ob_data = ind.get("order_blocks")
        liq_data = ind.get("liquidity")
        sl: float | None = None

        if direction == "long":
            if ob_data is not None and not ob_data.empty:
                for i in range(len(ob_data) - 1, -1, -1):
                    row = ob_data.iloc[i]
                    d = row.get("OB", np.nan)
                    bot = row.get("Bottom", np.nan)
                    if pd.notna(d) and d > 0 and pd.notna(bot) and bot < price:
                        sl = float(bot)
                        break
            if sl is None and liq_data is not None and not liq_data.empty:
                best: float | None = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl < price:
                        if best is None or lvl > best:
                            best = lvl
                if best is not None:
                    sl = float(best)
        else:
            if ob_data is not None and not ob_data.empty:
                for i in range(len(ob_data) - 1, -1, -1):
                    row = ob_data.iloc[i]
                    d = row.get("OB", np.nan)
                    top = row.get("Top", np.nan)
                    if pd.notna(d) and d < 0 and pd.notna(top) and top > price:
                        sl = float(top)
                        break
            if sl is None and liq_data is not None and not liq_data.empty:
                best = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl > price:
                        if best is None or lvl < best:
                            best = lvl
                if best is not None:
                    sl = float(best)

        return sl

    def _find_tp_from_buffer(
        self, buffer: pd.DataFrame, price: float, direction: str,
    ) -> float | None:
        """Extract TP from a given timeframe's SMC indicators."""
        swing_len = self.swing_length
        fvg_thresh = self.fvg_threshold
        ob_lookback = self.ob_lookback
        liq_range = self.liq_range

        if len(buffer) < swing_len * 2:
            return None

        try:
            ind = compute_smc_indicators(buffer, swing_len, fvg_thresh, ob_lookback, liq_range)
        except Exception:
            return None

        liq_data = ind.get("liquidity")
        fvg_data = ind.get("fvg")
        tp: float | None = None

        if direction == "long":
            if liq_data is not None and not liq_data.empty:
                best: float | None = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl > price:
                        if best is None or lvl < best:
                            best = lvl
                if best is not None:
                    tp = float(best)
            if tp is None and fvg_data is not None and not fvg_data.empty:
                best = None
                for i in range(len(fvg_data)):
                    row = fvg_data.iloc[i]
                    d = row.get("FVG", np.nan)
                    bot = row.get("Bottom", np.nan)
                    if pd.notna(d) and d < 0 and pd.notna(bot) and bot > price:
                        if best is None or bot < best:
                            best = bot
                if best is not None:
                    tp = float(best)
        else:
            if liq_data is not None and not liq_data.empty:
                best = None
                for i in range(len(liq_data)):
                    lvl = liq_data["Level"].iat[i]
                    if pd.notna(lvl) and lvl < price:
                        if best is None or lvl > best:
                            best = lvl
                if best is not None:
                    tp = float(best)
            if tp is None and fvg_data is not None and not fvg_data.empty:
                best = None
                for i in range(len(fvg_data)):
                    row = fvg_data.iloc[i]
                    d = row.get("FVG", np.nan)
                    top_val = row.get("Top", np.nan)
                    if pd.notna(d) and d > 0 and pd.notna(top_val) and top_val < price:
                        if best is None or top_val > best:
                            best = top_val
                if best is not None:
                    tp = float(best)

        return tp

    def _find_smc_sl_tp_for_style(
        self, price: float, direction: str, style: str,
    ) -> tuple[float, float] | None:
        """
        Compute SL and TP using timeframes appropriate for the trade style.

        SCALP  → SL from 15m/5m,  TP from 1H/15m
        DAY    → SL from 1H/15m,  TP from 4H/1H
        SWING  → SL from 4H/1H,   TP from 1D/4H
        """
        if style == STYLE_SWING:
            sl_buffers = [self.buffer_4h, self.buffer_1h]
            tp_buffers = [self.buffer_1d, self.buffer_4h]
        elif style == STYLE_SCALP:
            sl_buffers = [self.buffer_15m, self.buffer_5m]
            tp_buffers = [self.buffer_1h, self.buffer_15m]
        else:  # DAY
            sl_buffers = [self.buffer_1h, self.buffer_15m]
            tp_buffers = [self.buffer_4h, self.buffer_1h]

        sl: float | None = None
        tp: float | None = None

        for buf in sl_buffers:
            sl = self._find_sl_from_buffer(buf, price, direction)
            if sl is not None:
                break

        for buf in tp_buffers:
            tp = self._find_tp_from_buffer(buf, price, direction)
            if tp is not None:
                break

        # Fallback to original 5m-based method
        if sl is None or tp is None:
            fallback = self._find_smc_sl_tp(price, direction)
            if fallback is None:
                return None
            if sl is None:
                sl = fallback[0]
            if tp is None:
                tp = fallback[1]

        return (sl, tp)

    @staticmethod
    def _step_mult(val: float, bands: list[tuple[float, float]]) -> float:
        """
        Return multiplier for *val* based on descending ``(threshold, multiplier)`` bands.

        :param val: input value to compare against thresholds.
        :param bands: list of (threshold, multiplier) sorted in descending threshold order; ordering is required for first-match semantics.
        :returns: first matching multiplier or 1.0 when no threshold matches.
        """
        for threshold, mult in bands:
            if val >= threshold:
                return mult
        return 1.0

    @staticmethod
    def _extract_initial_leverage(
        brackets: Any,
        logger: logging.Logger | None = None,
        max_depth: int = 6,
    ) -> list[int]:
        """
        Collect positive initialLeverage values from nested leverage bracket payloads.

        :param brackets: dict or list structure containing leverage bracket data (may nest via ``brackets`` key).
        :param logger: optional logger for debug parsing errors.
        :param max_depth: recursion depth guard to avoid runaway nesting.
        :returns: list of positive integer leverage values extracted.
        """
        values: list[int] = []
        if max_depth <= 0:
            return values
        if isinstance(brackets, dict):
            values.extend(PaperBot._extract_initial_leverage(brackets.get("brackets", []), logger, max_depth - 1))
            if "initialLeverage" in brackets:
                    raw_val = brackets.get("initialLeverage")
                    if raw_val is not None:
                        try:
                            ival = int(raw_val)
                            if ival > 0:
                                values.append(ival)
                        except Exception as exc:
                            if logger:
                                logger.debug("Failed to parse initialLeverage from bracket dict: %s", exc)
        elif isinstance(brackets, list):
            for item in brackets:
                if isinstance(item, dict):
                    if "initialLeverage" in item:
                        raw_val = item.get("initialLeverage")
                        if raw_val is not None:
                            try:
                                ival = int(raw_val)
                                if ival > 0:
                                    values.append(ival)
                            except Exception as exc:
                                if logger:
                                    logger.debug("Failed to parse initialLeverage from bracket item: %s", exc)
                                continue
                    values.extend(PaperBot._extract_initial_leverage(item.get("brackets", []), logger, max_depth - 1))
        return [v for v in values if v > 0]

    # ── Summary helpers ───────────────────────────────────────────

    @property
    def winrate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    @property
    def return_pct(self) -> float:
        if self._account_equity <= 0:
            return 0.0
        return self.total_pnl / self._account_equity * 100

    def summary_dict(self) -> dict[str, Any]:
        return {
            "bot": self.tag,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "pnl": round(self.total_pnl, 2),
            "return_pct": round(self.return_pct, 2),
            "trades": self.trades,
            "winrate": round(self.winrate * 100, 1),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "open_pos": len(self._active_trades),
        }


# ═══════════════════════════════════════════════════════════════════
#  Exchange helper
# ═══════════════════════════════════════════════════════════════════

async def create_adapters(config: dict[str, Any]) -> dict[str, ExchangeAdapter]:
    """Create and connect all available exchange adapters.

    Returns dict mapping asset_class → adapter.
    Skips adapters whose API keys are missing or packages not installed.
    OANDA handles both 'forex' and 'commodities' (same adapter instance).
    """
    adapters: dict[str, ExchangeAdapter] = {}

    # ── Binance (crypto) ────────────────────────────────────────
    bk = os.getenv("BINANCE_API_KEY", "")
    bs = os.getenv("BINANCE_SECRET", "")
    if bk and bs:
        try:
            adapter = BinanceAdapter(api_key=bk, api_secret=bs, testnet=True)
            await adapter.connect()
            adapters["crypto"] = adapter
            logger.info("Binance (crypto): connected ✓")
        except Exception as exc:
            logger.warning("Binance connect failed: %s", exc)
    else:
        logger.warning("BINANCE keys missing — crypto disabled")

    # ── OANDA (forex + commodities) ─────────────────────────────
    ot = os.getenv("OANDA_ACCESS_TOKEN", "")
    oa = os.getenv("OANDA_ACCOUNT_ID", "")
    if ot and oa:
        try:
            from exchanges.oanda_adapter import OandaAdapter
            adapter = OandaAdapter(
                account_id=oa, access_token=ot, environment="practice",
            )
            await adapter.connect()
            adapters["forex"] = adapter
            adapters["commodities"] = adapter  # same instance
            logger.info("OANDA (forex+commodities): connected ✓")
        except ImportError:
            logger.warning("v20 not installed — forex/commodities disabled (pip install v20)")
        except Exception as exc:
            logger.warning("OANDA connect failed: %s", exc)
    else:
        logger.warning("OANDA keys missing — forex/commodities disabled")

    # ── Alpaca (stocks) ─────────────────────────────────────────
    ak = os.getenv("ALPACA_API_KEY", "")
    as_ = os.getenv("ALPACA_API_SECRET", "")
    if ak and as_:
        try:
            from exchanges.alpaca_adapter import AlpacaAdapter
            adapter = AlpacaAdapter(
                api_key=ak, secret_key=as_, paper=True,
            )
            await adapter.connect()
            adapters["stocks"] = adapter
            logger.info("Alpaca (stocks): connected ✓")
        except ImportError:
            logger.warning("alpaca-py not installed — stocks disabled (pip install alpaca-py)")
        except Exception as exc:
            logger.warning("Alpaca connect failed: %s", exc)
    else:
        logger.warning("ALPACA keys missing — stocks disabled")

    return adapters


# ═══════════════════════════════════════════════════════════════════
#  Rich Dashboard Builder
# ═══════════════════════════════════════════════════════════════════

def _pnl_color(value: float) -> str:
    """Return rich markup colour for a PnL value."""
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "white"


def _format_uptime(start: datetime) -> str:
    """Human-readable uptime string."""
    delta = datetime.now(timezone.utc) - start
    total_sec = int(delta.total_seconds())
    hours, remainder = divmod(total_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def _build_bot_table(
    title: str,
    rows: list[dict[str, Any]],
    style: str,
) -> Table:
    """Build a Rich Table for a set of bot summaries."""
    table = Table(
        title=title,
        title_style=f"bold {style}",
        border_style=style,
        show_lines=False,
        expand=True,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Bot-ID", style="cyan", width=9)
    table.add_column("Symbol", style="bright_yellow", width=18)
    table.add_column("Class", style="dim", width=7)
    table.add_column("PnL", justify="right", width=14)
    table.add_column("Return%", justify="right", width=10)
    table.add_column("Trades", justify="right", width=10)
    table.add_column("Winrate", justify="right", width=10)
    table.add_column("DD%", justify="right", width=9)
    table.add_column("Open", justify="right", width=7)

    for i, r in enumerate(rows, 1):
        pnl_c = _pnl_color(r["pnl"])
        ret_c = _pnl_color(r["return_pct"])
        table.add_row(
            str(i),
            r["bot"],
            r.get("symbol", ""),
            r.get("asset_class", "crypto")[:6],
            f"[{pnl_c}]{r['pnl']:+,.2f}[/{pnl_c}]",
            f"[{ret_c}]{r['return_pct']:+.2f}%[/{ret_c}]",
            str(r["trades"]),
            f"{r['winrate']:.1f}%",
            f"{r['drawdown_pct']:.2f}%",
            str(r["open_pos"]),
        )

    return table


def build_dashboard(
    bots: list[PaperBot],
    ws_status: dict[str, str],
    start_time: datetime,
    active_symbols: list[str],
    total_equity: float = 0.0,
    paper_grid: PaperGrid | None = None,
) -> Layout:
    """
    Build the complete Rich Layout for the live dashboard.

    Returns a Layout containing:
      - Header panel (title, total equity, uptime)
      - Top 20 bots table
      - Worst 20 bots table
      - WebSocket status panel
    """
    all_summaries = sorted(
        [b.summary_dict() for b in bots],
        key=lambda r: r["pnl"],
        reverse=True,
    )

    total_pnl = sum(b.total_pnl for b in bots)
    total_trades = sum(b.trades for b in bots)
    uptime = _format_uptime(start_time)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Header ────────────────────────────────────────────────────
    # Per-class bot counts
    class_counts: dict[str, int] = {}
    for b in bots:
        class_counts[b.asset_class] = class_counts.get(b.asset_class, 0) + 1
    class_str = " | ".join(f"{ac}: {cnt}" for ac, cnt in sorted(class_counts.items()))

    eq_color = _pnl_color(total_pnl)
    header_text = Text.from_markup(
        f"[bold cyan]📊 SMC MULTI-ASSET LIVE TRADING DASHBOARD[/bold cyan]\n"
        f"[dim]{now_str}[/dim]  ·  Uptime: [bold]{uptime}[/bold]  ·  "
        f"Bots: [bold]{len(bots)}[/bold] ({class_str})\n"
        f"Total Equity: [bold]{total_equity:,.2f}[/bold]  ·  "
        f"Total PnL: [bold {eq_color}]{total_pnl:+,.2f}[/bold {eq_color}]  ·  "
        f"Total Trades: [bold]{total_trades}[/bold]",
    )
    header_panel = Panel(
        header_text,
        title="[bold white]HEADER[/bold white]",
        border_style="bright_blue",
    )

    # ── Bot tables ────────────────────────────────────────────────
    top_20 = all_summaries[:20]
    worst_20 = list(reversed(all_summaries[-20:])) if len(all_summaries) > 20 else list(reversed(all_summaries))

    top_table = _build_bot_table("🏆  TOP 20 BOTS", top_20, "green")
    worst_table = _build_bot_table("📉  WORST 20 BOTS", worst_20, "red")

    # ── WebSocket Status ──────────────────────────────────────────
    # Determine global status
    statuses = list(ws_status.values())
    n_connected = statuses.count("connected")
    n_reconnecting = sum(1 for s in statuses if s.startswith("reconnecting"))
    n_disconnected = statuses.count("disconnected")

    if n_disconnected > 0:
        global_label = f"[bold red]⛔ DISCONNECTED ({n_disconnected})[/bold red]"
    elif n_reconnecting > 0:
        global_label = f"[bold yellow]🔄 RECONNECTING ({n_reconnecting})[/bold yellow]"
    else:
        global_label = f"[bold green]✅ ALL CONNECTED ({n_connected})[/bold green]"

    ws_lines = [f"Global: {global_label}\n"]

    # Group status display
    sorted_keys = sorted(ws_status.keys())
    groups: dict[int, list[str]] = {}
    for i, sym in enumerate(sorted_keys):
        idx = i // WS_GROUP_SIZE
        groups.setdefault(idx, []).append(sym)

    for gid, syms in sorted(groups.items()):
        group_statuses = [ws_status.get(s, "unknown") for s in syms]
        gc = sum(1 for gs in group_statuses if gs == "connected")
        gr = sum(1 for gs in group_statuses if gs.startswith("reconnecting"))
        gd = sum(1 for gs in group_statuses if gs == "disconnected")

        if gd > 0:
            gs_label = f"[red]⛔ {gc}/{len(syms)} connected, {gd} disconnected[/red]"
        elif gr > 0:
            gs_label = f"[yellow]🔄 {gc}/{len(syms)} connected, {gr} reconnecting[/yellow]"
        else:
            gs_label = f"[green]✅ {gc}/{len(syms)} connected[/green]"

        ws_lines.append(f"  Group {gid + 1} ({len(syms)} symbols): {gs_label}")

    ws_panel = Panel(
        Text.from_markup("\n".join(ws_lines)),
        title="[bold white]WEBSOCKET STATUS[/bold white]",
        border_style="bright_blue",
    )

    # ── Paper Grid panel ──────────────────────────────────────────
    grid_panel = None
    if paper_grid is not None:
        grid_rows = paper_grid.dashboard_data()
        grid_table = Table(title="Paper Grid (Top 10 Variants)", expand=True)
        grid_table.add_column("Variant", style="cyan", no_wrap=True)
        grid_table.add_column("PnL", justify="right")
        grid_table.add_column("PnL%", justify="right")
        grid_table.add_column("DD%", justify="right")
        grid_table.add_column("Trades", justify="right")
        grid_table.add_column("WR%", justify="right")
        grid_table.add_column("PF", justify="right")
        grid_table.add_column("BE%", justify="right")
        grid_table.add_column("Open", justify="right")
        grid_table.add_column("Params", style="dim")

        for row in grid_rows[:10]:
            pnl_c = "green" if row["pnl"] >= 0 else "red"
            dd_c = "red" if row["dd_pct"] < -5 else "yellow" if row["dd_pct"] < -2 else "green"
            grid_table.add_row(
                row["name"],
                f"[{pnl_c}]${row['pnl']:+,.0f}[/{pnl_c}]",
                f"[{pnl_c}]{row['pnl_pct']:+.1f}%[/{pnl_c}]",
                f"[{dd_c}]{row['dd_pct']:.1f}%[/{dd_c}]",
                str(row["trades"]),
                f"{row['wr_real']:.0f}%" if row["trades"] > 0 else "-",
                f"{row['pf_real']:.1f}" if row["trades"] > 0 else "-",
                f"{row['be_rate']:.0f}%" if row["trades"] > 0 else "-",
                str(row["open"]),
                f"A={row['align']:.2f} RR={row['rr']:.1f} L={row['lev']} R={row['risk']:.1f}%",
            )

        grid_panel = Panel(grid_table, border_style="bright_magenta")

    # ── Compose Layout ────────────────────────────────────────────
    layout = Layout()
    if grid_panel is not None:
        layout.split_column(
            Layout(header_panel, name="header", size=6),
            Layout(name="tables"),
            Layout(grid_panel, name="grid", size=14),
            Layout(ws_panel, name="status", size=3 + len(groups) + 2),
        )
    else:
        layout.split_column(
            Layout(header_panel, name="header", size=6),
            Layout(name="tables"),
            Layout(ws_panel, name="status", size=3 + len(groups) + 2),
        )
    layout["tables"].split_row(
        Layout(top_table, name="top"),
        Layout(worst_table, name="worst"),
    )

    return layout


# ═══════════════════════════════════════════════════════════════════
#  Main runner
# ═══════════════════════════════════════════════════════════════════

class LiveMultiBotRunner:
    """
    Orchestrates multi-asset PaperBot instances (112 instruments) with:
      - 3 exchange adapters: Binance (crypto), OANDA (forex+commodities), Alpaca (stocks)
      - WebSocket for crypto, REST polling for forex/stocks/commodities
      - Real bracket orders via adapter interface
      - Position polling per adapter (detects TP/SL fills)
      - Rich Live Dashboard grouped by asset class
      - Central shared RL brain
      - Circuit breaker for portfolio-level risk management
      - Paper Grid multi-variant A/B testing

    Each bot trades only its assigned instrument.
    """

    def __init__(
        self,
        bots: list[PaperBot],
        adapters: dict[str, ExchangeAdapter],
    ) -> None:
        self.bots = bots
        self.adapters = adapters
        # Crypto adapter for WebSocket feeds
        self._crypto_adapter = adapters.get("crypto")
        self.brain = None  # Legacy PPO brain, no longer used

        # Build lookups
        self._symbol_to_bot: dict[str, PaperBot] = {
            b.symbol: b for b in bots
        }
        self.symbols: list[str] = [b.symbol for b in bots]
        self._shutdown = asyncio.Event()
        self._start_time = datetime.now(timezone.utc)

        # Group bots by asset class
        from collections import defaultdict
        self._bots_by_class: dict[str, list[PaperBot]] = defaultdict(list)
        for bot in bots:
            self._bots_by_class[bot.asset_class].append(bot)

        # ── Circuit Breaker (shared across all bots) ───────────────
        self.circuit_breaker = CircuitBreaker()
        for bot in self.bots:
            bot.circuit_breaker = self.circuit_breaker

        # ── Opportunity Ranker + Capital Allocator (Phase B) ───────
        self.ranker = OpportunityRanker(max_opportunities=10)
        self.allocator = CapitalAllocator()

        # ── Paper Grid (Multi-Variant A/B Testing) ───────────────
        self.paper_grid = PaperGrid()
        self.paper_grid.load_state()  # resume from crash
        for bot in self.bots:
            bot.paper_grid = self.paper_grid

        # ── Trade Journal (lifecycle logger for ML training data) ─
        self.journal = TradeJournal("trade_journal/journal.db")
        for bot in self.bots:
            bot.journal = self.journal

        # Feed status per symbol: connected | reconnecting_N | disconnected | polling
        self.ws_status: dict[str, str] = {}
        for bot in bots:
            self.ws_status[bot.symbol] = "polling" if bot.asset_class != "crypto" else "connecting"

        # Active watcher tasks keyed by symbol
        self._watcher_tasks: dict[str, asyncio.Task[None]] = {}
        self._ticker_tasks: dict[str, asyncio.Task[None]] = {}

        # ── Candle tracking + watchdog state ──────────────────────
        self._last_candle_ts: dict[str, float] = {}  # symbol -> time.time()
        self._candles_by_class: dict[str, int] = {ac: 0 for ac in ["crypto", "forex", "stocks", "commodities"]}
        self._candles_since_heartbeat: dict[str, int] = {ac: 0 for ac in ["crypto", "forex", "stocks", "commodities"]}
        self._near_misses_total: int = 0
        self._symbol_restart_times: dict[str, list[float]] = {}  # symbol -> timestamps of watchdog restarts
        self._rest_fallback_symbols: set[str] = set()  # symbols degraded from WS to REST
        self._last_status_write: float = 0.0  # debounce heartbeat.json writes

    # ── WebSocket OHLCV watcher with auto-reconnect ───────────────

    async def _watch_symbol(self, symbol: str, stagger_delay: float = 0.0) -> None:
        """
        Subscribe to 5 m OHLCV candles for *symbol* and feed the
        assigned bot only.
        Auto-reconnects up to WS_MAX_RECONNECT times with exponential backoff + jitter.
        """
        bot = self._symbol_to_bot.get(symbol)
        if bot is None:
            return

        # Stagger startup to avoid thundering herd on ccxt.pro WebSocket
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)

        last_ts: int | None = None
        reconnect_count = 0

        while not self._shutdown.is_set():
            try:
                self.ws_status[symbol] = "subscribing"
                reconnect_count = 0  # reset on successful connection

                while not self._shutdown.is_set():
                    try:
                        ohlcv_list = await asyncio.wait_for(
                            self._crypto_adapter.watch_ohlcv(symbol, "5m"),
                            timeout=120.0,  # 2 min max — force reconnect if hung
                        )
                    except asyncio.TimeoutError:
                        bot.logger.warning("WS OHLCV timeout %s — forcing reconnect", symbol)
                        break  # break inner loop → triggers reconnect logic

                    if not ohlcv_list:
                        continue

                    for row in ohlcv_list:
                        ts = int(row[0])
                        if last_ts is not None and ts <= last_ts:
                            continue
                        last_ts = ts

                        candle = {
                            "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "volume": float(row[5]),
                        }

                        try:
                            bot.on_candle(symbol, candle)
                            # Track candle for heartbeat/watchdog
                            self._last_candle_ts[symbol] = time.time()
                            self._candles_by_class["crypto"] += 1
                            self._candles_since_heartbeat["crypto"] += 1
                            # Mark connected after first successful candle
                            if self.ws_status.get(symbol) != "connected":
                                self.ws_status[symbol] = "connected"
                                logger.info("WS %s: first candle received — connected", symbol)
                            # Debounced status file write (max every 30s)
                            now_t = time.time()
                            if now_t - self._last_status_write > 30:
                                self._last_status_write = now_t
                                self._write_heartbeat_status()
                        except Exception as exc:
                            bot.logger.error(
                                "Error processing candle for %s: %s", symbol, exc
                            )

            except asyncio.CancelledError:
                self.ws_status[symbol] = "disconnected"
                return

            except Exception as exc:
                reconnect_count += 1
                if reconnect_count > WS_MAX_RECONNECT:
                    self.ws_status[symbol] = "disconnected"
                    logger.warning(
                        "⚠ %s: max reconnect attempts (%d) exceeded: %s",
                        symbol, WS_MAX_RECONNECT, exc,
                    )
                    return

                # Exponential backoff + random jitter to prevent reconnect thundering herd
                delay = min(
                    WS_RECONNECT_BASE_DELAY * (2 ** (reconnect_count - 1)),
                    60,
                ) + random.uniform(0, 5)
                self.ws_status[symbol] = f"reconnecting_{reconnect_count}"
                logger.warning(
                    "🔄 %s: reconnect attempt %d/%d in %.1fs: %s",
                    symbol, reconnect_count, WS_MAX_RECONNECT, delay, exc,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=delay
                    )
                    self.ws_status[symbol] = "disconnected"
                    return  # shutdown during reconnect wait
                except asyncio.TimeoutError:
                    pass  # continue reconnect loop

    # ── WebSocket ticker watcher for real-time entry ──────────────

    async def _watch_ticker(self, symbol: str, stagger_delay: float = 0.0) -> None:
        """
        Subscribe to live ticker updates for *symbol* and feed the
        assigned bot's ``on_tick`` for real-time entry decisions.

        Uses the same reconnect logic as ``_watch_symbol``.
        """
        bot = self._symbol_to_bot.get(symbol)
        if bot is None:
            return

        # Stagger startup (paired with _watch_symbol for same symbol)
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)

        reconnect_count = 0

        while not self._shutdown.is_set():
            try:
                reconnect_count = 0
                while not self._shutdown.is_set():
                    try:
                        ticker = await asyncio.wait_for(
                            self._crypto_adapter.watch_ticker(symbol),
                            timeout=60.0,  # 1 min — tickers should arrive every few seconds
                        )
                    except asyncio.TimeoutError:
                        bot.logger.warning("WS ticker timeout %s — forcing reconnect", symbol)
                        break  # triggers reconnect
                    if ticker is None:
                        continue
                    last_price = ticker.get("last")
                    if last_price is None:
                        continue
                    try:
                        await bot.on_tick(symbol, float(last_price))
                    except Exception as exc:
                        bot.logger.error(
                            "Error processing tick for %s: %s", symbol, exc,
                        )

            except asyncio.CancelledError:
                return

            except Exception as exc:
                reconnect_count += 1
                if reconnect_count > WS_MAX_RECONNECT:
                    logger.warning(
                        "⚠ Ticker %s: max reconnect (%d) exceeded: %s",
                        symbol, WS_MAX_RECONNECT, exc,
                    )
                    return

                # Exponential backoff + jitter (same as _watch_symbol)
                delay = min(
                    WS_RECONNECT_BASE_DELAY * (2 ** (reconnect_count - 1)),
                    60,
                ) + random.uniform(0, 5)
                logger.warning(
                    "🔄 Ticker %s: reconnect %d/%d in %.1fs: %s",
                    symbol, reconnect_count, WS_MAX_RECONNECT, delay, exc,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=delay,
                    )
                    return
                except asyncio.TimeoutError:
                    pass

    # ── REST polling for non-WebSocket adapters (OANDA, Alpaca) ────

    async def _poll_candles(self, bot: PaperBot, stagger_sec: float = 0.0) -> None:
        """REST-based 5m candle polling for OANDA/Alpaca bots."""
        if stagger_sec > 0:
            await asyncio.sleep(stagger_sec)
        last_ts: int | None = None
        while not self._shutdown.is_set():
            # Skip polling when market is closed (saves API calls, avoids rate limits)
            try:
                if not await bot.adapter.is_market_open(bot.symbol):
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=300)  # check again in 5 min
                        return
                    except asyncio.TimeoutError:
                        continue
            except Exception:
                pass  # assume open if check fails
            try:
                candles = await bot.adapter.fetch_ohlcv(bot.symbol, "5m", limit=2)
                if candles:
                    for row in candles:
                        ts = int(row[0])
                        if last_ts is not None and ts <= last_ts:
                            continue
                        last_ts = ts
                        candle = {
                            "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "volume": float(row[5]),
                        }
                        try:
                            bot.on_candle(bot.symbol, candle)
                            # Track candle for heartbeat/watchdog
                            self._last_candle_ts[bot.symbol] = time.time()
                            ac = bot.asset_class
                            self._candles_by_class[ac] = self._candles_by_class.get(ac, 0) + 1
                            self._candles_since_heartbeat[ac] = self._candles_since_heartbeat.get(ac, 0) + 1
                            # Debounced status file write (max every 30s)
                            now_t = time.time()
                            if now_t - self._last_status_write > 30:
                                self._last_status_write = now_t
                                self._write_heartbeat_status()
                        except Exception as exc:
                            bot.logger.error("Poll candle error %s: %s", bot.symbol, exc)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                bot.logger.error("REST candle poll %s: %s", bot.symbol, exc)
            try:
                interval = REST_POLL_INTERVAL_STOCKS_CANDLE if bot.asset_class == "stocks" else REST_POLL_INTERVAL_SEC
                await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
                return  # shutdown
            except asyncio.TimeoutError:
                pass

    async def _poll_ticker(self, bot: PaperBot, stagger_sec: float = 0.0) -> None:
        """REST-based ticker polling for OANDA/Alpaca bots."""
        if stagger_sec > 0:
            await asyncio.sleep(stagger_sec)
        while not self._shutdown.is_set():
            # Skip polling when market is closed
            try:
                if not await bot.adapter.is_market_open(bot.symbol):
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=300)
                        return
                    except asyncio.TimeoutError:
                        continue
            except Exception:
                pass
            try:
                ticker = await bot.adapter.watch_ticker(bot.symbol)
                if ticker:
                    last_price = ticker.get("last")
                    if last_price is not None:
                        try:
                            await bot.on_tick(bot.symbol, float(last_price))
                        except Exception as exc:
                            bot.logger.error("Poll tick error %s: %s", bot.symbol, exc)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                bot.logger.error("REST ticker poll %s: %s", bot.symbol, exc)
            try:
                interval = REST_POLL_INTERVAL_STOCKS_TICKER if bot.asset_class == "stocks" else 5
                await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass

    # ── Heartbeat + Watchdog ────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Log a HEARTBEAT every 5 minutes with per-class candle counts."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=HEARTBEAT_SEC)
                return  # shutdown
            except asyncio.TimeoutError:
                pass
            # Per-class candle counts since last heartbeat
            counts = " ".join(
                f"{ac}={self._candles_since_heartbeat.get(ac, 0)}"
                for ac in ["crypto", "forex", "stocks", "commodities"]
            )
            totals = " ".join(
                f"{ac}={self._candles_by_class.get(ac, 0)}"
                for ac in ["crypto", "forex", "stocks", "commodities"]
            )
            active_positions = sum(
                1 for b in self.bots if b._active_trades
            )
            total_trades = sum(b.trades for b in self.bots)
            total_pnl = sum(b.total_pnl for b in self.bots)
            rest_fallbacks = len(self._rest_fallback_symbols)
            logger.info(
                "HEARTBEAT: candles_5m=[%s] total=[%s] | trades=%d pnl=%.2f positions=%d | rest_fallbacks=%d",
                counts, totals, total_trades, total_pnl, active_positions, rest_fallbacks,
            )
            # Reset per-heartbeat counters
            for ac in self._candles_since_heartbeat:
                self._candles_since_heartbeat[ac] = 0
            # Write heartbeat status file for dashboard
            self._write_heartbeat_status()

    def _write_heartbeat_status(self) -> None:
        """Write heartbeat.json with real candle timestamps for the dashboard."""
        now = time.time()
        per_class: dict[str, dict[str, Any]] = {}
        for ac in ["crypto", "forex", "stocks", "commodities"]:
            # Find the most recent candle timestamp for this class
            class_symbols = [b.symbol for b in self.bots if b.asset_class == ac]
            latest = 0.0
            for sym in class_symbols:
                ts = self._last_candle_ts.get(sym, 0.0)
                if ts > latest:
                    latest = ts
            per_class[ac] = {
                "last_candle_epoch": latest,
                "last_candle_iso": datetime.fromtimestamp(latest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if latest > 0 else None,
                "candles_total": self._candles_by_class.get(ac, 0),
                "symbols_active": sum(1 for s in class_symbols if self._last_candle_ts.get(s, 0) > now - 600),
                "symbols_total": len(class_symbols),
            }
        status = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_epoch": now,
            "per_class": per_class,
            "rest_fallbacks": list(self._rest_fallback_symbols),
        }
        try:
            status_path = OUTPUT_DIR / "heartbeat.json"
            with open(status_path, "w") as f:
                json.dump(status, f)
        except Exception:
            pass

    async def _watchdog_loop(self) -> None:
        """Monitor per-symbol candle flow, restart stuck symbols, degrade to REST."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=WATCHDOG_SEC)
                return  # shutdown
            except asyncio.TimeoutError:
                pass

            now = time.time()
            stuck_symbols: list[str] = []

            for bot in self.bots:
                sym = bot.symbol
                last = self._last_candle_ts.get(sym)

                # Skip symbols that haven't received any candle yet (still starting up)
                if last is None:
                    # If bot has been running > 5 min with no candle, it's stuck
                    startup_elapsed = now - self._start_time.timestamp()
                    if startup_elapsed < 300:
                        continue

                # Check market hours — only flag as stuck if market is open
                try:
                    if not await bot.adapter.is_market_open(sym):
                        continue
                except Exception:
                    pass  # assume open if check fails

                age = now - (last or 0)
                if age > WATCHDOG_SEC:  # No candle in 10 minutes
                    stuck_symbols.append(sym)

            if not stuck_symbols:
                continue

            logger.warning(
                "WATCHDOG: %d stuck symbols (no candle >%ds): %s",
                len(stuck_symbols), WATCHDOG_SEC,
                ", ".join(stuck_symbols[:10]) + ("..." if len(stuck_symbols) > 10 else ""),
            )

            # Check for total silence across ALL classes → catastrophic failure
            total_recent = sum(self._candles_since_heartbeat.values())
            all_class_total = sum(self._candles_by_class.values())
            if len(stuck_symbols) == len(self.bots) and all_class_total > 0:
                # Had candles before but now ALL are stuck → trigger full restart
                logger.critical(
                    "WATCHDOG CRITICAL: ALL %d symbols stuck — triggering SIGTERM for systemd restart",
                    len(stuck_symbols),
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return

            # Restart individual stuck symbols
            for sym in stuck_symbols:
                bot = self._symbol_to_bot.get(sym)
                if bot is None:
                    continue

                # Only handle crypto WS symbols (REST bots self-recover via polling)
                if bot.asset_class != "crypto":
                    continue

                # Track restart count (within last hour)
                restart_times = self._symbol_restart_times.get(sym, [])
                restart_times = [t for t in restart_times if now - t < 3600]
                restart_times.append(now)
                self._symbol_restart_times[sym] = restart_times

                if len(restart_times) > WATCHDOG_RESTART_LIMIT:
                    # Too many restarts → degrade to REST polling
                    if sym not in self._rest_fallback_symbols:
                        self._rest_fallback_symbols.add(sym)
                        logger.warning(
                            "WATCHDOG: %s failed %d times in 1h — degrading to REST polling",
                            sym, len(restart_times),
                        )
                        # Cancel WS tasks
                        if sym in self._watcher_tasks:
                            self._watcher_tasks[sym].cancel()
                        if sym in self._ticker_tasks:
                            self._ticker_tasks[sym].cancel()
                        # Start REST polling instead
                        self._watcher_tasks[sym] = asyncio.create_task(
                            self._poll_candles(bot, stagger_sec=0)
                        )
                        self._ticker_tasks[sym] = asyncio.create_task(
                            self._poll_ticker(bot, stagger_sec=0)
                        )
                        self.ws_status[sym] = "polling"
                else:
                    # Restart the WS watcher with a stagger
                    logger.info("WATCHDOG: restarting WS watcher for %s (attempt %d)", sym, len(restart_times))
                    if sym in self._watcher_tasks:
                        self._watcher_tasks[sym].cancel()
                    if sym in self._ticker_tasks:
                        self._ticker_tasks[sym].cancel()
                    stagger = random.uniform(0, 5)
                    self._watcher_tasks[sym] = asyncio.create_task(
                        self._watch_symbol(sym, stagger_delay=stagger)
                    )
                    self._ticker_tasks[sym] = asyncio.create_task(
                        self._watch_ticker(sym, stagger_delay=stagger)
                    )

    # ── Exchange position polling (replaces candle-based exit check) ─

    async def _poll_positions(self) -> None:
        """
        Poll exchange positions every POSITION_POLL_SEC seconds.

        Detects when bracket SL/TP has filled by inspecting trades and
        position state, then updates bot stats and RL rewards.
        """

        async def _record_close(bot: PaperBot, trade: dict[str, Any], exit_price: float) -> None:
            entry_price = trade["entry"]
            qty = trade["qty"]
            direction = trade["direction"]
            sl = trade["sl"]

            if direction == "long":
                raw_pnl = (exit_price - entry_price) * qty
            else:
                raw_pnl = (entry_price - exit_price) * qty

            commission = qty * entry_price * bot.commission_rate * COMMISSION_MULTIPLIER
            net_pnl = raw_pnl - commission

            pnl_pct = (net_pnl / bot.equity * 100) if bot.equity > 0 else 0.0

            bot.equity += net_pnl
            bot.total_pnl += net_pnl
            bot.trades += 1
            if net_pnl > 0:
                bot.wins += 1
            if bot.equity > bot.peak_equity:
                bot.peak_equity = bot.equity

            bot._append_equity()

            # ── RL performance kill switches ─────────────────────────
            if bot.trades >= 50 and bot.rl_suite is not None and bot.rl_suite.enabled:
                _wr = bot.wins / bot.trades
                _gross_win = bot.total_pnl if bot.total_pnl > 0 else 0.0
                _gross_loss = abs(bot.total_pnl - _gross_win) if bot.total_pnl < 0 else 0.0
                _pf = _gross_win / _gross_loss if _gross_loss > 0 else 99.0
                if _wr < 0.35:
                    bot.logger.critical(
                        "RL KILL: WR %.1f%% < 35%% over %d trades — disabling RL",
                        _wr * 100, bot.trades,
                    )
                    bot.rl_suite.enabled = False
                elif _pf < 1.0 and bot.total_pnl < 0:
                    bot.logger.critical(
                        "RL KILL: PF %.2f < 1.0, net PnL %.2f over %d trades — disabling RL",
                        _pf, bot.total_pnl, bot.trades,
                    )
                    bot.rl_suite.enabled = False

            # Record PnL in circuit breaker
            if bot.circuit_breaker is not None:
                pnl_pct_frac = net_pnl / bot._account_equity if bot._account_equity > 0 else 0.0
                bot.circuit_breaker.record_trade_pnl(
                    pnl_pct=pnl_pct_frac,
                    asset_class=bot.asset_class,
                    symbol=bot.symbol,
                )

            # Record trade close in Paper Grid (A/B testing)
            if bot.paper_grid is not None:
                bot.paper_grid.record_trade_close(exit_price, bot.symbol)

            # ── Trade Journal: record trade close ─────────────────────
            if bot.journal is not None:
                trade_id_j = trade.get("rl_trade_id", "")
                if trade_id_j:
                    try:
                        entry_price_j = float(trade.get("entry", exit_price))
                        sl_dist_j = abs(entry_price_j - float(trade.get("sl", entry_price_j)))
                        rr_actual = (
                            (abs(exit_price - entry_price_j) / sl_dist_j)
                            if sl_dist_j > 0 else 0.0
                        )
                        rr_target = abs(
                            float(trade.get("tp", entry_price_j)) - entry_price_j
                        ) / sl_dist_j if sl_dist_j > 0 else 3.0

                        outcome_str = "win" if net_pnl > 0 else "loss"
                        # Determine exit reason from price proximity to SL/TP
                        _override_reason = trade.get("_exit_reason_override")
                        if _override_reason:
                            exit_reason = _override_reason
                        else:
                            tp_price = float(trade.get("tp", 0.0))
                            sl_price = float(trade.get("sl", 0.0))
                            if tp_price > 0 and abs(exit_price - tp_price) / max(tp_price, 1) < 0.003:
                                exit_reason = "tp_hit"
                            elif sl_price > 0 and abs(exit_price - sl_price) / max(sl_price, 1) < 0.003:
                                exit_reason = "sl_hit"
                            else:
                                exit_reason = "manual"

                        # MFE/MAE from journal's running tracker
                        max_fav = bot.journal._max_favorable.get(trade_id_j, abs(pnl_pct / 100))
                        bars_held_j = trade.get("bars_held", 0)

                        bot.journal.close_trade(
                            trade_id=trade_id_j,
                            exit_time=datetime.now(timezone.utc),
                            exit_price=exit_price,
                            outcome=outcome_str,
                            exit_reason=exit_reason,
                            bars_held=bars_held_j,
                            pnl_pct=pnl_pct / 100.0,
                            rr_actual=rr_actual if net_pnl > 0 else -rr_actual,
                            max_favorable_pct=max_fav,
                            max_adverse_pct=min(pnl_pct / 100.0, 0.0),
                            be_triggered=bool(trade.get("be_triggered", False)),
                        )
                    except Exception as exc:
                        bot.logger.debug("journal.close_trade error: %s", exc)

            # === CLEANUP ===
            sl_order_id = trade.get("sl_order_id")
            tp_order_id = trade.get("tp_order_id")
            cancel_targets = [oid for oid in (sl_order_id, tp_order_id) if oid]

            _cancel_adapter = bot.adapter if bot.adapter is not None else None
            if cancel_targets and _cancel_adapter is not None:
                for cancel_id in cancel_targets:
                    try:
                        await _cancel_adapter.cancel_order(cancel_id, bot.symbol)
                        bot.logger.info(
                            "Cancelled dangling order %s for %s after exit", cancel_id, bot.symbol
                        )
                    except Exception as exc:
                        # -2011 "Unknown order sent" means order was already
                        # filled or cancelled on the exchange – expected when
                        # SL/TP triggered the close.  Log at DEBUG to reduce noise.
                        exc_str = str(exc)
                        if "-2011" in exc_str or "Unknown order" in exc_str:
                            bot.logger.debug(
                                "Order %s for %s already gone (filled/cancelled): %s",
                                cancel_id, bot.symbol, exc,
                            )
                        else:
                            bot.logger.warning(
                                "Failed to cancel dangling order %s for %s: %s",
                                cancel_id, bot.symbol, exc,
                            )

            # Belt-and-suspenders: fetch open orders and cancel any remaining
            # reduce-only SL/TP orders whose stop price matches this trade.
            # This catches zombie orders when ID-based cancels fail (e.g. the
            # triggered order already filled and its ID was re-used, or the
            # stored ID was wrong).
            # IMPORTANT: protect orders belonging to OTHER active trades
            # (different style) on the same coin — only cancel orders that
            # match the closed trade's SL/TP prices.
            if _cancel_adapter is not None:
                # Collect order IDs of other still-active trades to protect them
                protected_ids: set[str] = set()
                for other in bot._active_trades:
                    if other is trade:
                        continue
                    for k in ("order_id", "sl_order_id", "tp_order_id"):
                        oid = other.get(k)
                        if oid:
                            protected_ids.add(str(oid))
                try:
                    open_orders = await _cancel_adapter.fetch_open_orders(bot.symbol)
                    trade_sl = trade.get("sl")
                    trade_tp = trade.get("tp")
                    for o in open_orders:
                        o_id = str(o.get("id") or "")
                        if not o_id:
                            continue
                        # Never cancel orders belonging to other active trades
                        if o_id in protected_ids:
                            continue
                        # Skip orders we already successfully cancelled above
                        if o_id in cancel_targets:
                            continue
                        o_stop = float(
                            o.get("stopPrice")
                            or o.get("info", {}).get("stopPrice", 0)
                            or 0
                        )
                        o_type = (o.get("type", "") or "").lower()
                        is_exit_type = any(
                            k in o_type for k in ("stop", "take_profit")
                        )
                        if not (is_exit_type and o_stop > 0):
                            continue
                        tol = max(
                            abs(o_stop) * PRICE_TOLERANCE_FACTOR,
                            MIN_PRICE_TOLERANCE,
                        )
                        price_matches = (
                            (trade_sl and abs(o_stop - trade_sl) <= tol)
                            or (trade_tp and abs(o_stop - trade_tp) <= tol)
                        )
                        if price_matches:
                            try:
                                await _cancel_adapter.cancel_order(o_id, bot.symbol)
                                bot.logger.info(
                                    "Zombie order cancelled (price-match) %s"
                                    " stopPrice=%.6f for %s [style=%s]",
                                    o_id, o_stop, bot.symbol,
                                    trade.get("style", "?"),
                                )
                            except Exception as ce:
                                bot.logger.warning(
                                    "Zombie cancel failed %s for %s: %s",
                                    o_id, bot.symbol, ce,
                                )
                except Exception as exc:
                    bot.logger.warning(
                        "fetch_open_orders cleanup failed for %s: %s",
                        bot.symbol, exc,
                    )

            outcome = "WIN" if net_pnl > 0 else "LOSS"
            bot.logger.info(
                "CLOSE %s %s %s @ %.6f → %.6f | pnl=%.2f equity=%.2f",
                outcome,
                direction.upper(),
                bot.symbol,
                entry_price,
                exit_price,
                net_pnl,
                bot.equity,
            )
            bot._save_state()

        while not self._shutdown.is_set():
            try:
                # Fetch positions per adapter (deduplicate OANDA)
                pos_map: dict[str, Any] = {}
                seen_poll: set[int] = set()
                for ac, adapter in self.adapters.items():
                    aid = id(adapter)
                    if aid in seen_poll:
                        continue
                    seen_poll.add(aid)
                    for _poll_try in range(3):
                        try:
                            positions = await adapter.fetch_positions()
                            if positions:
                                for p in positions:
                                    # Support both dict and PositionInfo dataclass
                                    sym = p.get("symbol") if isinstance(p, dict) else getattr(p, "symbol", None)
                                    qty = abs(float(
                                        p.get("contracts", 0) or p.get("qty", 0)
                                        if isinstance(p, dict)
                                        else getattr(p, "qty", 0)
                                    ) or 0)
                                    if sym and qty > 0:
                                        pos_map[sym] = p
                            break
                        except asyncio.CancelledError:
                            raise
                        except Exception as _poll_exc:
                            if _poll_try < 2:
                                logger.debug("fetch_positions [%s] attempt %d failed: %s", ac, _poll_try + 1, _poll_exc)
                                await asyncio.sleep(2 * (_poll_try + 1))

                for bot in self.bots:
                    if not bot._active_trades:
                        continue
                    if bot.adapter is None:
                        continue

                    # Fetch recent trades since earliest entry
                    recent: list[Any] = []
                    try:
                        earliest = min(
                            int(t["entry_time"].timestamp() * 1000)
                            for t in bot._active_trades
                        )
                        recent = await bot.adapter.fetch_my_trades(
                            bot.symbol, since=earliest, limit=50,
                        )
                    except Exception as exc:
                        bot.logger.warning("fetch_my_trades failed: %s", exc)

                    remaining: list[dict[str, Any]] = []
                    # Process each tracked trade independently
                    for trade in list(bot._active_trades):
                        # ── BE management: move SL to breakeven ──
                        if (not trade.get("be_triggered", False)
                                and trade.get("rl_be_level", 0) > 0
                                and bot.rl_suite is not None
                                and bot.rl_suite.be_enabled
                                and bot.adapter is not None):
                            entry_p = trade["entry"]
                            direction = trade["direction"]
                            sl_dist_orig = abs(entry_p - trade.get("original_sl", trade["sl"]))
                            if sl_dist_orig > 0:
                                be_target_rr = trade["rl_be_level"]
                                current_rr = 0.0
                                last_price = entry_p
                                if recent:
                                    try:
                                        _lp = float(recent[-1].get("price", 0) or 0)
                                        if _lp > 0:
                                            last_price = _lp
                                    except Exception:
                                        pass
                                if direction == "long" and last_price > entry_p:
                                    current_rr = (last_price - entry_p) / sl_dist_orig
                                elif direction == "short" and last_price < entry_p:
                                    current_rr = (entry_p - last_price) / sl_dist_orig
                                if current_rr >= be_target_rr:
                                    # Move SL to entry + fee buffer
                                    fee_buffer = entry_p * bot.commission_rate * 4
                                    if direction == "long":
                                        new_sl = entry_p + fee_buffer
                                    else:
                                        new_sl = entry_p - fee_buffer
                                    old_sl_id = trade.get("sl_order_id")
                                    if old_sl_id:
                                        exit_side = "sell" if direction == "long" else "buy"
                                        new_sl_order = await bot.adapter.modify_stop_loss(
                                            old_sl_id, trade["symbol"], exit_side,
                                            trade["qty"], new_sl,
                                        )
                                        if new_sl_order and new_sl_order.order_id:
                                            trade["sl_order_id"] = new_sl_order.order_id
                                            trade["sl"] = new_sl
                                            trade["be_triggered"] = True
                                            bot.logger.info(
                                                "BE TRIGGERED %s: SL moved to %.6f (was %.6f) at RR=%.1f",
                                                trade["symbol"], new_sl, trade.get("sl", 0),
                                                current_rr,
                                            )

                        # ── ML exit: close via market when flag is set ────
                        if trade.get("_ml_exit_requested") and bot.adapter is not None:
                            _ml_exit_side = "sell" if trade["direction"] == "long" else "buy"
                            try:
                                # Cancel existing SL/TP orders
                                for _oid_key in ("sl_order_id", "tp_order_id"):
                                    _oid = trade.get(_oid_key)
                                    if _oid:
                                        try:
                                            await bot.adapter.cancel_order(_oid, bot.symbol)
                                        except Exception:
                                            pass
                                # Place market close order
                                _ml_close = await bot.adapter.create_market_order(
                                    bot.symbol, _ml_exit_side, trade["qty"],
                                    {"reduceOnly": True},
                                )
                                _ml_exit_price = float(
                                    _ml_close.avg_price or _ml_close.price or trade["entry"]
                                )
                                bot.logger.info(
                                    "ML_EXIT CLOSED %s %s @ %.6f | order=%s",
                                    bot.symbol, _ml_exit_side.upper(),
                                    _ml_exit_price, _ml_close.order_id,
                                )
                                trade["_exit_reason_override"] = "ml_exit"
                                await _record_close(bot, trade, _ml_exit_price)
                                continue
                            except Exception as _ml_exc:
                                bot.logger.error("ML exit market close failed %s: %s", bot.symbol, _ml_exc)
                                trade.pop("_ml_exit_requested", None)
                                # Fall through to normal exit detection

                        exit_side = "sell" if trade["direction"] == "long" else "buy"
                        entry_ms = int(trade["entry_time"].timestamp() * 1000)

                        exit_price: float | None = None
                        exit_id: str | None = None

                        for t in reversed(recent or []):
                            # Binance trade id lives in "id"; fallback to "order" for safety
                            tid = str(t.get("id") or t.get("order") or "")
                            if tid and tid in bot._processed_exit_ids:
                                continue
                            if t.get("side") != exit_side:
                                continue
                            t_ts = t.get("timestamp") or 0
                            t_amount = abs(float(t.get("amount", 0) or 0))
                            if t_amount <= 0:
                                continue
                            # Match exit to trade size to avoid mixing fills
                            diff_ratio = abs(t_amount - trade["qty"]) / max(t_amount, trade["qty"])
                            # Normalised by larger qty to tolerate either side being slightly off and avoid div/0
                            if diff_ratio > EXIT_QTY_MATCH_TOLERANCE:
                                continue
                            if t_ts and t_ts >= entry_ms:
                                exit_price = float(t["price"])
                                exit_id = tid or None
                                break

                        if exit_price is not None:
                            if exit_id:
                                bot._processed_exit_ids.add(exit_id)
                            await _record_close(bot, trade, exit_price)
                            continue

                        # No exit trade found but position flat → use last known trade price or SL as fallback
                        if bot.symbol not in pos_map:
                            fallback_price = None
                            if recent:
                                try:
                                    last_price_val = float(recent[-1].get("price", 0) or 0)
                                    if last_price_val > 0:
                                        fallback_price = last_price_val
                                except Exception:
                                    fallback_price = None
                            exit_price = fallback_price or trade["sl"]
                            fallback_label = (
                                "last trade price" if fallback_price is not None else "SL price"
                            )
                            bot.logger.error(
                                "Could not match exit trade for %s – using fallback (%s) %.6f",
                                bot.symbol,
                                fallback_label,
                                exit_price,
                            )
                            await _record_close(bot, trade, exit_price)
                            continue

                        remaining.append(trade)

                    bot._active_trades = remaining

                if not positions:
                    logger.debug(
                        "fetch_positions returned empty – skipping poll cycle"
                    )

            except asyncio.CancelledError:
                return
            except Exception as exc:
                err_msg = str(exc).lower()
                if "demo trading" in err_msg or "sandbox" in err_msg:
                    logger.warning("Position poll: %s – skipping cycle", exc)
                    continue
                logger.error("Position poll error: %s", exc)

            # Sleep until next poll (or shutdown)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=POSITION_POLL_SEC,
                )
                return  # shutdown requested
            except asyncio.TimeoutError:
                pass

    # ── Periodic zombie order sweep ────────────────────────────────

    async def _sweep_zombie_orders(self) -> None:
        """
        Periodically scan ALL open orders on the exchange and cancel any
        that do not belong to a currently active trade.

        This is the last line of defence against zombie orders that slip
        through the per-trade cleanup in _record_close (e.g. due to race
        conditions, network errors, or exchange quirks).

        IMPORTANT: style-aware — orders belonging to active trades of any
        style (scalp/day/swing) on the same coin are protected.
        """
        while not self._shutdown.is_set():
            try:
                # Build a set of ALL known order IDs across ALL active trades
                known_order_ids: set[str] = set()
                symbols_with_trades: set[str] = set()
                for bot in self.bots:
                    for trade in bot._active_trades:
                        for k in ("order_id", "sl_order_id", "tp_order_id"):
                            oid = trade.get(k)
                            if oid:
                                known_order_ids.add(str(oid))
                        symbols_with_trades.add(bot.symbol)

                # For every bot that has NO active trades, any open
                # SL/TP order is definitely a zombie — cancel it.
                for bot in self.bots:
                    if bot._active_trades:
                        continue  # has active trades – handled per-trade
                    if bot.adapter is None:
                        continue
                    try:
                        open_orders = await bot.adapter.fetch_open_orders(bot.symbol)
                    except Exception:
                        continue
                    for o in open_orders:
                        o_id = str(o.get("id") or "")
                        if not o_id:
                            continue
                        o_type = (o.get("type", "") or "").lower()
                        is_exit_type = any(
                            k in o_type for k in ("stop", "take_profit")
                        )
                        if not is_exit_type:
                            continue
                        # This coin has zero active trades → any exit order is zombie
                        try:
                            await bot.adapter.cancel_order(o_id, bot.symbol)
                            bot.logger.warning(
                                "ZOMBIE SWEEP: cancelled orphan order %s "
                                "type=%s for %s (no active trades)",
                                o_id, o.get("type", "?"), bot.symbol,
                            )
                        except Exception as ce:
                            exc_str = str(ce)
                            if "-2011" not in exc_str and "Unknown order" not in exc_str:
                                bot.logger.warning(
                                    "ZOMBIE SWEEP: cancel failed %s for %s: %s",
                                    o_id, bot.symbol, ce,
                                )

                # For bots WITH active trades, verify each open order belongs
                # to one of them.  Cancel if it doesn't match any known ID.
                for bot in self.bots:
                    if not bot._active_trades:
                        continue
                    if bot.adapter is None:
                        continue
                    try:
                        open_orders = await bot.adapter.fetch_open_orders(bot.symbol)
                    except Exception:
                        continue

                    # Collect order IDs for THIS bot's active trades
                    bot_order_ids: set[str] = set()
                    for trade in bot._active_trades:
                        for k in ("order_id", "sl_order_id", "tp_order_id"):
                            oid = trade.get(k)
                            if oid:
                                bot_order_ids.add(str(oid))

                    for o in open_orders:
                        o_id = str(o.get("id") or "")
                        if not o_id:
                            continue
                        o_type = (o.get("type", "") or "").lower()
                        is_exit_type = any(
                            k in o_type for k in ("stop", "take_profit")
                        )
                        if not is_exit_type:
                            continue
                        # If this order ID is NOT in any of this bot's active trades → zombie
                        if o_id not in bot_order_ids:
                            try:
                                await bot.adapter.cancel_order(o_id, bot.symbol)
                                bot.logger.warning(
                                    "ZOMBIE SWEEP: cancelled unmatched order %s "
                                    "type=%s for %s (not in any active trade)",
                                    o_id, o.get("type", "?"), bot.symbol,
                                )
                            except Exception as ce:
                                exc_str = str(ce)
                                if "-2011" not in exc_str and "Unknown order" not in exc_str:
                                    bot.logger.warning(
                                        "ZOMBIE SWEEP: cancel failed %s for %s: %s",
                                        o_id, bot.symbol, ce,
                                    )

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Zombie sweep error: %s", exc)

            # Sleep until next sweep (or shutdown)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=ZOMBIE_SWEEP_SEC,
                )
                return  # shutdown requested
            except asyncio.TimeoutError:
                pass

    # ── Model hot-swap loop ────────────────────────────────────────

    async def _model_reload_loop(self) -> None:
        """Check for updated model files every 60 seconds and reload if changed."""
        _MODEL_RELOAD_SEC = 60
        while not self._shutdown.is_set():
            try:
                rl_suite = self.bots[0].rl_suite if self.bots else None
                if rl_suite is not None and hasattr(rl_suite, "check_and_reload_models"):
                    try:
                        rl_suite.check_and_reload_models()
                    except Exception as exc:
                        logger.debug("Model reload check error: %s", exc)

                # Periodic state persistence (every 60s, crash-safe)
                for bot in self.bots:
                    try:
                        bot._save_state()
                    except Exception:
                        pass
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("Model reload loop error: %s", exc)

            # Sleep until next check (or shutdown)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=_MODEL_RELOAD_SEC,
                )
                return  # shutdown requested
            except asyncio.TimeoutError:
                pass

    # ── Rich Dashboard loop ───────────────────────────────────────

    async def _fetch_real_total_equity(self) -> float:
        """Fetch combined free balance across all adapters."""
        total = 0.0
        seen: set[int] = set()
        for ac, adapter in self.adapters.items():
            aid = id(adapter)
            if aid in seen:
                continue
            seen.add(aid)
            try:
                bal = await adapter.fetch_balance()
                total += bal.free
            except Exception as exc:
                logger.debug("fetch_balance [%s] failed: %s", ac, exc)
        return total

    async def _dashboard_loop(self) -> None:
        """Render the Rich Live Dashboard every DASHBOARD_REFRESH_SEC."""
        console = Console()

        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while not self._shutdown.is_set():
                try:
                    total_equity = await self._fetch_real_total_equity()
                    for b in self.bots:
                        b._account_equity = total_equity

                    # Update circuit breaker portfolio heat
                    total_risk = sum(
                        t.get("risk_pct", 0.0)
                        for b in self.bots
                        for t in b._active_trades
                    )
                    self.circuit_breaker.update_portfolio_heat(total_risk)
                    self.circuit_breaker.check()

                    # Save paper grid state periodically
                    try:
                        self.paper_grid.save_state()
                    except Exception:
                        pass

                    layout = build_dashboard(
                        bots=self.bots,
                        ws_status=self.ws_status,
                        start_time=self._start_time,
                        active_symbols=self.symbols,
                        total_equity=total_equity,
                        paper_grid=self.paper_grid,
                    )
                    live.update(layout)
                except Exception as exc:
                    logger.error("Dashboard render error: %s", exc)

                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=DASHBOARD_REFRESH_SEC
                    )
                    return
                except asyncio.TimeoutError:
                    pass

    # ── Main loop ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all watchers (OHLCV + ticker) + position poller + dashboard. Blocks until shutdown."""
        # Log per-class bot counts
        class_counts = {ac: len(bots) for ac, bots in self._bots_by_class.items()}
        logger.info(
            "Starting %d bots: %s", len(self.bots),
            ", ".join(f"{cnt} {ac}" for ac, cnt in class_counts.items()),
        )

        # ── Startup zombie order sweep per adapter ────────────────
        seen_adapters: set[int] = set()
        for ac, adapter in self.adapters.items():
            aid = id(adapter)
            if aid in seen_adapters:
                continue
            seen_adapters.add(aid)
            try:
                n_cancelled = 0
                for bot in self._bots_by_class.get(ac, []):
                    try:
                        open_orders = await adapter.fetch_open_orders(bot.symbol)
                        for o in open_orders:
                            o_id = str(o.get("id", "") if isinstance(o, dict) else getattr(o, "order_id", ""))
                            if o_id:
                                await adapter.cancel_order(o_id, bot.symbol)
                                n_cancelled += 1
                    except Exception:
                        pass
                # For OANDA: also sweep commodities bots (same adapter)
                if ac == "forex":
                    for bot in self._bots_by_class.get("commodities", []):
                        try:
                            open_orders = await adapter.fetch_open_orders(bot.symbol)
                            for o in open_orders:
                                o_id = str(o.get("id", "") if isinstance(o, dict) else getattr(o, "order_id", ""))
                                if o_id:
                                    await adapter.cancel_order(o_id, bot.symbol)
                                    n_cancelled += 1
                        except Exception:
                            pass
                if n_cancelled > 0:
                    logger.info("Startup [%s]: cancelled %d zombie orders", ac, n_cancelled)
            except Exception as exc:
                logger.warning("Startup zombie sweep [%s] failed: %s", ac, exc)

        # Start watchers: WebSocket for crypto, REST polling for others
        # Crypto: stagger WS subscriptions (1s apart) to avoid thundering herd on ccxt.pro
        # REST: stagger to avoid rate limits (stocks: 2s apart, others: 2s)
        _crypto_idx = 0
        _stock_idx = 0
        _rest_idx = 0
        for bot in self.bots:
            if bot.asset_class == "crypto":
                # WebSocket-based — staggered to prevent subscription deadlock
                stagger = _crypto_idx * WS_STAGGER_SEC
                self._watcher_tasks[bot.symbol] = asyncio.create_task(
                    self._watch_symbol(bot.symbol, stagger_delay=stagger)
                )
                self._ticker_tasks[bot.symbol] = asyncio.create_task(
                    self._watch_ticker(bot.symbol, stagger_delay=stagger)
                )
                _crypto_idx += 1
            else:
                # REST polling for OANDA/Alpaca — staggered start
                if bot.asset_class == "stocks":
                    stagger = _stock_idx * 2.0
                    _stock_idx += 1
                else:
                    stagger = _rest_idx * REST_STAGGER_SEC
                    _rest_idx += 1
                self._watcher_tasks[bot.symbol] = asyncio.create_task(
                    self._poll_candles(bot, stagger_sec=stagger)
                )
                self._ticker_tasks[bot.symbol] = asyncio.create_task(
                    self._poll_ticker(bot, stagger_sec=stagger)
                )

        logger.info(
            "All watchers started: %d crypto (staggered %.0fs apart), %d REST",
            _crypto_idx, WS_STAGGER_SEC, _stock_idx + _rest_idx,
        )

        # Position poller (detects TP/SL fills on exchange)
        poll_task = asyncio.create_task(self._poll_positions())

        # Periodic zombie order sweep (catches any orphans missed by per-trade cleanup)
        zombie_task = asyncio.create_task(self._sweep_zombie_orders())

        # Model hot-swap check (reload updated model files every 60s)
        model_reload_task = asyncio.create_task(self._model_reload_loop())

        # Rich dashboard
        dashboard_task = asyncio.create_task(self._dashboard_loop())

        # Heartbeat + watchdog
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        watchdog_task = asyncio.create_task(self._watchdog_loop())

        # Wait until shutdown
        await self._shutdown.wait()
        logger.info("Shutdown signal received – stopping …")

        # Cancel all tasks
        dashboard_task.cancel()
        poll_task.cancel()
        zombie_task.cancel()
        model_reload_task.cancel()
        heartbeat_task.cancel()
        watchdog_task.cancel()
        for t in self._watcher_tasks.values():
            t.cancel()
        for t in self._ticker_tasks.values():
            t.cancel()

        all_tasks = (
            [dashboard_task, poll_task, zombie_task, model_reload_task,
             heartbeat_task, watchdog_task]
            + list(self._watcher_tasks.values())
            + list(self._ticker_tasks.values())
        )
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Fetch final equity before closing exchange
        final_equity = await self._fetch_real_total_equity()
        for b in self.bots:
            b._account_equity = final_equity
            # === PERSISTENCE ===
            try:
                b._save_state()
            except Exception:
                pass

        # Save Paper Grid state + export results
        try:
            self.paper_grid.save_state()
            self.paper_grid.export_csv()
            self.paper_grid.export_summary()
            logger.info("Paper Grid state saved")
        except Exception as exc:
            logger.warning("Paper Grid save failed: %s", exc)

        # Close all exchange adapter connections (deduplicate OANDA)
        closed: set[int] = set()
        for adapter in self.adapters.values():
            aid = id(adapter)
            if aid in closed:
                continue
            closed.add(aid)
            try:
                await adapter.close()
            except Exception:
                pass

        # Final summary to console
        self._print_final_summary(final_equity)

    def _print_final_summary(self, total_equity: float = 0.0) -> None:
        """Print a plain-text final summary after dashboard stops."""
        console = Console()
        rows = sorted(
            [b.summary_dict() for b in self.bots],
            key=lambda r: r["pnl"],
            reverse=True,
        )
        total_pnl = sum(b.total_pnl for b in self.bots)

        console.print(f"\n[bold cyan]{'═' * 80}[/bold cyan]")
        console.print("[bold cyan]  📊  FINAL SUMMARY[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 80}[/bold cyan]")
        console.print(
            f"  Total Equity: [bold]{total_equity:,.2f}[/bold] USDT  |  "
            f"Total PnL: [bold {'green' if total_pnl >= 0 else 'red'}]"
            f"{total_pnl:+,.2f}[/bold {'green' if total_pnl >= 0 else 'red'}] USDT"
        )

        table = _build_bot_table("ALL BOTS – Final Ranking", rows, "cyan")
        console.print(table)
        console.print(f"[bold cyan]{'═' * 80}[/bold cyan]\n")

    def request_shutdown(self) -> None:
        self._shutdown.set()
        # === PERSISTENCE ===
        for b in self.bots:
            try:
                b._save_state()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

async def async_main(config: dict[str, Any], output_dir: Path) -> None:
    """Async entry point: create adapters, bots, load history, run."""
    console = Console()

    # ── Create adapters (graceful skip for missing keys) ──────────
    logger.info("Creating exchange adapters...")
    adapters = await create_adapters(config)
    if not adapters:
        sys.exit(
            "No exchange adapters available. Set API keys in .env\n"
            "Copy .env.example → .env and fill in at least one broker's keys."
        )

    # ── Determine active instruments ──────────────────────────────
    active: list[tuple[str, str]] = []  # (symbol, asset_class)
    for ac in ["crypto", "forex", "stocks", "commodities"]:
        if ac in adapters:
            for sym in ALL_INSTRUMENTS[ac]:
                active.append((sym, ac))

    class_counts = {}
    for _, ac in active:
        class_counts[ac] = class_counts.get(ac, 0) + 1
    count_str = ", ".join(f"{cnt} {ac}" for ac, cnt in class_counts.items())
    logger.info("%d instruments active: %s", len(active), count_str)
    console.print(f"[bold cyan]Creating {len(active)} bots ({count_str}) ...[/bold cyan]")

    # ── Create RL Brain Suite (XGBoost models) ────────────────────
    rl_suite = RLBrainSuite(config)

    # ── Create bots ───────────────────────────────────────────────
    bots: list[PaperBot] = []
    for idx, (sym, ac) in enumerate(active):
        bot = PaperBot(
            bot_id=idx + 1,
            symbol=sym,
            config=config,
            output_dir=output_dir,
            asset_class=ac,
            adapter=adapters[ac],
            rl_suite=rl_suite,
        )
        bots.append(bot)

    console.print(f"[bold green]{len(bots)} bots created.[/bold green]")

    # ── Load history (batched, rate-limit-friendly) ───────────────
    logger.info("Loading history (250+ bars per TF) for %d instruments...", len(bots))
    batch_size = 10
    for i in range(0, len(bots), batch_size):
        batch = bots[i:i + batch_size]
        await asyncio.gather(*[b.load_history() for b in batch])
        if i + batch_size < len(bots):
            await asyncio.sleep(1)  # rate-limit pause between batches
    logger.info("History loaded for all %d instruments.", len(bots))

    # ── Runner ────────────────────────────────────────────────────
    runner = LiveMultiBotRunner(bots=bots, adapters=adapters)

    # ── Sync initial equity from exchange for bots without saved state ─
    seen_adapters: dict[int, float] = {}
    for ac, adapter in adapters.items():
        aid = id(adapter)
        if aid in seen_adapters:
            continue
        try:
            bal = await adapter.fetch_balance()
            real_equity = bal.free if bal else 0.0
            seen_adapters[aid] = real_equity
            logger.info("Initial equity [%s]: %.2f", ac, real_equity)
        except Exception as exc:
            logger.warning("Failed to fetch initial equity [%s]: %s", ac, exc)
            seen_adapters[aid] = 0.0

    for bot in bots:
        if bot.equity <= 0:
            adapter_equity = seen_adapters.get(id(adapters.get(bot.asset_class)), 0.0)
            if adapter_equity > 0:
                bot.equity = adapter_equity
                bot.peak_equity = adapter_equity
                bot._account_equity = adapter_equity

    equity_summary = {ac: seen_adapters.get(id(adapters.get(ac)), 0.0) for ac in adapters}
    logger.info("Equity synced: %s", equity_summary)

    # Signal handlers
    loop = asyncio.get_event_loop()

    def _signal_handler() -> None:
        logger.info("Ctrl+C detected – shutting down gracefully ...")
        runner.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await runner.run()
    logger.info("All bots stopped. Results in %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMC Multi-Asset Live Trading Bot (Crypto + Forex + Stocks + Commodities)",
    )
    parser.add_argument(
        "--config", default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Directory for live bot outputs (equity CSVs, logs, RL models)",
    )
    args = parser.parse_args()

    # ── Load env ──────────────────────────────────────────────────
    load_dotenv()

    # ── Load config ───────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", cfg_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run async main ────────────────────────────────────────────
    asyncio.run(async_main(config, output_dir))


if __name__ == "__main__":
    main()