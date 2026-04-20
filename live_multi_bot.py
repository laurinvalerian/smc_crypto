"""
═══════════════════════════════════════════════════════════════════
 live_multi_bot.py  –  Crypto-Only SMC Trading Bot
 ──────────────────────────────────────────────────────────────
 30 Crypto bots (Binance USDT-M Futures), matching the backtester
 universe.

 Features:
   • 30 crypto bots (Binance Futures)
   • WebSocket candle + ticker feeds
   • Central XGBoost RL brain (RLBrainSuite)
   • Circuit Breaker for portfolio-level risk management
   • Rich Live Dashboard

 Requirements:
   pip install 'ccxt[pro]' pandas numpy python-dotenv pyyaml rich torch

 Quick Start:
   1. Copy .env.example → .env and fill in BINANCE_API_KEY / BINANCE_SECRET
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
import weakref
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live

from core.alignment import CORE_WEIGHTS_CRYPTO, compute_alignment_score
from rl_brain_v2 import RLBrainSuite
from models.student_brain import StudentBrain
from trade_journal import TradeJournal
from utils.indicators import compute_rsi_wilders, compute_atr_wilders
from strategies.smc_multi_style import (
    compute_smc_indicators_causal as compute_smc_indicators,
    _precompute_running_bias,
    _precompute_running_structure,
    _bias_from_running,
    _structure_confirms_from_running,
    _find_entry_zone_at,
    _precompute_5m_trigger_mask,
    _find_structure_tp_safe,
    _precompute_htf_arrays,
    _check_volume_ok,
)
from filters.trend_strength import compute_adx, check_momentum_confluence, multi_tf_trend_agreement
from filters.volume_liquidity import compute_volume_score
from filters.session_filter import compute_session_score
from filters.zone_quality import compute_zone_quality
from features.feature_extractor import FeatureExtractor
from features.schema import ENTRY_QUALITY_FEATURES, validate_against_model
from rl_dqn.dqn_inference import DQNExitManager
from exchanges import BinanceAdapter
from exchanges.base import ExchangeAdapter
from risk.circuit_breaker import CircuitBreaker
from paper_grid import PaperGrid
from live_teacher import analyze_closed_trade as _teacher_analyze, save_feedback as _teacher_save

# ── Continuous learner (optional — may not be deployed yet) ─────
try:
    from continuous_learner import run_continuous_learner
except ImportError:
    run_continuous_learner = None  # type: ignore[assignment,misc]
    logging.getLogger(__name__).warning("continuous_learner not found — auto-retrain disabled")

# ── Drift monitor (optional) ────────────────────────────────────
try:
    from drift_monitor import run_drift_monitor
except ImportError:
    run_drift_monitor = None  # type: ignore[assignment,misc]
    logging.getLogger(__name__).warning("drift_monitor not found — feature drift monitoring disabled")

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

# ── Combined instrument universe (crypto-only) ────────────────────
ALL_INSTRUMENTS: dict[str, list[str]] = {
    "crypto": TOP_30_CRYPTO,
}
NUM_BOTS = len(TOP_30_CRYPTO)  # 30

# Backward-compat alias
TOP_100_COINS = TOP_30_CRYPTO

# ── Trading costs imported from core SSOT (Phase 2.1, 2026-04-18) ─
# Values previously duplicated here AND in backtest/generate_rl_data.py +
# backtest/optuna_backtester.py + paper_grid.py + exchanges/replay_adapter.py.
# Authoritative source is now core.constants.{COMMISSION, SLIPPAGE}.
from core.constants import COMMISSION, SLIPPAGE, DEFAULT_RISK_PER_TRADE, MAX_RISK_PER_TRADE
from core.sizing import compute_risk_fraction
ASSET_COMMISSION: dict[str, float] = {"crypto": COMMISSION}
_TRAIN_COMMISSION: dict[str, float] = {"crypto": COMMISSION}
_TRAIN_SLIPPAGE: dict[str, float] = {"crypto": SLIPPAGE}

# ── Asset-Class-Specific SMC Parameters (from config smc_profiles) ─
ASSET_SMC_PARAMS: dict[str, dict[str, Any]] = {
    "crypto": {
        "swing_length": 8, "fvg_threshold": 0.0006,
        "order_block_lookback": 20, "liquidity_range_percent": 0.01,
        "alignment_threshold": 0.78, "weight_day": 1.25, "bos_choch_filter": "medium",
        "min_daily_atr_pct": 0.005, "min_5m_atr_pct": 0.0010,
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
    "crypto": 10,
}

# Max new signals per SYMBOL per 4-hour window, by asset class (safety throttle).
# Portfolio-level protection comes from circuit breakers (daily -3%, weekly -5%).
MAX_SIGNALS_PER_SYMBOL_4H: dict[str, int] = {
    "crypto": 10,
}

# ── Asset-Class IDs for RL Brain ─────────────────────────────────
ASSET_CLASS_ID: dict[str, float] = {
    "crypto": 0.0,
}

# DEPRECATED: REST polling constants (OANDA/Alpaca only) — not used for Crypto WebSocket
REST_POLL_INTERVAL_SEC = 10
REST_POLL_INTERVAL_STOCKS_CANDLE = 60
REST_POLL_INTERVAL_STOCKS_TICKER = 30
REST_STAGGER_SEC = 2.0


def symbol_to_asset_class(symbol: str) -> str:
    """Determine asset class from symbol format. Crypto-only: always returns 'crypto'."""
    return "crypto"

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
# Absolute minimum SL as % of entry price — avoids 1-bar SLs on very calm markets.
# Scalp-Day Hybrid 2026-04-20: replaces the per-style `min_sl_pct` that was
# pulled from STYLE_CONFIG (0.2% matches the old DAY floor).
MIN_SL_PCT_FLOOR = 0.002

# ── Trade Style Configuration ────────────────────────────────────
# Scalp-Day Hybrid (2026-04-20): style classification is NEUTRALIZED.
# `_classify_trade_style` returns STYLE_DAY unconditionally and
# `_validate_style_constraints` returns (True, ""). STYLE_CONFIG and the
# SCALP/SWING constants are kept only for backward compatibility with
# historical code (journal `style` column, XGB feature `style_id`, old
# REJECTION_HORIZON_* lookups). Do NOT re-introduce style-based clamping
# without data-driven validation — 63% of profitable backtest setups were
# blocked by the old clamping logic (see data/style_kill_analysis_2026-04-20.md).
STYLE_SCALP = "scalp"
STYLE_DAY = "day"
STYLE_SWING = "swing"
MAX_CONCURRENT_REJECTION_WATCHERS = 50  # cap async outcome watchers for rejected signals
REJECTION_DEDUP_BARS = 6        # 30 min in 5m candles — skip duplicate rejections (candle-time)
REJECTION_HORIZON_SCALP = 72    # 6h in 5m bars
REJECTION_HORIZON_DAY = 288     # 24h in 5m bars
REJECTION_HORIZON_SWING = 1440  # 5d in 5m bars


@dataclass
class _RejectionTracker:
    """Tracks counterfactual outcome of a rejected signal over 3 horizons."""
    db_id: int
    entry_price: float
    sl_price: float
    tp_price: float
    direction: str  # "long" or "short"
    bars_elapsed: int = 0
    mfe: float = 0.0  # max favorable excursion (fraction of entry)
    mae: float = 0.0  # max adverse excursion (fraction of entry)
    scalp_done: bool = False
    day_done: bool = False
    # swing_done implied by removal from list

# Scalp-Day Hybrid (2026-04-19): all styles cap at 4h hold (48 × 5m bars).
# Style detection (_detect_style) still classifies by SL-size, but every
# bucket times out at the same horizon — SMC structure plays out within
# 1-4h, anything still open after that has invalidated its thesis.
STYLE_CONFIG: dict[str, dict[str, Any]] = {
    STYLE_SCALP: {
        "min_sl_pct": 0.002,    # 0.2% min SL (lowered — brain validates)
        "max_sl_pct": 0.008,    # 0.8% max SL
        "min_tp_pct": 0.002,    # 0.2% min TP
        "max_tp_pct": 0.015,    # 1.5% max TP
        "min_rr": 2.0,          # minimum 2:1 RR enforced
        "max_hold_candles": 48,  # 4h on 5m bars (Scalp-Day Hybrid)
    },
    STYLE_DAY: {
        "min_sl_pct": 0.002,    # 0.2% min SL (lowered — brain validates)
        "max_sl_pct": 0.025,    # 2.5% max SL
        "min_tp_pct": 0.004,    # 0.4% min TP
        "max_tp_pct": 0.06,     # 6% max TP
        "min_rr": 2.0,          # minimum 2:1 RR enforced
        "max_hold_candles": 48,  # 4h on 5m bars (Scalp-Day Hybrid)
    },
    STYLE_SWING: {
        "min_sl_pct": 0.008,    # 0.8% min SL
        "max_sl_pct": 0.05,     # 5% max SL
        "min_tp_pct": 0.02,     # 2% min TP
        "max_tp_pct": 0.15,     # 15% max TP
        "min_rr": 2.0,          # brain filters bad RR
        "max_hold_candles": 48,  # 4h on 5m bars — wide-SL setups timeout (by design)
    },
}

# Risk sizing uses core.sizing.compute_risk_fraction as SSOT:
# linear 0.25% at ALIGNMENT_THRESHOLD to 1.0% at score 1.0, multiplied by the
# Student size-head when present (defaults to 1.0 when Student is disabled).
# Leverage capped by ASSET_MAX_LEVERAGE. No AAA++/AAA+ tier dispatch.

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

class _BracketSLTPFailed(Exception):
    """Raised when SL/TP placement fails after entry was already filled.

    The retry loop must NOT create a new entry — the position was already
    flattened (or flatten was attempted).  Retrying would create zombie positions.
    """


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

    # Class-level prefilter telemetry. Shared across all PaperBot instances
    # in the same process (asyncio is single-threaded so += is safe). The
    # Runner._heartbeat_loop reads these counters every HEARTBEAT_SEC,
    # logs an aggregated PREFILTER-STATS line, then resets them. Purpose:
    # diagnose which stage of _prepare_signal blocks most 5m candles when
    # the gate-crossing rate is suspiciously low. Added 2026-04-20 during
    # the signal-drought investigation.
    _prefilter_stats: dict[str, int] = defaultdict(int)

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
        student_brain: StudentBrain | None = None,
    ) -> None:
        self.bot_id = bot_id
        self.tag = f"bot_{bot_id:03d}"
        self.symbol = symbol
        self.asset_class = asset_class
        self.adapter = adapter
        # Backward-compat: raw ccxt.pro exchange (only BinanceAdapter has .raw)
        # Legacy self.exchange removed — all calls go through self.adapter now

        # Use per-symbol optimized params merged on top of per-class defaults.
        # Training samples (data/rl_training/*_samples.parquet) MUST be
        # regenerated with the same per-cluster optimized params and the
        # model retrained on them, otherwise train/inference feature
        # distributions drift apart. See backtest/generate_rl_data.py
        # _resolve_smc_params() — kept in sync here.
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
        self._display_multiplier: float = 1.0  # dashboard-only multiplier (does NOT affect trading)

        # Tracking
        self.total_pnl: float = 0.0
        self.trades: int = 0
        self.wins: int = 0

        # Active trades on exchange (max 3, cleared by position poller)
        self._active_trades: list[dict[str, Any]] = []
        self._processed_exit_ids: set[str] = set()

        # Teacher-analysis hook injected by Runner (keeps _teacher_semaphore in Runner).
        self._teacher_enabled: bool = False
        self._teacher_trigger: Callable[[PaperBot, dict[str, Any], float], Awaitable[None]] | None = None

        # Candle history  {symbol: list[dict]}
        self._candle_buf: dict[str, list[dict[str, Any]]] = {}

        # Pending signal for real-time entry (set by on_candle, consumed by on_tick)
        self._pending_signal: dict[str, Any] | None = None

        # Per-symbol signal rate limit throttle
        self._max_signals_per_4h: int = MAX_SIGNALS_PER_SYMBOL_4H.get(asset_class, 5)
        self._recent_signal_bars: list[int] = []  # bar counts of accepted signals (candle-time)

        # Bar counter for candle-time-based rate limiting and dedup
        self._total_bars: int = 0

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
        # Teacher-Student unified brain (replaces entry_filter + sl_adjuster +
        # tp_optimizer + position_sizer when `student_brain.enabled: true`).
        # None or disabled → bot falls back to the old rl_suite stack.
        self.student_brain = student_brain

        # ── Load pre-computed symbol ranks for XGB features ──────────
        # Ships from MacBook as models/symbol_ranks.json alongside model pickle.
        # Falls back to per-asset-class training medians if missing.
        # Class-level cache avoids 112 bots each re-reading the same JSON.
        if not hasattr(PaperBot, "_symbol_ranks_cache"):
            _ranks_path = Path("models/symbol_ranks.json")
            if _ranks_path.exists():
                try:
                    with open(_ranks_path) as f:
                        PaperBot._symbol_ranks_cache = json.load(f)
                except Exception:
                    PaperBot._symbol_ranks_cache = {}
            else:
                PaperBot._symbol_ranks_cache = {}

        _fallback = self._AC_FALLBACK_MEDIANS.get(asset_class, self._AC_FALLBACK_MEDIANS["crypto"])
        _sym_ranks = PaperBot._symbol_ranks_cache.get(asset_class, {}).get(symbol, {})
        self._symbol_volatility_rank = float(_sym_ranks.get("volatility_rank", _fallback["volatility"]))
        self._symbol_liquidity_rank = float(_sym_ranks.get("liquidity_rank", _fallback["liquidity"]))
        self._symbol_spread_rank = float(_sym_ranks.get("spread_rank", _fallback["spread"]))

        # DQN exit manager (shadow mode -- logs alongside XGBoost for comparison)
        self._dqn_exit = None
        self._dqn_cfg = config.get("dqn_exit_manager", {})

        # RL monitoring stats
        self._rl_accepted: int = 0
        self._rl_rejected: int = 0
        self._rejection_watcher_count: int = 0
        # Store last obs so we can record reward when trade closes
        self._pending_obs: np.ndarray | None = None

        # Counterfactual outcome tracking for rejected signals
        self._last_rejection_bar: dict[str, int] = {}  # direction -> last bar count (dedup)
        self._rejection_trackers: list[_RejectionTracker] = []

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
        self._buffer_5m_deque: deque[dict] = deque(maxlen=1500)
        self._buffer_5m_cache: pd.DataFrame = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        self._buffer_5m_dirty: bool = False
        # History is loaded async after construction: await bot.load_history()
        self._load_state()

    # ── buffer_5m property (O(1) deque append, dirty-flag DataFrame cache) ──

    @property
    def buffer_5m(self) -> pd.DataFrame:
        """Cached DataFrame view of 5m candle buffer. Rebuilt at most once per candle."""
        if self._buffer_5m_dirty:
            if self._buffer_5m_deque:
                self._buffer_5m_cache = pd.DataFrame(list(self._buffer_5m_deque))
            else:
                self._buffer_5m_cache = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            self._buffer_5m_dirty = False
        return self._buffer_5m_cache

    @buffer_5m.setter
    def buffer_5m(self, df: pd.DataFrame) -> None:
        """Setter for backward compatibility with load_history setattr."""
        self._buffer_5m_deque = deque(
            df.to_dict("records"), maxlen=1500
        )
        self._buffer_5m_dirty = True

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

    # === COUNTERFACTUAL OUTCOME TRACKING ===

    def _rejection_dedup_ok(self, direction: str) -> bool:
        """Return True if this rejection is NOT a duplicate (>6 bars since last same-dir).

        Uses candle-time (bar count) not wall-clock — weekends/market-close don't count.
        6 bars = 30 min in 5m candles.
        """
        last_bar = self._last_rejection_bar.get(direction)
        if last_bar is not None and (self._total_bars - last_bar) < 6:
            return False
        self._last_rejection_bar[direction] = self._total_bars
        return True

    def _record_and_track_rejection(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        xgb_confidence: float,
        alignment_score: float,
        entry_features: dict[str, float] | None,
    ) -> None:
        """Deduplicated rejection recording + start counterfactual outcome tracker."""
        if self.journal is None:
            return
        if not self._rejection_dedup_ok(direction):
            return  # duplicate within 30 min window

        db_id = self.journal.record_rejected_signal(
            symbol=symbol,
            asset_class=self.asset_class,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            xgb_confidence=xgb_confidence,
            alignment_score=alignment_score,
            entry_features=entry_features,
        )
        if db_id is not None and entry_price > 0:
            # Cap at 100 trackers per bot; evict oldest with timeout if full
            if len(self._rejection_trackers) >= 100:
                oldest = self._rejection_trackers[0]
                if self.journal:
                    if not oldest.scalp_done:
                        self.journal.update_rejection_outcome(oldest.db_id, "scalp", "timeout", oldest.mfe, oldest.mae)
                    if not oldest.day_done:
                        self.journal.update_rejection_outcome(oldest.db_id, "day", "timeout", oldest.mfe, oldest.mae)
                    self.journal.update_rejection_outcome(oldest.db_id, "swing", "timeout", oldest.mfe, oldest.mae)
                self._rejection_trackers.pop(0)
            self._rejection_trackers.append(_RejectionTracker(
                db_id=db_id,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                direction=direction,
            ))

    def _update_rejection_outcomes(self, candle: dict[str, Any]) -> None:
        """Update all pending rejection trackers with this candle's price action.

        Called once per 5m candle from on_candle. Checks SL/TP hits,
        updates MFE/MAE, writes outcomes to journal at each horizon.
        """
        if not self._rejection_trackers:
            return

        high = float(candle.get("high", 0.0))
        low = float(candle.get("low", 0.0))
        to_remove: list[_RejectionTracker] = []

        for t in self._rejection_trackers:
            t.bars_elapsed += 1
            entry = t.entry_price
            if entry <= 0:
                to_remove.append(t)
                continue

            # Update MFE / MAE
            if t.direction == "long":
                fav = (high - entry) / entry
                adv = (entry - low) / entry
                sl_hit = low <= t.sl_price
                tp_hit = high >= t.tp_price
            else:
                fav = (entry - low) / entry
                adv = (high - entry) / entry
                sl_hit = high >= t.sl_price
                tp_hit = low <= t.tp_price

            t.mfe = max(t.mfe, max(fav, 0.0))
            t.mae = max(t.mae, max(adv, 0.0))

            # If SL or TP hit, finalize ALL remaining horizons and remove
            # Conservative: if BOTH hit on same candle, label as "loss"
            # (intra-bar order unknown — avoids optimistic bias in training data)
            if sl_hit or tp_hit:
                outcome = "loss" if (sl_hit and tp_hit) else ("win" if tp_hit else "loss")
                if not t.scalp_done and self.journal:
                    self.journal.update_rejection_outcome(t.db_id, "scalp", outcome, t.mfe, t.mae)
                if not t.day_done and self.journal:
                    self.journal.update_rejection_outcome(t.db_id, "day", outcome, t.mfe, t.mae)
                if self.journal:
                    self.journal.update_rejection_outcome(t.db_id, "swing", outcome, t.mfe, t.mae)
                to_remove.append(t)
                continue

            # Timeout checks per horizon
            if not t.scalp_done and t.bars_elapsed >= REJECTION_HORIZON_SCALP:
                if self.journal:
                    self.journal.update_rejection_outcome(t.db_id, "scalp", "timeout", t.mfe, t.mae)
                t.scalp_done = True

            if not t.day_done and t.bars_elapsed >= REJECTION_HORIZON_DAY:
                if self.journal:
                    self.journal.update_rejection_outcome(t.db_id, "day", "timeout", t.mfe, t.mae)
                t.day_done = True

            if t.bars_elapsed >= REJECTION_HORIZON_SWING:
                if self.journal:
                    self.journal.update_rejection_outcome(t.db_id, "swing", "timeout", t.mfe, t.mae)
                to_remove.append(t)

        if to_remove:
            remove_set = set(id(t) for t in to_remove)
            self._rejection_trackers = [t for t in self._rejection_trackers if id(t) not in remove_set]

    # === CAPACITY-REJECTED SIGNAL TRACKING ===
    def _record_capacity_rejected(self, sig: dict[str, Any], reason: str) -> None:
        """Record a signal blocked by capacity limits for counterfactual learning."""
        if self.journal is None:
            return
        try:
            self._record_and_track_rejection(
                symbol=sig.get("symbol", self.symbol),
                direction=sig.get("direction", ""),
                entry_price=sig.get("ref_price", 0.0),
                sl_price=sig.get("sl", 0.0),
                tp_price=sig.get("tp", 0.0),
                xgb_confidence=-1.0,  # marker: not XGB-evaluated
                alignment_score=sig.get("score", 0.0),
                entry_features=sig.get("features"),
            )
            self.logger.debug(
                "CAPACITY REJECT recorded %s score=%.2f reason=%s",
                sig.get("symbol", "?"), sig.get("score", 0), reason,
            )
        except Exception as exc:
            self.logger.debug("Failed to record capacity-rejected signal: %s", exc)

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
                # In-flight migration: legacy trades from before trade-attached SL/TP fix
                if "sl_attached" not in t:
                    t["sl_attached"] = False  # Legacy: standalone SL/TP orders
                if "oanda_trade_id" not in t:
                    t["oanda_trade_id"] = None
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
        Granular multi-timeframe SMC alignment score.

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

        # Components tracked for alignment scoring
        comp: dict[str, Any] = {
            "bias": False, "bias_strong": False,
            "h4_confirms": False, "h4_poi": False,
            "h1_confirms": False, "h1_choch": False,
            "entry_zone": None, "zone_fresh": False,
            "precision_trigger": False, "volume_ok": False,
            # Extended components
            "adx_strong": False, "adx_value": 0.0,
            "session_optimal": False, "session_score": 0.0,
            "momentum_confluent": False, "momentum_score": 0.0,
            "tf_agreement": 0, "tf_agreement_score": 0.0,
            "zone_quality": 0.0, "zone_quality_ok": False,
            "volume_score": 0.0, "volume_details": None,
        }

        daily_bias = "neutral"
        score = 0.0

        # Core weights imported from core.alignment (Phase 2.1 SSOT).
        # The gate threshold (0.78) was calibrated against these training weights.
        # Crypto (non-forex) path only.
        _w_bias        = CORE_WEIGHTS_CRYPTO["bias"]         # any non-neutral bias (additive base)
        _w_bias_strong = CORE_WEIGHTS_CRYPTO["bias_strong"]  # strong bias bonus (additive on top)
        _w_h4          = CORE_WEIGHTS_CRYPTO["h4"]
        _w_h4_poi      = CORE_WEIGHTS_CRYPTO["h4_poi"]
        _w_h1          = CORE_WEIGHTS_CRYPTO["h1"]
        _w_h1_choch    = CORE_WEIGHTS_CRYPTO["h1_choch"]
        _w_zone        = CORE_WEIGHTS_CRYPTO["zone"]
        _w_trigger     = CORE_WEIGHTS_CRYPTO["trigger"]
        _w_volume      = CORE_WEIGHTS_CRYPTO["volume"]
        # Bonus components (not in training, small weight — never needed for gate)
        _w_adx       = 0.02
        _w_session   = 0.02
        _w_momentum  = 0.02
        _w_tf_agree  = 0.02
        _w_freshness = 0.02

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
                    score += _w_bias  # Base: 0.12 for any non-neutral bias (matches training)
                    # Additive bonus for strong bias (BOS/CHoCH confirmed)
                    pure_struct = _precompute_running_structure(ind_1d)
                    pure_bias = _bias_from_running(pure_struct, len(self.buffer_1d))
                    if pure_bias != "neutral" and pure_bias == daily_bias:
                        score += _w_bias_strong  # +0.08/0.12 bonus (matches training)
                        comp["bias_strong"] = True
            except Exception as exc:
                self.logger.debug("1D bias computation failed: %s", exc)

        if daily_bias == "neutral":
            direction = "long"
            comp["daily_bias"] = "neutral"
            return 0.0, direction, comp

        direction = "long" if daily_bias == "bullish" else "short"
        comp["daily_bias"] = daily_bias

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
                _zone_bars = 6  # crypto-only (was `12 if _fx else 6`)
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
                _trig_lb = 1  # crypto-only (was `3 if _fx else 1`)
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
                # SSOT volume_ok flag must match the backtest path
                # (strategies/smc_multi_style.py:1648 calls _check_volume_ok with
                # a simple 1.2× relative-ratio check). The 3-layer compute_volume_score
                # above stays as a continuous rich feature (volume_score), but the
                # binary gate flag is computed identically to the backtest to
                # keep the alignment-score distribution 1:1 across both paths.
                try:
                    _vol_ok_ssot = _check_volume_ok(
                        self.buffer_5m, len(self.buffer_5m) - 1,
                    )
                except Exception:
                    _vol_ok_ssot = bool(vol_result.get("volume_ok", False))
                comp["volume_ok"] = _vol_ok_ssot
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

        # O(1) append to deque, invalidate DataFrame cache
        self._buffer_5m_deque.append({
            "timestamp": candle["timestamp"],
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
        })
        self._buffer_5m_dirty = True

        # Increment candle-time bar counter (used for rate limiting + dedup)
        self._total_bars += 1

        # Update counterfactual outcome trackers for rejected signals
        self._update_rejection_outcomes(candle)

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
                # Count market-open candles (only increments when a real candle
                # arrives — respects market hours automatically).
                trade["_candles_seen"] = trade.get("_candles_seen", 0) + 1
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
                # Dollar-denominated unrealized PnL for dashboard display
                trade["unrealized_pnl_usd"] = unrealized_pnl_pct * entry_price * float(trade.get("qty", 0))

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

                        # Safety gates — scaled by trade style
                        _exit_conf_threshold = self.rl_suite.exit_threshold
                        _trade_style = trade.get("style", STYLE_DAY)
                        _style_max_candles = STYLE_CONFIG.get(_trade_style, STYLE_CONFIG[STYLE_DAY]).get("max_hold_candles", 288)
                        # Min bars before ML exit allowed: 10% of max hold time
                        # Scalp: 7 bars (35min), Day: 29 bars (2.4h), Swing: 144 bars (12h)
                        _exit_min_bars = max(6, int(_style_max_candles * 0.10))
                        # Min favorable RR before exit: higher for longer-term trades
                        _exit_min_favorable = {"scalp": 0.5, "day": 1.0, "swing": 1.5}.get(_trade_style, 0.5)
                        _exit_be_priority = True

                        bars_count = int(trade.get("_bars_held_count", bars_held))
                        unrealized_rr = bar_features.get("bar_unrealized_rr", 0)
                        be_trig = bool(trade.get("be_triggered", False))
                        # Scalp-Day Hybrid parity (2026-04-20): fallback 2.5 matches
                        # backtest evergreen_params.json be_ratchet_r=2.5 (was 1.5).
                        be_ratchet_r = self.rl_suite.min_be_rr if self.rl_suite.be_enabled else 2.5

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

    # Per-asset-class fallback medians for symbol_*_rank features.
    # Used when models/symbol_ranks.json is missing or symbol not in JSON.
    # Verified from training parquets (2026-04-01).
    _AC_FALLBACK_MEDIANS: dict[str, dict[str, float]] = {
        "crypto":      {"volatility": 0.246, "liquidity": 0.709, "spread": 0.246},
        "forex":       {"volatility": 0.481, "liquidity": 0.519, "spread": 0.481},
        "stocks":      {"volatility": 0.510, "liquidity": 0.490, "spread": 0.510},
        "commodities": {"volatility": 0.333, "liquidity": 0.667, "spread": 0.667},
    }

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

    @staticmethod
    def _training_style_alignment_score(
        components: dict,
        asset_class: str,
    ) -> float:
        """Compute alignment_score using the SAME formula as generate_rl_data.

        This is a 9-component boolean-weighted sum that matches the training
        distribution the XGBoost entry filter was trained on. The live
        13-component score (with ADX/session/momentum/tf/freshness extras
        and continuous zone_quality weighting) is a DIFFERENT distribution
        — feeding it as the feature causes train/inference mismatch and
        breaks the model's confidence calibration.

        Mirrors generate_rl_data.py:793-813 vectorized formula, per-bar.
        """
        # Crypto (non-forex) weights — matches training default path
        w_bias, w_strong, w_h4, w_h4poi = 0.12, 0.08, 0.08, 0.08
        w_h1, w_choch, w_zone, w_trigger, w_vol = 0.08, 0.06, 0.15, 0.15, 0.10

        has_bias = 1.0 if components.get("bias") else 0.0
        bias_strong = 1.0 if components.get("bias_strong") else 0.0
        h4_confirms = 1.0 if components.get("h4_confirms") else 0.0
        h4_poi = 1.0 if components.get("h4_poi") else 0.0
        h1_confirms = 1.0 if components.get("h1_confirms") else 0.0
        h1_choch = 1.0 if components.get("h1_choch") else 0.0
        # Training uses has_entry_zone boolean (from FVG/OB on 15m). At live
        # entry time we only reach this code when an entry exists, so set to 1.
        has_entry_zone = 1.0
        precision_trigger = 1.0 if components.get("precision_trigger") else 0.0
        # Crypto uses real volume — no tick-volume neutral override needed
        volume_ok_val = 1.0 if components.get("volume_ok") else 0.0

        alignment = (
            has_bias * w_bias
            + has_bias * bias_strong * w_strong
            + h4_confirms * w_h4
            + h4_poi * w_h4poi
            + h1_confirms * w_h1
            + h1_confirms * h1_choch * w_choch
            + has_entry_zone * w_zone
            + precision_trigger * w_trigger
            + volume_ok_val * w_vol
        )
        return min(alignment, 1.0)

    @staticmethod
    def _training_style_adx_1h(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Compute the single-value ADX at the last bar using the EXACT formula
        from generate_rl_data._compute_adx (which is a custom Wilder variant).

        The canonical Wilder ADX in filters/trend_strength.compute_adx computes
        slightly different smoothed DI values, causing train/inference mismatch
        on the adx_1h feature. The XGBoost model was trained on the training
        parquet's values, so live must match training.

        Mirrors generate_rl_data.py:363-397.
        """
        n = len(highs)
        if n < period * 2 + 1:
            return 25.0  # default neutral
        from utils.indicators import compute_atr_wilders
        atr = compute_atr_wilders(highs, lows, closes, period)

        # +DM / -DM
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = np.zeros(n, dtype=np.float64)
        minus_di = np.zeros(n, dtype=np.float64)
        if atr[period] > 0:
            plus_di[period] = np.mean(plus_dm[:period]) / atr[period] * 100
            minus_di[period] = np.mean(minus_dm[:period]) / atr[period] * 100
        for i in range(period + 1, n):
            if atr[i] > 0:
                plus_di[i] = min((plus_di[i - 1] * (period - 1) + plus_dm[i - 1]) / period / atr[i] * 100, 100.0)
                minus_di[i] = min((minus_di[i - 1] * (period - 1) + minus_dm[i - 1]) / period / atr[i] * 100, 100.0)

        dx = np.zeros(n, dtype=np.float64)
        for i in range(period, n):
            s = plus_di[i] + minus_di[i]
            dx[i] = abs(plus_di[i] - minus_di[i]) / s * 100 if s > 0 else 0

        adx_arr = np.full(n, 25.0, dtype=np.float64)
        if n > period * 2:
            adx_arr[period * 2] = np.mean(dx[period:period * 2 + 1])
            for i in range(period * 2 + 1, n):
                adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period
        return float(adx_arr[-1])

    def _build_xgb_features(self, components: dict, score: float) -> dict[str, float]:
        """Build 40-feature dict matching XGBoost model expectations.

        Mirrors the feature computation in backtest/generate_rl_data.py
        but only for the current (last) bar.  has_entry_zone and
        alignment_score are kept for TP/BE/sizing models; the entry_quality
        model excludes them via feat_names reindex.

        The `score` parameter is the LIVE 13-component alignment score used
        for the acceptance gate. It is NOT used for the alignment_score
        feature — that is recomputed via _training_style_alignment_score to
        match the training distribution (see method docstring).
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

        # Crypto uses real volume — no tick-volume neutral override needed

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
        # CRITICAL: must use the SAME formula as generate_rl_data._compute_adx,
        # which is a custom Wilder variant that produces slightly different
        # values than filters/trend_strength.compute_adx (used in scoring).
        # The XGBoost model was trained on the training parquet's ADX values,
        # so live must compute matching values to avoid feature distribution
        # drift. Falls back to component value for the SCORING adx_strong flag.
        if not self.buffer_1h.empty and len(self.buffer_1h) >= 30:
            try:
                _h1 = self.buffer_1h["high"].values.astype(np.float64)
                _l1 = self.buffer_1h["low"].values.astype(np.float64)
                _c1 = self.buffer_1h["close"].values.astype(np.float64)
                _adx_train = self._training_style_adx_1h(_h1, _l1, _c1, period=14)
                feat["adx_1h"] = min(_adx_train / 50.0, 2.0)
            except Exception:
                feat["adx_1h"] = min(float(components.get("adx_value", 25.0)) / 50.0, 2.0)
        else:
            feat["adx_1h"] = 0.5  # matches training's default

        # ── Alignment score ─────────────────────────────────────────
        # CRITICAL: the `score` parameter is the live 13-component gate score
        # (with ADX/session/momentum/tf/freshness extras and continuous
        # zone_quality weighting). The training alignment_score column in
        # parquet uses a simpler 9-component boolean-weighted sum. Feeding
        # the 13-component score as the feature causes train/inference
        # distribution drift. Use the training-matching formula instead.
        feat["alignment_score"] = self._training_style_alignment_score(
            components, self.asset_class,
        )

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

        # ── Symbol ranks (pre-computed, loaded from JSON at init) ────
        feat["symbol_volatility_rank"] = self._symbol_volatility_rank
        feat["symbol_liquidity_rank"] = self._symbol_liquidity_rank
        feat["symbol_spread_rank"] = self._symbol_spread_rank

        # ── Asset class ID ──────────────────────────────────────────
        feat["asset_class_id"] = float(self._XGB_AC_MAP.get(self.asset_class, 0))

        # ── Trade style ID (set properly at call site from _classify_trade_style) ──
        feat["style_id"] = 0.5  # default=day; overridden after call

        return feat

    def validate_xgb_features(self, model_feat_names: list[str]) -> bool:
        """Validate feature alignment at startup. Returns True if no missing features."""
        dummy = self._build_xgb_features({}, 0.0)
        live_keys = set(dummy.keys())
        model_keys = set(model_feat_names)
        missing = model_keys - live_keys
        extra = live_keys - model_keys

        if missing:
            self.logger.error(
                "XGB FEATURE MISMATCH: %d features MISSING from live: %s — model will fill with 0.0!",
                len(missing), sorted(missing),
            )
        if extra:
            self.logger.warning(
                "XGB features in live but NOT in model (will be dropped): %s",
                sorted(extra),
            )
        if not missing and not extra:
            self.logger.info("XGB features validated: %d/%d exact match", len(model_keys), len(model_keys))
        elif not missing:
            self.logger.info(
                "XGB features validated: %d/%d model features present (%d extra in live)",
                len(model_keys), len(model_keys), len(extra),
            )
        # Cross-check against shared schema
        schema_missing, schema_extra = validate_against_model(model_feat_names)
        if schema_missing:
            self.logger.warning(
                "Schema drift: model has features not in schema: %s", sorted(schema_missing),
            )
        if schema_extra:
            self.logger.warning(
                "Schema drift: schema has features not in model: %s", sorted(schema_extra),
            )
        return len(missing) == 0

    # ── Signal preparation (called from on_candle) ──────────────────

    # Class-level cache for component toggles (shared across all 112 PaperBot instances)
    _component_cache: dict[str, bool] = {}
    _component_cache_ts: float = 0.0
    _COMPONENT_CACHE_TTL: float = 30.0

    def _check_component_enabled(self, component: str) -> bool:
        """Check if a component is enabled via dashboard toggles (cached 30s)."""
        now = time.monotonic()
        if now - PaperBot._component_cache_ts > PaperBot._COMPONENT_CACHE_TTL:
            toggles_path = Path("live_results/component_toggles.json")
            if toggles_path.exists():
                try:
                    with open(toggles_path) as f:
                        PaperBot._component_cache = json.load(f)
                except Exception:
                    PaperBot._component_cache = {}
            else:
                PaperBot._component_cache = {}
            PaperBot._component_cache_ts = now
        return PaperBot._component_cache.get(component, True)

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
        4. XGBoost confidence gate
        5. ATR-based minimum SL distance (not just fixed %)
        """
        # ── Pause flag check ──────────────────────────────────────────
        if Path("live_results/.pause_flag").exists():
            PaperBot._prefilter_stats["paused"] += 1
            return  # Paused — don't generate new signals

        # ── Trading hours check (forex/stocks have limited hours) ────
        if self.adapter is not None and not self.adapter.is_market_open(self.symbol):
            PaperBot._prefilter_stats["market_closed"] += 1
            self._pending_signal = None
            return

        # (no early block — style check happens after classification at line ~2191)

        # ── Circuit breaker check ──────────────────────────────────
        if self.circuit_breaker is not None:
            can_trade, cb_reason = self.circuit_breaker.can_trade(self.asset_class)
            if not can_trade:
                PaperBot._prefilter_stats["circuit_breaker"] += 1
                self._pending_signal = None
                self.logger.info("CIRCUIT BREAKER SKIP %s: %s", symbol, cb_reason)
                return

        # ── Volatility check (soft — training has no volatility gate) ─
        tradeable, daily_atr, fivem_atr = self._check_volatility()
        _vol_penalty = 0.05 if not tradeable else 0.0

        # NOTE 2026-04-20: The 0.5×avg(20) volume pre-filter that used to live
        # here was removed as part of the Backtest/Live parity restore. The
        # backtest path (strategies/smc_multi_style.py) has no such hard gate
        # — it uses only the binary `volume_ok` flag inside the alignment
        # score (weight 0.10). On live Binance Testnet the 0.5×avg threshold
        # was rejecting ~95 % of candles before they ever reached the SSOT
        # gate (PREFILTER-STATS: vol_low=57/60 within 5 min of the first
        # heartbeat after the SSOT-routing fix). Volume quality is already
        # captured by the continuous `volume_ok` / volume_score inside
        # _multi_tf_alignment_score + compute_alignment_score.

        # ── Multi-TF alignment score (granular for features, SSOT for gate) ──
        # The rich 13-component score (continuous multipliers on zone_quality,
        # volume_score, D/P penalty, volatility penalty) diverges ~0.07 below
        # the SSOT formula that the backtest + training parquet use
        # (strategies/smc_multi_style.py:1651, backtest/generate_rl_data.py).
        # See .omc/research/alignment_drought_probe.py for the 43205-bar
        # calibration run showing ~36 projected triggers/day on 30 symbols
        # under SSOT vs 0 observed under Live-formula over 16h.
        # Fix: gate on SSOT formula (identical to backtest), keep rich score
        # as feature for dashboard/XGB context.
        rich_score, direction, components = self._multi_tf_alignment_score(candle)
        rich_score -= _vol_penalty  # vol penalty preserved on rich score
        components["rich_score"] = rich_score
        _daily_bias = components.get("daily_bias", "neutral")
        score = compute_alignment_score(
            daily_bias=_daily_bias,
            h1_confirms=bool(components.get("h1_confirms")),
            entry_zone=components.get("entry_zone"),
            precision_trigger=bool(components.get("precision_trigger")),
            style_weight=1.0,
            bias_strong=bool(components.get("bias_strong")),
            h4_confirms=bool(components.get("h4_confirms")),
            h4_poi=bool(components.get("h4_poi")),
            h1_choch=bool(components.get("h1_choch")),
            volume_ok=bool(components.get("volume_ok")),
            asset_class=self.asset_class,
        )
        components["gate_score"] = score
        if score < self.alignment_threshold:
            _near_miss_floor = max(0.40, self.alignment_threshold - 0.15)
            if _daily_bias == "neutral":
                PaperBot._prefilter_stats["neutral_bias"] += 1
            elif score >= _near_miss_floor:
                PaperBot._prefilter_stats["near_miss_logged"] += 1
                self.logger.info(
                    "NEAR-MISS ALIGNMENT %s | class=%s score=%.3f rich=%.3f thresh=%.2f dir=%s | flags=%s",
                    symbol, self.asset_class, score, rich_score,
                    self.alignment_threshold, direction,
                    {k: v for k, v in components.items() if not k.startswith("_")},
                )
            else:
                PaperBot._prefilter_stats["low_score"] += 1
            self._pending_signal = None
            return
        PaperBot._prefilter_stats["gate_passed"] += 1

        price = candle["close"]
        if price <= 0:
            self._pending_signal = None
            return

        # ── SL: entry zone or swing fallback (matching training pipeline) ──
        entry_zone = components.get("entry_zone")
        liq_range = self.liq_range
        if entry_zone is not None:
            if direction == "long":
                sl = entry_zone["bottom"] * (1 - liq_range)
            else:
                sl = entry_zone["top"] * (1 + liq_range)
        else:
            _lb = 20
            if direction == "long":
                recent_lows = self.buffer_5m["low"].iloc[-_lb:]
                sl = float(recent_lows.min()) * (1 - liq_range)
            else:
                recent_highs = self.buffer_5m["high"].iloc[-_lb:]
                sl = float(recent_highs.max()) * (1 + liq_range)

        sl_dist = abs(price - sl)
        if sl_dist == 0 or sl_dist / price < 0.001:
            self._pending_signal = None
            return

        # ── TP: structure TP from 4H/1H (matching training pipeline) ──
        _htf_4h = _precompute_htf_arrays(self.buffer_4h, self.swing_length) if not self.buffer_4h.empty else None
        _htf_1h = _precompute_htf_arrays(self.buffer_1h, self.swing_length) if not self.buffer_1h.empty else None
        bias = "bullish" if direction == "long" else "bearish"
        # Scalp-Day Hybrid (2026-04-20): min_rr=2.0 matches backtest
        # (strategies/smc_multi_style.py:1710) — was 1.0 historically with a
        # separate `rr < 1.0` check after TP clamping. Since TP clamping is
        # removed, a single 2.0 upfront gate is cleaner AND parity-matched.
        tp, _tp_source = _find_structure_tp_safe(
            _htf_4h, _htf_1h,
            vlen_4h=len(self.buffer_4h) if not self.buffer_4h.empty else 0,
            vlen_1h=len(self.buffer_1h) if not self.buffer_1h.empty else 0,
            entry_price=price,
            bias=bias,
            sl_dist=sl_dist,
            min_rr=2.0,
        )

        # Classify style from the resulting SL/TP
        style = self._classify_trade_style(price, sl, tp)

        # ── Per-style limit: max 1 trade per style on same symbol ──
        # Different styles (scalp + day, day + swing) allowed simultaneously.
        # Circuit breaker + rate limit handle overall risk.
        if any(t.get("style") == style for t in self._active_trades):
            return

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
        # Scalp-Day Hybrid (2026-04-20): replaced style_cfg["min_sl_pct"] with constant.
        # Backtest min SL floor comes from SMC structure, not a per-style config.
        # 0.2% is a sensible absolute floor to avoid 1-bar SLs on very calm markets.
        pct_min_sl = price * MIN_SL_PCT_FLOOR
        min_sl_dist = max(atr_min_sl, pct_min_sl, tick_min_sl)

        if sl_dist < min_sl_dist:
            old_sl_pct = sl_dist / price * 100 if price > 0 else 0
            sl_dist = min_sl_dist
            sl = (price - sl_dist) if direction == "long" else (price + sl_dist)
            self.logger.info(
                "[SL WIDENED] %s %s | from %.3f%% → %.3f%% | reason=min_sl floor (atr=%.3f%% pct=%.3f%% tick=%.3f%%)",
                direction.upper(), symbol, old_sl_pct, sl_dist / price * 100,
                (atr_min_sl / price * 100) if price > 0 else 0,
                (pct_min_sl / price * 100) if price > 0 else 0,
                (tick_min_sl / price * 100) if price > 0 else 0,
            )

        # ── SL safety cap (training has no style-based SL limits) ──
        # Just cap at swing max (5%) as safety — no style upgrades needed
        max_sl_pct = 0.05  # 5% absolute max SL
        max_sl_dist = price * max_sl_pct
        if sl_dist > max_sl_dist:
            self._pending_signal = None
            self.logger.info(
                "[SL REJECT] %s %s | style=%s | sl=%.3f%% > max_cap=5.000%% | price=%.6f sl=%.6f",
                direction.upper(), symbol, style,
                sl_dist / price * 100, price, sl,
            )
            return

        if tp_dist <= 0:
            self._pending_signal = None
            self.logger.info(
                "[TP REJECT] %s %s | style=%s | tp_dist<=0 (price=%.6f tp=%.6f)",
                direction.upper(), symbol, style, price, tp,
            )
            return

        # ── TP clamping REMOVED 2026-04-20 (Scalp-Day Hybrid data-driven fix) ──
        # Backtest has no TP clamping — structural SMC TP is authoritative.
        # Data showed 63% of profitable backtest setups (86 of 136 trades) would have
        # been killed by the old max_tp_pct=6% (DAY) / 1.5% (SCALP) clamps.
        # Safety net: 5% max-SL cap above + min_rr=2.0 below.

        # ── RR check (matches backtest min_rr=2.0 + post-widening safety) ──
        # _find_structure_tp_safe already enforces min_rr=2.0 at TP selection,
        # but ATR-floor SL-widening above can reduce effective RR. Re-check.
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < 2.0:
            self._pending_signal = None
            self.logger.info(
                "[RR REJECT] %s %s | style=%s | rr=%.2f < 2.0 | sl_pct=%.3f%% tp_pct=%.3f%% | reason=sl_widened_by_floor",
                direction.upper(), symbol, style, rr,
                sl_dist / price * 100, tp_dist / price * 100,
            )
            return

        # ── Signal rate limiting (per-class safety throttle) ──────
        # Only counts XGB-accepted signals (moved to after XGB gate below).
        # Rejected signals don't consume rate limit slots.
        # Uses candle-time (bar count) — 48 bars = 4h in 5m candles.
        cutoff_bar = self._total_bars - 48
        self._recent_signal_bars = [
            b for b in self._recent_signal_bars if b > cutoff_bar
        ]
        if len(self._recent_signal_bars) >= self._max_signals_per_4h:
            self.logger.info(
                "RATE-LIMITED %s | class=%s | %d signals in 48 bars (max=%d)",
                symbol, self.asset_class,
                len(self._recent_signal_bars), self._max_signals_per_4h,
            )
            self._pending_signal = None
            return

        # ── Final style constraint validation ─────────────────────
        ok, reject_reason = self._validate_style_constraints(style, price, sl, tp)
        if not ok:
            self._pending_signal = None
            sl_pct_log = abs(price - sl) / price if price > 0 else 0
            tp_pct_log = abs(tp - price) / price if price > 0 else 0
            rr_log = tp_pct_log / sl_pct_log if sl_pct_log > 0 else 0
            self.logger.info(
                "[STYLE REJECT] %s %s | style=%s | price=%.6f sl=%.6f tp=%.6f | sl_pct=%.3f%% tp_pct=%.3f%% rr=%.2f | fail=%s",
                direction.upper(), symbol, style, price, sl, tp,
                sl_pct_log * 100, tp_pct_log * 100, rr_log, reject_reason,
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
        # Style encoding for XGBoost (matches training: scalp=0.0, day=0.5, swing=1.0)
        _style_map = {STYLE_SCALP: 0.0, STYLE_DAY: 0.5, STYLE_SWING: 1.0}
        _xgb_features["style_id"] = _style_map.get(style, 0.5)
        # Legacy PPO obs vector — no longer consumed (PPO brain removed).
        _ppo_obs = np.zeros(24, dtype=np.float32)

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
            "components": components,
            "daily_atr_pct": daily_atr,
            "obs": _ppo_obs,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "ref_price": price,
            "features": _xgb_features,
        }
        self.logger.info(
            "PENDING %s %s [%s] | zone=[%.6f, %.6f] SL=%.6f TP=%.6f "
            "RR=%.1f score=%.2f daily_atr=%.3f%%",
            style.upper(), direction.upper(), symbol,
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
        # ── Dynamic risk budget check ────────────────────────────
        # Instead of fixed per-style limits, check if DD budget allows
        # another trade. The closer to DD limits, the fewer trades opened.
        if self.circuit_breaker is not None:
            # Use the risk_pct that this trade would use (conservative estimate)
            _est_risk = 0.005  # conservative estimate: 0.5%
            allowed, reason = self.circuit_breaker.risk_budget_allows(_est_risk)
            if not allowed:
                self.logger.info("RISK BUDGET BLOCK %s | %s", symbol, reason)
                # Record for counterfactual learning before discarding
                self._record_capacity_rejected(sig, reason)
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
        style = sig.get("style", STYLE_DAY)

        # ── RL Brain gate ────────────────────────────────────────────
        # XGBoost models are pre-trained (19M rows) — no warmup needed.
        use_ppo_brain = False  # PPO gate removed (v2)
        rl_tracked = False
        rl_trade_id: str | None = None
        take_trade = True
        rl_confidence = 1.0

        # Student-brain overrides populated iff student is enabled.
        # When `student_used` is True the legacy rl_suite entry/SL/TP/size
        # blocks below are skipped entirely — the student is authoritative.
        student_used: bool = False
        student_sl_rr_mult: float = 1.0  # multiplies strategy sl_dist → final sl_dist
        student_tp_rr_mult: float = 1.0  # multiplies strategy sl_dist → final tp_dist (NOT tp_dist!)
        student_size_mult: float = 1.0

        # ── Student Brain (unified Teacher-Student) ──────────────────
        # Replaces rl_suite entry filter + SL adjuster + TP optimizer +
        # position sizer when enabled. Runs ONE multi-head inference and
        # sets SL/TP/size from hindsight-optimal regression heads.
        if self.student_brain is not None and self.student_brain.enabled:
            sig_features = sig.get("features", {})
            pred = self.student_brain.predict(sig_features)
            rl_confidence = float(pred.entry_prob)
            if not pred.accept:
                self._rl_rejected += 1
                _total_rl = self._rl_accepted + self._rl_rejected
                _rate = self._rl_accepted / _total_rl * 100 if _total_rl > 0 else 0
                self.logger.info(
                    "STUDENT REJECT %s prob=%.3f sl=%.2fR tp=%.2fR | accepted=%d rejected=%d rate=%.0f%%",
                    symbol, pred.entry_prob, pred.sl_rr, pred.tp_rr,
                    self._rl_accepted, self._rl_rejected, _rate,
                )
                # Log rejected signal for the counterfactual loop (same pipeline as rl_suite)
                try:
                    self._record_and_track_rejection(
                        symbol=symbol, direction=direction,
                        entry_price=price, sl_price=sl, tp_price=tp,
                        xgb_confidence=rl_confidence, alignment_score=score,
                        entry_features=sig_features,
                    )
                except Exception as exc:
                    self.logger.debug("Failed to log rejected signal: %s", exc)
                self._pending_signal = None
                return

            # Student accepted the setup — stash its levels for later use and
            # mark the flag so the legacy rl_suite blocks below are skipped.
            student_used = True
            student_sl_rr_mult = pred.sl_rr
            student_tp_rr_mult = pred.tp_rr
            student_size_mult = pred.size
            self._rl_accepted += 1
            self._recent_signal_bars.append(self._total_bars)
            self.logger.info(
                "STUDENT ACCEPT %s prob=%.3f sl=%.2fR tp=%.2fR size=%.2f score=%.2f",
                symbol, pred.entry_prob, pred.sl_rr, pred.tp_rr, pred.size, score,
            )

        # XGBoost entry filter (RLBrainSuite) — legacy path, used only when student disabled
        if not student_used and self.rl_suite is not None and self.rl_suite.entry_filter_enabled and self._check_component_enabled("entry_filter"):
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
                # ── DEBUG: Feature parity dump ──────────────────────
                # Write the full feature vector + buffer state for the first
                # ~10 rejects to /root/bot/debug/. Used to diagnose train/
                # inference mismatch. Disabled after dumping enough.
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    _dbg_dir = _Path("debug/feature_parity")
                    _dbg_dir.mkdir(parents=True, exist_ok=True)
                    _dbg_count = sum(1 for _ in _dbg_dir.glob("*.json"))
                    if _dbg_count < 30:
                        _last_5m_ts = (
                            str(self.buffer_5m["timestamp"].iloc[-1])
                            if not self.buffer_5m.empty else None
                        )
                        _last_1d_ts = (
                            str(self.buffer_1d["timestamp"].iloc[-1])
                            if not self.buffer_1d.empty else None
                        )
                        _buf_1d_tail = (
                            self.buffer_1d.tail(10)[["timestamp","open","high","low","close"]].to_dict(orient="records")
                            if not self.buffer_1d.empty else []
                        )
                        # Convert timestamps to ISO strings
                        for _r in _buf_1d_tail:
                            _r["timestamp"] = str(_r["timestamp"])
                        _dump = {
                            "symbol": symbol,
                            "asset_class": self.asset_class,
                            "log_ts": _last_5m_ts,
                            "score_live_13comp": float(score),
                            "rl_confidence": float(rl_confidence),
                            "features": {
                                k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
                                else None
                                for k, v in sig.get("features", {}).items()
                            },
                            "components_flags": {
                                k: bool(v) if isinstance(v, (bool,))
                                else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else None)
                                for k, v in sig.get("components", {}).items()
                                if not k.startswith("_") and not isinstance(v, (dict, list, pd.DataFrame))
                            },
                            "buffer_1d_len": int(len(self.buffer_1d)),
                            "buffer_1d_tail": _buf_1d_tail,
                            "buffer_4h_len": int(len(self.buffer_4h)),
                            "buffer_1h_len": int(len(self.buffer_1h)),
                            "buffer_5m_len": int(len(self.buffer_5m)),
                            "buffer_5m_last_ts": _last_5m_ts,
                            "buffer_1d_last_ts": _last_1d_ts,
                        }
                        _fname = _dbg_dir / f"{symbol}_{int(pd.Timestamp.utcnow().timestamp())}.json"
                        with open(_fname, "w") as _f:
                            _json.dump(_dump, _f, indent=2, default=str)
                        self.logger.debug("Feature parity dump: %s", _fname)
                except Exception as _exc:
                    self.logger.debug("Feature dump failed: %s", _exc)
                # Alert: abnormal acceptance rate after 20+ decisions
                if _total_rl >= 20 and (_rate < 10 or _rate > 95):
                    self.logger.debug(
                        "RL ALERT: acceptance rate %.0f%% (%d/%d) — check feature extraction",
                        _rate, self._rl_accepted, _total_rl,
                    )
                # Near-miss: close rejections visible on dashboard
                if rl_confidence >= 0.45:
                    self.logger.info(
                        "NEAR-MISS XGB %s conf=%.3f score=%.2f thresh=%.2f | %s %s",
                        symbol, rl_confidence, score,
                        self.rl_suite.confidence_threshold,
                        self.asset_class, direction,
                    )
                # Log rejected signal + start counterfactual outcome tracker
                try:
                    self._record_and_track_rejection(
                        symbol=symbol,
                        direction=direction,
                        entry_price=price,
                        sl_price=sl,
                        tp_price=tp,
                        xgb_confidence=rl_confidence,
                        alignment_score=score,
                        entry_features=sig.get("features"),
                    )
                except Exception as exc:
                    self.logger.debug("Failed to log rejected signal: %s", exc)
                self._pending_signal = None
                return
            self._rl_accepted += 1
            self._recent_signal_bars.append(self._total_bars)  # rate limit counts accepted only (candle-time)
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

        # ── Student-brain SL/TP override (unified Teacher-Student) ────
        rl_be_level = 0.0
        sl_dist_for_tp = sl_dist  # preserve original for TP model (trained on unadjusted SL)
        # Track ORIGINAL (pre-adjustment) SL for journal + SL-model label derivation.
        # bug_008 fix: without this, sl_original stored the ADJUSTED value, and the SL
        # model computed label_mae_rr against a too-small denominator → self-caused stops
        # got mislabeled as KEEP instead of WIDEN (DOT 2026-04-17 case).
        sl_pre_adjustment = sl
        sl_dist_pre_adjustment = sl_dist
        sl_was_adjusted = False

        if student_used:
            # Student's sl_rr and tp_rr are MULTIPLES of the strategy's
            # original sl_dist (the R-unit anchor). This matches the Teacher v2
            # label formula: optimal_sl_rr = clip(MAE × 1.10 / strategy_sl_dist).
            if student_sl_rr_mult != 1.0:
                new_sl_dist = sl_dist * student_sl_rr_mult
                old_sl = sl
                sl = price - new_sl_dist if direction == "long" else price + new_sl_dist
                sl_dist = new_sl_dist
                sl_was_adjusted = True
                self.logger.info(
                    "STUDENT SL %s: %.6f → %.6f (%.2fR × base %.4f = %.4f)",
                    symbol, old_sl, sl, student_sl_rr_mult,
                    sl_dist_pre_adjustment, new_sl_dist,
                )
            # TP from student is also a multiple of the strategy sl_dist.
            new_tp_dist = sl_dist_pre_adjustment * student_tp_rr_mult
            old_tp = tp
            tp = price + new_tp_dist if direction == "long" else price - new_tp_dist
            tp_dist = new_tp_dist
            self.logger.info(
                "STUDENT TP %s: %.6f → %.6f (%.2fR × base %.4f = %.4f)",
                symbol, old_tp, tp, student_tp_rr_mult,
                sl_dist_pre_adjustment, new_tp_dist,
            )

        # ── RL SL adjustment (legacy path — used only when student disabled) ──
        if not student_used and self.rl_suite is not None:
            if self.rl_suite.sl_enabled and sl_dist > 0 and self._check_component_enabled("sl_adjuster"):
                sl_dist_orig = sl_dist
                adjusted_sl_dist = self.rl_suite.predict_sl_adjustment(
                    sig.get("features", {}), sl_dist, price,
                )
                if abs(adjusted_sl_dist - sl_dist) > 0.01 * sl_dist:
                    old_sl = sl
                    sl = price - adjusted_sl_dist if direction == "long" else price + adjusted_sl_dist
                    sl_dist = adjusted_sl_dist
                    sl_was_adjusted = True
                    self.logger.info(
                        "RL SL adjusted %s: %.6f -> %.6f (dist %.4f -> %.4f)",
                        symbol, old_sl, sl, sl_dist_orig, adjusted_sl_dist,
                    )

        # ── RL TP adjustment (legacy path — used only when student disabled) ──
        if not student_used and self.rl_suite is not None:
            if self.rl_suite.tp_enabled and sl_dist > 0 and self._check_component_enabled("tp_optimizer"):
                planned_tp_rr = tp_dist / sl_dist_for_tp  # use original SL (model trained on unadjusted)
                adjusted_tp_rr = self.rl_suite.predict_tp_adjustment(
                    sig.get("features", {}), planned_tp_rr,
                )
                if abs(adjusted_tp_rr - planned_tp_rr) > 0.01:
                    old_tp = tp
                    tp = price + adjusted_tp_rr * sl_dist_for_tp if direction == "long" else price - adjusted_tp_rr * sl_dist_for_tp
                    tp_dist = abs(tp - price)
                    self.logger.info(
                        "RL TP adjusted %s: %.6f -> %.6f (RR %.1f -> %.1f)",
                        symbol, old_tp, tp, planned_tp_rr, adjusted_tp_rr,
                    )

        # BE manager runs regardless of student/rl_suite choice — it's orthogonal
        # (dynamic SL-to-BE ratchet triggered by running PnL in R, not by entry-time features).
        if self.rl_suite is not None:
            if self.rl_suite.be_enabled and sl_dist > 0 and self._check_component_enabled("be_manager"):
                # Match training formula: cost_rr in R-multiples
                _tc = _TRAIN_COMMISSION.get(self.asset_class, 0.0004)
                _ts = _TRAIN_SLIPPAGE.get(self.asset_class, 0.0002)
                cost_rr = (price * (_tc + _ts) * 2) / sl_dist_for_tp  # use original SL (model trained on unadjusted)
                rl_be_level = self.rl_suite.predict_be_level(
                    sig.get("features", {}), cost_rr,
                )

        _bracket_result = (
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
                style=style,
                features=sig.get("features", {}),
                xgb_confidence=rl_confidence,
                size_multiplier=student_size_mult,  # 1.0 when student disabled
            )
        )
        # Unpack — oanda_trade_id is optional (7th element, None for non-OANDA)
        order_id = _bracket_result[0]
        sl_order_id = _bracket_result[1]
        tp_order_id = _bracket_result[2]
        qty = _bracket_result[3]
        risk_pct = _bracket_result[4]
        used_leverage = _bracket_result[5]
        oanda_trade_id = _bracket_result[6] if len(_bracket_result) > 6 else None

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
            "sl": sl,                               # actually-used SL (possibly adjusted)
            "original_sl": sl_pre_adjustment,       # pre-RL-adjustment SL (planned by strategy)
            "original_sl_dist": sl_dist_pre_adjustment,
            "sl_was_adjusted": sl_was_adjusted,
            "tp": tp,
            "qty": qty,
            "leverage": used_leverage,
            "risk_pct": risk_pct,
            "entry_time": datetime.now(timezone.utc),
            "score": score,
            "style": style,
            "order_id": order_id,
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id,
            "oanda_trade_id": oanda_trade_id,
            "sl_attached": self.adapter.supports_attached_sl_tp if self.adapter else False,
            "rl_tracked": rl_tracked,
            "rl_trade_id": rl_trade_id,
            "rl_confidence": rl_confidence,
            "rl_be_level": rl_be_level,
            "be_triggered": False,
        }
        self._active_trades.append(trade_info)
        # (cooldown code removed — circuit breaker + rate limit are sufficient)

        # ── Trade Journal: record trade open ─────────────────────────
        trade_id_for_journal = rl_trade_id or order_id or f"t_{len(self._active_trades)}"
        # Backfill trade_id so candle handler and close hook use same key
        self._active_trades[-1]["rl_trade_id"] = trade_id_for_journal
        if self.journal is not None:
            try:
                # RR target measured against the strategy-planned SL (pre-adjustment).
                rr_target = abs(tp - price) / abs(price - sl_pre_adjustment) if abs(price - sl_pre_adjustment) > 0 else 3.0
                self.journal.open_trade(
                    trade_id=trade_id_for_journal,
                    symbol=symbol,
                    asset_class=self.asset_class,
                    direction=direction,
                    style=style,
                    tier="",
                    entry_time=datetime.now(timezone.utc),
                    entry_price=price,
                    sl_original=sl_pre_adjustment,   # TRUE pre-adjustment SL — needed for correct SL-label R-units
                    sl_used=sl if sl_was_adjusted else None,
                    tp=tp,
                    score=score,
                    rr_target=rr_target,
                    leverage=used_leverage,
                    risk_pct=risk_pct,
                    entry_features=sig.get("features", {}),
                    xgb_confidence=rl_confidence,
                )
            except Exception as exc:
                self.logger.debug("journal.open_trade error: %s", exc)

        self.logger.info(
            "OPEN [%s] %s %s @ %.6f | SL=%.6f TP=%.6f RR=%.1f | qty=%.4f "
            "risk=%.2f%% lev=%dx score=%.2f bal=%.2f order=%s",
            style.upper(), direction.upper(), symbol, price, sl, tp,
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
        style: str = STYLE_DAY,
        features: dict[str, Any] | None = None,
        xgb_confidence: float = 0.6,
        size_multiplier: float = 1.0,
    ) -> tuple[str | None, str | None, str | None, float, float, int]:
        """
        Execute a bracket order with confidence-based risk allocation.

        Risk scales linearly with XGB confidence (granular, no steps):
          conf at threshold (0.55) → 0.20%
          conf 0.775              → 0.85%
          conf 1.00               → 1.50%

        ``size_multiplier`` (default 1.0) multiplies the confidence-derived risk,
        clamped back into [min_risk, max_risk]. Used by the Student brain to
        up/down-scale position size based on its ``optimal_size`` head.

        Returns:
            tuple: (order_id, sl_order_id, tp_order_id, quantity, used_risk_pct, applied_leverage)
        """
        # === ALIGNMENT-BASED DYNAMIC RISK (core.sizing SSOT) ===
        # base_risk scales linearly with alignment_score (0.25% at threshold,
        # 1.0% at score 1.0). Student size-head multiplies on top when active.
        # Result is clamped into [DEFAULT_RISK_PER_TRADE, MAX_RISK_PER_TRADE]
        # for funded-compliance.
        rr = tp_dist / sl_dist if sl_dist > EPSILON_SL_DIST else 0.0
        base_risk = compute_risk_fraction(score)
        dynamic_risk = base_risk * float(size_multiplier)
        dynamic_risk = max(DEFAULT_RISK_PER_TRADE, min(dynamic_risk, MAX_RISK_PER_TRADE))

        risk_source = "alignment" if size_multiplier == 1.0 else "alignment+student"
        self.logger.info(
            "[RISK] %s | source=%s score=%.3f size_mult=%.2f RR=%.1f conf=%.3f"
            " → risk=%.3f%% (bounds %.2f%%–%.2f%%)",
            style.upper(), risk_source, score, size_multiplier, rr, xgb_confidence,
            dynamic_risk * 100,
            DEFAULT_RISK_PER_TRADE * 100, MAX_RISK_PER_TRADE * 100,
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

        # ── Asset-class leverage cap ─────────────────────────────
        asset_max_lev = ASSET_MAX_LEVERAGE.get(self.asset_class, 10)
        if planned_leverage > asset_max_lev:
            self.logger.info(
                "Capping leverage %dx → %dx for %s on %s",
                planned_leverage, asset_max_lev, self.asset_class, symbol,
            )
            planned_leverage = asset_max_lev

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
            return (q, q * price) if q > 0 else (0.0, 0.0)

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
                or ("margin" in msg_str and "not sufficient" in msg_str)
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
                    order_id, sl_order_id, tp_order_id, oanda_trade_id = await self._place_bracket_order(
                        symbol, direction, price, sl, tp, qty,
                    )
                    # _place_bracket_order returns order_id=None on silent
                    # rejection (e.g. OANDA precision exceeded). Don't claim
                    # success in that case — treat as failure.
                    if order_id is None:
                        self.logger.error(
                            "[ORDER_FAILED] %s %s | qty=%.6f notional=%.2f — "
                            "broker rejected order silently",
                            direction.upper(), symbol, qty, notional,
                        )
                        return (
                            None, None, None,
                            qty, risk_pct, planned_leverage,
                            None,
                        )
                    self.logger.info(
                        "[ORDER_SUCCESS] %s %s | risk_used=%.2f%% leverage=%dx (%s) "
                        "qty=%.6f notional=%.2f expected_margin=%.2f order=%s",
                        direction.upper(), symbol,
                        risk_pct * 100, planned_leverage, leverage_source,
                        qty, notional, expected_margin,
                        order_id,
                    )
                    return (
                        order_id, sl_order_id, tp_order_id,
                        qty, risk_pct, planned_leverage,
                        oanda_trade_id,
                    )
                except _BracketSLTPFailed as sltp_exc:
                    # Entry was filled but SL/TP failed — position already
                    # flattened (or flatten attempted).  Do NOT retry with a
                    # new entry; that would create zombie positions.
                    self.logger.error(
                        "[BRACKET_ABORT] %s %s — SL/TP failed after entry, "
                        "position flattened. NOT retrying. %s",
                        direction.upper(), symbol, sltp_exc,
                    )
                    return None, None, None, qty, risk_pct, planned_leverage, None
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
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """
        Place a market entry plus SL/TP orders via the exchange adapter.

        Works for all exchanges (Binance, OANDA, Alpaca) through the
        unified ExchangeAdapter interface.

        For OANDA: SL/TP attached to entry via stopLossOnFill/takeProfitOnFill.
        For others: separate SL/TP orders created after entry.

        Returns (entry_id, sl_order_id, tp_order_id, oanda_trade_id).
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

        # Entry market order — attach SL/TP for exchanges that support it (OANDA)
        _attached = self.adapter.supports_attached_sl_tp
        entry_params: dict[str, Any] = {}
        if _attached:
            entry_params["stopLossPrice"] = sl
            entry_params["takeProfitPrice"] = tp

        try:
            entry = await self.adapter.create_market_order(
                symbol, side, qty, entry_params or None,
            )
            entry_id = entry.order_id
            oanda_trade_id = entry.trade_id  # None for non-OANDA
            # Detect silent rejection (OANDA returns a valid OrderResult with
            # None id when rejected, e.g. PRICE_PRECISION_EXCEEDED).
            if entry_id is None:
                entry_status = getattr(entry, "status", "unknown")
                self.logger.error(
                    "ENTRY REJECTED %s %s qty=%.6f | status=%s (no order placed)",
                    side.upper(), symbol, qty, entry_status,
                )
                # Return None to signal failure to the caller
                return None, None, None, None
            self.logger.info(
                "ENTRY %s %s qty=%.6f | id=%s trade_id=%s",
                side.upper(), symbol, qty, entry_id, oanda_trade_id or "n/a",
            )
        except Exception as exc:
            self.logger.error("Entry order FAILED %s %s: %s", side.upper(), symbol, exc)
            raise

        sl_order_id: str | None = None
        tp_order_id: str | None = None

        if _attached:
            # OANDA: SL/TP already attached via stopLossOnFill/takeProfitOnFill
            # No standalone orders created — they auto-cancel when trade closes
            self.logger.info(
                "SL/TP attached to trade %s (stopLossOnFill=%.6f takeProfitOnFill=%.6f)",
                oanda_trade_id or entry_id, sl, tp,
            )
        else:
            # Binance/Alpaca: create separate SL/TP orders
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
                raise _BracketSLTPFailed(f"SL failed after entry: {exc}") from exc

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
                raise _BracketSLTPFailed(f"TP failed after entry: {exc}") from exc

        return entry_id, sl_order_id, tp_order_id, oanda_trade_id

    # ── Fetch real testnet balance ────────────────────────────────

    async def _fetch_balance(self) -> float | None:
        """Return total account equity from the exchange adapter.

        Uses bal.total (account equity) instead of bal.free (buying power)
        so that position sizing is based on full account value, not remaining
        margin.  Note: bal.total includes unrealized PnL on Binance (futures
        totalMarginBalance) and Alpaca (equity); OANDA total is deposited
        balance only.  Acceptable for paper trading — position sizing drift
        from unrealized PnL is negligible at current scale.

        Returns real account equity from the exchange. No multipliers —
        trading always uses real balance for position sizing.
        """
        if self.adapter is None:
            return None
        try:
            bal = await self.adapter.fetch_balance()
            return float(bal.total)
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
        Scalp-Day Hybrid (2026-04-20): hardcoded `day` for backtest parity.

        Data-driven decision (136 OOS backtest trades):
          - Backtest: 100% style="day", median TP=8%, 95p TP=61%, max=79%
          - No style classification in backtest — structural SMC TP/SL is authoritative
          - Old dynamic classifier (scalp<0.5% SL, swing>2%) would have REJECTED 63%
            of profitable setups via max_tp_pct=6% clamping

        Safety is enforced elsewhere:
          - ATR-min-SL floor (adaptive to volatility)
          - tick-min-SL floor (precision safety)
          - 5% absolute max-SL cap
          - min_rr=2.0 global gate
          - 48-bar max_hold timeout
        """
        sl_dist_pct = abs(price - sl) / price if price > 0 else 0
        tp_dist_pct = abs(tp - price) / price if price > 0 else 0

        self.logger.info(
            "[STYLE CLASSIFY] %s | price=%.6f sl=%.6f tp=%.6f | sl_pct=%.3f%% tp_pct=%.3f%% → style=day (hardcoded, Scalp-Day Hybrid)",
            self.symbol, price, sl, tp,
            sl_dist_pct * 100, tp_dist_pct * 100,
        )
        return STYLE_DAY

    def _validate_style_constraints(
        self, style: str, price: float, sl: float, tp: float,
    ) -> tuple[bool, str]:
        """
        Scalp-Day Hybrid (2026-04-20): neutralized — returns always (True, "").

        Backtest has NO style constraint validation (strategies/smc_multi_style.py:1742
        hardcodes style="day" without max_tp_pct/min_tp_pct/max_sl_pct bounds).

        Safety is enforced at the caller: ATR floor + tick floor + 5% SL cap + min_rr.
        Removing this filter restored 86 of 136 profitable backtest setups (63%)
        that would have been rejected by the old max_tp_pct=6% clamping.
        """
        return (True, "")

    # Risk sizing is 100% confidence-based. No tier gates.

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

    # ── Trade close accounting (moved from Runner._poll_positions) ──

    async def _record_close(self, trade: dict[str, Any], exit_price: float) -> None:
        """Finalize a closed trade: update equity/pnl, journal, schedule teacher, cleanup orders."""
        entry_price = trade["entry"]
        qty = trade["qty"]
        direction = trade["direction"]
        sl = trade["sl"]
        # Pre-seed so teacher block + any early-return paths always have a value
        exit_reason = "unknown"
        outcome_str = "unknown"

        if direction == "long":
            raw_pnl = (exit_price - entry_price) * qty
        else:
            raw_pnl = (entry_price - exit_price) * qty

        commission = qty * entry_price * self.commission_rate * COMMISSION_MULTIPLIER
        net_pnl = raw_pnl - commission

        pnl_pct = (net_pnl / self.equity * 100) if self.equity > 0 else 0.0

        self.equity += net_pnl
        self.total_pnl += net_pnl
        self.trades += 1
        if net_pnl > 0:
            self.wins += 1
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self._append_equity()

        # ── RL performance kill switches ─────────────────────────
        if self.trades >= 50 and self.rl_suite is not None and self.rl_suite.enabled:
            _wr = self.wins / self.trades
            _gross_win = self.total_pnl if self.total_pnl > 0 else 0.0
            _gross_loss = abs(self.total_pnl - _gross_win) if self.total_pnl < 0 else 0.0
            _pf = _gross_win / _gross_loss if _gross_loss > 0 else 99.0
            if _wr < 0.35:
                self.logger.critical(
                    "RL KILL: WR %.1f%% < 35%% over %d trades — disabling RL",
                    _wr * 100, self.trades,
                )
                self.rl_suite.enabled = False
            elif _pf < 1.0 and self.total_pnl < 0:
                self.logger.critical(
                    "RL KILL: PF %.2f < 1.0, net PnL %.2f over %d trades — disabling RL",
                    _pf, self.total_pnl, self.trades,
                )
                self.rl_suite.enabled = False

        # Record PnL in circuit breaker
        if self.circuit_breaker is not None:
            pnl_pct_frac = net_pnl / self._account_equity if self._account_equity > 0 else 0.0
            self.circuit_breaker.record_trade_pnl(
                pnl_pct=pnl_pct_frac,
                asset_class=self.asset_class,
                symbol=self.symbol,
            )

        # Record trade close in Paper Grid (A/B testing)
        if self.paper_grid is not None:
            self.paper_grid.record_trade_close(exit_price, self.symbol)

        # ── Trade Journal: record trade close ─────────────────────
        if self.journal is not None:
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
                    _override_reason = trade.get("_exit_reason_override")
                    if _override_reason:
                        exit_reason = _override_reason
                    else:
                        tp_price = float(trade.get("tp", 0.0))
                        sl_price = float(trade.get("sl", 0.0))
                        # 1% tolerance — wider to handle slippage on illiquid instruments
                        if tp_price > 0 and abs(exit_price - tp_price) / max(tp_price, 1) < 0.01 and net_pnl > 0:
                            exit_reason = "tp_hit"
                        elif sl_price > 0 and abs(exit_price - sl_price) / max(sl_price, 1) < 0.01 and net_pnl <= 0:
                            exit_reason = "sl_hit"
                        else:
                            # PnL-based fallback when price didn't match SL/TP within tolerance
                            if net_pnl > 0:
                                exit_reason = "tp_hit"
                            else:
                                exit_reason = "sl_hit"

                    # MFE/MAE from journal's running tracker.
                    # bug_012 fix (2026-04-17): ensure fallback uses same formula
                    # as the tracker (gross, entry-relative) so labels stay on the same scale.
                    _exit_move = abs(exit_price - entry_price_j) / entry_price_j if entry_price_j > 0 else 0.0
                    max_fav = self.journal._max_favorable.get(
                        trade_id_j,
                        _exit_move if net_pnl > 0 else 0.0,
                    )
                    max_adv = self.journal._max_adverse.get(
                        trade_id_j,
                        _exit_move if net_pnl <= 0 else 0.0,
                    )
                    bars_held_j = trade.get("bars_held", 0)

                    self.journal.close_trade(
                        trade_id=trade_id_j,
                        exit_time=datetime.now(timezone.utc),
                        exit_price=exit_price,
                        outcome=outcome_str,
                        exit_reason=exit_reason,
                        bars_held=bars_held_j,
                        pnl_pct=pnl_pct / 100.0,
                        rr_actual=rr_actual if net_pnl > 0 else -rr_actual,
                        max_favorable_pct=max_fav,
                        max_adverse_pct=max_adv,
                        be_triggered=bool(trade.get("be_triggered", False)),
                    )
                except Exception as exc:
                    self.logger.debug("journal.close_trade error: %s", exc)

        # ── Teacher analysis (non-blocking, retroactive) ────────
        if self._teacher_enabled and self._teacher_trigger is not None:
            # Mirror live fields into teacher-expected keys
            trade["_exit_reason"] = exit_reason
            trade["outcome"] = outcome_str if outcome_str != "unknown" else ("win" if net_pnl > 0 else "loss")
            trade["pnl_pct"] = (net_pnl / self.equity) if self.equity > 0 else 0.0
            asyncio.create_task(self._teacher_trigger(self, trade, exit_price))

        # === CLEANUP ===
        _cancel_adapter = self.adapter if self.adapter is not None else None

        # Track which ID-based cancels actually succeeded (or order was already gone)
        cancelled_ids: set[str] = set()

        if trade.get("sl_attached"):
            # OANDA: SL/TP are trade-attached — they auto-cancel when trade closes.
            self.logger.debug("Trade-attached SL/TP auto-cancelled with trade close")
        else:
            # Binance/Alpaca: cancel standalone SL/TP orders
            sl_order_id = trade.get("sl_order_id")
            tp_order_id = trade.get("tp_order_id")
            cancel_targets = [oid for oid in (sl_order_id, tp_order_id) if oid]

            if cancel_targets and _cancel_adapter is not None:
                for cancel_id in cancel_targets:
                    try:
                        await _cancel_adapter.cancel_order(cancel_id, self.symbol)
                        cancelled_ids.add(str(cancel_id))
                        self.logger.info(
                            "Cancelled dangling order %s for %s after exit", cancel_id, self.symbol
                        )
                    except Exception as exc:
                        # -2011 "Unknown order sent" expected when SL/TP triggered the close.
                        exc_str = str(exc)
                        if "-2011" in exc_str or "Unknown order" in exc_str:
                            cancelled_ids.add(str(cancel_id))  # already gone = effectively cancelled
                            self.logger.debug(
                                "Order %s for %s already gone (filled/cancelled): %s",
                                cancel_id, self.symbol, exc,
                            )
                        else:
                            self.logger.warning(
                                "Failed to cancel dangling order %s for %s: %s",
                                cancel_id, self.symbol, exc,
                            )

        # Belt-and-suspenders: fetch open orders and cancel any remaining
        # reduce-only SL/TP orders whose stop price matches this trade.
        # IMPORTANT: protect orders belonging to OTHER active trades (different style).
        if _cancel_adapter is not None:
            # Collect order IDs of other still-active trades to protect them
            protected_ids: set[str] = set()
            for other in self._active_trades:
                if other is trade:
                    continue
                for k in ("order_id", "sl_order_id", "tp_order_id"):
                    oid = other.get(k)
                    if oid:
                        protected_ids.add(str(oid))
            try:
                open_orders = await _cancel_adapter.fetch_open_orders(self.symbol)
                trade_sl = trade.get("sl")
                trade_tp = trade.get("tp")

                # Warn if stored order IDs not found on exchange at all
                open_order_ids = {str(o.get("id") or "") for o in open_orders}
                for label, stored_id in [("sl_order_id", trade.get("sl_order_id")),
                                         ("tp_order_id", trade.get("tp_order_id"))]:
                    if stored_id and str(stored_id) not in cancelled_ids and str(stored_id) not in open_order_ids:
                        self.logger.warning(
                            "ORDER ID MISMATCH: %s=%s for %s not found on exchange "
                            "(possible order replacement/amendment)",
                            label, stored_id, self.symbol,
                        )

                for o in open_orders:
                    o_id = str(o.get("id") or "")
                    if not o_id:
                        continue
                    if o_id in protected_ids:
                        continue
                    if o_id in cancelled_ids:
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
                            await _cancel_adapter.cancel_order(o_id, self.symbol)
                            self.logger.info(
                                "Zombie order cancelled (price-match) %s"
                                " stopPrice=%.6f for %s [style=%s]",
                                o_id, o_stop, self.symbol,
                                trade.get("style", "?"),
                            )
                        except Exception as ce:
                            self.logger.warning(
                                "Zombie cancel failed %s for %s: %s",
                                o_id, self.symbol, ce,
                            )
            except Exception as exc:
                self.logger.warning(
                    "fetch_open_orders cleanup failed for %s: %s",
                    self.symbol, exc,
                )

        outcome = "WIN" if net_pnl > 0 else "LOSS"
        self.logger.info(
            "CLOSE %s %s %s @ %.6f → %.6f | pnl=%.2f equity=%.2f",
            outcome,
            direction.upper(),
            self.symbol,
            entry_price,
            exit_price,
            net_pnl,
            self.equity,
        )
        self._save_state()


# ═══════════════════════════════════════════════════════════════════
#  Exchange helper
# ═══════════════════════════════════════════════════════════════════

async def create_adapters(config: dict[str, Any]) -> dict[str, ExchangeAdapter]:
    """Create and connect the Binance exchange adapter (crypto-only).

    Returns dict mapping asset_class → adapter.
    """
    adapters: dict[str, ExchangeAdapter] = {}

    # ── Binance (crypto) ────────────────────────────────────────
    bk = os.getenv("BINANCE_API_KEY", "")
    bs = os.getenv("BINANCE_SECRET", "")
    _binance_cfg = (config.get("exchanges", {}) or {}).get("binance", {}) or {}
    _testnet = bool(_binance_cfg.get("testnet", True))
    _paper_only = bool(_binance_cfg.get("paper_only", False))
    _paper_balance = float(_binance_cfg.get("paper_balance", 5000.0))
    if bk and bs:
        try:
            adapter = BinanceAdapter(
                api_key=bk,
                api_secret=bs,
                testnet=_testnet,
                paper_only=_paper_only,
                paper_balance=_paper_balance,
            )
            await adapter.connect()
            adapters["crypto"] = adapter
            logger.info(
                "Binance (crypto): connected ✓ (testnet=%s, paper_only=%s)",
                _testnet, _paper_only,
            )
        except Exception as exc:
            logger.warning("Binance connect failed: %s", exc)
    else:
        logger.warning("BINANCE keys missing — crypto disabled")

    return adapters


# Dashboard helpers extracted to bot/dashboard.py (Phase 3, 2026-04-18).
# Kept as re-exports so existing callers (LiveMultiBotRunner at lines ~5865, ~6045)
# continue to work without import changes.
from bot.dashboard import (
    _build_bot_table,
    _format_uptime,
    _pnl_color,
    build_dashboard,
)


# ═══════════════════════════════════════════════════════════════════
#  Main runner
# ═══════════════════════════════════════════════════════════════════

class LiveMultiBotRunner:
    """
    Orchestrates 30 Crypto PaperBot instances (Binance Futures) with:
      - WebSocket candle + ticker feeds
      - Real bracket orders via BinanceAdapter
      - Position polling (detects TP/SL fills)
      - Rich Live Dashboard
      - Central shared RL brain (RLBrainSuite)
      - Circuit breaker for portfolio-level risk management

    Each bot trades only its assigned instrument.
    """

    def __init__(
        self,
        bots: list[PaperBot],
        adapters: dict[str, ExchangeAdapter],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.bots = bots
        self.adapters = adapters
        self.config = config or {}
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

        # ── Circuit Breakers (per-broker for funded account isolation) ──
        # Each broker = separate funded account → independent DD limits.
        _BROKER_MAP = {"crypto": "binance"}
        self._broker_cbs: dict[str, CircuitBreaker] = {}
        for bot in self.bots:
            broker = _BROKER_MAP.get(bot.asset_class, bot.asset_class)
            if broker not in self._broker_cbs:
                self._broker_cbs[broker] = CircuitBreaker()
            bot.circuit_breaker = self._broker_cbs[broker]
        # Keep reference to primary (worst) for dashboard compatibility
        self.circuit_breaker = next(iter(self._broker_cbs.values()), CircuitBreaker())

        # ── Paper Grid (Multi-Variant A/B Testing) ───────────────
        # Disabled: main bot's confidence-based risk is well-calibrated from backtesting.
        # Continuous learner now handles adaptation. Re-enable when needed.
        self.paper_grid = None

        # ── Trade Journal (lifecycle logger for ML training data) ─
        self.journal = TradeJournal("trade_journal/journal.db")
        for bot in self.bots:
            bot.journal = self.journal

        # ── Runner back-reference for global style limits ─────────
        _runner_ref = weakref.ref(self)
        for bot in self.bots:
            bot._runner_ref = _runner_ref

        # Feed status per symbol: connected | reconnecting_N | disconnected | polling
        self.ws_status: dict[str, str] = {}
        for bot in bots:
            self.ws_status[bot.symbol] = "polling" if bot.asset_class != "crypto" else "connecting"

        # Active watcher tasks keyed by symbol
        self._watcher_tasks: dict[str, asyncio.Task[None]] = {}
        self._ticker_tasks: dict[str, asyncio.Task[None]] = {}

        # ── Candle tracking + watchdog state ──────────────────────
        self._last_candle_ts: dict[str, float] = {}  # symbol -> time.time()
        self._candles_by_class: dict[str, int] = {ac: 0 for ac in ["crypto"]}
        self._candles_since_heartbeat: dict[str, int] = {ac: 0 for ac in ["crypto"]}
        self._symbol_restart_times: dict[str, list[float]] = {}  # symbol -> timestamps of watchdog restarts
        self._rest_fallback_symbols: set[str] = set()  # symbols degraded from WS to REST
        self._last_status_write: float = 0.0  # debounce heartbeat.json writes

        # Teacher analysis (retroactive non-causal SMC after trade close)
        self._teacher_enabled = True
        self._teacher_semaphore = asyncio.Semaphore(1)  # max 1 concurrent analysis

        # Inject teacher hook into each PaperBot so _record_close can schedule
        # retroactive analysis without coupling PaperBot to the Runner class.
        for _bot in self.bots:
            _bot._teacher_enabled = self._teacher_enabled
            _bot._teacher_trigger = self._run_teacher_analysis

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
                            timeout=300.0,  # 5 min — Binance testnet has sparse ticker data
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
                if not bot.adapter.is_market_open(bot.symbol):
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
                # Transient OANDA/Alpaca read timeouts are auto-recovered on the
                # next 10s poll cycle (fetch_ohlcv uses limit=2 so a missed candle
                # is picked up on retry). Log these as WARNING so ERROR remains
                # reserved for issues that need attention.
                if "timed out" in str(exc).lower():
                    bot.logger.warning("REST candle poll %s: %s", bot.symbol, exc)
                else:
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
                if not bot.adapter.is_market_open(bot.symbol):
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

    # ── Batch ticker polling for OANDA (replaces 36 individual calls with 1-2) ──

    async def _poll_tickers_batch_oanda(self, oanda_bots: list[PaperBot]) -> None:
        """Batch-fetch tickers for all OANDA bots in 1-2 API calls instead of 36."""
        if not oanda_bots:
            return
        # Group bots by adapter instance
        adapter_bots: dict[int, list[PaperBot]] = {}
        for bot in oanda_bots:
            aid = id(bot.adapter)
            adapter_bots.setdefault(aid, []).append(bot)

        while not self._shutdown.is_set():
            for aid, bots in adapter_bots.items():
                adapter = bots[0].adapter
                # Collect symbols with open markets
                symbols = []
                sym_to_bot: dict[str, PaperBot] = {}
                for bot in bots:
                    try:
                        if not adapter.is_market_open(bot.symbol):
                            continue
                    except Exception:
                        pass
                    symbols.append(bot.symbol)
                    sym_to_bot[bot.symbol] = bot

                if not symbols:
                    continue

                try:
                    prices = await adapter.fetch_batch_pricing(symbols)
                    for sym, tick in prices.items():
                        bot = sym_to_bot.get(sym)
                        if bot and tick.get("last"):
                            try:
                                await bot.on_tick(sym, float(tick["last"]))
                            except Exception as exc:
                                bot.logger.error("Batch tick error %s: %s", sym, exc)
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.error("Batch ticker poll error: %s", exc)

            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=5)
                return  # shutdown
            except asyncio.TimeoutError:
                pass

    # ── Global style counting (for per-style trade limits) ─────────

    def get_global_style_counts(self) -> dict[str, int]:
        """Count active trades by style across ALL bots."""
        counts: dict[str, int] = {"scalp": 0, "day": 0, "swing": 0}
        for bot in self.bots:
            for trade in bot._active_trades:
                style = trade.get("style", "day")
                counts[style] = counts.get(style, 0) + 1
        return counts

    # ── Post-startup: attach SL/TP to unprotected trades ───────────

    async def _attach_missing_sl_tp(self) -> None:
        """Scan all bots for active trades with sl=0.0 and attach ATR-based SL/TP.

        This catches trades loaded from state files that were never given
        protective orders (e.g. orphaned positions from pre-fix reconciliation).
        """
        for bot in self.bots:
            for trade in bot._active_trades:
                if trade["sl"] != 0.0:
                    continue  # already has SL
                entry_price = float(trade.get("entry", 0))
                if entry_price <= 0:
                    continue

                # Compute ATR from bot's 5m buffer
                if not (hasattr(bot, '_buffer_5m_deque') and bot._buffer_5m_deque and len(bot._buffer_5m_deque) >= 14):
                    logger.error("UNPROTECTED %s: no candle data for ATR SL/TP — CLOSING POSITION", bot.symbol)
                    if bot.adapter is not None:
                        try:
                            _close_side = "sell" if trade["direction"] == "long" else "buy"
                            await bot.adapter.create_market_order(
                                bot.symbol, _close_side, trade["qty"],
                                {"reduceOnly": True},
                            )
                            logger.warning("UNPROTECTED %s: CLOSED position (no candle data for SL)", bot.symbol)
                            bot._active_trades.remove(trade)
                            bot._save_state()
                        except Exception as close_exc:
                            logger.error("UNPROTECTED %s: FAILED to close: %s — REQUIRES MANUAL INTERVENTION", bot.symbol, close_exc)
                    continue

                try:
                    _buf = list(bot._buffer_5m_deque)[-14:]
                    _highs = [float(c["high"]) for c in _buf]
                    _lows = [float(c["low"]) for c in _buf]
                    _closes = [float(c["close"]) for c in _buf]
                    _tr = [max(_highs[i] - _lows[i], abs(_highs[i] - _closes[i - 1]), abs(_lows[i] - _closes[i - 1])) for i in range(1, len(_highs))]
                    _atr = sum(_tr) / len(_tr) if _tr else 0
                    if _atr <= 0:
                        continue

                    _closing_side = "buy" if trade["direction"] == "short" else "sell"
                    if trade["direction"] == "long":
                        trade["sl"] = entry_price - 3.0 * _atr
                        trade["tp"] = entry_price + 4.5 * _atr
                    else:
                        trade["sl"] = entry_price + 3.0 * _atr
                        trade["tp"] = entry_price - 4.5 * _atr
                    logger.info(
                        "ATTACH SL/TP %s: SL=%.5f TP=%.5f (ATR=%.5f, 3x/4.5x)",
                        bot.symbol, trade["sl"], trade["tp"], _atr,
                    )

                    bot._save_state()
                except Exception as exc:
                    logger.error("ATTACH SL/TP %s: ATR computation failed: %s — position remains UNPROTECTED", bot.symbol, exc)

    # ── Trade-aware startup zombie sweep ────────────────────────────

    async def _startup_sweep_zombie_orders(self) -> None:
        """Cancel orphan orders remaining from a previous session/crash.

        Runs AFTER reconciliation and SL/TP attachment so we know exactly
        which orders are legitimate.  For each bot:
          - 0 active trades → cancel ALL exit orders (clean slate)
          - Has active trades → cancel only orders not matching stored IDs
        """
        # Build set of all known legitimate order IDs
        known_ids: set[str] = set()
        for bot in self.bots:
            for trade in bot._active_trades:
                for k in ("order_id", "sl_order_id", "tp_order_id"):
                    oid = trade.get(k)
                    if oid:
                        known_ids.add(str(oid))

        seen_adapters: set[int] = set()
        total_cancelled = 0

        for ac, adapter in self.adapters.items():
            aid = id(adapter)
            if aid in seen_adapters:
                continue
            seen_adapters.add(aid)

            bot_lists = [self._bots_by_class.get(ac, [])]

            for bot_list in bot_lists:
                for bot in bot_list:
                    try:
                        open_orders = await adapter.fetch_open_orders(bot.symbol)
                    except Exception:
                        continue

                    for o in open_orders:
                        o_id = str(o.get("id", "") if isinstance(o, dict) else getattr(o, "order_id", ""))
                        if not o_id:
                            continue

                        o_type = (o.get("type", "") or "").lower()
                        is_exit = any(k in o_type for k in ("stop", "take_profit", "limit"))

                        if not bot._active_trades:
                            # No active trades → any order is a zombie
                            if is_exit:
                                try:
                                    await adapter.cancel_order(o_id, bot.symbol)
                                    total_cancelled += 1
                                    bot.logger.info(
                                        "STARTUP SWEEP: cancelled orphan %s type=%s (no active trades)",
                                        o_id, o.get("type", "?"),
                                    )
                                except Exception:
                                    pass
                        else:
                            # Has active trades → only cancel orders not matching known IDs
                            if o_id not in known_ids and is_exit:
                                try:
                                    await adapter.cancel_order(o_id, bot.symbol)
                                    total_cancelled += 1
                                    bot.logger.info(
                                        "STARTUP SWEEP: cancelled unknown %s type=%s (not in active trade IDs)",
                                        o_id, o.get("type", "?"),
                                    )
                                except Exception:
                                    pass

        if total_cancelled > 0:
            logger.info("Startup sweep: cancelled %d zombie orders", total_cancelled)
        else:
            logger.info("Startup sweep: no zombie orders found")

    # ── Startup position reconciliation ─────────────────────────────

    async def _reconcile_exchange_positions(self) -> None:
        """Poll exchange positions and restore any that bots don't know about.

        After a restart, bots start with active_trades from state files.
        If the state file had active=0 but the exchange still has open
        positions, this method restores them so the bot can manage SL/TP
        and track PnL correctly.
        """
        restored = 0
        seen_adapters: set[int] = set()

        for bot in self.bots:
            if bot.adapter is None:
                continue
            aid = id(bot.adapter)

            # Already have active trades from state file — skip
            if bot._active_trades:
                continue

            # Only poll each adapter once, then distribute
            if aid not in seen_adapters:
                seen_adapters.add(aid)
                try:
                    positions = await bot.adapter.fetch_positions()
                except Exception as exc:
                    logger.warning("Position reconciliation failed for %s: %s", bot.adapter.exchange_id, exc)
                    continue

                # Cache positions by symbol for all bots sharing this adapter
                self._reconcile_cache = getattr(self, '_reconcile_cache', {})
                for p in positions:
                    sym = p.symbol if hasattr(p, 'symbol') else p.get('symbol', '')
                    qty = abs(float(p.qty if hasattr(p, 'qty') else p.get('qty', 0)))
                    if sym and qty > 0:
                        self._reconcile_cache[sym] = p

            # Check if this bot's symbol has an exchange position
            pos = getattr(self, '_reconcile_cache', {}).get(bot.symbol)
            if pos is None:
                continue

            # Restore as active trade
            entry_price = float(pos.entry_price if hasattr(pos, 'entry_price') else pos.get('entry_price', 0))
            side = pos.side if hasattr(pos, 'side') else pos.get('side', 'long')
            qty = abs(float(pos.qty if hasattr(pos, 'qty') else pos.get('qty', 0)))
            unrealized = float(pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else pos.get('unrealized_pnl', 0))

            if entry_price <= 0 or qty <= 0:
                continue

            # Reconstruct trade dict with reasonable defaults
            trade = {
                "direction": side,
                "entry": entry_price,
                "qty": qty,
                "sl": 0.0,  # unknown — position poll doesn't return SL
                "tp": 0.0,  # unknown
                "entry_time": datetime.now(timezone.utc),
                "style": STYLE_DAY,
                "sl_attached": bot.adapter.supports_attached_sl_tp,
                "oanda_trade_id": None,
                "be_triggered": False,
                "bars_held": 0,
                "_reconciled": True,  # flag: restored from exchange, not from signal
            }

            # ── ATR-based SL/TP for orphaned/reconciled trades with no protection ──
            if trade["sl"] == 0.0 and hasattr(bot, '_buffer_5m_deque') and bot._buffer_5m_deque and len(bot._buffer_5m_deque) >= 14:
                try:
                    _buf = list(bot._buffer_5m_deque)[-14:]
                    _highs = [float(c["high"]) for c in _buf]
                    _lows = [float(c["low"]) for c in _buf]
                    _closes = [float(c["close"]) for c in _buf]
                    _tr = [max(_highs[i] - _lows[i], abs(_highs[i] - _closes[i - 1]), abs(_lows[i] - _closes[i - 1])) for i in range(1, len(_highs))]
                    _atr = sum(_tr) / len(_tr) if _tr else 0
                    if _atr > 0:
                        _closing_side = "buy" if trade["direction"] == "short" else "sell"
                        if trade["direction"] == "long":
                            trade["sl"] = entry_price - 3.0 * _atr
                            trade["tp"] = entry_price + 4.5 * _atr
                        else:
                            trade["sl"] = entry_price + 3.0 * _atr
                            trade["tp"] = entry_price - 4.5 * _atr
                        logger.info(
                            "RECONCILED %s: attached conservative SL=%.5f TP=%.5f (ATR=%.5f, 3x/4.5x)",
                            bot.symbol, trade["sl"], trade["tp"], _atr,
                        )
                except Exception as exc:
                    logger.error(
                        "RECONCILED %s: ATR SL/TP computation failed: %s — CLOSING UNPROTECTED POSITION",
                        bot.symbol, exc,
                    )
                    # Close unprotected position immediately
                    try:
                        _close_side = "sell" if trade["direction"] == "long" else "buy"
                        await bot.adapter.create_market_order(
                            bot.symbol, _close_side, trade["qty"],
                            {"reduceOnly": True},
                        )
                        logger.warning(
                            "RECONCILED %s: CLOSED unprotected position (ATR computation failed)",
                            bot.symbol,
                        )
                    except Exception as close_exc:
                        logger.error(
                            "RECONCILED %s: FAILED to close unprotected position: %s — REQUIRES MANUAL INTERVENTION",
                            bot.symbol, close_exc,
                        )
                    continue  # skip adding to active_trades
            elif trade["sl"] == 0.0:
                logger.error(
                    "RECONCILED %s: no candle data for ATR SL/TP — CLOSING UNPROTECTED POSITION",
                    bot.symbol,
                )
                # Close unprotected position immediately
                try:
                    _close_side = "sell" if trade["direction"] == "long" else "buy"
                    await bot.adapter.create_market_order(
                        bot.symbol, _close_side, trade["qty"],
                        {"reduceOnly": True},
                    )
                    logger.warning(
                        "RECONCILED %s: CLOSED unprotected position (no candle data for SL)",
                        bot.symbol,
                    )
                except Exception as close_exc:
                    logger.error(
                        "RECONCILED %s: FAILED to close unprotected position: %s — REQUIRES MANUAL INTERVENTION",
                        bot.symbol, close_exc,
                    )
                continue  # skip adding to active_trades

            bot._active_trades.append(trade)
            bot._save_state()
            restored += 1
            logger.info(
                "RECONCILED %s: %s %s qty=%.4f entry=%.5f sl=%.5f tp=%.5f%s",
                bot.symbol, trade["direction"].upper(), bot.symbol,
                trade["qty"], trade["entry"], trade["sl"], trade["tp"],
                " (OANDA trade_id=" + str(trade.get("oanda_trade_id")) + ")" if trade.get("oanda_trade_id") else "",
            )

        # Cleanup cache
        if hasattr(self, '_reconcile_cache'):
            del self._reconcile_cache

        if restored > 0:
            logger.info("Position reconciliation: restored %d trades from exchange", restored)
        else:
            logger.info("Position reconciliation: no orphaned exchange positions found")

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
                for ac in ["crypto"]
            )
            totals = " ".join(
                f"{ac}={self._candles_by_class.get(ac, 0)}"
                for ac in ["crypto"]
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
            # Aggregated prefilter telemetry — diagnose where _prepare_signal
            # blocks candles during drought periods. Reset each heartbeat so
            # the numbers reflect the last HEARTBEAT_SEC window.
            _stats = PaperBot._prefilter_stats
            _total = sum(_stats.values())
            if _total > 0:
                logger.info(
                    "PREFILTER-STATS (%d events): paused=%d mkt_closed=%d cb=%d vol_low=%d neutral_bias=%d low_score=%d near_miss=%d gate_passed=%d",
                    _total,
                    _stats.get("paused", 0),
                    _stats.get("market_closed", 0),
                    _stats.get("circuit_breaker", 0),
                    _stats.get("volume_too_low", 0),
                    _stats.get("neutral_bias", 0),
                    _stats.get("low_score", 0),
                    _stats.get("near_miss_logged", 0),
                    _stats.get("gate_passed", 0),
                )
            PaperBot._prefilter_stats.clear()
            # Reset per-heartbeat counters
            for ac in self._candles_since_heartbeat:
                self._candles_since_heartbeat[ac] = 0
            # Write heartbeat status file for dashboard
            self._write_heartbeat_status()

    def _write_heartbeat_status(self) -> None:
        """Write heartbeat.json with real candle timestamps for the dashboard."""
        now = time.time()
        per_class: dict[str, dict[str, Any]] = {}
        for ac in ["crypto"]:
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
                    if not bot.adapter.is_market_open(sym):
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

    # ── Teacher analysis (retroactive, non-blocking) ─────────────

    async def _run_teacher_analysis(self, bot, trade: dict, exit_price: float) -> None:
        """Run retroactive teacher analysis on a closed trade (non-blocking)."""
        async with self._teacher_semaphore:
            try:
                # Memory guard: skip if RSS > 6GB
                import resource
                import platform
                if platform.system() == "Darwin":
                    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                else:
                    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if rss_mb > 6144:
                    bot.logger.warning("Teacher skipped: RSS %.0fMB > 6GB", rss_mb)
                    return

                # Fetch 200 bars of 5m data around the trade
                candle_data = await bot.adapter.fetch_ohlcv(bot.symbol, "5m", limit=200)
                if not candle_data or len(candle_data) < 20:
                    bot.logger.debug("Teacher skipped: insufficient candle data for %s", bot.symbol)
                    return

                # CPU-bound analysis in thread executor
                exit_reason = trade.get("_exit_reason", "unknown")
                feedback = await asyncio.to_thread(
                    _teacher_analyze, trade, candle_data, bot.symbol, bot.asset_class, exit_reason
                )
                _teacher_save(feedback)
                bot.logger.info(
                    "TEACHER: %s grade=%s confluences=%s missed=%d",
                    bot.symbol,
                    feedback.get("teacher", {}).get("grade", "N/A"),
                    feedback.get("teacher", {}).get("confluences_at_entry", []),
                    feedback.get("teacher", {}).get("missed_setups_in_window", 0),
                )
            except Exception as exc:
                bot.logger.debug("Teacher analysis failed for %s: %s", bot.symbol, exc)

    # ── Candle staleness monitor (faster than watchdog, per-class) ──

    async def _check_candle_staleness(self) -> None:
        """Check for stale candles every 60s. Catches WebSocket staleness faster than the 10-min watchdog."""
        # 5m candles arrive once per 5 min (300s). Threshold must be > 300s to avoid
        # false positives between candle deliveries. 420s = 7 min = missed one full cycle.
        STALE_THRESHOLD = 420
        CHECK_INTERVAL = 60
        STARTUP_GRACE = 300  # skip first 5 min (history loading)
        startup_time = time.time()
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=CHECK_INTERVAL)
                return  # shutdown
            except asyncio.TimeoutError:
                pass

            now = time.time()
            # Skip during startup grace period (history still loading)
            if now - startup_time < STARTUP_GRACE:
                continue

            stale_by_class: dict[str, list[str]] = {}

            for bot in self.bots:
                sym = bot.symbol
                last = self._last_candle_ts.get(sym)
                if last is None:
                    continue  # hasn't received first candle yet
                # Only check during market hours
                try:
                    if not bot.adapter.is_market_open(sym):
                        continue
                except Exception:
                    pass
                age = now - last
                if age > STALE_THRESHOLD:
                    ac = bot.asset_class
                    stale_by_class.setdefault(ac, []).append(sym)

            for ac, symbols in stale_by_class.items():
                total_in_class = sum(1 for b in self.bots if b.asset_class == ac)
                pct = len(symbols) / total_in_class * 100 if total_in_class > 0 else 0

                if pct > 50:
                    logger.critical(
                        "STALE CRITICAL: %s — %d/%d symbols (%.0f%%) have no candle for >%ds: %s",
                        ac, len(symbols), total_in_class, pct, STALE_THRESHOLD,
                        ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""),
                    )
                else:
                    logger.warning(
                        "STALE WARNING: %s — %d/%d symbols have no candle for >%ds: %s",
                        ac, len(symbols), total_in_class, STALE_THRESHOLD,
                        ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""),
                    )

    # ── Exchange position polling (replaces candle-based exit check) ─

    async def _poll_positions(self) -> None:
        """
        Poll exchange positions every POSITION_POLL_SEC seconds.

        Detects when bracket SL/TP has filled by inspecting trades and
        position state, then updates bot stats and RL rewards.
        """

        # NOTE: _record_close moved to PaperBot.__dict__ (Block 0, 2026-04-19).
        # Call sites below use ``await bot._record_close(trade, price)`` directly.

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
                                    _oanda_tid = trade.get("oanda_trade_id")
                                    if old_sl_id or _oanda_tid:
                                        exit_side = "sell" if direction == "long" else "buy"
                                        new_sl_order = await bot.adapter.modify_stop_loss(
                                            old_sl_id or "", trade["symbol"], exit_side,
                                            trade["qty"], new_sl,
                                            trade_id=_oanda_tid,
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

                        # ── Max hold time: force close when market-hours candle limit reached ──
                        _candles_seen = trade.get("_candles_seen", 0)
                        _trade_style = trade.get("style", STYLE_DAY)
                        _max_candles = STYLE_CONFIG.get(_trade_style, STYLE_CONFIG[STYLE_DAY]).get("max_hold_candles", 0)
                        if _max_candles > 0 and _candles_seen >= _max_candles and bot.adapter is not None:
                            _timeout_side = "sell" if trade["direction"] == "long" else "buy"
                            try:
                                if not trade.get("sl_attached"):
                                    for _oid_key in ("sl_order_id", "tp_order_id"):
                                        _oid = trade.get(_oid_key)
                                        if _oid:
                                            try:
                                                await bot.adapter.cancel_order(_oid, bot.symbol)
                                            except Exception:
                                                pass
                                _timeout_close = await bot.adapter.create_market_order(
                                    bot.symbol, _timeout_side, trade["qty"],
                                    {"reduceOnly": True},
                                )
                                _timeout_price = float(
                                    _timeout_close.price or trade["entry"]
                                )
                                bot.logger.warning(
                                    "TIMEOUT EXIT %s %s after %d candles (max %d for %s style) @ %.6f",
                                    bot.symbol, _timeout_side.upper(),
                                    _candles_seen, _max_candles, _trade_style, _timeout_price,
                                )
                                trade["_exit_reason_override"] = "timeout"
                                await bot._record_close(trade, _timeout_price)
                                continue
                            except Exception as _to_exc:
                                bot.logger.error("Timeout exit failed %s: %s", bot.symbol, _to_exc)

                        # ── ML exit: close via market when flag is set ────
                        if trade.get("_ml_exit_requested") and bot.adapter is not None:
                            _ml_exit_side = "sell" if trade["direction"] == "long" else "buy"
                            try:
                                # Cancel existing SL/TP orders (skip for trade-attached)
                                if not trade.get("sl_attached"):
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
                                    _ml_close.price or trade["entry"]
                                )
                                bot.logger.info(
                                    "ML_EXIT CLOSED %s %s @ %.6f | order=%s",
                                    bot.symbol, _ml_exit_side.upper(),
                                    _ml_exit_price, _ml_close.order_id,
                                )
                                trade["_exit_reason_override"] = "ml_exit"
                                await bot._record_close(trade, _ml_exit_price)
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
                            await bot._record_close(trade, exit_price)
                            continue

                        # No exit trade found but position flat → use last known trade price or SL as fallback
                        # Grace period: don't close freshly opened trades (< 60s) — position may not be synced yet
                        _trade_age_s = (datetime.now(timezone.utc) - trade["entry_time"]).total_seconds() if isinstance(trade.get("entry_time"), datetime) else 999
                        if bot.symbol not in pos_map and _trade_age_s > 60:
                            bot.logger.warning("GHOST EXIT %s: not in pos_map (age=%.0fs, pos_map_keys=%s)",
                                bot.symbol, _trade_age_s, list(pos_map.keys())[:10])
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
                            trade["_exit_reason_override"] = "ghost_exit"
                            await bot._record_close(trade, exit_price)
                            continue

                        remaining.append(trade)

                    bot._active_trades = remaining

                    # ── Refresh unrealized PnL from exchange position (real-time) ──
                    # _prepare_signal only updates every 5m candle; this keeps
                    # dashboard PnL current between candles (~30s refresh).
                    # Only apply when this bot is the sole holder of the symbol
                    # to avoid double-counting when multiple style-bots share it.
                    pos = pos_map.get(bot.symbol)
                    if pos and remaining:
                        _other_bots_same_sym = sum(
                            1 for b2 in self.bots
                            if b2.symbol == bot.symbol and b2._active_trades and b2 is not bot
                        )
                        if _other_bots_same_sym == 0:
                            _pos_pnl = float(
                                pos.unrealized_pnl if hasattr(pos, "unrealized_pnl")
                                else pos.get("unrealized_pnl", 0)
                            )
                            _total_qty = sum(t["qty"] for t in remaining)
                            if _total_qty > 0:
                                for _rt in remaining:
                                    _rt["unrealized_pnl_usd"] = _pos_pnl * (_rt["qty"] / _total_qty)

                    # ── Belt-and-suspenders: if all trades closed, cancel ALL exit orders ──
                    # Catches zombies that _record_close's ID-based + price-match cleanup missed.
                    if not remaining and bot.adapter and not bot.adapter.supports_attached_sl_tp:
                        try:
                            _leftover = await bot.adapter.fetch_open_orders(bot.symbol)
                            _cancelled = 0
                            for _lo in _leftover:
                                _lo_id = str(_lo.get("id", "") if isinstance(_lo, dict) else "")
                                _lo_type = (_lo.get("type", "") or "").lower()
                                if _lo_id and any(k in _lo_type for k in ("stop", "take_profit", "limit")):
                                    try:
                                        await bot.adapter.cancel_order(_lo_id, bot.symbol)
                                        _cancelled += 1
                                    except Exception:
                                        pass
                            if _cancelled > 0:
                                bot.logger.info(
                                    "POSITION GONE %s: cancelled %d remaining exit orders",
                                    bot.symbol, _cancelled,
                                )
                        except Exception:
                            pass

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

                # Student-Brain reload — all bots share the same StudentBrain
                # instance, so a single call covers all. Break after first bot.
                for bot in self.bots:
                    if bot.student_brain is not None and hasattr(bot.student_brain, "check_and_reload_models"):
                        try:
                            bot.student_brain.check_and_reload_models()
                        except Exception as exc:
                            logger.warning("Student model reload failed: %s", exc)
                        break

                # Periodic state persistence (every 60s, crash-safe)
                # Also export candle buffers for dashboard charts
                candles_dir = OUTPUT_DIR / "candles"
                candles_dir.mkdir(exist_ok=True)
                for bot in self.bots:
                    try:
                        bot._save_state()
                    except Exception:
                        pass
                    # Export last 300 candles for dashboard (max API limit)
                    try:
                        buf = bot._buffer_5m_deque
                        if buf and len(buf) > 0:
                            sym_key = bot.symbol.replace("/", "_").replace(":", "_")
                            candles_out = []
                            for c in list(buf)[-300:]:
                                ts = c.get("timestamp")
                                if hasattr(ts, "timestamp"):
                                    t = int(ts.timestamp())
                                elif isinstance(ts, (int, float)):
                                    t = int(ts)
                                else:
                                    continue
                                candles_out.append({
                                    "time": t,
                                    "open": float(c.get("open", 0)),
                                    "high": float(c.get("high", 0)),
                                    "low": float(c.get("low", 0)),
                                    "close": float(c.get("close", 0)),
                                    "volume": float(c.get("volume", 0)),
                                })
                            if candles_out:
                                p = candles_dir / f"{sym_key}.json"
                                with open(p, "w") as f:
                                    json.dump(candles_out, f)
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
        """Fetch combined total equity across all adapters.

        Uses bal.total (account equity) instead of bal.free (buying power).
        """
        totals = await self._fetch_per_broker_equity()
        return sum(totals.values())

    async def _fetch_per_broker_equity(self) -> dict[str, float]:
        """Fetch equity per broker for per-account DD tracking.

        Returns dict like {"binance": 5000.0}.
        """
        _BROKER_MAP = {"crypto": "binance"}
        broker_equity: dict[str, float] = {}
        seen: set[int] = set()
        for ac, adapter in self.adapters.items():
            aid = id(adapter)
            if aid in seen:
                continue
            seen.add(aid)
            broker = _BROKER_MAP.get(ac, ac)
            try:
                bal = await adapter.fetch_balance()
                broker_equity[broker] = bal.total
            except Exception as exc:
                logger.debug("fetch_balance [%s] failed: %s", ac, exc)
        return broker_equity

    async def _dashboard_loop(self) -> None:
        """Render the Rich Live Dashboard every DASHBOARD_REFRESH_SEC."""
        console = Console()

        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while not self._shutdown.is_set():
                try:
                    # Per-broker equity + circuit breaker updates
                    _BROKER_MAP = {"crypto": "binance"}
                    broker_equity = await self._fetch_per_broker_equity()
                    total_equity = sum(broker_equity.values())
                    for b in self.bots:
                        broker = _BROKER_MAP.get(b.asset_class, b.asset_class)
                        b._account_equity = broker_equity.get(broker, 0.0)

                    # Update heat + check per-broker circuit breakers
                    broker_heat: dict[str, float] = {}
                    for b in self.bots:
                        broker = _BROKER_MAP.get(b.asset_class, b.asset_class)
                        for t in b._active_trades:
                            risk = 0.0 if t.get("be_triggered") else t.get("risk_pct", 0.0)
                            broker_heat[broker] = broker_heat.get(broker, 0.0) + risk
                    for broker, cb in self._broker_cbs.items():
                        cb.update_portfolio_heat(broker_heat.get(broker, 0.0))
                        cb.check()

                    # Save paper grid state periodically
                    if self.paper_grid is not None:
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

        # ── Reconcile exchange positions with bot state ──────────
        # After restart, bots may have active=0 while real positions exist
        # on the exchange. Poll once and restore any orphaned positions.
        # Runs BEFORE watchers so positions are known before live data flows.
        await self._reconcile_exchange_positions()

        # ── Post-startup: attach SL/TP to any unprotected trades ──
        # Catches trades loaded from state file with sl=0.0 (e.g. DIS orphan)
        await self._attach_missing_sl_tp()

        # ── Trade-aware startup zombie sweep ─────────────────────
        # Now that reconciliation + SL/TP attachment is done, we know
        # exactly which orders are legitimate. Cancel everything else.
        # Replaces the old indiscriminate "cancel ALL orders" sweep.
        await self._startup_sweep_zombie_orders()

        # Start watchers: WebSocket for crypto
        # Stagger WS subscriptions (1s apart) to avoid thundering herd on ccxt.pro
        _crypto_idx = 0
        for bot in self.bots:
            stagger = _crypto_idx * WS_STAGGER_SEC
            self._watcher_tasks[bot.symbol] = asyncio.create_task(
                self._watch_symbol(bot.symbol, stagger_delay=stagger)
            )
            self._ticker_tasks[bot.symbol] = asyncio.create_task(
                self._watch_ticker(bot.symbol, stagger_delay=stagger)
            )
            _crypto_idx += 1

        logger.info(
            "All watchers started: %d crypto (staggered %.0fs apart)",
            _crypto_idx, WS_STAGGER_SEC,
        )

        # Position poller (detects TP/SL fills on exchange)
        poll_task = asyncio.create_task(self._poll_positions())

        # Periodic zombie order sweep (catches any orphans missed by per-trade cleanup)
        zombie_task = asyncio.create_task(self._sweep_zombie_orders())

        # Model hot-swap check (reload updated model files every 60s)
        model_reload_task = asyncio.create_task(self._model_reload_loop())

        # Rich dashboard
        dashboard_task = asyncio.create_task(self._dashboard_loop())

        # Heartbeat + watchdog + staleness monitor
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        watchdog_task = asyncio.create_task(self._watchdog_loop())
        staleness_task = asyncio.create_task(self._check_candle_staleness())

        # Continuous learner (auto-retrain loop)
        learner_task = None
        if run_continuous_learner is not None:
            learner_task = asyncio.create_task(run_continuous_learner(self.config, self._shutdown))

        # Drift monitor (feature distribution drift alerts)
        drift_task = None
        if run_drift_monitor is not None:
            drift_task = asyncio.create_task(run_drift_monitor(self.config, self._shutdown))

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
        staleness_task.cancel()
        if learner_task is not None:
            learner_task.cancel()
        if hasattr(self, '_batch_ticker_task'):
            self._batch_ticker_task.cancel()
        for t in self._watcher_tasks.values():
            t.cancel()
        for t in self._ticker_tasks.values():
            t.cancel()

        all_tasks = (
            [dashboard_task, poll_task, zombie_task, model_reload_task,
             heartbeat_task, watchdog_task, staleness_task]
            + ([learner_task] if learner_task is not None else [])
            + ([self._batch_ticker_task] if hasattr(self, '_batch_ticker_task') else [])
            + list(self._watcher_tasks.values())
            + list(self._ticker_tasks.values())
        )
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Fetch final equity per-broker before closing exchange
        _BROKER_MAP = {"crypto": "binance"}
        final_broker_eq = await self._fetch_per_broker_equity()
        for b in self.bots:
            broker = _BROKER_MAP.get(b.asset_class, b.asset_class)
            b._account_equity = final_broker_eq.get(broker, 0.0)
            # === PERSISTENCE ===
            try:
                b._save_state()
            except Exception:
                pass

        # Save Paper Grid state + export results
        if self.paper_grid is not None:
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
        final_equity = sum(final_broker_eq.values())
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
    for ac in ["crypto"]:
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

    # ── Create Student Brain (unified Teacher-Student) ────────────
    # When `student_brain.enabled: true`, replaces rl_suite's entry/SL/TP/size
    # stack with one multi-head model trained on hindsight-optimal targets.
    # Initialised here so all bots share one loaded instance (saves RAM).
    student_brain = StudentBrain(config)

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
            student_brain=student_brain,
        )
        # Display multiplier for dashboard (does NOT affect trading)
        disp_mult = config.get("equity_display_multipliers", {}).get(ac, 1)
        if disp_mult and disp_mult > 1:
            bot._display_multiplier = float(disp_mult)
        bots.append(bot)

    console.print(f"[bold green]{len(bots)} bots created.[/bold green]")

    # ── Validate XGB feature alignment (once, using first bot) ────
    if rl_suite and rl_suite._entry_filter and bots:
        ef_feat_names = rl_suite._entry_filter.get("feat_names", [])
        if ef_feat_names:
            ok = bots[0].validate_xgb_features(ef_feat_names)
            if not ok:
                logger.error("XGB feature validation FAILED — model will use 0.0 for missing features!")

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
    runner = LiveMultiBotRunner(bots=bots, adapters=adapters, config=config)

    # ── Sync initial equity from exchange for bots without saved state ─
    display_multipliers = config.get("equity_display_multipliers", {})
    seen_adapters: dict[int, float] = {}
    class_equity: dict[str, float] = {}
    for ac, adapter in adapters.items():
        aid = id(adapter)
        if aid in seen_adapters:
            class_equity[ac] = seen_adapters[aid]
            continue
        try:
            bal = await adapter.fetch_balance()
            real_equity = bal.total if bal else 0.0
            disp_mult = display_multipliers.get(ac, 1)
            if disp_mult and disp_mult > 1:
                logger.info("Initial equity [%s]: %.2f (dashboard: ×%d = %.0f)", ac, real_equity, disp_mult, real_equity * disp_mult)
            else:
                logger.info("Initial equity [%s]: %.2f", ac, real_equity)
            seen_adapters[aid] = real_equity
            class_equity[ac] = real_equity
        except Exception as exc:
            logger.warning("Failed to fetch initial equity [%s]: %s", ac, exc)
            seen_adapters[aid] = 0.0
            class_equity[ac] = 0.0

    for bot in bots:
        if bot.equity <= 0:
            adapter_equity = class_equity.get(bot.asset_class, 0.0)
            if adapter_equity > 0:
                bot.equity = adapter_equity
                bot.peak_equity = adapter_equity
                bot._account_equity = adapter_equity

    logger.info("Equity synced: %s", {ac: class_equity.get(ac, 0.0) for ac in adapters})

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
        description="SMC Crypto Live Trading Bot (Binance Futures)",
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