"""
═══════════════════════════════════════════════════════════════════
 live_multi_bot.py  –  Final Coin-Specialised 100-Bot Version
 ──────────────────────────────────────────────────────────────
 Exactly 100 bots, each permanently assigned to one coin from
 the Top 100 Evergreen list.  All bots share identical fixed
 SMC parameters and money-management rules.  A single central
 PPO brain gates trades for every coin.

 Features:
   • Fixed 100 bots (no --num-bots parameter)
   • 1 bot = 1 coin (1:1 mapping, no dynamic volume ranking)
   • Fixed SMC params & money management for all bots
   • Central PPO RL brain (rl_brain.py) shared by all coins
   • Reward = pure PnL change in % (no shaping)
   • Real Binance Testnet bracket orders (market + SL + TP)
   • Real-time entry via watch_ticker (no waiting for closed 5m candle)
   • Risk = 1 % of real account balance (fetch_balance per trade)
   • Dynamic SL/TP from SMC (OB + Liquidity + FVG), RR ≥ 3.0
   • Warm-up: first 100 trades per bot always accepted
   • WebSocket with stable auto-reconnect (max 5 retries)
   • Rich Live Dashboard:
       – Header: title + total equity + uptime
       – TOP 20 / WORST 20 bots tables
       – WebSocket status panel (global + per group)
       – Green/Red colour coding for PnL
   • Each bot: own equity CSV + log file

 Requirements:
   pip install 'ccxt[pro]' pandas numpy python-dotenv pyyaml rich torch

 Quick Start:
   1. Copy .env.example → .env and fill in your testnet keys:
        BINANCE_API_KEY=your_testnet_api_key
        BINANCE_SECRET=your_testnet_secret
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
import signal
import sys
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

from rl_brain import CentralRLBrain, extract_features
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

# ── ccxt (sync) for history loading ───────────────────────────────
import ccxt as ccxt_sync

# ── ccxt.pro for WebSocket ────────────────────────────────────────
try:
    import ccxt.pro as ccxtpro
except ImportError:
    sys.exit(
        "ccxt.pro is required.  Install with:  pip install 'ccxt[pro]'"
    )

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
WS_MAX_RECONNECT = 5              # Max reconnect attempts per symbol
WS_RECONNECT_BASE_DELAY = 2       # Base delay (seconds) for exponential backoff
WS_GROUP_SIZE = 10                 # Symbols per WebSocket watcher group

# ── Fixed Top 100 Evergreen Coins (1 bot = 1 coin) ───────────────
TOP_100_COINS: list[str] = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "TON/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "1000SHIB/USDT:USDT",
    "LINK/USDT:USDT",
    "DOT/USDT:USDT",
    "TRX/USDT:USDT",
    "BCH/USDT:USDT",
    "NEAR/USDT:USDT",
    "LTC/USDT:USDT",
    "1000PEPE/USDT:USDT",
    "SUI/USDT:USDT",
    "UNI/USDT:USDT",
    "HBAR/USDT:USDT",
    "APT/USDT:USDT",
    "ARB/USDT:USDT",
    "OP/USDT:USDT",
    "POL/USDT:USDT",
    "FIL/USDT:USDT",
    "INJ/USDT:USDT",
    "RENDER/USDT:USDT",
    "TIA/USDT:USDT",
    "SEI/USDT:USDT",
    "WLD/USDT:USDT",
    "FET/USDT:USDT",
    "SAND/USDT:USDT",
    "MANA/USDT:USDT",
    "GALA/USDT:USDT",
    "AXS/USDT:USDT",
    "EGLD/USDT:USDT",
    "KAS/USDT:USDT",
    "XLM/USDT:USDT",
    "VET/USDT:USDT",
    "1000CAT/USDT:USDT",
    "ATOM/USDT:USDT",
    "FTM/USDT:USDT",
    "EOS/USDT:USDT",
    "THETA/USDT:USDT",
    "AAVE/USDT:USDT",
    "MKR/USDT:USDT",
    "LDO/USDT:USDT",
    "RUNE/USDT:USDT",
    "GRT/USDT:USDT",
    "QNT/USDT:USDT",
    "STX/USDT:USDT",
    "ALGO/USDT:USDT",
    "XMR/USDT:USDT",
    "ZEC/USDT:USDT",
    "ETC/USDT:USDT",
    "NEO/USDT:USDT",
    "IOTA/USDT:USDT",
    "ONT/USDT:USDT",
    "WAVES/USDT:USDT",
    "ZIL/USDT:USDT",
    "KLAY/USDT:USDT",
    "FLOW/USDT:USDT",
    "CRV/USDT:USDT",
    "DYDX/USDT:USDT",
    "GMX/USDT:USDT",
    "APE/USDT:USDT",
    "CHZ/USDT:USDT",
    "ENJ/USDT:USDT",
    "1INCH/USDT:USDT",
    "SUSHI/USDT:USDT",
    "COMP/USDT:USDT",
    "SNX/USDT:USDT",
    "YFI/USDT:USDT",
    "1000BONK/USDT:USDT",
    "JUP/USDT:USDT",
    "PYTH/USDT:USDT",
    "ORDI/USDT:USDT",
    "STRK/USDT:USDT",
    "IMX/USDT:USDT",
    "KAVA/USDT:USDT",
    "CELO/USDT:USDT",
    "ROSE/USDT:USDT",
    "1000LUNC/USDT:USDT",
    "PENDLE/USDT:USDT",
    "NOT/USDT:USDT",
    "BRETT/USDT:USDT",
    "POPCAT/USDT:USDT",
    "MEW/USDT:USDT",
    "GIGGLE/USDT:USDT",
    "TURBO/USDT:USDT",
    "1000000MOG/USDT:USDT",
    "1000FLOKI/USDT:USDT",
    "WIF/USDT:USDT",
    "BOME/USDT:USDT",
    "PIXEL/USDT:USDT",
    "ONDO/USDT:USDT",
    "TAO/USDT:USDT",
    "XAI/USDT:USDT",
    "PEOPLE/USDT:USDT",
    "BIGTIME/USDT:USDT"
]

NUM_BOTS = len(TOP_100_COINS)  # exactly 100 (or as many coins as listed)

# ── Fixed SMC Parameters (identical for all 100 bots) ────────────
FIXED_SMC_PARAMS: dict[str, Any] = {
    "swing_length": 8,
    "fvg_threshold": 0.0006,
    "order_block_lookback": 20,
    "liquidity_range_percent": 0.01,
    "alignment_threshold": 0.35,
    "weight_day": 1.25,
    "bos_choch_filter": "medium",
}

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
MAX_DYNAMIC_RISK_PCT = 0.02    # 2.0 % cap for dynamic sizing
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
        "min_sl_pct": 0.004,    # 0.4% min SL (prevents 1-tick SLs on low-price coins)
        "max_sl_pct": 0.008,    # 0.8% max SL
        "min_tp_pct": 0.006,    # 0.6% min TP
        "max_tp_pct": 0.015,    # 1.5% max TP
        "min_rr": 2.0,
        "cooldown_minutes": 20,
    },
    STYLE_DAY: {
        "min_sl_pct": 0.0035,   # 0.35% min SL
        "max_sl_pct": 0.025,    # 2.5% max SL
        "min_tp_pct": 0.008,    # 0.8% min TP
        "max_tp_pct": 0.06,     # 6% max TP
        "min_rr": 2.5,
        "cooldown_minutes": 60,
    },
    STYLE_SWING: {
        "min_sl_pct": 0.008,    # 0.8% min SL
        "max_sl_pct": 0.05,     # 5% max SL
        "min_tp_pct": 0.02,     # 2% min TP
        "max_tp_pct": 0.15,     # 15% max TP
        "min_rr": 3.0,
        "cooldown_minutes": 240,
    },
}

# ── Setup Quality Tiers ─────────────────────────────────────────
# 80% capital in AAA+, 15% in A, 5% in SPEC
TIER_AAA_PLUS = "AAA+"
TIER_A = "A"
TIER_SPECULATIVE = "SPEC"

TIER_THRESHOLDS: dict[str, dict[str, float]] = {
    TIER_AAA_PLUS: {"min_score": 0.78, "min_rr": 4.0},
    TIER_A:        {"min_score": 0.58, "min_rr": 3.0},
    TIER_SPECULATIVE: {"min_score": 0.40, "min_rr": 2.5},
}

TIER_RISK: dict[str, dict[str, float]] = {
    TIER_AAA_PLUS:    {"base_risk": 0.006, "max_risk": 0.020},  # 0.6%–2.0%
    TIER_A:           {"base_risk": 0.003, "max_risk": 0.008},  # 0.3%–0.8%
    TIER_SPECULATIVE: {"base_risk": 0.001, "max_risk": 0.003},  # 0.1%–0.3%
}

# Maximum leverage per tier – SPEC gets conservative leverage,
# AAA+ can use full exchange leverage for precise entries
TIER_MAX_LEVERAGE: dict[str, int] = {
    TIER_AAA_PLUS: 50,
    TIER_A: 25,
    TIER_SPECULATIVE: 15,
}

# ── Shared sync exchange for history fetching (public endpoints) ──
_history_exchange: Any = None


def _get_history_exchange() -> Any:
    """Return a shared synchronous ccxt exchange for OHLCV history."""
    global _history_exchange
    if _history_exchange is None:
        _history_exchange = ccxt_sync.binanceusdm({"enableRateLimit": True})
    return _history_exchange


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
        exchange: Any = None,
        central_brain: CentralRLBrain | None = None,
    ) -> None:
        self.bot_id = bot_id
        self.tag = f"bot_{bot_id:03d}"
        self.symbol = symbol
        self.exchange = exchange  # ccxt.pro async exchange for real orders

        # Fixed strategy parameters
        self.swing_length: int = FIXED_SMC_PARAMS["swing_length"]
        self.alignment_threshold: float = FIXED_SMC_PARAMS["alignment_threshold"]
        self.risk_pct: float = FIXED_RISK_PCT
        self.rr_ratio: float = FIXED_RR_MIN
        self.leverage: int = 10  # default leverage

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

        # Cooldown: dynamic per trade style
        self._last_entry_time: datetime | None = None
        self._entry_cooldown = timedelta(hours=1)
        self._last_trade_style: str = STYLE_DAY

        # Volatility cache (refreshed every candle)
        self._daily_atr_pct: float = 0.0
        self._5m_atr_pct: float = 0.0

        # Active exchange order ID (for bracket order tracking)
        self._active_order_id: str | None = None

        # Shared RL Brain (central PPO)
        if central_brain is None:
            raise ValueError("central_brain must be provided")
        self.brain = central_brain
        # Store last obs so we can record reward when trade closes
        self._pending_obs: np.ndarray | None = None

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
            "Initialised %s | symbol=%s | equity=%.2f | params=%s",
            self.tag, symbol, self.equity, json.dumps(FIXED_SMC_PARAMS, default=str),
        )

        # Multi-TF OHLCV buffers (last 60 days, loaded from exchange)
        self.buffer_1d: pd.DataFrame = pd.DataFrame()
        self.buffer_4h: pd.DataFrame = pd.DataFrame()
        self.buffer_1h: pd.DataFrame = pd.DataFrame()
        self.buffer_15m: pd.DataFrame = pd.DataFrame()
        self.buffer_5m: pd.DataFrame = pd.DataFrame()
        self._load_history()
        self._load_state()

    # ── History loading (multi-TF buffers) ───────────────────────────

    def _load_history(self) -> None:
        """Load last 60 days OHLCV for 5 timeframes via ccxt (sync)."""
        ex = _get_history_exchange()
        # limits for 60 days: Binance max per request is 1500 candles
        tf_limits = {
            "1d": 60,      # 60 days
            "4h": 360,     # 60 × 6
            "1h": 1440,    # 60 × 24
            "15m": 1500,   # 60 × 96 = 5760 → capped at API max
            "5m": 1500,    # 60 × 288 = 17280 → capped at API max
        }
        for tf, limit in tf_limits.items():
            try:
                raw = ex.fetch_ohlcv(self.symbol, tf, limit=limit)
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

    # === PERSISTENCE ===
    def _save_state(self) -> None:
        """Persist core bot state to JSON."""
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
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
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
        Granular multi-timeframe SMC alignment score.

        Proper SMC top-down flow:
          1D  → HTF bias + large liquidity zones       (0.20 max)
          4H  → Strong OB / FVG as primary POI         (0.20 max)
          1H  → Internal structure observation          (0.15 max)
          15m → Entry zone / setup identification       (0.20 max)
          5m  → Precision trigger for entry             (0.25 max)

        Returns (score 0–1, direction, components_dict).
        """
        swing_len = self.swing_length
        fvg_thresh = FIXED_SMC_PARAMS["fvg_threshold"]
        ob_lookback = FIXED_SMC_PARAMS["order_block_lookback"]
        liq_range = FIXED_SMC_PARAMS["liquidity_range_percent"]

        # Components tracked for tier classification
        comp: dict[str, Any] = {
            "bias": False, "bias_strong": False,
            "h4_confirms": False, "h4_poi": False,
            "h1_confirms": False, "h1_choch": False,
            "entry_zone": None, "zone_fresh": False,
            "precision_trigger": False, "volume_ok": False,
        }

        daily_bias = "neutral"
        score = 0.0

        # ═══ STEP 1: Daily Bias (1D) – HTF direction ═════════════
        # 0.15 for any bias, +0.05 bonus if from BOS/CHoCH (not EMA fallback)
        if len(self.buffer_1d) >= swing_len * 2:
            try:
                ind_1d = compute_smc_indicators(
                    self.buffer_1d, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                # Full bias (BOS/CHoCH + EMA fallback)
                running_bias = _precompute_running_bias(ind_1d, self.buffer_1d)
                daily_bias = _bias_from_running(running_bias, len(self.buffer_1d))

                if daily_bias != "neutral":
                    score += 0.15
                    comp["bias"] = True

                    # Check if bias comes from BOS/CHoCH (stronger) vs EMA fallback
                    pure_struct = _precompute_running_structure(ind_1d)
                    pure_bias = _bias_from_running(pure_struct, len(self.buffer_1d))
                    if pure_bias != "neutral" and pure_bias == daily_bias:
                        score += 0.05
                        comp["bias_strong"] = True
            except Exception as exc:
                self.logger.debug("1D bias computation failed: %s", exc)

        if daily_bias == "neutral":
            direction = "long"  # placeholder
            return 0.0, direction, comp

        # ═══ STEP 2: 4H – Strong OB/FVG as primary POI ═══════════
        # 0.10 for structure confirmation, +0.10 for active OB/FVG POI
        if len(self.buffer_4h) >= swing_len * 2:
            try:
                ind_4h = compute_smc_indicators(
                    self.buffer_4h, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                # Structure confirmation
                running_4h = _precompute_running_structure(ind_4h)
                if _structure_confirms_from_running(running_4h, daily_bias, len(self.buffer_4h)):
                    score += 0.10
                    comp["h4_confirms"] = True

                # 4H POI: find active OB or FVG aligned with bias
                price = float(self.buffer_4h["close"].iloc[-1])
                h4_poi = self._find_poi_from_indicators(
                    ind_4h, price, daily_bias, lookback_bars=10,
                )
                if h4_poi is not None:
                    score += 0.10
                    comp["h4_poi"] = True
                    comp["h4_poi_data"] = h4_poi
            except Exception as exc:
                self.logger.debug("4H computation failed: %s", exc)

        # ═══ STEP 3: 1H – Internal structure observation ═════════
        # 0.10 for structure confirmation, +0.05 if latest signal is CHoCH
        if len(self.buffer_1h) >= swing_len * 2:
            try:
                ind_1h = compute_smc_indicators(
                    self.buffer_1h, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                running_1h = _precompute_running_structure(ind_1h)
                if _structure_confirms_from_running(running_1h, daily_bias, len(self.buffer_1h)):
                    score += 0.10
                    comp["h1_confirms"] = True

                # Check for CHoCH (stronger than BOS)
                bos_choch_1h = ind_1h.get("bos_choch")
                if bos_choch_1h is not None and not bos_choch_1h.empty:
                    for i in range(len(bos_choch_1h) - 1, max(0, len(bos_choch_1h) - 4), -1):
                        choch_val = bos_choch_1h["CHOCH"].iat[i]
                        if pd.notna(choch_val) and choch_val != 0:
                            choch_dir = "bullish" if choch_val > 0 else "bearish"
                            if choch_dir == daily_bias:
                                score += 0.05
                                comp["h1_choch"] = True
                            break
            except Exception as exc:
                self.logger.debug("1H structure computation failed: %s", exc)

        # ═══ STEP 4: 15m – Entry zone / setup identification ═════
        # 0.15 for entry zone found, +0.05 if zone is fresh (last 6 bars)
        entry_zone = None
        if len(self.buffer_15m) >= swing_len * 2:
            try:
                ind_15m = compute_smc_indicators(
                    self.buffer_15m, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                entry_zone = _find_entry_zone_at(
                    ind_15m, self.buffer_15m, daily_bias,
                    fvg_thresh, len(self.buffer_15m),
                )
                if entry_zone is not None:
                    score += 0.15
                    comp["entry_zone"] = entry_zone
                    # Freshness: zone from last 4 bars (1 hour) is fresh
                    comp["zone_fresh"] = True  # _find_entry_zone_at already limits to 6 bars
                    score += 0.05
            except Exception as exc:
                self.logger.debug("15m entry zone computation failed: %s", exc)

        # ═══ STEP 5: 5m – Precision trigger ══════════════════════
        # 0.15 for BOS/CHoCH trigger, +0.10 for volume confirmation
        if len(self.buffer_5m) >= swing_len * 2:
            try:
                ind_5m = compute_smc_indicators(
                    self.buffer_5m, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                bull_mask, bear_mask = _precompute_5m_trigger_mask(ind_5m)
                if len(bull_mask) > 0:
                    if daily_bias == "bullish" and bull_mask[-1]:
                        score += 0.15
                        comp["precision_trigger"] = True
                    elif daily_bias == "bearish" and bear_mask[-1]:
                        score += 0.15
                        comp["precision_trigger"] = True

                # Volume confirmation on 5m
                if not self.buffer_5m.empty and len(self.buffer_5m) >= 21:
                    vol_current = float(self.buffer_5m["volume"].iloc[-1])
                    vol_avg = float(self.buffer_5m["volume"].iloc[-21:-1].mean())
                    if vol_avg > 0 and vol_current >= vol_avg:
                        score += 0.10
                        comp["volume_ok"] = True
            except Exception as exc:
                self.logger.debug("5m trigger computation failed: %s", exc)

        # Apply style weight and clamp
        weight = FIXED_SMC_PARAMS.get("weight_day", 1.0)
        score = min(score * weight, 1.0)

        direction = "long" if daily_bias == "bullish" else "short"
        return float(np.clip(score, 0.0, 1.0)), direction, comp

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
        fvg_thresh = FIXED_SMC_PARAMS["fvg_threshold"]
        ob_lookback = FIXED_SMC_PARAMS["order_block_lookback"]
        liq_range = FIXED_SMC_PARAMS["liquidity_range_percent"]

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

    # ── Signal preparation (called from on_candle) ──────────────────

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
        # ── Duplicate zone check (replaces cooldown timer) ─────────
        # Don't prepare a new signal if there's already an active trade
        # on this symbol – prevents double-entries on the same zone
        if self._active_trades:
            self._pending_signal = None
            return

        # ── Volatility gate (skip coins with too little movement) ─
        tradeable, daily_atr, fivem_atr = self._check_volatility()
        if not tradeable:
            self._pending_signal = None
            self.logger.debug(
                "VOLATILITY SKIP %s | daily_atr=%.4f%% (min %.4f%%) 5m_atr=%.4f%% (min %.4f%%)",
                symbol, daily_atr * 100, MIN_DAILY_ATR_PCT * 100,
                fivem_atr * 100, MIN_5M_ATR_PCT * 100,
            )
            return

        # ── Volume filter ─────────────────────────────────────────
        volumes = [c["volume"] for c in buf[-20:]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
        if avg_vol > 0 and candle["volume"] < FIXED_MIN_VOL_MULT * avg_vol:
            self._pending_signal = None
            return

        # ── Multi-TF alignment score (granular) ───────────────────
        score, direction, components = self._multi_tf_alignment_score(candle)
        if score < self.alignment_threshold:
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
            return

        # ── Enforce TP constraints for style ──────────────────────
        tp_pct = tp_dist / price
        if tp_pct < style_cfg["min_tp_pct"]:
            self._pending_signal = None
            return
        if tp_pct > style_cfg["max_tp_pct"]:
            # Clamp TP to max for this style
            tp_dist = price * style_cfg["max_tp_pct"]
            tp = (price + tp_dist) if direction == "long" else (price - tp_dist)

        # ── RR check (style-specific minimum) ─────────────────────
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < style_cfg["min_rr"]:
            self._pending_signal = None
            return

        # ── Setup quality tier classification ─────────────────────
        tier = self._classify_setup_tier(score, rr, components)
        if not tier:
            self._pending_signal = None
            self.logger.debug(
                "NO TIER %s | score=%.2f RR=%.1f – setup quality too low",
                symbol, score, rr,
            )
            return

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
            "obs": extract_features(
                buf, score, direction,
                setup_tier=tier, trade_style=style,
                rr_ratio=rr, daily_atr_pct=daily_atr,
            ),
            "zone_low": zone_low,
            "zone_high": zone_high,
            "ref_price": price,
        }
        self.logger.info(
            "PENDING %s %s %s [%s] | zone=[%.6f, %.6f] SL=%.6f TP=%.6f "
            "RR=%.1f score=%.2f daily_atr=%.3f%%",
            tier, style.upper(), direction.upper(), symbol,
            zone_low, zone_high, sl, tp, rr, score, daily_atr * 100,
        )

    # ── Real-time tick handler (called from watch_ticker) ─────────

    async def on_tick(self, symbol: str, price: float) -> None:
        """
        Called on every live ticker update.

        If a pending signal exists and the live price is inside the
        entry zone, place a real bracket order on the testnet.
        """
        sig = self._pending_signal
        if sig is None:
            return
        if sig["symbol"] != symbol:
            return
        # ── Duplicate zone check (replaces cooldown timer) ─────────
        if self._active_trades:
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
        tier = sig.get("tier", TIER_A)
        style = sig.get("style", STYLE_DAY)

        # ── RL Brain gate (warm-up: first 100 trades always accepted) ─
        use_brain = self.trades >= WARMUP_TRADES
        rl_tracked = False
        rl_trade_id: str | None = None
        take_trade = True
        if use_brain:
            rl_decision, rl_trade_id = self.brain.should_trade(obs, coin_id=self.symbol)
            rl_tracked = True
            take_trade = rl_decision
            if not take_trade:
                self.logger.info(
                    "RL skipped trade for %s because decision=0 (score=%.2f)",
                    symbol,
                    score,
                )
                self.brain.record_outcome(trade_id=rl_trade_id, reward=0.0, done=True)
                self._pending_signal = None
                return

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
        # total_fee_pct = COMMISSION_RATE * COMMISSION_MULTIPLIER (entry + exit)
        min_tp_for_profit = price * COMMISSION_RATE * COMMISSION_MULTIPLIER
        if tp_dist <= min_tp_for_profit:
            self.logger.info(
                "FEE GATE: skipping %s %s – tp_dist=%.6f <= fee_cost=%.6f (%.4f%%)",
                direction.upper(), symbol, tp_dist, min_tp_for_profit,
                COMMISSION_RATE * COMMISSION_MULTIPLIER * 100,
            )
            self._pending_signal = None
            return

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
        }
        self._active_trades.append(trade_info)
        self._last_entry_time = datetime.now(timezone.utc)
        self._last_trade_style = style
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
        tier: str = TIER_A,
        style: str = STYLE_DAY,
    ) -> tuple[str | None, str | None, str | None, float, float, int]:
        """
        Execute a bracket order with tier-based risk allocation.

        Risk allocation:
          AAA+ → 0.6%–2.0% (80% of trading capital)
          A    → 0.3%–0.8% (15% of trading capital)
          SPEC → 0.1%–0.3% (5%  of trading capital – moonshots)

        Returns:
            tuple: (order_id, sl_order_id, tp_order_id, quantity, used_risk_pct, applied_leverage)
        """
        # === TIER-BASED DYNAMIC RISK ===
        rr = tp_dist / sl_dist if sl_dist > EPSILON_SL_DIST else 0.0

        # Get tier-specific risk bounds
        tier_cfg = TIER_RISK.get(tier, TIER_RISK[TIER_A])
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

        if self.exchange is None:
            return None, None, None, 0.0, dynamic_risk, self.leverage
        ORIGINAL_RISK_PCT = dynamic_risk

        # ═══════════════════════════════════════════════════════════
        #  STEP 1: Load market info FIRST (correct market_id, limits)
        # ═══════════════════════════════════════════════════════════
        max_leverage = 20  # raised default – will be overridden if bracket data found
        leverage_source = "default"
        max_qty_limit: float | None = None
        max_notional_limit: float | None = None
        min_qty_limit: float | None = None
        qty_step: float | None = None
        lev_max = None
        market_id: str | None = None
        market: dict | None = None

        try:
            await self.exchange.load_markets()
            market = self.exchange.market(symbol)
            if market:
                # Derive the proper Binance REST API symbol (e.g. "KASUSDT") from base+quote.
                # market.get("id") often returns CCXT's internal format "KASUSDTUSDT" for
                # linear perps (base + quote + settle), which is rejected by the Binance API.
                _base = (market.get("base") or "").upper()
                _quote = (market.get("quote") or "").upper()
                if _base and _quote:
                    market_id = _base + _quote  # "KASUSDT"
                else:
                    market_id = market.get("id")
                    if not market_id:
                        try:
                            market_id = self.exchange.market_id(symbol)
                        except Exception:
                            pass
                lev_max = market.get("limits", {}).get("leverage", {}).get("max")
                limits = market.get("limits", {}) if isinstance(market, dict) else {}
                amt_limits = limits.get("amount", {}) if isinstance(limits, dict) else {}
                cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}
                max_qty_limit = float(amt_limits.get("max")) if amt_limits.get("max") else None
                min_qty_limit = float(amt_limits.get("min")) if amt_limits.get("min") else None
                max_notional_limit = (
                    float(cost_limits.get("max")) if cost_limits.get("max") else None
                )
                # Also parse raw exchange filters (Binance LOT_SIZE / MARKET_LOT_SIZE)
                # CCXT doesn't always populate limits.amount.max from these filters.
                raw_info = market.get("info", {})
                for _f in (raw_info.get("filters") or []):
                    if not isinstance(_f, dict):
                        continue
                    if _f.get("filterType") in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                        _raw_max = _f.get("maxQty") or _f.get("maxAmount")
                        if _raw_max:
                            try:
                                _parsed = float(_raw_max)
                                # Prefer the smaller (more restrictive) of the two
                                if max_qty_limit is None or _parsed < max_qty_limit:
                                    max_qty_limit = _parsed
                            except Exception:
                                pass
                        break
                # Extract quantity step size from precision info
                prec = market.get("precision", {})
                if prec.get("amount") is not None:
                    try:
                        # ccxt precision can be int (decimals) or float (step)
                        p = prec["amount"]
                        if isinstance(p, int):
                            qty_step = 10 ** (-p)
                        elif isinstance(p, float) and p < 1:
                            qty_step = p
                    except Exception:
                        pass
                self.logger.info(
                    "Market info for %s: id=%s max_qty=%s min_qty=%s max_notional=%s lev_max=%s qty_step=%s",
                    symbol, market_id, max_qty_limit, min_qty_limit, max_notional_limit, lev_max, qty_step,
                )
        except Exception as exc:
            self.logger.warning("Could not load market limits for %s: %s", symbol, exc)

        # Fallback market_id if load_markets failed
        if not market_id:
            # Derive Binance futures symbol: "KAS/USDT:USDT" -> "KASUSDT"
            base_quote = symbol.split(":")[0] if ":" in symbol else symbol
            market_id = base_quote.replace("/", "")
            self.logger.info("Using derived market_id=%s for %s", market_id, symbol)

        # ═══════════════════════════════════════════════════════════
        #  STEP 2: Fetch leverage brackets (with correct market_id)
        # ═══════════════════════════════════════════════════════════
        leverage_options_with_source: list[tuple[int, str]] = []

        # Method 1: ccxt unified fetch_leverage_tiers (correct method for binanceusdm)
        for _lev_method in ("fetch_leverage_tiers", "fetch_leverage_bracket"):
            _method_fn = getattr(self.exchange, _lev_method, None)
            if _method_fn is None:
                continue
            try:
                tiers = await _method_fn([symbol])
                # fetch_leverage_tiers returns {symbol: [tier_dicts]}; extract max leverage
                if isinstance(tiers, dict):
                    for tier_list in tiers.values():
                        if isinstance(tier_list, list):
                            for bracket in tier_list:
                                if isinstance(bracket, dict):
                                    lv = bracket.get("maxLeverage") or bracket.get("initialLeverage")
                                    if lv:
                                        try:
                                            leverage_options_with_source.append((int(lv), _lev_method))
                                        except Exception:
                                            pass
                else:
                    vals = self._extract_initial_leverage(tiers, self.logger)
                    if vals:
                        leverage_options_with_source.append((max(vals), _lev_method))
                if leverage_options_with_source:
                    break
            except Exception as exc:
                self.logger.debug(
                    "%s not available for %s: %s", _lev_method, symbol, exc,
                )

        # Method 2: Binance private API with CORRECT market_id
        if market_id and getattr(self.exchange, "id", "").lower().startswith("binance"):
            try:
                raw_brackets = await self.exchange.fapiPrivateGetLeverageBracket({"symbol": market_id})
                vals = self._extract_initial_leverage(raw_brackets, self.logger)
                if vals:
                    leverage_options_with_source.append((max(vals), "fapiPrivateGetLeverageBracket"))
                    self.logger.info(
                        "Leverage bracket for %s (market_id=%s): max=%dx",
                        symbol, market_id, max(vals),
                    )
            except Exception as exc:
                self.logger.debug(
                    "fapiPrivateGetLeverageBracket failed for %s (market_id=%s): %s",
                    symbol, market_id, exc,
                )

        # Method 3: market limits from loaded markets
        if lev_max:
            try:
                leverage_options_with_source.append((int(lev_max), "market limits"))
            except Exception:
                pass

        # Determine final leverage
        used_default_leverage = False
        if not leverage_options_with_source:
            leverage_options_with_source.append((max_leverage, leverage_source))
            used_default_leverage = True

        max_leverage, leverage_source = max(leverage_options_with_source, key=lambda t: t[0])

        self.logger.info(
            "Max leverage for %s = %dx (source: %s)",
            symbol, max_leverage, leverage_source,
        )
        if used_default_leverage:
            self.logger.info("Using default leverage fallback %dx for %s", max_leverage, symbol)

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
            # Use exchange's amount_to_precision if available
            try:
                q = float(self.exchange.amount_to_precision(symbol, q))
            except Exception:
                # Manual rounding via qty_step
                if qty_step and qty_step > 0:
                    q = float(int(q / qty_step) * qty_step)
            # Clamp to exchange limits
            if max_qty_limit and q > max_qty_limit:
                q = max_qty_limit
                # Re-round after clamping (max_qty_limit should already be valid)
                try:
                    q = float(self.exchange.amount_to_precision(symbol, q))
                except Exception:
                    if qty_step and qty_step > 0:
                        q = float(int(q / qty_step) * qty_step)
            if min_qty_limit and q < min_qty_limit:
                q = 0.0  # signal: too small to trade
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

            # Set leverage (may need to re-set if reduced during retries)
            if not leverage_already_set:
                try:
                    await self.exchange.set_leverage(planned_leverage, symbol)
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
                                    await self.exchange.set_leverage(planned_leverage, symbol)
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
        Place a market entry plus separate reduceOnly SL/TP orders on the
        Binance Testnet via ``create_order``.

        Returns (entry_id, sl_order_id, tp_order_id). Raises Exceptions on placement failures so
        the caller can handle risk reduction or cleanup.
        """
        if self.exchange is None:
            return None, None, None

        side = "buy" if direction == "long" else "sell"
        exit_side = "sell" if direction == "long" else "buy"

        # Round SL/TP to exchange price precision to avoid truncation issues
        try:
            sl = float(self.exchange.price_to_precision(symbol, sl))
            tp = float(self.exchange.price_to_precision(symbol, tp))
        except Exception:
            pass  # keep raw values if precision lookup fails

        async def _close_position(reason: str) -> None:
            try:
                close = await self.exchange.create_order(
                    symbol,
                    "market",
                    exit_side,
                    qty,
                    params={"reduceOnly": True},
                )
                self.logger.warning(
                    "Flattened position after %s | close_order=%s",
                    reason,
                    close.get("id"),
                )
            except Exception as close_exc:
                self.logger.error(
                    "Failed to flatten position after %s: %s", reason, close_exc,
                )
        try:
            entry = await self.exchange.create_order(symbol, "market", side, qty)
            entry_id = entry.get("id")
            self.logger.info(
                "ENTRY %s %s qty=%.6f | id=%s",
                side.upper(), symbol, qty, entry_id,
            )
        except Exception as exc:
            self.logger.error("Entry order FAILED %s %s: %s", side.upper(), symbol, exc)
            raise

        # Place SL (STOP_MARKET reduceOnly)
        sl_order_id: str | None = None
        try:
            sl_order = await self.exchange.create_order(
                symbol,
                "STOP_MARKET",
                exit_side,
                qty,
                params={
                    "stopPrice": sl,
                    "reduceOnly": True,
                },
            )
            sl_order_id = sl_order.get("id")
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

        # Place TP (TAKE_PROFIT_MARKET reduceOnly)
        try:
            tp_order = await self.exchange.create_order(
                symbol,
                "TAKE_PROFIT_MARKET",
                exit_side,
                qty,
                params={
                    "stopPrice": tp,
                    "reduceOnly": True,
                },
            )
            self.logger.info(
                "TP %s %s qty=%.6f @ %.6f | id=%s",
                exit_side.upper(), symbol, qty, tp, tp_order.get("id"),
            )
        except Exception as exc:
            self.logger.error(
                "TP order FAILED %s %s: %s", exit_side.upper(), symbol, exc,
            )
            # Cancel SL to avoid dangling reduceOnly without TP
            if sl_order_id:
                try:
                    await self.exchange.cancel_order(sl_order_id, symbol)
                except Exception as cancel_exc:
                    self.logger.warning(
                        "Failed to cancel SL %s after TP failure: %s",
                        sl_order_id,
                        cancel_exc,
                    )
            await _close_position("TP placement failure")
            raise

        return entry_id, sl_order_id, tp_order.get("id")

    # ── Fetch real testnet balance ────────────────────────────────

    async def _fetch_balance(self) -> float | None:
        """Return the free USDT balance from the Binance Testnet account."""
        if self.exchange is None:
            return None
        try:
            bal = await self.exchange.fetch_balance()
            usdt = bal.get("USDT", {})
            free = usdt.get("free", 0.0)
            return float(free)
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

        tradeable = daily_atr_pct >= MIN_DAILY_ATR_PCT and fivem_atr_pct >= MIN_5M_ATR_PCT
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
                if gap_size < FIXED_SMC_PARAMS["fvg_threshold"]:
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
        Classify setup quality: AAA+ (80% of capital), A (15%), SPEC (5%).

        AAA+ requires all major timeframes aligned + strong bias.
        A requires good score and RR.
        SPEC is for unusual setups with decent minimum quality.
        """
        # AAA+: Premium setup – all major TFs aligned, strong bias, high RR
        t = TIER_THRESHOLDS[TIER_AAA_PLUS]
        if (score >= t["min_score"]
                and rr >= t["min_rr"]
                and components.get("bias_strong", False)
                and components.get("h4_confirms", False)
                and components.get("h1_confirms", False)
                and components.get("precision_trigger", False)):
            return TIER_AAA_PLUS

        # A: Good setup – solid score and RR
        t = TIER_THRESHOLDS[TIER_A]
        if score >= t["min_score"] and rr >= t["min_rr"]:
            return TIER_A

        # SPEC: Speculative – minimum requirements
        t = TIER_THRESHOLDS[TIER_SPECULATIVE]
        if score >= t["min_score"] and rr >= t["min_rr"]:
            return TIER_SPECULATIVE

        return ""  # Don't trade

    # ── Style-aware SL/TP from multiple timeframes ───────────────

    def _find_sl_from_buffer(
        self, buffer: pd.DataFrame, price: float, direction: str,
    ) -> float | None:
        """Extract SL from a given timeframe's SMC indicators."""
        swing_len = self.swing_length
        fvg_thresh = FIXED_SMC_PARAMS["fvg_threshold"]
        ob_lookback = FIXED_SMC_PARAMS["order_block_lookback"]
        liq_range = FIXED_SMC_PARAMS["liquidity_range_percent"]

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
        fvg_thresh = FIXED_SMC_PARAMS["fvg_threshold"]
        ob_lookback = FIXED_SMC_PARAMS["order_block_lookback"]
        liq_range = FIXED_SMC_PARAMS["liquidity_range_percent"]

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

def create_exchange(api_key: str, api_secret: str) -> Any:
    """Create a ccxt.pro Binance USDT-M Futures exchange (testnet)."""
    exchange = ccxtpro.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
        },
    })
    exchange.enable_demo_trading(True)
    logger.info("Exchange created: %s (demo trading)", exchange.id)
    return exchange


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
    table.add_column("Coin", style="bright_yellow", width=18)
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
    eq_color = _pnl_color(total_pnl)
    header_text = Text.from_markup(
        f"[bold cyan]📊 SMC CRYPTO LIVE MULTI-BOT DASHBOARD[/bold cyan]\n"
        f"[dim]{now_str}[/dim]  ·  Uptime: [bold]{uptime}[/bold]  ·  "
        f"Symbols: [bold]{len(active_symbols)}[/bold]  ·  "
        f"Bots: [bold]{len(bots)}[/bold]\n"
        f"Total Equity: [bold]{total_equity:,.2f}[/bold] USDT  ·  "
        f"Total PnL: [bold {eq_color}]{total_pnl:+,.2f}[/bold {eq_color}] USDT  ·  "
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

    # ── Compose Layout ────────────────────────────────────────────
    layout = Layout()
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
    Orchestrates 100 coin-specialised PaperBot instances with:
      - Fixed symbol list (no dynamic volume ranking)
      - WebSocket auto-reconnect per symbol (OHLCV + ticker)
      - Real bracket orders on Binance Testnet
      - Position polling every 5 s (detects TP/SL fills on exchange)
      - Rich Live Dashboard
      - Central shared RL brain

    Each bot trades only its assigned coin.
    """

    def __init__(
        self,
        bots: list[PaperBot],
        exchange: Any,
    ) -> None:
        self.bots = bots
        self.exchange = exchange
        self.brain: CentralRLBrain | None = None
        if bots:
            self.brain = bots[0].brain
        # Build a lookup: symbol → bot
        self._symbol_to_bot: dict[str, PaperBot] = {
            b.symbol: b for b in bots
        }
        self.symbols: list[str] = [b.symbol for b in bots]
        self._shutdown = asyncio.Event()
        self._start_time = datetime.now(timezone.utc)

        # WebSocket status per symbol: connected | reconnecting_N | disconnected
        self.ws_status: dict[str, str] = {
            s: "connecting" for s in self.symbols
        }

        # Active watcher tasks keyed by symbol
        self._watcher_tasks: dict[str, asyncio.Task[None]] = {}
        self._ticker_tasks: dict[str, asyncio.Task[None]] = {}

    # ── WebSocket OHLCV watcher with auto-reconnect ───────────────

    async def _watch_symbol(self, symbol: str) -> None:
        """
        Subscribe to 5 m OHLCV candles for *symbol* and feed the
        assigned bot only.
        Auto-reconnects up to WS_MAX_RECONNECT times with exponential backoff.
        """
        bot = self._symbol_to_bot.get(symbol)
        if bot is None:
            return

        last_ts: int | None = None
        reconnect_count = 0

        while not self._shutdown.is_set():
            try:
                self.ws_status[symbol] = "connected"
                reconnect_count = 0  # reset on successful connection

                while not self._shutdown.is_set():
                    ohlcv_list = await self.exchange.watch_ohlcv(symbol, "5m")

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

                delay = min(
                    WS_RECONNECT_BASE_DELAY * (2 ** (reconnect_count - 1)),
                    60,
                )
                self.ws_status[symbol] = f"reconnecting_{reconnect_count}"
                logger.warning(
                    "🔄 %s: reconnect attempt %d/%d in %.0fs: %s",
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

    async def _watch_ticker(self, symbol: str) -> None:
        """
        Subscribe to live ticker updates for *symbol* and feed the
        assigned bot's ``on_tick`` for real-time entry decisions.

        Uses the same reconnect logic as ``_watch_symbol``.
        """
        bot = self._symbol_to_bot.get(symbol)
        if bot is None:
            return

        reconnect_count = 0

        while not self._shutdown.is_set():
            try:
                reconnect_count = 0
                while not self._shutdown.is_set():
                    ticker = await self.exchange.watch_ticker(symbol)
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

                delay = min(
                    WS_RECONNECT_BASE_DELAY * (2 ** (reconnect_count - 1)),
                    60,
                )
                logger.warning(
                    "🔄 Ticker %s: reconnect %d/%d in %.0fs: %s",
                    symbol, reconnect_count, WS_MAX_RECONNECT, delay, exc,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=delay,
                    )
                    return
                except asyncio.TimeoutError:
                    pass

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

            commission = qty * entry_price * COMMISSION_RATE * COMMISSION_MULTIPLIER
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

            # Feed reward to central RL brain when the decision was tracked
            if trade.get("rl_tracked") and trade.get("rl_trade_id"):
                bot.brain.record_outcome(trade_id=trade["rl_trade_id"], reward=pnl_pct, done=True)

            # === CLEANUP ===
            sl_order_id = trade.get("sl_order_id")
            tp_order_id = trade.get("tp_order_id")
            cancel_targets = [oid for oid in (sl_order_id, tp_order_id) if oid]

            if cancel_targets and self.exchange is not None:
                for cancel_id in cancel_targets:
                    try:
                        await self.exchange.cancel_order(cancel_id, bot.symbol)
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
            if self.exchange is not None:
                try:
                    open_orders = await self.exchange.fetch_open_orders(bot.symbol)
                    trade_sl = trade.get("sl")
                    trade_tp = trade.get("tp")
                    for o in open_orders:
                        o_id = str(o.get("id") or "")
                        if not o_id:
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
                                await self.exchange.cancel_order(o_id, bot.symbol)
                                bot.logger.info(
                                    "Zombie order cancelled (price-match) %s"
                                    " stopPrice=%.6f for %s",
                                    o_id, o_stop, bot.symbol,
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
                # Retry fetch_positions up to 3 times on transient errors
                positions = None
                for _poll_try in range(3):
                    try:
                        positions = await self.exchange.fetch_positions()
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as _poll_exc:
                        if _poll_try < 2:
                            logger.debug("fetch_positions attempt %d failed: %s", _poll_try + 1, _poll_exc)
                            await asyncio.sleep(2 * (_poll_try + 1))
                        else:
                            raise  # let outer handler log it

                pos_map: dict[str, Any] = {}
                if positions:
                    for p in positions:
                        sym = p.get("symbol")
                        contracts = abs(float(p.get("contracts", 0) or 0))
                        if sym and contracts > 0:
                            pos_map[sym] = p

                for bot in self.bots:
                    if not bot._active_trades:
                        continue

                    # Fetch recent trades since earliest entry
                    recent: list[Any] = []
                    try:
                        earliest = min(
                            int(t["entry_time"].timestamp() * 1000)
                            for t in bot._active_trades
                        )
                        recent = await self.exchange.fetch_my_trades(
                            bot.symbol, since=earliest, limit=50,
                        )
                    except Exception as exc:
                        bot.logger.warning("fetch_my_trades failed: %s", exc)

                    remaining: list[dict[str, Any]] = []
                    # Process each tracked trade independently
                    for trade in list(bot._active_trades):
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

    # ── Rich Dashboard loop ───────────────────────────────────────

    async def _fetch_real_total_equity(self) -> float:
        """Fetch real USDT free balance from Binance demo account."""
        last_exc = None
        for _attempt in range(3):
            try:
                bal = await self.exchange.fetch_balance()
                usdt = bal.get("USDT", {})
                free = float(usdt.get("free", 0.0))
                return free
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if _attempt < 2:
                    await asyncio.sleep(1 * (_attempt + 1))
        logger.warning("_fetch_real_total_equity failed after 3 attempts: %s", last_exc)
        return 0.0

    async def _dashboard_loop(self) -> None:
        """Render the Rich Live Dashboard every DASHBOARD_REFRESH_SEC."""
        console = Console()

        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while not self._shutdown.is_set():
                try:
                    total_equity = await self._fetch_real_total_equity()
                    for b in self.bots:
                        b._account_equity = total_equity
                    layout = build_dashboard(
                        bots=self.bots,
                        ws_status=self.ws_status,
                        start_time=self._start_time,
                        active_symbols=self.symbols,
                        total_equity=total_equity,
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
        logger.info(
            "Starting %d bots on %d symbols …", len(self.bots), len(self.symbols)
        )

        # Start one OHLCV watcher task per symbol (5 m candle analysis)
        for sym in self.symbols:
            self._watcher_tasks[sym] = asyncio.create_task(
                self._watch_symbol(sym)
            )

        # Start one ticker watcher task per symbol (real-time entry)
        for sym in self.symbols:
            self._ticker_tasks[sym] = asyncio.create_task(
                self._watch_ticker(sym)
            )

        # Position poller (detects TP/SL fills on exchange)
        poll_task = asyncio.create_task(self._poll_positions())

        # Rich dashboard
        dashboard_task = asyncio.create_task(self._dashboard_loop())

        # Wait until shutdown
        await self._shutdown.wait()
        logger.info("Shutdown signal received – stopping …")

        # Cancel all tasks
        dashboard_task.cancel()
        poll_task.cancel()
        for t in self._watcher_tasks.values():
            t.cancel()
        for t in self._ticker_tasks.values():
            t.cancel()

        all_tasks = (
            [dashboard_task, poll_task]
            + list(self._watcher_tasks.values())
            + list(self._ticker_tasks.values())
        )
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Flush central RL brain (save remaining buffer)
        if self.brain is not None:
            try:
                self.brain.flush()
            except Exception:
                pass

        # Fetch final equity before closing exchange
        final_equity = await self._fetch_real_total_equity()
        for b in self.bots:
            b._account_equity = final_equity
            # === PERSISTENCE ===
            try:
                b._save_state()
            except Exception:
                pass

        # Close exchange WebSocket connections
        try:
            await self.exchange.close()
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live 100-Bot Coin-Specialised System with RL Brain (Binance Testnet)",
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
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET", "")
    if not api_key or not api_secret:
        sys.exit(
            "BINANCE_API_KEY and BINANCE_SECRET must be set in .env\n"
            "Copy .env.example → .env and fill in your Binance Testnet keys."
        )

    # ── Load config ───────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", cfg_path)

    # ── Create exchange ───────────────────────────────────────────
    exchange = create_exchange(api_key, api_secret)

    # ── Create 100 bots (1 bot = 1 coin) ─────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.print(
        f"[bold cyan]Creating {NUM_BOTS} coin-specialised bots …[/bold cyan]"
    )

    central_brain = CentralRLBrain(
        model_dir=output_dir / "rl_models",
        coin_ids=TOP_100_COINS,
    )

    bots: list[PaperBot] = []
    for idx, coin in enumerate(TOP_100_COINS):
        bot = PaperBot(
            bot_id=idx + 1,
            symbol=coin,
            config=config,
            output_dir=output_dir,
            exchange=exchange,
            central_brain=central_brain,
        )
        bots.append(bot)

    logger.info("Created %d coin-specialised bots", len(bots))
    console.print(
        f"[bold green]✅ {len(bots)} bots created – "
        f"each assigned to a unique coin.[/bold green]"
    )

    # ── Runner ────────────────────────────────────────────────────
    runner = LiveMultiBotRunner(
        bots=bots,
        exchange=exchange,
    )

    # ── Graceful shutdown on Ctrl+C ───────────────────────────────
    loop = asyncio.new_event_loop()

    def _signal_handler() -> None:
        logger.info("Ctrl+C detected – shutting down gracefully …")
        runner.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(runner.run())
    except KeyboardInterrupt:
        runner.request_shutdown()
        loop.run_until_complete(runner.run())
    finally:
        loop.close()

    logger.info("✅  All bots stopped. Results in %s", output_dir)


if __name__ == "__main__":
    main()