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

        # Cooldown: minimum 2 h between entries on the same coin
        self._last_entry_time: datetime | None = None
        self._entry_cooldown = timedelta(hours=1)

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
    ) -> tuple[float, str]:
        """
        Multi-timeframe SMC alignment score.

        Flow:
          1D + 4H → Bias   (BOS/CHoCH + EMA200 fallback)
          1H + 15m → Structure
          5m → Entry trigger  (FVG + Liquidity + Alignment)

        Returns (score 0–1, direction "long" | "short").
        """
        swing_len = self.swing_length
        fvg_thresh = FIXED_SMC_PARAMS["fvg_threshold"]
        ob_lookback = FIXED_SMC_PARAMS["order_block_lookback"]
        liq_range = FIXED_SMC_PARAMS["liquidity_range_percent"]

        daily_bias = "neutral"
        h4_confirms = False
        h1_confirms = False
        entry_zone = None
        precision_trigger = False

        # 1. Daily Bias from 1D (BOS/CHoCH + EMA200 fallback) ─────
        if len(self.buffer_1d) >= swing_len * 2:
            try:
                ind_1d = compute_smc_indicators(
                    self.buffer_1d, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                running_bias = _precompute_running_bias(ind_1d, self.buffer_1d)
                daily_bias = _bias_from_running(running_bias, len(self.buffer_1d))
            except Exception as exc:
                self.logger.debug("1D bias computation failed: %s", exc)

        # 2. 4H Bias confirmation ─────────────────────────────────
        if daily_bias != "neutral" and len(self.buffer_4h) >= swing_len * 2:
            try:
                ind_4h = compute_smc_indicators(
                    self.buffer_4h, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                running_4h = _precompute_running_structure(ind_4h)
                h4_confirms = _structure_confirms_from_running(
                    running_4h, daily_bias, len(self.buffer_4h),
                )
            except Exception as exc:
                self.logger.debug("4H structure computation failed: %s", exc)

        # 3. 1H Structure confirmation ────────────────────────────
        if daily_bias != "neutral" and len(self.buffer_1h) >= swing_len * 2:
            try:
                ind_1h = compute_smc_indicators(
                    self.buffer_1h, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                running_1h = _precompute_running_structure(ind_1h)
                h1_confirms = _structure_confirms_from_running(
                    running_1h, daily_bias, len(self.buffer_1h),
                )
            except Exception as exc:
                self.logger.debug("1H structure computation failed: %s", exc)

        # 4. 15m Entry zone (FVG / OB) ────────────────────────────
        if daily_bias != "neutral" and len(self.buffer_15m) >= swing_len * 2:
            try:
                ind_15m = compute_smc_indicators(
                    self.buffer_15m, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                entry_zone = _find_entry_zone_at(
                    ind_15m, self.buffer_15m, daily_bias,
                    fvg_thresh, len(self.buffer_15m),
                )
            except Exception as exc:
                self.logger.debug("15m entry zone computation failed: %s", exc)

        # 5. 5m Precision trigger (BOS/CHoCH) ─────────────────────
        if daily_bias != "neutral" and len(self.buffer_5m) >= swing_len * 2:
            try:
                ind_5m = compute_smc_indicators(
                    self.buffer_5m, swing_len, fvg_thresh, ob_lookback, liq_range,
                )
                bull_mask, bear_mask = _precompute_5m_trigger_mask(ind_5m)
                if len(bull_mask) > 0:
                    if daily_bias == "bullish":
                        precision_trigger = bool(bull_mask[-1])
                    elif daily_bias == "bearish":
                        precision_trigger = bool(bear_mask[-1])
            except Exception as exc:
                self.logger.debug("5m trigger computation failed: %s", exc)

        # ── Combine into score (5 × 0.20 = 1.0 max) ─────────────
        score = _compute_alignment_score(
            daily_bias,
            h4_confirms and h1_confirms,
            entry_zone,
            precision_trigger,
            style_weight=FIXED_SMC_PARAMS.get("weight_day", 1.0),
        )

        direction = "long" if daily_bias == "bullish" else "short"
        return float(np.clip(score, 0.0, 1.0)), direction

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
        Evaluate multi-TF alignment + SMC SL/TP on the latest 5 m candle.

        If all filters pass (alignment, RR ≥ 2.5), store a *pending
        signal* that ``on_tick`` will consume for real-time entry once
        the live price touches the entry zone. The RL gate runs in
        ``on_tick`` just before placing the order.
        """
        if len(self._active_trades) >= 3:
            self._pending_signal = None
            return

        # Cooldown: skip if last entry on this coin was < 2 h ago
        if self._last_entry_time is not None:
            elapsed = datetime.now(timezone.utc) - self._last_entry_time
            if elapsed < self._entry_cooldown:
                self._pending_signal = None
                return

        # Volume filter: skip if current volume < 1.0× avg(20)
        volumes = [c["volume"] for c in buf[-20:]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
        if avg_vol > 0 and candle["volume"] < FIXED_MIN_VOL_MULT * avg_vol:
            self._pending_signal = None
            return

        score, direction = self._multi_tf_alignment_score(candle)
        if score < self.alignment_threshold:
            self._pending_signal = None
            return

        price = candle["close"]
        if price <= 0:
            self._pending_signal = None
            return

        # ── Dynamic SMC SL/TP (Order Blocks + Liquidity + FVG) ────
        sl_tp = self._find_smc_sl_tp(price, direction)
        if sl_tp is None:
            self._pending_signal = None
            return
        sl, tp = sl_tp

        sl_dist = abs(price - sl)
        tp_dist = abs(tp - price)

        # Enforce minimum SL distance (0.35% of price); keep SMC level when wider
        min_sl_dist = price * 0.0035
        if sl_dist < min_sl_dist:
            sl_dist = min_sl_dist
            sl = (price - sl_dist) if direction == "long" else (price + sl_dist)

        if tp_dist <= 0:
            self._pending_signal = None
            return

        # Dynamic RR – never trade below FIXED_RR_MIN (1:2.5)
        rr = tp_dist / sl_dist
        if rr < FIXED_RR_MIN:
            self._pending_signal = None
            return

        # ── Compute entry zone from 15 m analysis ─────────────────
        # Zone boundaries: for long, enter when price reaches SL side
        # (pullback into zone); for short, enter when price rallies into zone.
        if direction == "long":
            zone_low = sl
            zone_high = price  # enter between SL and current close
        else:
            zone_low = price
            zone_high = sl

        # Store pending signal – on_tick will trigger the actual entry
        self._pending_signal = {
            "symbol": symbol,
            "direction": direction,
            "sl": sl,
            "tp": tp,
            "score": score,
            "obs": extract_features(buf, score, direction),
            "zone_low": zone_low,
            "zone_high": zone_high,
            "ref_price": price,
        }
        self.logger.debug(
            "PENDING %s %s | zone=[%.6f, %.6f] SL=%.6f TP=%.6f score=%.2f",
            direction.upper(), symbol, zone_low, zone_high, sl, tp, score,
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
        if len(self._active_trades) >= 3:
            self._pending_signal = None
            return
        if self._last_entry_time is not None:
            elapsed = datetime.now(timezone.utc) - self._last_entry_time
            if elapsed < self._entry_cooldown:
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
            "order_id": order_id,
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id,
            "rl_tracked": rl_tracked,
            "rl_trade_id": rl_trade_id,
        }
        self._active_trades.append(trade_info)
        self._last_entry_time = datetime.now(timezone.utc)
        self.logger.info(
            "OPEN %s %s @ %.6f | SL=%.6f TP=%.6f | qty=%.4f "
            "lev=%dx score=%.2f bal=%.2f order=%s",
            direction.upper(), symbol, price, sl, tp, qty,
            used_leverage, score, balance, order_id or "no-exchange",
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
    ) -> tuple[str | None, str | None, str | None, float, float, int]:
        """
        Execute a bracket order with leverage-bracket lookup and risk reduction.

        Returns:
            tuple: (order_id, sl_order_id, tp_order_id, quantity, used_risk_pct, applied_leverage)
        """
        # === DYNAMIC RISK ===
        rr = tp_dist / sl_dist if sl_dist > EPSILON_SL_DIST else 0.0
        base_risk = FIXED_RISK_PCT

        # Max Multiplikator = 3.2 * 2.5 = 8.0 -> 0.25% * 8 = exakt 2.0% Max Risk
        rr_mult = self._step_mult(rr, [(9.0, 3.2), (6.0, 2.0), (3.0, 1.5)])
        score_mult = self._step_mult(score, [(0.85, 2.5), (0.70, 1.5), (0.55, 1.1)])

        def _fmt_mult(mult: float) -> str:
            return f"{mult:.3f}".rstrip("0").rstrip(".")

        final_risk = base_risk * rr_mult * score_mult
        dynamic_risk = max(MIN_DYNAMIC_RISK_PCT, min(final_risk, MAX_DYNAMIC_RISK_PCT))
        
        self.logger.info(
            "[DYNAMIC RISK] score=%.2f RR=%.2f → final risk=%.2f%% (base %.2f%%) (RR_mult=%.1fx, score_mult=%.1fx)",
            score,
            rr,
            dynamic_risk * 100,
            base_risk * 100,
            rr_mult,
            score_mult,
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
                            for tier in tier_list:
                                if isinstance(tier, dict):
                                    lv = tier.get("maxLeverage") or tier.get("initialLeverage")
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

            # Set leverage once (or retry on failure)
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
            qty_halve_retries = 0
            while rate_limit_retries <= 2 and qty_halve_retries <= 12:
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
                        # Binary-search the true max_qty: halve on every failure so we
                        # converge in O(log n) attempts regardless of how far off we are.
                        # Example: true limit=20k, first try=235k → 117k→58k→29k→14k (4 steps)
                        if max_qty_limit is None:
                            max_qty_limit = qty * 0.5
                            self.logger.warning(
                                "→ -4005: no max_qty known, binary-searching from %.0f for %s",
                                max_qty_limit, symbol,
                            )
                        else:
                            max_qty_limit = max(max_qty_limit * 0.5, qty * 0.5)
                            self.logger.warning(
                                "→ -4005: still failing, halving max_qty to %.0f for %s",
                                max_qty_limit, symbol,
                            )
                        # Immediately recalculate qty with updated cap and retry the SAME
                        # risk level (don't waste a risk step if only qty was the issue).
                        new_qty = _round_qty(qty)  # _round_qty now clamps to new max_qty_limit
                        if new_qty > 0 and new_qty < qty * 0.99 and qty_halve_retries <= 12:
                            # qty actually changed – retry at same risk without advancing step
                            qty = new_qty
                            qty_halve_retries += 1
                            notional = qty * price
                            expected_margin = notional / planned_leverage if planned_leverage > 0 else notional
                            self.logger.info(
                                "[ORDER_RETRY_SAME_STEP #%d] %s %s | new qty=%.6f notional=%.2f max_qty=%.0f",
                                qty_halve_retries, direction.upper(), symbol, qty, notional, max_qty_limit,
                            )
                            continue  # retry inner while loop with reduced qty
                        # qty didn't shrink (already at or below cap) – advance to next risk step
                        if attempt_num < total_steps:
                            self.logger.warning(
                                "→ Reducing risk from %.2f%% to %.2f%% for %s (qty=%.0f max_qty=%.0f)",
                                risk_pct * 100, risk_steps[idx + 1] * 100,
                                symbol, qty, max_qty_limit or 0,
                            )
                        break  # go to next risk step

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
                        bot.logger.warning(
                            "Failed to cancel dangling order %s for %s: %s",
                            cancel_id,
                            bot.symbol,
                            exc,
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
                positions = await self.exchange.fetch_positions()

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
        try:
            bal = await self.exchange.fetch_balance()
            usdt = bal.get("USDT", {})
            free = float(usdt.get("free", 0.0))
            return free
        except Exception as exc:
            logger.warning("_fetch_real_total_equity failed: %s", exc)
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