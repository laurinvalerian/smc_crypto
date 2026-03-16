"""
═══════════════════════════════════════════════════════════════════
 live_multi_bot.py  –  Final Coin-Specialised 100-Bot Version
 ──────────────────────────────────────────────────────────────
 Exactly 100 bots, each permanently assigned to one coin from
 the Top 100 Evergreen list.  All bots share identical fixed
 SMC parameters and money-management rules.  Each bot has its
 own PPO RL brain that learns a yes/no trade filter.

 Features:
   • Fixed 100 bots (no --num-bots parameter)
   • 1 bot = 1 coin (1:1 mapping, no dynamic volume ranking)
   • Fixed SMC params & money management for all bots
   • Per-bot PPO RL brain (rl_brain.py)
   • Reward = pure PnL change in % (no shaping)
   • Real Binance Testnet bracket orders (market + SL + TP)
   • Real-time entry via watch_ticker (no waiting for closed 5m candle)
   • Risk = 1 % of real account balance (fetch_balance per trade)
   • Dynamic SL/TP from SMC (OB + Liquidity + FVG), RR ≥ 2.5
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
from datetime import datetime, timezone
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

from rl_brain import RLBrain, extract_features
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
    "ICP/USDT:USDT",
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
    "GIGA/USDT:USDT",
    "TURBO/USDT:USDT",
    "MOG/USDT:USDT",
    "1000FLOKI/USDT:USDT",
    "WIF/USDT:USDT",
    "BOME/USDT:USDT",
    "PIXEL/USDT:USDT",
    "ONDO/USDT:USDT",
    "TAO/USDT:USDT",
]

NUM_BOTS = len(TOP_100_COINS)  # exactly 100 (or as many coins as listed)

# ── Fixed SMC Parameters (identical for all 100 bots) ────────────
FIXED_SMC_PARAMS: dict[str, Any] = {
    "swing_length": 10,
    "fvg_threshold": 0.00045,
    "order_block_lookback": 28,
    "liquidity_range_percent": 0.0075,
    "alignment_threshold": 0.52,
    "weight_day": 1.25,
    "bos_choch_filter": "medium",
}

# ── Fixed Money Management ────────────────────────────────────────
FIXED_RISK_PCT = 0.01       # 1 % risk per trade
FIXED_RR_MIN = 2.5          # minimum 1:2.5 reward-to-risk
FIXED_ATR_PERIOD = 14
FIXED_EMA_FAST = 20
FIXED_EMA_SLOW = 50
FIXED_MIN_VOL_MULT = 1.0    # min volume = 1.0× average

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
        self.equity: float = 0.0  # wird später vom Runner mit real balance gesetzt
        self.peak_equity: float = 0.0  # managed by Runner
        self._account_equity: float = 0.0  # real account equity, set by Runner

        # Tracking
        self.total_pnl: float = 0.0
        self.trades: int = 0
        self.wins: int = 0

        # Active trade on exchange (set on entry, cleared by position poller)
        self._active_trade: dict[str, Any] | None = None

        # Candle history  {symbol: list[dict]}
        self._candle_buf: dict[str, list[dict[str, Any]]] = {}

        # Pending signal for real-time entry (set by on_candle, consumed by on_tick)
        self._pending_signal: dict[str, Any] | None = None

        # Active exchange order ID (for bracket order tracking)
        self._active_order_id: str | None = None

        # RL Brain (per-bot PPO)
        self.brain = RLBrain(
            bot_tag=self.tag,
            model_dir=output_dir / "rl_models",
        )
        # Store last obs so we can record reward when trade closes
        self._pending_obs: np.ndarray | None = None

        # Output
        output_dir.mkdir(parents=True, exist_ok=True)
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

        If all filters pass (alignment, RR ≥ 2.5, RL brain), store a
        *pending signal* that ``on_tick`` will consume for real-time
        entry once the live price touches the entry zone.
        """
        if self._active_trade is not None:
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

        # ── RL Brain gate (warm-up: first 100 trades always accepted) ─
        obs = extract_features(buf, score, direction)
        if self.trades < 100:
            take_trade = True
        else:
            take_trade = self.brain.should_trade(obs)
            if not take_trade:
                # Skipped → reward 0 for brain
                self.brain.record_outcome(reward=0.0, done=True)
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
            "obs": obs,
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
        if self._active_trade is not None:
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

        # ── Fetch real balance (1 % risk) ─────────────────────────
        balance = await self._fetch_balance()
        if balance is None or balance <= 0:
            self.logger.warning(
                "fetch_balance returned %s – falling back to tracked equity (%.2f)",
                balance, self.equity,
            )
            balance = self.equity  # fallback to tracked equity

        sl_dist = abs(price - sl)
        if sl_dist <= 0:
            return

        risk_amount = balance * self.risk_pct
        qty = risk_amount / sl_dist
        notional = qty * price

        if qty <= 0 or notional <= 0:
            return

        # ── Place real bracket order on testnet ───────────────────
        order_id = await self._place_bracket_order(
            symbol, direction, price, sl, tp, qty,
        )

        # Consume the pending signal
        self._pending_signal = None

        # If the real order failed, do not track as active trade
        if order_id is None and self.exchange is not None:
            self.logger.warning(
                "Bracket order failed for %s %s – skipping",
                direction.upper(), symbol,
            )
            return

        self._pending_obs = obs

        # ── Track active trade (cleared by position poller on fill) ──
        self._active_trade = {
            "symbol": symbol,
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "qty": qty,
            "leverage": self.leverage,
            "entry_time": datetime.now(timezone.utc),
            "score": score,
            "order_id": order_id,
        }
        self.logger.info(
            "OPEN %s %s @ %.6f | SL=%.6f TP=%.6f | qty=%.4f "
            "lev=%dx score=%.2f bal=%.2f order=%s",
            direction.upper(), symbol, price, sl, tp, qty,
            self.leverage, score, balance, order_id or "no-exchange",
        )

    # ── Real testnet bracket order ────────────────────────────────

    async def _place_bracket_order(
        self,
        symbol: str,
        direction: str,
        price: float,
        sl: float,
        tp: float,
        qty: float,
    ) -> str | None:
        """
        Place a market order with attached stopLoss + takeProfit on the
        Binance Testnet via ``create_order``.

        Returns the order ID on success, ``None`` on failure.
        """
        if self.exchange is None:
            return None

        side = "buy" if direction == "long" else "sell"
        try:
            order = await self.exchange.create_order(
                symbol,
                "market",
                side,
                qty,
                params={
                    "stopLoss": {
                        "triggerPrice": sl,
                        "type": "market",
                    },
                    "takeProfit": {
                        "triggerPrice": tp,
                        "type": "market",
                    },
                },
            )
            order_id = order.get("id")
            self.logger.info(
                "BRACKET ORDER %s %s qty=%.6f | SL=%.6f TP=%.6f | id=%s",
                side.upper(), symbol, qty, sl, tp, order_id,
            )
            return order_id
        except Exception as exc:
            self.logger.error(
                "Bracket order FAILED %s %s: %s", side.upper(), symbol, exc,
            )
            return None

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
            "equity": 0,
            "pnl": round(self.total_pnl, 2),
            "return_pct": round(self.return_pct, 2),
            "trades": self.trades,
            "winrate": round(self.winrate * 100, 1),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "open_pos": 1 if self._active_trade is not None else 0,
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
    table.add_column("Equity", justify="right", width=14)
    table.add_column("PnL", justify="right", width=12)
    table.add_column("Return%", justify="right", width=9)
    table.add_column("Trades", justify="right", width=7)
    table.add_column("Winrate", justify="right", width=8)
    table.add_column("DD%", justify="right", width=7)
    table.add_column("Open", justify="right", width=5)

    for i, r in enumerate(rows, 1):
        pnl_c = _pnl_color(r["pnl"])
        ret_c = _pnl_color(r["return_pct"])
        table.add_row(
            str(i),
            r["bot"],
            r.get("symbol", ""),
            f"{r['equity']:,.2f}",
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
      - Per-bot RL brain

    Each bot trades only its assigned coin.
    """

    def __init__(
        self,
        bots: list[PaperBot],
        exchange: Any,
    ) -> None:
        self.bots = bots
        self.exchange = exchange
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

        Detects when a bracket order's SL or TP has been filled, then:
          - determines exit price via ``fetch_my_trades``
          - calculates PnL and commission
          - feeds reward to the bot's RL brain
          - updates equity / dashboard counters
          - clears the bot's ``_active_trade``
        """
        while not self._shutdown.is_set():
            try:
                positions = await self.exchange.fetch_positions()

                # Binance USDT-M one-way mode: one position per symbol.
                # Build map: symbol → position (non-zero contracts only).
                # Skip poll cycle if exchange returned nothing (API glitch).
                if positions:
                    pos_map: dict[str, Any] = {}
                    for p in positions:
                        sym = p.get("symbol")
                        contracts = abs(float(p.get("contracts", 0) or 0))
                        if sym and contracts > 0:
                            pos_map[sym] = p

                    for bot in self.bots:
                        trade = bot._active_trade
                        if trade is None:
                            continue

                        # Position still open on exchange → nothing to do
                        if bot.symbol in pos_map:
                            continue

                        # ── Position closed (TP or SL filled) ─────
                        entry_price = trade["entry"]
                        sl = trade["sl"]
                        tp = trade["tp"]
                        qty = trade["qty"]
                        direction = trade["direction"]

                        # Determine exit price from recent exchange
                        # trades.  Filter to exit-side trades (sell for
                        # long, buy for short) that occurred after entry.
                        exit_price: float | None = None
                        exit_side = "sell" if direction == "long" else "buy"
                        try:
                            since_ms = int(
                                trade["entry_time"].timestamp() * 1000
                            )
                            recent = await self.exchange.fetch_my_trades(
                                bot.symbol, since=since_ms, limit=20,
                            )
                            # Find the last trade on the exit side
                            for t in reversed(recent or []):
                                if t.get("side") == exit_side:
                                    exit_price = float(t["price"])
                                    break
                        except Exception as exc:
                            bot.logger.warning(
                                "fetch_my_trades failed: %s", exc,
                            )

                        # Fallback: assume SL (conservative)
                        if exit_price is None:
                            exit_price = sl
                            bot.logger.warning(
                                "Could not determine exit price for %s "
                                "– assuming SL hit",
                                bot.symbol,
                            )

                        # ── PnL calculation ───────────────────────
                        if direction == "long":
                            raw_pnl = (exit_price - entry_price) * qty
                        else:
                            raw_pnl = (entry_price - exit_price) * qty

                        commission = qty * entry_price * 0.0004 * 2
                        net_pnl = raw_pnl - commission

                        pnl_pct = (
                            (net_pnl / bot.equity * 100)
                            if bot.equity > 0
                            else 0.0
                        )

                        bot.equity += net_pnl
                        bot.total_pnl += net_pnl
                        bot.trades += 1
                        if net_pnl > 0:
                            bot.wins += 1
                        if bot.equity > bot.peak_equity:
                            bot.peak_equity = bot.equity

                        bot._append_equity()

                        # Feed reward to RL brain
                        bot.brain.record_outcome(
                            reward=pnl_pct, done=True,
                        )

                        outcome = "WIN" if net_pnl > 0 else "LOSS"
                        bot.logger.info(
                            "CLOSE %s %s %s @ %.6f → %.6f | "
                            "pnl=%.2f equity=%.2f",
                            outcome,
                            direction.upper(),
                            bot.symbol,
                            entry_price,
                            exit_price,
                            net_pnl,
                            bot.equity,
                        )

                        bot._active_trade = None
                else:
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

        # Flush RL brains (save remaining buffer)
        for bot in self.bots:
            try:
                bot.brain.flush()
            except Exception:
                pass

        # Fetch final equity before closing exchange
        final_equity = await self._fetch_real_total_equity()
        for b in self.bots:
            b._account_equity = final_equity

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

    bots: list[PaperBot] = []
    for idx, coin in enumerate(TOP_100_COINS):
        bot = PaperBot(
            bot_id=idx + 1,
            symbol=coin,
            config=config,
            output_dir=output_dir,
            exchange=exchange,
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
