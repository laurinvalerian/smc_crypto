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
FIXED_RR_MIN = 2.0          # minimum 1:2 reward-to-risk
FIXED_ATR_PERIOD = 14
FIXED_EMA_FAST = 20
FIXED_EMA_SLOW = 50
FIXED_MIN_VOL_MULT = 1.0    # min volume = 1.0× average

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
    A single paper-trading bot specialised on one coin.

    Uses fixed SMC parameters and money management.  An RL brain
    (PPO) gates each potential entry (yes / no).  Reward for the
    brain = pure PnL change in % (no shaping, no R:R bonus).

    Strategy logic:
      – Monitors 5 m candles for its assigned coin
      – Computes alignment score (EMA-20/50 trend + BOS momentum)
      – If score ≥ threshold AND RL brain says "trade" → enter
      – Full SL or full TP (no partial exits)
      – Volume filter: skip bar if volume < 1.0× average(20)
    """

    def __init__(
        self,
        bot_id: int,
        symbol: str,
        config: dict[str, Any],
        output_dir: Path,
    ) -> None:
        self.bot_id = bot_id
        self.tag = f"bot_{bot_id:03d}"
        self.symbol = symbol

        # Fixed strategy parameters
        self.swing_length: int = FIXED_SMC_PARAMS["swing_length"]
        self.alignment_threshold: float = FIXED_SMC_PARAMS["alignment_threshold"]
        self.risk_pct: float = FIXED_RISK_PCT
        self.rr_ratio: float = FIXED_RR_MIN
        self.leverage: int = 10  # default leverage

        # Virtual account
        self.start_equity: float = float(config["account"]["size"])
        self.equity: float = self.start_equity
        self.peak_equity: float = self.start_equity

        # Tracking
        self.total_pnl: float = 0.0
        self.trades: int = 0
        self.wins: int = 0
        self.open_positions: dict[str, dict[str, Any]] = {}
        self.max_open: int = 1  # 1 bot = 1 coin = max 1 position

        # Candle history  {symbol: list[dict]}
        self._candle_buf: dict[str, list[dict[str, Any]]] = {}

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

        candle keys: timestamp, open, high, low, close, volume
        """
        if symbol not in self._candle_buf:
            self._candle_buf[symbol] = []
        buf = self._candle_buf[symbol]
        buf.append(candle)
        if len(buf) > 300:
            buf.pop(0)

        self._check_exits(symbol, candle)

        if len(buf) >= self.swing_length + 5:
            self._evaluate_entry(symbol, buf, candle)

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

    # ── Entry evaluation ──────────────────────────────────────────

    def _evaluate_entry(
        self,
        symbol: str,
        buf: list[dict[str, Any]],
        candle: dict[str, Any],
    ) -> None:
        if symbol in self.open_positions:
            return
        if len(self.open_positions) >= self.max_open:
            return

        # Volume filter: skip if current volume < 1.0× avg(20)
        volumes = [c["volume"] for c in buf[-20:]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 0.0
        if avg_vol > 0 and candle["volume"] < FIXED_MIN_VOL_MULT * avg_vol:
            return

        score, direction = self._alignment_score(buf, self.swing_length)
        if score < self.alignment_threshold:
            return

        price = candle["close"]
        if price <= 0:
            return

        atr = self._simple_atr(buf, period=FIXED_ATR_PERIOD)
        if atr <= 0:
            return

        # ── RL Brain gate ─────────────────────────────────────────
        obs = extract_features(buf, score, direction)
        take_trade = self.brain.should_trade(obs)
        if not take_trade:
            # Skipped → reward 0 for brain
            self.brain.record_outcome(reward=0.0, done=True)
            return

        self._pending_obs = obs

        sl_dist = max(atr * 1.5, price * 0.0035)
        tp_dist = sl_dist * self.rr_ratio

        if direction == "long":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        risk_amount = self.equity * self.risk_pct
        qty = (risk_amount / sl_dist) if sl_dist > 0 else 0.0
        notional = qty * price

        if qty <= 0 or notional <= 0:
            return

        self.open_positions[symbol] = {
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "qty": qty,
            "leverage": self.leverage,
            "entry_time": candle["timestamp"],
            "score": score,
        }
        self.logger.info(
            "OPEN %s %s @ %.6f | SL=%.6f TP=%.6f | qty=%.4f lev=%dx score=%.2f",
            direction.upper(), symbol, price, sl, tp, qty, self.leverage, score,
        )

    # ── Exit checking ─────────────────────────────────────────────

    def _check_exits(self, symbol: str, candle: dict[str, Any]) -> None:
        pos = self.open_positions.get(symbol)
        if pos is None:
            return

        high = candle["high"]
        low = candle["low"]
        direction = pos["direction"]

        hit_tp = False
        hit_sl = False

        if direction == "long":
            if low <= pos["sl"]:
                hit_sl = True
            elif high >= pos["tp"]:
                hit_tp = True
        else:
            if high >= pos["sl"]:
                hit_sl = True
            elif low <= pos["tp"]:
                hit_tp = True

        if not hit_tp and not hit_sl:
            return

        if hit_tp:
            exit_price = pos["tp"]
            outcome = "WIN"
        else:
            exit_price = pos["sl"]
            outcome = "LOSS"

        if direction == "long":
            raw_pnl = (exit_price - pos["entry"]) * pos["qty"]
        else:
            raw_pnl = (pos["entry"] - exit_price) * pos["qty"]

        commission = pos["qty"] * pos["entry"] * 0.0004 * 2
        net_pnl = raw_pnl - commission

        # Reward = pure PnL change in % (relative to equity before trade)
        pnl_pct = (net_pnl / self.equity * 100) if self.equity > 0 else 0.0

        self.equity += net_pnl
        self.total_pnl += net_pnl
        self.trades += 1
        if net_pnl > 0:
            self.wins += 1
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self._append_equity()

        # Feed reward to RL brain
        self.brain.record_outcome(reward=pnl_pct, done=True)

        self.logger.info(
            "CLOSE %s %s %s @ %.6f → %.6f | pnl=%.2f equity=%.2f",
            outcome, direction.upper(), symbol,
            pos["entry"], exit_price, net_pnl, self.equity,
        )
        del self.open_positions[symbol]

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
        if self.start_equity <= 0:
            return 0.0
        return (self.equity - self.start_equity) / self.start_equity * 100

    def summary_dict(self) -> dict[str, Any]:
        return {
            "bot": self.tag,
            "symbol": self.symbol,
            "equity": round(self.equity, 2),
            "pnl": round(self.total_pnl, 2),
            "return_pct": round(self.return_pct, 2),
            "trades": self.trades,
            "winrate": round(self.winrate * 100, 1),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "open_pos": len(self.open_positions),
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
    exchange.set_sandbox_mode(True)
    logger.info("Exchange created: %s (sandbox/testnet)", exchange.id)
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

    total_equity = sum(b.equity for b in bots)
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
      - WebSocket auto-reconnect per symbol
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

    # ── Rich Dashboard loop ───────────────────────────────────────

    async def _dashboard_loop(self) -> None:
        """Render the Rich Live Dashboard every DASHBOARD_REFRESH_SEC."""
        console = Console()

        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while not self._shutdown.is_set():
                try:
                    layout = build_dashboard(
                        bots=self.bots,
                        ws_status=self.ws_status,
                        start_time=self._start_time,
                        active_symbols=self.symbols,
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
        """Start all watchers + dashboard. Blocks until shutdown."""
        logger.info(
            "Starting %d bots on %d symbols …", len(self.bots), len(self.symbols)
        )

        # Start one watcher task per symbol (= per bot)
        for sym in self.symbols:
            self._watcher_tasks[sym] = asyncio.create_task(
                self._watch_symbol(sym)
            )

        # Rich dashboard
        dashboard_task = asyncio.create_task(self._dashboard_loop())

        # Wait until shutdown
        await self._shutdown.wait()
        logger.info("Shutdown signal received – stopping …")

        # Cancel all tasks
        dashboard_task.cancel()
        for t in self._watcher_tasks.values():
            t.cancel()

        all_tasks = [dashboard_task] + list(self._watcher_tasks.values())
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Flush RL brains (save remaining buffer)
        for bot in self.bots:
            try:
                bot.brain.flush()
            except Exception:
                pass

        # Close exchange WebSocket connections
        try:
            await self.exchange.close()
        except Exception:
            pass

        # Final summary to console
        self._print_final_summary()

    def _print_final_summary(self) -> None:
        """Print a plain-text final summary after dashboard stops."""
        console = Console()
        rows = sorted(
            [b.summary_dict() for b in self.bots],
            key=lambda r: r["pnl"],
            reverse=True,
        )
        total_equity = sum(b.equity for b in self.bots)
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
        )
        bots.append(bot)

    logger.info("Created %d coin-specialised bots", len(bots))
    console.print(
        f"[bold green]✅ {len(bots)} bots created – "
        f"each assigned to a unique coin.[/bold green]"
    )

    # ── Create exchange ───────────────────────────────────────────
    exchange = create_exchange(api_key, api_secret)

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
