"""
═══════════════════════════════════════════════════════════════════
 live_multi_bot.py
 ─────────────────
 Phase 2 – Live Demo: Run the Top 30 best parameter-sets from
 walk-forward backtesting as parallel paper-trading bots on the
 Binance Futures Testnet.

 Each bot:
   • Gets its own parameter-set (loaded from CSV)
   • Tracks its own virtual equity & PnL
   • Writes an equity-curve CSV  (live_results/bot_001_equity.csv)
   • Writes its own log file      (live_results/bot_001.log)

 A 5-minute summary prints a ranking to the console.

 Requirements:
   pip install ccxt[pro] pandas numpy python-dotenv pyyaml tqdm

 Usage:
   1. Copy .env.example → .env and fill in your testnet keys
   2. Run the backtest first so that CSV files exist in backtest/results/
   3. python live_multi_bot.py [--top 30] [--config config/default_config.yaml]

 Ctrl+C → graceful shutdown, final summary.
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

# ── ccxt.pro for WebSocket ────────────────────────────────────────
try:
    import ccxt.pro as ccxtpro
except ImportError:
    sys.exit(
        "ccxt.pro is required.  Install with:  pip install 'ccxt[pro]'"
    )

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("backtest/results")
OUTPUT_DIR = Path("live_results")
SUMMARY_INTERVAL_SEC = 300  # 5 minutes

# Same 20 coins used throughout the project (CCXT futures format)
COINS: list[str] = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "TON/USDT:USDT", "ADA/USDT:USDT",
    "AVAX/USDT:USDT", "SHIB/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
    "TRX/USDT:USDT", "BCH/USDT:USDT", "NEAR/USDT:USDT", "LTC/USDT:USDT",
    "PEPE/USDT:USDT", "SUI/USDT:USDT", "UNI/USDT:USDT", "HBAR/USDT:USDT",
]

# ═══════════════════════════════════════════════════════════════════
#  Logging helpers
# ═══════════════════════════════════════════════════════════════════

_console_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                 datefmt="%H:%M:%S")


def _make_logger(name: str, log_path: Path) -> logging.Logger:
    """Create a logger that writes to *log_path* and to stderr."""
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    lgr.addHandler(fh)

    return lgr


# Root logger → console only
root_logger = logging.getLogger("live_multi")
root_logger.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setFormatter(_console_fmt)
root_logger.addHandler(_ch)
logger = root_logger

# ═══════════════════════════════════════════════════════════════════
#  Parameter loading
# ═══════════════════════════════════════════════════════════════════

# Columns that are strategy parameters (not metrics)
PARAM_COLS: list[str] = [
    "leverage", "risk_per_trade", "alignment_threshold",
    "swing_length", "fvg_threshold", "order_block_lookback",
    "liquidity_range_percent", "risk_reward", "weight_day",
]


def load_top_params(results_dir: Path, top_n: int = 30) -> pd.DataFrame:
    """
    Load parameter-sets from backtest CSV files and return the top *top_n*
    ranked by a composite score  (0.7 × normalised total_pnl
                                 + 0.3 × normalised sharpe).
    """
    frames: list[pd.DataFrame] = []

    # 1. global_top_params.csv (primary source)
    global_path = results_dir / "global_top_params.csv"
    if global_path.exists():
        df = pd.read_csv(global_path)
        df["_source"] = "global"
        frames.append(df)
        logger.info("Loaded %d rows from %s", len(df), global_path.name)

    # 2. Per-window top_params_wX.csv files
    for p in sorted(results_dir.glob("top_params_w*.csv")):
        df = pd.read_csv(p)
        df["_source"] = p.stem
        frames.append(df)
        logger.info("Loaded %d rows from %s", len(df), p.name)

    if not frames:
        raise FileNotFoundError(
            f"No parameter CSV files found in {results_dir}. "
            "Run the backtest first (python -m backtest.optuna_backtester)."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Ensure required metric columns exist (fill 0 if missing)
    for col in ("total_pnl", "sharpe"):
        if col not in combined.columns:
            combined[col] = 0.0

    # ── Composite ranking score ───────────────────────────────────
    pnl = combined["total_pnl"].astype(float)
    sharpe = combined["sharpe"].astype(float)

    pnl_range = pnl.max() - pnl.min()
    sharpe_range = sharpe.max() - sharpe.min()

    norm_pnl = (pnl - pnl.min()) / pnl_range if pnl_range > 0 else 0.0
    norm_sharpe = (sharpe - sharpe.min()) / sharpe_range if sharpe_range > 0 else 0.0

    combined["_rank_score"] = 0.7 * norm_pnl + 0.3 * norm_sharpe
    combined = combined.sort_values("_rank_score", ascending=False).reset_index(drop=True)

    # De-duplicate (identical param vectors)
    param_cols_present = [c for c in PARAM_COLS if c in combined.columns]
    if param_cols_present:
        combined = combined.drop_duplicates(subset=param_cols_present, keep="first")

    top = combined.head(top_n).copy()
    logger.info("Selected top %d parameter sets (of %d total)", len(top), len(combined))
    return top


def params_from_row(row: pd.Series) -> dict[str, Any]:
    """Extract a clean parameter dict from a DataFrame row."""
    params: dict[str, Any] = {}
    for col in PARAM_COLS:
        if col in row.index and pd.notna(row[col]):
            params[col] = row[col]
    return params


# ═══════════════════════════════════════════════════════════════════
#  Paper-Trading Bot
# ═══════════════════════════════════════════════════════════════════

class PaperBot:
    """
    A single paper-trading bot that evaluates a specific parameter set
    against live market prices.

    Simplified strategy logic:
      – Uses alignment_threshold, risk_per_trade, leverage, risk_reward
      – Monitors 5 m candles; enters when alignment score (simple
        momentum + volatility proxy) exceeds threshold
      – Simulates fill at close, SL/TP based on ATR
      – Tracks equity over time
    """

    def __init__(
        self,
        bot_id: int,
        params: dict[str, Any],
        config: dict[str, Any],
        output_dir: Path,
    ) -> None:
        self.bot_id = bot_id
        self.tag = f"bot_{bot_id:03d}"
        self.params = params
        self.cfg = config

        # Strategy parameters
        self.leverage: int = int(params.get("leverage", 10))
        self.risk_pct: float = float(params.get("risk_per_trade", 0.01))
        self.rr_ratio: float = float(params.get("risk_reward", 3.0))
        self.alignment_threshold: float = float(
            params.get("alignment_threshold", 0.55)
        )
        self.swing_length: int = int(params.get("swing_length", 8))

        # Virtual account
        self.start_equity: float = float(config["account"]["size"])
        self.equity: float = self.start_equity
        self.peak_equity: float = self.start_equity

        # Tracking
        self.total_pnl: float = 0.0
        self.trades: int = 0
        self.wins: int = 0
        self.open_positions: dict[str, dict[str, Any]] = {}
        self.max_open: int = int(config.get("live", {}).get("max_open_trades", 5))

        # Candle history per symbol  {symbol: list[dict]}
        self._candle_buf: dict[str, list[dict[str, Any]]] = {
            s: [] for s in COINS
        }

        # Output
        output_dir.mkdir(parents=True, exist_ok=True)
        self._equity_path = output_dir / f"{self.tag}_equity.csv"
        self._init_equity_csv()
        self.logger = _make_logger(
            f"live_multi.{self.tag}",
            output_dir / f"{self.tag}.log",
        )
        self.logger.info(
            "Initialised %s | params=%s | equity=%.2f",
            self.tag, json.dumps(params, default=str), self.equity,
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
        buf = self._candle_buf[symbol]
        buf.append(candle)
        # Keep last 300 candles (~25 h of 5 m data)
        if len(buf) > 300:
            buf.pop(0)

        # Check open positions for SL/TP hit
        self._check_exits(symbol, candle)

        # Evaluate new entry only if we have enough history
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

        # EMA-20 and EMA-50
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]

        # Recent swing high/low
        recent_high = float(np.max(highs[-swing_len:]))
        recent_low = float(np.min(lows[-swing_len:]))
        prev_high = float(np.max(highs[-2 * swing_len: -swing_len])) if len(highs) >= 2 * swing_len else recent_high
        prev_low = float(np.min(lows[-2 * swing_len: -swing_len])) if len(lows) >= 2 * swing_len else recent_low

        price = closes[-1]

        # Trend component [0..1]
        trend_bull = 1.0 if ema20 > ema50 else 0.0

        # Momentum: break of structure
        bos_bull = 1.0 if recent_high > prev_high else 0.0
        bos_bear = 1.0 if recent_low < prev_low else 0.0

        # Position relative to EMAs
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
        # Skip if already have position in this symbol or at max
        if symbol in self.open_positions:
            return
        if len(self.open_positions) >= self.max_open:
            return

        score, direction = self._alignment_score(buf, self.swing_length)
        if score < self.alignment_threshold:
            return

        price = candle["close"]
        if price <= 0:
            return

        # ATR-based SL
        atr = self._simple_atr(buf, period=14)
        if atr <= 0:
            return

        sl_dist = max(atr * 1.5, price * 0.0035)  # min 0.35 %
        tp_dist = sl_dist * self.rr_ratio

        if direction == "long":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        # Position sizing
        risk_amount = self.equity * self.risk_pct
        qty = (risk_amount / sl_dist) if sl_dist > 0 else 0.0
        notional = qty * price

        # Sanity check
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
        else:  # short
            if high >= pos["sl"]:
                hit_sl = True
            elif low <= pos["tp"]:
                hit_tp = True

        if not hit_tp and not hit_sl:
            return

        # Compute PnL
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

        # Commission (round-trip)
        commission = pos["qty"] * pos["entry"] * 0.0004 * 2
        net_pnl = raw_pnl - commission

        self.equity += net_pnl
        self.total_pnl += net_pnl
        self.trades += 1
        if net_pnl > 0:
            self.wins += 1
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self._append_equity()

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
#  Main runner
# ═══════════════════════════════════════════════════════════════════

class LiveMultiBotRunner:
    """Orchestrates multiple PaperBot instances with shared WebSocket data."""

    def __init__(
        self,
        bots: list[PaperBot],
        exchange: Any,
        symbols: list[str],
    ) -> None:
        self.bots = bots
        self.exchange = exchange
        self.symbols = symbols
        self._shutdown = asyncio.Event()

    # ── Summary printing ──────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a ranked summary of all bots to the console."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        rows = sorted(
            [b.summary_dict() for b in self.bots],
            key=lambda r: r["pnl"],
            reverse=True,
        )

        header = (
            f"\n{'═' * 90}\n"
            f"  📊  LIVE MULTI-BOT RANKING  –  {now}\n"
            f"{'═' * 90}"
        )
        col_hdr = (
            f"  {'#':>3}  {'Bot':<8}  {'Equity':>12}  {'PnL':>10}  "
            f"{'Return%':>8}  {'Trades':>6}  {'WR%':>5}  {'DD%':>6}  {'Open':>4}"
        )
        sep = f"  {'─' * 84}"

        lines = [header, col_hdr, sep]
        for i, r in enumerate(rows, 1):
            lines.append(
                f"  {i:>3}  {r['bot']:<8}  {r['equity']:>12,.2f}  "
                f"{r['pnl']:>+10,.2f}  {r['return_pct']:>+7.2f}%  "
                f"{r['trades']:>6}  {r['winrate']:>5.1f}  "
                f"{r['drawdown_pct']:>5.2f}%  {r['open_pos']:>4}"
            )
        lines.append(f"{'═' * 90}\n")
        print("\n".join(lines), flush=True)

    # ── WebSocket OHLCV watcher ───────────────────────────────────

    async def _watch_symbol(self, symbol: str) -> None:
        """
        Subscribe to 5 m OHLCV candles for *symbol* and feed each bot.

        ccxt.pro's watch_ohlcv returns the latest candle array every time
        a new tick arrives; we detect new *closed* candles by timestamp.
        """
        last_ts: int | None = None

        while not self._shutdown.is_set():
            try:
                ohlcv_list = await self.exchange.watch_ohlcv(symbol, "5m")
            except Exception as exc:
                logger.warning("watch_ohlcv %s error: %s", symbol, exc)
                await asyncio.sleep(5)
                continue

            if not ohlcv_list:
                continue

            # ohlcv_list is [[ts, o, h, l, c, v], ...]
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

                for bot in self.bots:
                    try:
                        bot.on_candle(symbol, candle)
                    except Exception as exc:
                        bot.logger.error(
                            "Error processing candle for %s: %s", symbol, exc
                        )

    # ── Periodic summary ──────────────────────────────────────────

    async def _summary_loop(self) -> None:
        """Print a summary every SUMMARY_INTERVAL_SEC seconds."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=SUMMARY_INTERVAL_SEC
                )
                break  # shutdown was set
            except asyncio.TimeoutError:
                pass
            self.print_summary()

    # ── Main loop ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all watchers + summary loop. Blocks until shutdown."""
        logger.info(
            "Starting %d bots on %d symbols …", len(self.bots), len(self.symbols)
        )
        self.print_summary()

        tasks: list[asyncio.Task[None]] = []

        # One watcher task per symbol
        for sym in self.symbols:
            tasks.append(asyncio.create_task(self._watch_symbol(sym)))

        # Summary printer
        tasks.append(asyncio.create_task(self._summary_loop()))

        # Wait until shutdown
        await self._shutdown.wait()
        logger.info("Shutdown signal received – stopping watchers …")

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Close exchange WebSocket connections
        try:
            await self.exchange.close()
        except Exception:
            pass

        # Final summary
        logger.info("Final summary:")
        self.print_summary()

    def request_shutdown(self) -> None:
        self._shutdown.set()


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 – Live Multi-Bot Demo (Binance Testnet)",
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of top parameter-sets to run (default: 30)",
    )
    parser.add_argument(
        "--config", default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Directory containing backtest CSV results",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Directory for live bot outputs (equity CSVs, logs)",
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

    # ── Load parameters ───────────────────────────────────────────
    results_dir = Path(args.results_dir)
    top_params_df = load_top_params(results_dir, top_n=args.top)
    if top_params_df.empty:
        sys.exit("No parameter sets found. Run the backtest first.")

    # ── Create bots ───────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bots: list[PaperBot] = []
    for idx, row in top_params_df.iterrows():
        params = params_from_row(row)
        bot = PaperBot(
            bot_id=int(idx) + 1,
            params=params,
            config=config,
            output_dir=output_dir,
        )
        bots.append(bot)

    logger.info("Created %d paper-trading bots", len(bots))

    # ── Create exchange ───────────────────────────────────────────
    exchange = create_exchange(api_key, api_secret)

    # ── Runner ────────────────────────────────────────────────────
    runner = LiveMultiBotRunner(bots=bots, exchange=exchange, symbols=COINS)

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
