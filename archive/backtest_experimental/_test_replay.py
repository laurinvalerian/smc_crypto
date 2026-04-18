"""Quick test: single-symbol replay loop for BTC."""
import xgboost  # MUST import before torch (shared lib conflict causes segfault)
import logging
import asyncio
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger()

log.info("Step 1: imports")
from live_multi_bot import PaperBot, ASSET_SMC_PARAMS, COMMISSION_MULTIPLIER, STYLE_CONFIG
from exchanges.replay_adapter import ReplayAdapter
from rl_brain_v2 import RLBrainSuite
from trade_journal import TradeJournal
from risk.circuit_breaker import CircuitBreaker
from backtest.replay_runner import _load_tf_data, _handle_trade_close

log.info("Step 2: config")
ASSET_SMC_PARAMS["crypto"]["alignment_threshold"] = 0.78

model_path = Path("replay_results/B_initial_model.pkl")
suite = RLBrainSuite({
    "entry_filter": {"enabled": True, "model_path": str(model_path), "confidence_threshold": 0.55},
    "be_manager": {"enabled": False},
    "tp_optimizer": {"enabled": False},
    "exit_classifier": {"enabled": False},
})
log.info("Step 3: create bot")
out = Path("replay_results/test_single")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

adapter = ReplayAdapter("crypto", 100_000.0, leverage=10)
journal = TradeJournal(str(out / "journal.db"))
cb = CircuitBreaker()
sym = "BTC/USDT:USDT"

bot = PaperBot(0, sym, {}, out / "bots", "crypto", adapter, rl_suite=suite)
bot.equity = 100_000.0
bot.peak_equity = 100_000.0
bot._account_equity = 100_000.0
bot.journal = journal
bot.circuit_breaker = cb

log.info("Step 4: load TF data (trimmed to live-equivalent limits)")
# Trim higher TF buffers to match live bot's load_history limits
# Without this, SMC library processes 113K+ bars per candle = extremely slow
TF_LIMITS = {"1d": 250, "4h": 500, "1h": 1000, "15m": 1500}
for tf, limit in TF_LIMITS.items():
    df_tf = _load_tf_data(sym, "crypto", tf)
    if not df_tf.empty:
        df_tf = df_tf.tail(limit).reset_index(drop=True)
        setattr(bot, f"buffer_{tf}", df_tf)
        log.info("  %s: %d bars (trimmed from full)", tf, len(df_tf))

df_5m = _load_tf_data(sym, "crypto", "5m")
log.info("  5m: %d bars", len(df_5m))

log.info("Step 5: replay 700 candles")
import pandas as pd

candle_count = 0
t0 = time.time()

for _, row in df_5m.head(700).iterrows():
    ts = row["timestamp"]
    if not isinstance(ts, datetime):
        ts = pd.Timestamp(ts).to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    candle = {
        "timestamp": ts,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }

    adapter.update_price(sym, candle["close"], ts)

    # Check fills
    fills = adapter.check_and_fill_orders(sym, candle)
    for fill in fills:
        matching = [t for t in bot._active_trades if t.get("symbol") == sym]
        if matching:
            _handle_trade_close(bot, matching[0], fill["exit_price"], fill["exit_reason"], ts)
            log.info("CLOSED %s @ %.2f | equity=%.2f", fill["exit_reason"], fill["exit_price"], bot.equity)

    # Feed candle
    bot.on_candle(sym, candle)
    candle_count += 1

    # Simulate tick for entry zone
    if bot._pending_signal is not None and candle_count > 300:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bot.on_tick(sym, candle["close"]))
        except Exception:
            pass
        finally:
            loop.close()

elapsed = time.time() - t0
log.info(
    "DONE: %d candles in %.1fs (%.0f/s), trades=%d, wins=%d, equity=%.2f, pending=%s",
    candle_count, elapsed, candle_count / max(elapsed, 0.01),
    bot.trades, bot.wins, bot.equity,
    "YES" if bot._pending_signal else "NO",
)
