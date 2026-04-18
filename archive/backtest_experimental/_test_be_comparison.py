"""
BE Manager Impact Test — BTC single-symbol comparison.
Runs 4 configs: B/C × BE_on/BE_off on 10K candles.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xgboost  # MUST import before torch
import asyncio
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("be_test")
log.setLevel(logging.INFO)

from live_multi_bot import PaperBot, ASSET_SMC_PARAMS
from exchanges.replay_adapter import ReplayAdapter
from rl_brain_v2 import RLBrainSuite
from backtest.replay_runner import _load_tf_data, _handle_trade_close
from trade_journal import TradeJournal
from risk.circuit_breaker import CircuitBreaker
import pandas as pd

N_CANDLES = 10_000
SYM = "BTC/USDT:USDT"
AC = "crypto"


def run_config(name: str, model_path: str, threshold: float, be_enabled: bool) -> dict:
    """Run a single config and return metrics."""
    ASSET_SMC_PARAMS["crypto"]["alignment_threshold"] = threshold

    rl_config = {
        "entry_filter": {"enabled": True, "model_path": model_path, "confidence_threshold": 0.55},
        "be_manager": {"enabled": be_enabled, "model_path": "models/rl_be_manager.pkl", "min_be_rr": 0.5},
        "tp_optimizer": {"enabled": False},
        "exit_classifier": {"enabled": False},
    }
    suite = RLBrainSuite(rl_config)

    out = Path(f"replay_results/be_test/{name}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    adapter = ReplayAdapter(AC, 100_000.0, leverage=10)
    journal = TradeJournal(str(out / "journal.db"))
    cb = CircuitBreaker()

    bot = PaperBot(0, SYM, {}, out / "bots", AC, adapter, rl_suite=suite)
    bot.equity = 100_000.0
    bot.peak_equity = 100_000.0
    bot._account_equity = 100_000.0
    bot.journal = journal
    bot.circuit_breaker = cb

    # Load trimmed TF buffers (match live bot limits)
    for tf, limit in [("1d", 250), ("4h", 500), ("1h", 1000), ("15m", 1500)]:
        df_tf = _load_tf_data(SYM, AC, tf)
        if not df_tf.empty:
            setattr(bot, f"buffer_{tf}", df_tf.tail(limit).reset_index(drop=True))

    df_5m = _load_tf_data(SYM, AC, "5m")
    if df_5m.empty:
        return {"name": name, "error": "no data"}

    t0 = time.time()
    candle_count = 0
    be_triggers = 0

    for _, row in df_5m.tail(N_CANDLES).iterrows():
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

        adapter.update_price(SYM, candle["close"], ts)

        # Check SL/TP fills
        fills = adapter.check_and_fill_orders(SYM, candle)
        for fill in fills:
            matching = [t for t in bot._active_trades if t.get("symbol") == SYM]
            if matching:
                _handle_trade_close(bot, matching[0], fill["exit_price"], fill["exit_reason"], ts)

        # Feed candle
        bot.on_candle(SYM, candle)
        candle_count += 1

        # BE check — replicate live bot logic
        if be_enabled and bot._active_trades:
            for trade in bot._active_trades:
                if trade.get("be_triggered", False) or trade.get("rl_be_level", 0) <= 0:
                    continue
                entry_p = trade["entry"]
                direction = trade["direction"]
                sl_dist = abs(entry_p - trade.get("original_sl", trade["sl"]))
                if sl_dist <= 0:
                    continue
                be_target_rr = trade["rl_be_level"]
                if direction == "long":
                    current_rr = (candle["high"] - entry_p) / sl_dist
                else:
                    current_rr = (entry_p - candle["low"]) / sl_dist
                if current_rr >= be_target_rr:
                    fee_buffer = entry_p * bot.commission_rate * 4
                    new_sl = entry_p + fee_buffer if direction == "long" else entry_p - fee_buffer
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(
                            adapter.modify_stop_loss("", SYM, "sell" if direction == "long" else "buy",
                                                     trade.get("qty", 0), new_sl)
                        )
                    finally:
                        loop.close()
                    trade["sl"] = new_sl
                    trade["be_triggered"] = True
                    be_triggers += 1

        # Simulate tick for entry
        if bot._pending_signal is not None and candle_count > 300:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(bot.on_tick(SYM, candle["close"]))
            except Exception:
                pass
            finally:
                loop.close()

    elapsed = time.time() - t0

    # Compute metrics
    wins = bot.wins
    trades = bot.trades
    wr = (wins / trades * 100) if trades > 0 else 0
    ret = ((bot.equity - 100_000) / 100_000) * 100
    max_dd = ((bot.peak_equity - bot.equity) / bot.peak_equity) * 100 if bot.peak_equity > 0 else 0

    return {
        "name": name,
        "trades": trades,
        "wins": wins,
        "wr": round(wr, 1),
        "equity": round(bot.equity, 2),
        "return_pct": round(ret, 2),
        "max_dd_pct": round(max_dd, 2),
        "be_triggers": be_triggers,
        "be_enabled": be_enabled,
        "elapsed": round(elapsed, 1),
        "candles_per_sec": round(candle_count / elapsed, 1) if elapsed > 0 else 0,
    }


def main():
    configs = [
        ("B_noBE", "replay_results/B_initial_model.pkl", 0.78, False),
        ("B_withBE", "replay_results/B_initial_model.pkl", 0.78, True),
        ("C_noBE", "replay_results/C_initial_model.pkl", 0.78, False),
        ("C_withBE", "replay_results/C_initial_model.pkl", 0.78, True),
    ]

    print(f"BE Impact Test: {SYM}, {N_CANDLES} candles, 4 configs")
    print("=" * 80)

    results = []
    for name, model, thresh, be in configs:
        print(f"\nRunning {name}...")
        r = run_config(name, model, thresh, be)
        results.append(r)
        print(f"  Trades={r['trades']}, WR={r['wr']}%, Return={r['return_pct']}%, "
              f"MaxDD={r['max_dd_pct']}%, BE_triggers={r.get('be_triggers', 0)}, "
              f"Speed={r['candles_per_sec']}/s ({r['elapsed']}s)")

    print("\n" + "=" * 80)
    print(f"{'Config':<15} {'Trades':>7} {'WR%':>6} {'Return%':>10} {'MaxDD%':>8} {'BE_trig':>8} {'Equity':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} {r['trades']:>7} {r['wr']:>5.1f}% {r['return_pct']:>9.2f}% "
              f"{r['max_dd_pct']:>7.2f}% {r.get('be_triggers', 0):>8} {r['equity']:>11.2f}")

    # Delta analysis
    print("\n" + "=" * 80)
    print("BE Impact Analysis:")
    for variant in ["B", "C"]:
        no_be = next(r for r in results if r["name"] == f"{variant}_noBE")
        with_be = next(r for r in results if r["name"] == f"{variant}_withBE")
        print(f"\n  {variant}: BE {'HELPS' if with_be['return_pct'] > no_be['return_pct'] else 'HURTS'}")
        print(f"    Return: {no_be['return_pct']}% → {with_be['return_pct']}% "
              f"(Δ{with_be['return_pct'] - no_be['return_pct']:+.2f}%)")
        print(f"    WR:     {no_be['wr']}% → {with_be['wr']}%")
        print(f"    MaxDD:  {no_be['max_dd_pct']}% → {with_be['max_dd_pct']}%")
        print(f"    BE triggers: {with_be.get('be_triggers', 0)}")


if __name__ == "__main__":
    main()
