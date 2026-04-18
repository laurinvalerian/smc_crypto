"""Test: parallel symbol processing to measure full MacBook throughput."""
import xgboost  # MUST import before torch
import multiprocessing as mp
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING)

N_CANDLES = 1000


def process_symbol(args):
    """Process one symbol — runs in its own process."""
    sym, ac = args
    from live_multi_bot import PaperBot, ASSET_SMC_PARAMS
    from exchanges.replay_adapter import ReplayAdapter
    from rl_brain_v2 import RLBrainSuite
    from backtest.replay_runner import _load_tf_data
    import pandas as pd

    ASSET_SMC_PARAMS["crypto"]["alignment_threshold"] = 0.78

    model_path = Path("replay_results/B_initial_model.pkl")
    suite = RLBrainSuite({
        "entry_filter": {"enabled": True, "model_path": str(model_path), "confidence_threshold": 0.55},
        "be_manager": {"enabled": False}, "tp_optimizer": {"enabled": False}, "exit_classifier": {"enabled": False},
    })
    adapter = ReplayAdapter(ac, 100_000.0, leverage=10)
    out = Path(f"replay_results/test_par/{sym.replace('/', '_')}")
    out.mkdir(parents=True, exist_ok=True)

    bot = PaperBot(0, sym, {}, out, ac, adapter, rl_suite=suite)
    bot.equity = 100_000.0
    bot.peak_equity = 100_000.0
    bot._account_equity = 100_000.0

    # Load trimmed TF buffers
    for tf, limit in [("1d", 250), ("4h", 500), ("1h", 1000), ("15m", 1500)]:
        df_tf = _load_tf_data(sym, ac, tf)
        if not df_tf.empty:
            setattr(bot, f"buffer_{tf}", df_tf.tail(limit).reset_index(drop=True))

    df_5m = _load_tf_data(sym, ac, "5m")
    if df_5m.empty:
        return sym, 0, 0.0

    t0 = time.time()
    count = 0
    for _, row in df_5m.head(N_CANDLES).iterrows():
        ts = row["timestamp"]
        if not isinstance(ts, datetime):
            ts = pd.Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        candle = {"timestamp": ts, "open": float(row["open"]), "high": float(row["high"]),
                  "low": float(row["low"]), "close": float(row["close"]), "volume": float(row["volume"])}
        adapter.update_price(sym, candle["close"], ts)
        fills = adapter.check_and_fill_orders(sym, candle)
        bot.on_candle(sym, candle)
        count += 1
        if bot._pending_signal is not None and count > 300:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(bot.on_tick(sym, candle["close"]))
            except Exception:
                pass
            finally:
                loop.close()

    elapsed = time.time() - t0
    return sym, count, elapsed


def main():
    # Top 10 crypto symbols
    symbols = [
        ("BTC/USDT:USDT", "crypto"), ("ETH/USDT:USDT", "crypto"),
        ("SOL/USDT:USDT", "crypto"), ("BNB/USDT:USDT", "crypto"),
        ("XRP/USDT:USDT", "crypto"), ("DOGE/USDT:USDT", "crypto"),
        ("ADA/USDT:USDT", "crypto"), ("AVAX/USDT:USDT", "crypto"),
        ("LINK/USDT:USDT", "crypto"), ("DOT/USDT:USDT", "crypto"),
    ]

    print(f"Testing {len(symbols)} symbols × {N_CANDLES} candles with {mp.cpu_count()} cores")
    t0 = time.time()

    with mp.Pool(min(mp.cpu_count() - 1, len(symbols))) as pool:
        results = pool.map(process_symbol, symbols)

    total_elapsed = time.time() - t0
    total_candles = sum(r[1] for r in results)

    print(f"\nResults:")
    for sym, count, elapsed in results:
        rate = count / elapsed if elapsed > 0 else 0
        print(f"  {sym:20s}: {count} candles in {elapsed:.1f}s ({rate:.0f}/s)")

    print(f"\nTotal: {total_candles} candles in {total_elapsed:.1f}s")
    print(f"Throughput: {total_candles / total_elapsed:.0f} candles/sec (parallel)")
    print(f"Per-symbol avg: {sum(r[2] for r in results) / len(results):.1f}s for {N_CANDLES} candles")

    # Extrapolate
    candles_per_symbol = 340_000
    symbols_total = 112
    total_candles_full = candles_per_symbol * symbols_total
    hours = total_candles_full / (total_candles / total_elapsed) / 3600
    print(f"\nExtrapolation for full replay (112 symbols × 340K candles):")
    print(f"  Estimated: {hours:.1f} hours per variant")
    print(f"  5 variants: {hours * 5:.1f} hours total")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
