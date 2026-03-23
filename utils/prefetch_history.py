"""
utils/prefetch_history.py
=========================
Ensures all instruments have enough historical data for indicator warmup
(e.g., EMA200 on Daily needs 200+ daily bars before the backtest start date).

Downloads ONLY higher timeframes (1D, 4H, 1H) going back further than the
existing data. This is fast because higher TFs have far fewer bars.

Usage:
    python3 -m utils.prefetch_history                          # all asset classes
    python3 -m utils.prefetch_history --asset-class crypto     # only crypto
    python3 -m utils.prefetch_history --min-daily-bars 250     # custom warmup
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Timeframes to prefetch (only higher TFs — 1m/5m are too expensive to backfill)
PREFETCH_TFS = {
    "1d": 86400,     # seconds per bar
    "4h": 14400,
    "1h": 3600,
}

# How many bars we need BEFORE the backtest start date
DEFAULT_MIN_BARS = {"1d": 250, "4h": 500, "1h": 1000}


def load_config(path: str = "config/default_config.yaml") -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _bars_before(tf_path: Path, backtest_start: pd.Timestamp) -> int:
    """Count bars in an existing parquet file that are before backtest_start."""
    if not tf_path.exists():
        return 0
    try:
        df = pd.read_parquet(tf_path, columns=["timestamp"])
        return int((df["timestamp"] < backtest_start).sum())
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════
#  Crypto (Binance via CCXT)
# ═══════════════════════════════════════════════════════════════════

def prefetch_crypto(cfg: dict[str, Any], min_bars: dict[str, int], backtest_start: pd.Timestamp) -> None:
    import ccxt

    crypto_dir = Path(cfg["data"].get("crypto_dir", "data/crypto"))
    if not crypto_dir.exists():
        logger.warning("Crypto dir %s not found — skipping", crypto_dir)
        return

    exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
    exchange.load_markets()

    # Build symbol mapping: ETHUSDT → ETH/USDT:USDT
    sym_map = {}
    for mkt_sym, mkt in exchange.markets.items():
        if mkt.get("swap") and mkt.get("linear") and mkt.get("settle") == "USDT":
            safe = mkt_sym.replace("/", "").replace(":USDT", "")
            sym_map[safe] = mkt_sym

    parquets = sorted(crypto_dir.glob("*_1m.parquet"))
    symbols = [p.stem.replace("_1m", "") for p in parquets if "volume" not in p.stem]
    logger.info("Crypto: %d symbols to check", len(symbols))

    for sym in tqdm(symbols, desc="Crypto prefetch"):
        ccxt_sym = sym_map.get(sym)
        if not ccxt_sym:
            continue

        for tf, min_n in min_bars.items():
            tf_path = crypto_dir / f"{sym}_{tf}.parquet"

            if _bars_before(tf_path, backtest_start) >= min_n:
                continue

            # Calculate how far back to fetch
            tf_seconds = PREFETCH_TFS[tf]
            target_start = backtest_start - pd.Timedelta(seconds=min_n * tf_seconds * 1.2)
            since_ms = int(target_start.timestamp() * 1000)
            until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            try:
                all_candles = []
                fetch_since = since_ms

                while fetch_since < until_ms:
                    candles = exchange.fetch_ohlcv(ccxt_sym, timeframe=tf, since=fetch_since, limit=1500)
                    if not candles:
                        break
                    all_candles.extend(candles)
                    fetch_since = candles[-1][0] + tf_seconds * 1000
                    time.sleep(0.2)

                if not all_candles:
                    continue

                new_df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
                new_df = new_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                if tf_path.exists():
                    old_df = pd.read_parquet(tf_path)
                    combined = pd.concat([new_df, old_df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                else:
                    combined = new_df

                bars_b = int((combined["timestamp"] < backtest_start).sum())
                old_count = len(pd.read_parquet(tf_path)) if tf_path.exists() else 0
                combined.to_parquet(tf_path, engine="pyarrow", compression="gzip", index=False)
                logger.info("[%s] %s: %d → %d bars (%d before %s)", sym, tf, old_count, len(combined), bars_b, backtest_start.date())

            except Exception as exc:
                logger.warning("[%s] %s prefetch failed: %s", sym, tf, exc)


# ═══════════════════════════════════════════════════════════════════
#  Forex + Commodities (OANDA v20)
# ═══════════════════════════════════════════════════════════════════

def prefetch_oanda(cfg: dict[str, Any], min_bars: dict[str, int], backtest_start: pd.Timestamp, asset_class: str = "forex") -> None:
    if asset_class == "forex":
        data_dir = Path(cfg["data"].get("forex_dir", "data/forex"))
    else:
        data_dir = Path(cfg["data"].get("commodities_dir", "data/commodities"))

    if not data_dir.exists():
        logger.warning("%s dir %s not found — skipping", asset_class, data_dir)
        return

    try:
        import v20
    except ImportError:
        logger.error("v20 not installed — pip install v20")
        return

    token = os.getenv("OANDA_ACCESS_TOKEN")
    if not token:
        logger.error("OANDA_ACCESS_TOKEN not set in .env")
        return

    ctx = v20.Context("api-fxpractice.oanda.com", token=token)
    oanda_tf = {"1d": "D", "4h": "H4", "1h": "H1"}

    parquets = sorted(data_dir.glob("*_1m.parquet"))
    symbols = sorted(set(p.stem.replace("_1m", "") for p in parquets))
    logger.info("%s: %d instruments to check", asset_class.title(), len(symbols))

    # Target: go back far enough for warmup
    tf_seconds = PREFETCH_TFS["1d"]
    target_start_dt = backtest_start - pd.Timedelta(seconds=min_bars["1d"] * tf_seconds * 1.3)
    target_from = target_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for sym in tqdm(symbols, desc=f"{asset_class.title()} prefetch"):
        for tf, min_n in min_bars.items():
            tf_path = data_dir / f"{sym}_{tf}.parquet"

            if _bars_before(tf_path, backtest_start) >= min_n:
                continue

            o_tf = oanda_tf[tf]

            try:
                all_candles = []
                current_from = target_from

                for _ in range(20):  # max pagination loops
                    response = ctx.instrument.candles(
                        sym,
                        granularity=o_tf,
                        fromTime=current_from,
                        price="M",
                        count=500,
                    )
                    candles = response.body.get("candles", [])
                    if not candles:
                        break

                    for c in candles:
                        if c.complete:
                            all_candles.append({
                                "timestamp": pd.Timestamp(c.time),
                                "open": float(c.mid.o),
                                "high": float(c.mid.h),
                                "low": float(c.mid.l),
                                "close": float(c.mid.c),
                                "volume": int(c.volume),
                            })

                    last_time = candles[-1].time
                    if last_time == current_from:
                        break
                    current_from = last_time
                    time.sleep(0.25)

                    if len(candles) < 500:
                        break

                if not all_candles:
                    continue

                new_df = pd.DataFrame(all_candles)

                if tf_path.exists():
                    old_df = pd.read_parquet(tf_path)
                    combined = pd.concat([new_df, old_df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                else:
                    combined = new_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                bars_b = int((combined["timestamp"] < backtest_start).sum())
                old_count = len(pd.read_parquet(tf_path)) if tf_path.exists() else 0
                combined.to_parquet(tf_path, engine="pyarrow", compression="gzip", index=False)
                logger.info("[%s] %s: %d → %d bars (%d before %s)", sym, tf, old_count, len(combined), bars_b, backtest_start.date())

            except Exception as exc:
                logger.warning("[%s] %s prefetch failed: %s", sym, tf, exc)

            time.sleep(0.2)


# ═══════════════════════════════════════════════════════════════════
#  Stocks (Alpaca)
# ═══════════════════════════════════════════════════════════════════

def prefetch_stocks(cfg: dict[str, Any], min_bars: dict[str, int], backtest_start: pd.Timestamp) -> None:
    stocks_dir = Path(cfg["data"].get("stocks_dir", "data/stocks"))
    if not stocks_dir.exists():
        logger.warning("Stocks dir %s not found — skipping", stocks_dir)
        return

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        logger.error("alpaca-py not installed — pip install alpaca-py")
        return

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        logger.error("Alpaca credentials not set in .env")
        return

    client = StockHistoricalDataClient(key, secret)
    alpaca_tf = {
        "1d": TimeFrame(1, TimeFrameUnit.Day),
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
    }

    # Use _1d files for stock symbols (stocks don't have _1m, they use _5m)
    parquets = sorted(stocks_dir.glob("*_1d.parquet"))
    symbols = [p.stem.replace("_1d", "") for p in parquets]
    logger.info("Stocks: %d symbols to check", len(symbols))

    # Calculate how far back to go
    tf_seconds = PREFETCH_TFS["1d"]
    target_start_dt = backtest_start - pd.Timedelta(seconds=min_bars["1d"] * tf_seconds * 1.5)

    for sym in tqdm(symbols, desc="Stocks prefetch"):
        ticker = sym.replace("_", ".")  # BRK_B -> BRK.B

        for tf, min_n in min_bars.items():
            tf_path = stocks_dir / f"{sym}_{tf}.parquet"

            if _bars_before(tf_path, backtest_start) >= min_n:
                continue

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=alpaca_tf[tf],
                    start=target_start_dt.to_pydatetime(),
                    end=datetime.now(timezone.utc),
                )
                bars = client.get_stock_bars(request)

                rows = []
                for bar in bars[ticker]:
                    rows.append({
                        "timestamp": pd.Timestamp(bar.timestamp),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    })

                if not rows:
                    continue

                new_df = pd.DataFrame(rows)

                if tf_path.exists():
                    old_df = pd.read_parquet(tf_path)
                    combined = pd.concat([new_df, old_df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                else:
                    combined = new_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                bars_b = int((combined["timestamp"] < backtest_start).sum())
                old_count = len(pd.read_parquet(tf_path)) if tf_path.exists() else 0
                combined.to_parquet(tf_path, engine="pyarrow", compression="gzip", index=False)
                logger.info("[%s] %s: %d → %d bars (%d before %s)", ticker, tf, old_count, len(combined), bars_b, backtest_start.date())

            except Exception as exc:
                logger.warning("[%s] %s prefetch failed: %s", ticker, tf, exc)
            time.sleep(0.3)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def run(config_path: str = "config/default_config.yaml",
        asset_class: str | None = None,
        min_daily_bars: int = 250) -> None:
    cfg = load_config(config_path)

    # Backtest start date — we need warmup bars BEFORE this
    backtest_start = pd.Timestamp(cfg["data"]["start_date"], tz="UTC")

    min_bars = {
        "1d": min_daily_bars,
        "4h": min_daily_bars * 2,
        "1h": min_daily_bars * 4,
    }

    logger.info("Prefetching history: min %d daily bars before %s", min_daily_bars, backtest_start.date())

    if asset_class is None or asset_class == "crypto":
        prefetch_crypto(cfg, min_bars, backtest_start)
    if asset_class is None or asset_class == "forex":
        prefetch_oanda(cfg, min_bars, backtest_start, "forex")
    if asset_class is None or asset_class == "commodities":
        prefetch_oanda(cfg, min_bars, backtest_start, "commodities")
    if asset_class is None or asset_class == "stocks":
        prefetch_stocks(cfg, min_bars, backtest_start)

    logger.info("Prefetch complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefetch historical data for indicator warmup")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--asset-class", choices=["crypto", "forex", "stocks", "commodities"])
    parser.add_argument("--min-daily-bars", type=int, default=250, help="Minimum daily bars needed")
    args = parser.parse_args()
    run(config_path=args.config, asset_class=args.asset_class, min_daily_bars=args.min_daily_bars)
