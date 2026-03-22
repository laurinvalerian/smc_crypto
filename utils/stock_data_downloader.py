"""
utils/stock_data_downloader.py
==============================
Downloads historical OHLCV data for US Stocks via the Alpaca Data API.
Stores as Parquet files.

Note: Alpaca free tier provides IEX data. For 1m data, history is limited.
We use 5m as the base timeframe for stocks (sufficient for Walk-Forward).

Usage:
    python3 -m utils.stock_data_downloader
    python3 -m utils.stock_data_downloader --config config/default_config.yaml
    python3 -m utils.stock_data_downloader --symbols AAPL MSFT GOOGL
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
RATE_LIMIT_SLEEP = 0.3  # Alpaca rate limit: ~200 requests/min

# Top 50 US Stocks by Market Cap
US_STOCK_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK.B", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA",
    "JNJ", "PG", "COST", "HD", "ABBV",
    "MRK", "CRM", "AMD", "NFLX", "BAC",
    "CVX", "KO", "PEP", "LIN", "TMO",
    "ADBE", "WMT", "MCD", "CSCO", "ACN",
    "ABT", "DHR", "TXN", "NEE", "PM",
    "ORCL", "INTC", "CMCSA", "DIS", "VZ",
    "IBM", "QCOM", "AMGN", "GE", "INTU",
]

# Alpaca TimeFrame mapping
ALPACA_TF_MAP: dict[str, str] = {
    "5m": "5Min",
    "15m": "15Min",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
}

# Max bars per request varies by timeframe
# For minute-level bars, Alpaca returns max 10,000 per request
MAX_BARS_PER_REQUEST = 10_000

# Chunk size for date ranges (days per chunk for 5m data)
CHUNK_DAYS_5M = 30  # ~30 days * ~78 bars/day = ~2340 bars (well under 10K limit)


# ═══════════════════════════════════════════════════════════════════
#  Alpaca Data Client
# ═══════════════════════════════════════════════════════════════════

def create_alpaca_data_client(api_key: str, secret_key: str) -> Any:
    """Create an Alpaca StockHistoricalDataClient."""
    from alpaca.data.historical import StockHistoricalDataClient
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    logger.info("Alpaca Data client connected")
    return client


# ═══════════════════════════════════════════════════════════════════
#  OHLCV Download
# ═══════════════════════════════════════════════════════════════════

def download_ohlcv_alpaca(
    client: Any,
    symbol: str,
    timeframe_str: str,
    from_dt: datetime,
    to_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca in date-range chunks.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    # Parse timeframe
    tf_map = {
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    alpaca_tf = ALPACA_TF_MAP.get(timeframe_str, "5Min")
    tf = tf_map.get(alpaca_tf)
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe_str}")

    all_rows: list[dict] = []
    current_from = from_dt

    # Chunk by date to avoid hitting limits
    chunk_delta = timedelta(days=CHUNK_DAYS_5M)

    while current_from < to_dt:
        chunk_to = min(current_from + chunk_delta, to_dt)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=current_from,
                end=chunk_to,
            )
            bars = client.get_stock_bars(request)
        except Exception as exc:
            logger.warning("Error fetching %s: %s — retrying", symbol, exc)
            time.sleep(2)
            continue

        # bars[symbol] is a list of Bar objects
        symbol_bars = bars.get(symbol, []) if hasattr(bars, 'get') else []
        if not symbol_bars:
            # Try attribute-based access (alpaca-py returns BarSet)
            try:
                symbol_bars = bars[symbol]
            except (KeyError, TypeError):
                symbol_bars = []

        for bar in symbol_bars:
            all_rows.append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            })

        current_from = chunk_to
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════
#  Resampling
# ═══════════════════════════════════════════════════════════════════

def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe."""
    tf_offset = {
        "5m": "5min", "15m": "15min",
        "1h": "1h", "4h": "4h", "1d": "1D",
    }
    offset = tf_offset.get(target_tf, target_tf)
    tmp = df.set_index("timestamp")
    resampled = tmp.resample(offset).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    return resampled.reset_index()


# ═══════════════════════════════════════════════════════════════════
#  Parquet I/O
# ═══════════════════════════════════════════════════════════════════

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet with gzip compression."""
    df.to_parquet(path, engine="pyarrow", compression="gzip", index=False)
    logger.info("Saved %s rows -> %s", len(df), path)


# ═══════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def run(
    config_path: str = "config/default_config.yaml",
    symbols: list[str] | None = None,
) -> None:
    """
    Download US Stock OHLCV data from Alpaca.

    1. Connect to Alpaca Data API
    2. Download 5m candles for each stock (base timeframe)
    3. Resample to 15m, 1h, 4h, 1d and save as Parquet
    """
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_API_SECRET", "")
    if not api_key or not secret_key:
        raise SystemExit(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env\n"
            "Generate API keys in your Alpaca Paper Trading dashboard."
        )

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    start_date = cfg["data"]["start_date"]
    stocks_dir = Path(cfg["data"].get("stocks_dir", "data/stocks"))
    stocks_dir.mkdir(parents=True, exist_ok=True)

    from_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.now(timezone.utc)

    # Stocks: higher timeframes only (5m is base, not 1m)
    stock_resample_tfs = ["15m", "1h", "4h", "1d"]

    # Symbols to download
    target_symbols = symbols if symbols else US_STOCK_UNIVERSE

    # Connect
    client = create_alpaca_data_client(api_key, secret_key)

    total = len(target_symbols)
    for idx, symbol in enumerate(target_symbols, 1):
        logger.info("[%d/%d] Downloading %s ...", idx, total, symbol)

        # Use safe filename (BRK.B -> BRK_B)
        safe_name = symbol.replace(".", "_")
        out_path = stocks_dir / f"{safe_name}_5m.parquet"

        # Resume support
        actual_from = from_dt
        if out_path.exists():
            existing = pd.read_parquet(out_path, columns=["timestamp"])
            if not existing.empty:
                last_ts = existing["timestamp"].max()
                resume_from = last_ts.to_pydatetime() + timedelta(minutes=5)
                if resume_from >= to_dt - timedelta(hours=1):
                    logger.info("Skipping %s (up-to-date)", symbol)
                    continue
                logger.info("Resuming %s from %s", symbol, last_ts)
                actual_from = resume_from

        df_5m = download_ohlcv_alpaca(client, symbol, "5m", actual_from, to_dt)

        if df_5m.empty:
            logger.warning("No data for %s — skipping", symbol)
            continue

        # Merge with existing if resuming
        if out_path.exists() and actual_from != from_dt:
            old_df = pd.read_parquet(out_path)
            df_5m = pd.concat([old_df, df_5m]).drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp").reset_index(drop=True)

        save_parquet(df_5m, out_path)

        # Resample to higher timeframes
        for tf in stock_resample_tfs:
            tf_df = resample_ohlcv(df_5m, tf)
            tf_path = stocks_dir / f"{safe_name}_{tf}.parquet"
            save_parquet(tf_df, tf_path)

    logger.info("Done! Stock data in %s", stocks_dir)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download US Stock OHLCV from Alpaca")
    parser.add_argument("--config", default="config/default_config.yaml", help="YAML config path")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols (e.g. AAPL MSFT)")
    args = parser.parse_args()
    run(config_path=args.config, symbols=args.symbols)
