"""
═══════════════════════════════════════════════════════════════════
 utils/data_downloader.py
 ────────────────────────
 Downloads Binance USDT-M Futures 1-minute OHLCV data via CCXT,
 computes a rolling 30-day volume ranking to identify the
 historical top-100 coins, and stores everything as Parquet files.

 Usage:
     python -m utils.data_downloader                # uses default config
     python -m utils.data_downloader --config path  # custom config
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
MS_PER_MINUTE = 60_000
BATCH_LIMIT = 1_500          # Binance max candles per request
RATE_LIMIT_SLEEP = 0.35      # Seconds between API calls


# ═══════════════════════════════════════════════════════════════════
#  Config helpers
# ═══════════════════════════════════════════════════════════════════

def load_config(path: str = "config/default_config.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ═══════════════════════════════════════════════════════════════════
#  Exchange & symbol helpers
# ═══════════════════════════════════════════════════════════════════

def create_exchange(cfg: dict[str, Any]) -> ccxt.Exchange:
    """Instantiate a CCXT exchange object for Binance USDT-M Futures."""
    exchange_id = cfg["data"]["exchange"]            # "binanceusdm"
    exchange_cls = getattr(ccxt, exchange_id)
    exchange: ccxt.Exchange = exchange_cls(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )
    exchange.load_markets()
    return exchange


def get_usdt_perp_symbols(exchange: ccxt.Exchange) -> list[str]:
    """Return all active USDT-margined perpetual symbols."""
    symbols: list[str] = []
    for sym, mkt in exchange.markets.items():
        if (
            mkt.get("swap")
            and mkt.get("linear")
            and mkt.get("active")
            and mkt.get("settle") == "USDT"
        ):
            symbols.append(sym)
    return sorted(symbols)


# ═══════════════════════════════════════════════════════════════════
#  OHLCV download (single symbol)
# ═══════════════════════════════════════════════════════════════════

def download_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles in batches.
    Returns a DataFrame with columns:
        timestamp, open, high, low, close, volume
    """
    all_candles: list[list] = []
    fetch_since = since_ms

    while fetch_since < until_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=fetch_since,
                limit=BATCH_LIMIT,
            )
        except ccxt.NetworkError as exc:
            logger.warning("Network error for %s – retrying: %s", symbol, exc)
            time.sleep(2)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error for %s – skipping: %s", symbol, exc)
            break

        if not candles:
            break

        all_candles.extend(candles)
        fetch_since = candles[-1][0] + MS_PER_MINUTE  # Next batch starts 1 min after last
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_candles:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Trim to requested range
    until_dt = pd.Timestamp(until_ms, unit="ms", tz="UTC")
    df = df[df["timestamp"] <= until_dt]
    return df


# ═══════════════════════════════════════════════════════════════════
#  Resampling
# ═══════════════════════════════════════════════════════════════════

def resample_ohlcv(df_1m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV to a higher timeframe.
    *target_tf* uses pandas offset aliases: '5min', '15min', '1h', '4h', '1D'.
    """
    # Map config shorthand to pandas offset strings
    tf_map = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "1h": "1h", "4h": "4h", "1d": "1D",
    }
    offset = tf_map.get(target_tf, target_tf)

    df = df_1m.set_index("timestamp")
    resampled = df.resample(offset).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    resampled = resampled.reset_index()
    return resampled


# ═══════════════════════════════════════════════════════════════════
#  30-Day rolling volume ranking (historical)
# ═══════════════════════════════════════════════════════════════════

def compute_volume_rankings(
    data_dir: Path,
    symbols: list[str],
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    For each calendar month, compute the 30-day rolling USDT volume
    for every symbol and rank them. Returns a DataFrame:
        month | symbol | volume_30d | rank
    This allows the backtest to filter historically correct top-100 coins.
    """
    records: list[dict] = []

    for symbol in tqdm(symbols, desc="Computing volume rankings"):
        safe = symbol.replace("/", "_").replace(":", "_")
        parquet_path = data_dir / f"{safe}_1m.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path, columns=["timestamp", "close", "volume"])
        if df.empty:
            continue

        df = df.set_index("timestamp")
        # Resample to daily volume in USDT terms  (close * volume ≈ USDT volume)
        daily = (df["close"] * df["volume"]).resample("1D").sum()
        # Rolling 30-day sum
        rolling = daily.rolling(f"{lookback_days}D", min_periods=lookback_days // 2).sum()
        # Sample at month end
        monthly = rolling.resample("ME").last().dropna()

        for ts, vol in monthly.items():
            records.append(
                {"month": ts.to_period("M"), "symbol": symbol, "volume_30d": vol}
            )

    if not records:
        return pd.DataFrame(columns=["month", "symbol", "volume_30d", "rank"])

    rank_df = pd.DataFrame(records)
    rank_df["rank"] = (
        rank_df.groupby("month")["volume_30d"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    rank_df = rank_df.sort_values(["month", "rank"]).reset_index(drop=True)
    return rank_df


# ═══════════════════════════════════════════════════════════════════
#  Parquet I/O
# ═══════════════════════════════════════════════════════════════════

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet with gzip compression."""
    df.to_parquet(path, engine="pyarrow", compression="gzip", index=False)
    logger.info("Saved %s rows → %s", len(df), path)


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run(config_path: str = "config/default_config.yaml") -> None:
    """
    Full download pipeline:
    1. Fetch list of USDT-perp symbols from Binance Futures.
    2. Download 1 m OHLCV for each symbol since 2019-11-01.
    3. Resample to 5 m, 15 m, 1 h, 4 h, 1 d and save.
    4. Compute historical 30-day volume rankings.
    """
    cfg = load_config(config_path)
    data_dir = Path(cfg["data"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    start_date = cfg["data"]["start_date"]
    resample_tfs = cfg["data"]["resample_tfs"]

    since_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # ── 1. Connect & list symbols ─────────────────────────────────
    logger.info("Connecting to exchange …")
    exchange = create_exchange(cfg)
    symbols = get_usdt_perp_symbols(exchange)
    logger.info("Found %d USDT-perp symbols", len(symbols))

    # ── 2. Download 1 m candles per symbol ────────────────────────
    for symbol in tqdm(symbols, desc="Downloading 1m data"):
        safe = symbol.replace("/", "_").replace(":", "_")
        out_path = data_dir / f"{safe}_1m.parquet"

        # Resume: skip if file already exists and is recent
        if out_path.exists():
            existing = pd.read_parquet(out_path, columns=["timestamp"])
            if not existing.empty:
                last_ts = existing["timestamp"].max()
                resume_ms = int(last_ts.timestamp() * 1000) + MS_PER_MINUTE
                if resume_ms >= until_ms - MS_PER_MINUTE * 60:
                    logger.info("Skipping %s (up-to-date)", symbol)
                    continue
                logger.info("Resuming %s from %s", symbol, last_ts)
                new_df = download_ohlcv(exchange, symbol, "1m", resume_ms, until_ms)
                if not new_df.empty:
                    old_df = pd.read_parquet(out_path)
                    df_1m = pd.concat([old_df, new_df]).drop_duplicates(
                        subset=["timestamp"]
                    ).sort_values("timestamp").reset_index(drop=True)
                    save_parquet(df_1m, out_path)
                continue

        df_1m = download_ohlcv(exchange, symbol, "1m", since_ms, until_ms)
        if df_1m.empty:
            logger.warning("No data for %s – skipping", symbol)
            continue

        save_parquet(df_1m, out_path)

        # ── 3. Resample & save higher timeframes ─────────────────
        for tf in resample_tfs:
            tf_df = resample_ohlcv(df_1m, tf)
            tf_path = data_dir / f"{safe}_{tf}.parquet"
            save_parquet(tf_df, tf_path)

    # ── 4. Volume ranking ─────────────────────────────────────────
    logger.info("Computing historical volume rankings …")
    ranking_df = compute_volume_rankings(
        data_dir,
        symbols,
        lookback_days=cfg["volume_filter"]["lookback_days"],
    )
    ranking_path = data_dir / "volume_rankings.parquet"
    save_parquet(ranking_df, ranking_path)

    logger.info("✅  Data download complete. Files in %s", data_dir)


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Futures OHLCV data")
    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run(config_path=args.config)
