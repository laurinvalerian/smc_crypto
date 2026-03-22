"""
utils/forex_data_downloader.py
==============================
Downloads historical OHLCV data for Forex pairs and Commodities
via the OANDA v20 REST API. Stores as Parquet files.

Usage:
    python3 -m utils.forex_data_downloader
    python3 -m utils.forex_data_downloader --config config/default_config.yaml
    python3 -m utils.forex_data_downloader --instruments EUR_USD GBP_USD
    python3 -m utils.forex_data_downloader --commodities-only
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
MAX_CANDLES_PER_REQUEST = 5000   # OANDA max count
RATE_LIMIT_SLEEP = 0.5           # Seconds between API calls

FOREX_PAIRS: list[str] = [
    # Majors (7)
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CHF", "USD_CAD", "NZD_USD",
    # Crosses (21)
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
]

COMMODITY_INSTRUMENTS: list[str] = [
    "XAU_USD",    # Gold
    "XAG_USD",    # Silver
    "WTICO_USD",  # WTI Crude Oil
    "BCO_USD",    # Brent Crude Oil
]

# OANDA granularity mapping
TF_MAP: dict[str, str] = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "4h": "H4", "1d": "D",
}

# Approximate candle durations in seconds (for pagination)
TF_SECONDS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


# ═══════════════════════════════════════════════════════════════════
#  OANDA API Client
# ═══════════════════════════════════════════════════════════════════

def create_oanda_client(
    access_token: str,
    environment: str = "practice",
) -> Any:
    """Create an OANDA v20 API context."""
    import v20

    hostname = (
        "api-fxpractice.oanda.com" if environment == "practice"
        else "api-fxtrade.oanda.com"
    )
    ctx = v20.Context(hostname=hostname, token=access_token)
    logger.info("OANDA v20 client connected (%s)", hostname)
    return ctx


# ═══════════════════════════════════════════════════════════════════
#  OHLCV Download
# ═══════════════════════════════════════════════════════════════════

def download_ohlcv_oanda(
    ctx: Any,
    instrument: str,
    granularity: str,
    from_dt: datetime,
    to_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from OANDA in batches.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    Note: OANDA volume is tick count (not monetary volume), but still useful
    for relative volume analysis.
    """
    all_rows: list[dict] = []
    current_from = from_dt
    tf_key = next((k for k, v in TF_MAP.items() if v == granularity), "1h")
    step_seconds = TF_SECONDS.get(tf_key, 3600) * MAX_CANDLES_PER_REQUEST

    while current_from < to_dt:
        from_rfc = current_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        to_rfc = min(
            current_from + timedelta(seconds=step_seconds), to_dt
        ).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        try:
            response = ctx.instrument.candles(
                instrument,
                granularity=granularity,
                fromTime=from_rfc,
                toTime=to_rfc,
                price="M",  # Midpoint
                count=MAX_CANDLES_PER_REQUEST,
            )
        except Exception as exc:
            logger.warning("Error fetching %s %s: %s — retrying", instrument, granularity, exc)
            time.sleep(2)
            continue

        candles = response.get("candles", [])
        if not candles:
            break

        for c in candles:
            if not c.complete:
                continue
            mid = c.mid
            all_rows.append({
                "timestamp": pd.Timestamp(c.time),
                "open": float(mid.o),
                "high": float(mid.h),
                "low": float(mid.l),
                "close": float(mid.c),
                "volume": int(c.volume),
            })

        # Move cursor past last candle
        last_time = pd.Timestamp(candles[-1].time)
        current_from = last_time.to_pydatetime() + timedelta(seconds=TF_SECONDS.get(tf_key, 3600))
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════
#  Resampling (same as crypto downloader)
# ═══════════════════════════════════════════════════════════════════

def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe."""
    tf_offset = {
        "1m": "1min", "5m": "5min", "15m": "15min",
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
    instruments: list[str] | None = None,
    commodities_only: bool = False,
) -> None:
    """
    Download Forex + Commodities OHLCV data from OANDA.

    1. Connect to OANDA Practice API
    2. Download 1m candles for each instrument
    3. Resample to higher timeframes and save as Parquet
    """
    load_dotenv()

    access_token = os.getenv("OANDA_ACCESS_TOKEN", "")
    if not access_token:
        raise SystemExit(
            "OANDA_ACCESS_TOKEN must be set in .env\n"
            "Generate an API token in your OANDA Practice dashboard."
        )

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    start_date = cfg["data"]["start_date"]
    resample_tfs = cfg["data"]["resample_tfs"]
    forex_dir = Path(cfg["data"].get("forex_dir", "data/forex"))
    commodities_dir = Path(cfg["data"].get("commodities_dir", "data/commodities"))

    from_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.now(timezone.utc)

    # Determine instruments to download
    if instruments:
        forex_instruments = [i for i in instruments if i not in COMMODITY_INSTRUMENTS]
        commodity_instruments = [i for i in instruments if i in COMMODITY_INSTRUMENTS]
    elif commodities_only:
        forex_instruments = []
        commodity_instruments = COMMODITY_INSTRUMENTS
    else:
        forex_instruments = FOREX_PAIRS
        commodity_instruments = COMMODITY_INSTRUMENTS

    # Create directories
    if forex_instruments:
        forex_dir.mkdir(parents=True, exist_ok=True)
    if commodity_instruments:
        commodities_dir.mkdir(parents=True, exist_ok=True)

    # Connect
    environment = "practice"
    oanda_cfg = cfg.get("exchanges", {}).get("oanda", {})
    if oanda_cfg.get("practice") is False:
        environment = "live"

    ctx = create_oanda_client(access_token, environment)

    # Download all instruments
    all_instruments = [
        (instr, forex_dir, "forex") for instr in forex_instruments
    ] + [
        (instr, commodities_dir, "commodities") for instr in commodity_instruments
    ]

    total = len(all_instruments)
    for idx, (instrument, out_dir, asset_class) in enumerate(all_instruments, 1):
        logger.info("[%d/%d] Downloading %s (%s) ...", idx, total, instrument, asset_class)

        # Download 1m base data
        out_path = out_dir / f"{instrument}_1m.parquet"

        # Resume support
        actual_from = from_dt
        if out_path.exists():
            existing = pd.read_parquet(out_path, columns=["timestamp"])
            if not existing.empty:
                last_ts = existing["timestamp"].max()
                resume_from = last_ts.to_pydatetime() + timedelta(minutes=1)
                if resume_from >= to_dt - timedelta(hours=1):
                    logger.info("Skipping %s (up-to-date)", instrument)
                    continue
                logger.info("Resuming %s from %s", instrument, last_ts)
                actual_from = resume_from

        df_1m = download_ohlcv_oanda(ctx, instrument, "M1", actual_from, to_dt)

        if df_1m.empty:
            logger.warning("No data for %s — skipping", instrument)
            continue

        # Merge with existing if resuming
        if out_path.exists() and actual_from != from_dt:
            old_df = pd.read_parquet(out_path)
            df_1m = pd.concat([old_df, df_1m]).drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp").reset_index(drop=True)

        save_parquet(df_1m, out_path)

        # Resample to higher timeframes
        for tf in resample_tfs:
            tf_df = resample_ohlcv(df_1m, tf)
            tf_path = out_dir / f"{instrument}_{tf}.parquet"
            save_parquet(tf_df, tf_path)

    logger.info("Done! Forex data in %s, Commodities data in %s", forex_dir, commodities_dir)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Forex/Commodities OHLCV from OANDA")
    parser.add_argument("--config", default="config/default_config.yaml", help="YAML config path")
    parser.add_argument("--instruments", nargs="+", help="Specific instruments (e.g. EUR_USD XAU_USD)")
    parser.add_argument("--commodities-only", action="store_true", help="Download only commodities")
    args = parser.parse_args()
    run(config_path=args.config, instruments=args.instruments, commodities_only=args.commodities_only)
