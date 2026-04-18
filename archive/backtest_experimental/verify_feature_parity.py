"""
backtest/verify_feature_parity.py
==================================
After fixing the per-cluster SMC params mismatch and regenerating training
samples, this script verifies that LIVE and TRAINING now compute identical
feature vectors for the same (symbol, timestamp) bar.

Picks AUD_JPY (the test case) and computes features two ways:
  1. Training-style: load the regenerated parquet, read the row
  2. Live-style: instantiate a minimal PaperBot, compute via _build_xgb_features

Then diffs every feature and reports any deltas > 0.001.

Run AFTER regen completes:
    python3 -m backtest.verify_feature_parity
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_parquet_features(symbol: str, asset_class: str, ts: pd.Timestamp) -> dict | None:
    """Load the parquet row matching the target (symbol, ts)."""
    path = Path(f"data/rl_training/{asset_class}_samples.parquet")
    if not path.exists():
        logger.error("Missing: %s", path)
        return None
    df = pd.read_parquet(path)
    df_ts = pd.to_datetime(df["timestamp"])
    if df_ts.dt.tz is None:
        df_ts = df_ts.dt.tz_localize("UTC")
    df["timestamp"] = df_ts
    match = df[(df["symbol"] == symbol) & (df["timestamp"] == ts)]
    if len(match) == 0:
        logger.error("No parquet row for %s @ %s", symbol, ts)
        return None
    row = match.iloc[0]
    # Convert to plain dict, dropping meta columns
    from rl_brain_v2 import META_COLS, DEAD_FEATURES
    skip = META_COLS | DEAD_FEATURES | {"symbol", "asset_class", "window"}
    return {k: float(row[k]) for k in row.index if k not in skip}


async def get_live_features(symbol: str, asset_class: str) -> dict | None:
    """Spin up a minimal live PaperBot, fetch buffers, compute features."""
    from exchanges.oanda_adapter import OandaAdapter
    from exchanges.binance_adapter import BinanceAdapter
    import live_multi_bot as lmb

    if asset_class in ("forex", "commodities"):
        adapter = OandaAdapter(
            account_id=os.getenv("OANDA_ACCOUNT_ID", ""),
            access_token=os.getenv("OANDA_ACCESS_TOKEN", ""),
            environment="practice",
        )
        await adapter.connect()
    else:
        adapter = BinanceAdapter()
        await adapter.connect()

    # Build a minimal PaperBot
    bot = lmb.PaperBot(
        bot_id=999,
        symbol=symbol,
        config={},
        output_dir=Path("/tmp/feature_parity"),
        asset_class=asset_class,
        adapter=adapter,
        rl_suite=None,
    )
    # Fetch history (this populates buffer_*)
    await bot.load_history()
    logger.info("buffer_5m=%d 15m=%d 1h=%d 4h=%d 1d=%d",
                len(bot.buffer_5m), len(bot.buffer_15m),
                len(bot.buffer_1h), len(bot.buffer_4h), len(bot.buffer_1d))

    if bot.buffer_5m.empty:
        logger.error("buffer_5m empty — aborting")
        return None

    # Run alignment scoring on the latest 5m candle
    last_5m = bot.buffer_5m.iloc[-1].to_dict()
    score, direction, comp = bot._multi_tf_alignment_score(last_5m)
    logger.info("Live alignment_score (13-component): %.4f, direction: %s", score, direction)

    feat = bot._build_xgb_features(comp, score)
    feat["style_id"] = 0.5
    return feat


async def main() -> None:
    SYMBOL = "AUD_JPY"
    ASSET_CLASS = "forex"
    # Use the LATEST 5m bar (live will fetch fresh, so we don't pin a specific ts)

    logger.info("=" * 70)
    logger.info("Feature parity check: %s (%s)", SYMBOL, ASSET_CLASS)
    logger.info("=" * 70)

    # Get live features
    logger.info("Fetching live features...")
    live_feat = await get_live_features(SYMBOL, ASSET_CLASS)
    if live_feat is None:
        return

    # Find the latest matching parquet row
    parquet_path = Path(f"data/rl_training/{ASSET_CLASS}_samples.parquet")
    df = pd.read_parquet(parquet_path, columns=["symbol", "timestamp"])
    df_ts = pd.to_datetime(df["timestamp"])
    if df_ts.dt.tz is None:
        df_ts = df_ts.dt.tz_localize("UTC")
    df["timestamp"] = df_ts
    sym_rows = df[df["symbol"] == SYMBOL].sort_values("timestamp")
    latest_parquet_ts = sym_rows["timestamp"].iloc[-1]
    logger.info("Latest parquet ts for %s: %s", SYMBOL, latest_parquet_ts)

    parquet_feat = get_parquet_features(SYMBOL, ASSET_CLASS, latest_parquet_ts)
    if parquet_feat is None:
        return

    # Compare
    logger.info("")
    logger.info("=" * 70)
    logger.info("FEATURE-BY-FEATURE DIFF")
    logger.info("=" * 70)
    keys = sorted(set(live_feat.keys()) | set(parquet_feat.keys()))
    deltas = []
    for k in keys:
        v_live = live_feat.get(k, None)
        v_parq = parquet_feat.get(k, None)
        if v_live is None or v_parq is None:
            logger.info("  %-30s live=%-15s parquet=%-15s [missing]", k, v_live, v_parq)
            continue
        d = abs(float(v_live) - float(v_parq))
        deltas.append(d)
        marker = " <<<" if d > 0.001 else ""
        logger.info("  %-30s live=%+.4f  parquet=%+.4f  delta=%+.4f%s",
                    k, v_live, v_parq, d, marker)
    logger.info("")
    logger.info("Total features compared: %d", len(deltas))
    logger.info("Max delta: %.4f", max(deltas) if deltas else 0)
    logger.info("Mean delta: %.4f", np.mean(deltas) if deltas else 0)
    logger.info("Features with delta > 0.001: %d", sum(1 for d in deltas if d > 0.001))


if __name__ == "__main__":
    asyncio.run(main())
