"""
backtest/append_w12.py
=======================
Appends W12 (2026-04-01 → 2026-04-15) entry samples to the existing
training parquet files.  Bypasses the broken multiprocessing Pool
by processing sequentially with small parallel batches of Python processes.

Why this script exists:
  generate_rl_data.py's Pool crashes on Python 3.13 + this code with
  BrokenPipeError on put().  Rather than debug multiprocessing IPC,
  we process W12 sequentially here: it's only ~2 weeks of data so it's
  quick even without Pool.

Usage:
    python3 -m backtest.append_w12 --classes crypto
    python3 -m backtest.append_w12 --classes forex commodities
    python3 -m backtest.append_w12  # all 3 non-stocks classes
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.generate_rl_data import (
    process_instrument,
    get_symbols_for_class,
    OUTPUT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# The new window we want to append
W12_START = "2026-04-01"
W12_END = "2026-04-15"
W12_INDEX = 12


def append_w12_for_class(
    asset_class: str,
    config: dict[str, Any],
    symbols: list[str] | None = None,
    max_crypto: int = 30,
) -> int:
    """Process all symbols for W12 and append to existing parquet. Returns n new rows."""
    if symbols is None:
        symbols = get_symbols_for_class(asset_class, max_crypto=max_crypto)

    logger.info("═══ %s: W12 append  (%d symbols) ═══", asset_class.upper(), len(symbols))

    existing_path = OUTPUT_DIR / f"{asset_class}_samples.parquet"
    if not existing_path.exists():
        logger.error("Existing parquet missing: %s — cannot append", existing_path)
        return 0

    # Load existing
    existing = pd.read_parquet(existing_path)
    pre_rows = len(existing)
    pre_windows = sorted(existing["window"].unique().tolist())
    logger.info("  Existing: %d rows, windows %s", pre_rows, pre_windows)

    # Check if W12 already exists — if so, drop and recompute
    if W12_INDEX in pre_windows:
        n_old_w12 = int((existing["window"] == W12_INDEX).sum())
        logger.info("  W12 already present with %d rows — removing", n_old_w12)
        existing = existing[existing["window"] != W12_INDEX].copy()

    # Process each symbol sequentially for W12
    new_results: list[pd.DataFrame] = []
    for i, sym in enumerate(symbols, start=1):
        try:
            result = process_instrument(sym, asset_class, config, W12_START, W12_END)
        except Exception as e:
            logger.warning("  [%d/%d] %s FAILED: %s", i, len(symbols), sym, e)
            continue
        if result is None or len(result) == 0:
            logger.info("  [%d/%d] %s: no data / no entries", i, len(symbols), sym)
            continue
        result["window"] = W12_INDEX
        n_ent = int((result["label_action"] > 0).sum())
        logger.info("  [%d/%d] %s: %d rows (%d entries)", i, len(symbols), sym, len(result), n_ent)
        new_results.append(result)

    if not new_results:
        logger.warning("  No W12 data produced for %s", asset_class)
        return 0

    combined_new = pd.concat(new_results, ignore_index=True)
    n_new = len(combined_new)
    n_new_entries = int((combined_new["label_action"] > 0).sum())
    logger.info("  W12 total: %d rows (%d entries)", n_new, n_new_entries)

    # Backup existing parquet before overwriting
    backup_dir = OUTPUT_DIR / "before_w12"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{asset_class}_samples.parquet"
    if not backup_path.exists():
        shutil.copy2(existing_path, backup_path)
        logger.info("  Backup: %s", backup_path)

    # Align columns (in case new data has extras the existing doesn't)
    missing_in_existing = set(combined_new.columns) - set(existing.columns)
    missing_in_new = set(existing.columns) - set(combined_new.columns)
    if missing_in_existing:
        logger.warning("  New columns not in existing (filling with NaN): %s",
                       sorted(missing_in_existing))
        for c in missing_in_existing:
            existing[c] = pd.NA
    if missing_in_new:
        logger.warning("  Missing columns in new (filling with NaN): %s",
                       sorted(missing_in_new))
        for c in missing_in_new:
            combined_new[c] = pd.NA

    # Put columns in the same order
    combined_new = combined_new[existing.columns]

    merged = pd.concat([existing, combined_new], ignore_index=True)
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    merged.to_parquet(existing_path, index=False)
    logger.info("  Merged saved: %s (%d -> %d rows, +%d)",
                existing_path, pre_rows, len(merged), n_new)
    return n_new


def main() -> None:
    parser = argparse.ArgumentParser(description="Append W12 to existing RL training parquets")
    parser.add_argument("--classes", nargs="+",
                        choices=["crypto", "forex", "stocks", "commodities"],
                        default=["crypto", "forex", "commodities"],
                        help="Which classes to process (stocks excluded by default — Alpaca 401)")
    parser.add_argument("--max-crypto", type=int, default=30)
    args = parser.parse_args()

    config_path = Path("config/default_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    total_new = 0
    for ac in args.classes:
        n = append_w12_for_class(ac, config, max_crypto=args.max_crypto)
        total_new += n

    logger.info("")
    logger.info("═══ W12 APPEND COMPLETE: +%d rows across %d classes ═══",
                total_new, len(args.classes))


if __name__ == "__main__":
    main()
