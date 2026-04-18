"""
backtest/regen_forex_subset.py
================================
Process a subset of forex symbols in parallel, save to a custom
intermediate parquet. Used to speed up forex regen by running 3 parallel
processes (each handling ~9-10 symbols).

After all 3 finish, merge them via merge_forex_parts.py.

Run:
    python3 -m backtest.regen_forex_subset --part 1 --symbols AUD_CAD AUD_CHF ...
    python3 -m backtest.regen_forex_subset --part 2 --symbols EUR_JPY ...
    python3 -m backtest.regen_forex_subset --part 3 --symbols USD_JPY ...
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from backtest.generate_rl_data import (
    WINDOWS, process_instrument, OUTPUT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, help="Part number (1, 2, 3)")
    parser.add_argument("--symbols", nargs="+", required=True, help="Forex symbols to process")
    args = parser.parse_args()

    config_path = Path("config/default_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("═══ FOREX REGEN PART %d — %d symbols ═══",
                args.part, len(args.symbols))
    logger.info("  symbols: %s", args.symbols)
    logger.info("  windows: %d", len(WINDOWS))

    all_data: list[pd.DataFrame] = []
    work_items = [
        (sym, ac, ws, we, wi)
        for wi, (ws, we) in enumerate(WINDOWS)
        for sym in args.symbols
        for ac in ["forex"]
    ]

    total = len(work_items)
    start = time.time()
    for i, (sym, ac, ws, we, wi) in enumerate(work_items, start=1):
        try:
            result = process_instrument(sym, ac, config, ws, we)
            if result is not None and len(result) > 0:
                result["window"] = wi
                all_data.append(result)
        except Exception as exc:
            logger.error("  %s W%d FAILED: %s", sym, wi, exc)

        if i % 10 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / max(elapsed, 1)
            eta = (total - i) / max(rate, 0.01) / 60
            logger.info("  [part %d] Progress: %d/%d (%.0f%%) rate=%.2f/s eta=%.0fm",
                        args.part, i, total, 100 * i / total, rate, eta)
        gc.collect()

    if not all_data:
        logger.error("No data produced — abort")
        return

    combined = pd.concat(all_data, ignore_index=True).reset_index(drop=True)
    out_path = OUTPUT_DIR / f"forex_samples_part{args.part}.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Saved: %s (%d rows, %d entries)",
                out_path, len(combined),
                int((combined["label_action"] > 0).sum()))


if __name__ == "__main__":
    main()
