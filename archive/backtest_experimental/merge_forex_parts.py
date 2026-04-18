"""
backtest/merge_forex_parts.py
================================
Merges forex_samples_part{1,2,3}.parquet into forex_samples.parquet.
Safe: errors out if any part is missing or empty.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backtest.generate_rl_data import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parts = [OUTPUT_DIR / f"forex_samples_part{i}.parquet" for i in (1, 2, 3)]
    dfs = []
    for p in parts:
        if not p.exists():
            logger.error("Missing: %s — abort", p)
            return
        df = pd.read_parquet(p)
        logger.info("  %s: %d rows, %d entries",
                    p, len(df), int((df["label_action"] > 0).sum()))
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    # Sort by window, symbol, timestamp for consistency with generate_rl_data output
    combined = combined.sort_values(["window", "symbol", "timestamp"]).reset_index(drop=True)

    out = OUTPUT_DIR / "forex_samples.parquet"
    combined.to_parquet(out, index=False)
    logger.info("Merged saved: %s (%d rows, %d entries)",
                out, len(combined), int((combined["label_action"] > 0).sum()))

    # Clean up parts
    for p in parts:
        p.unlink()
        logger.info("  Removed: %s", p)


if __name__ == "__main__":
    main()
