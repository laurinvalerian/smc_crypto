"""
backtest/compare_parquets.py
=============================
Compares the current rl_training parquets against the before_optfix backups.
Useful to verify that the regen with optimized SMC params actually changed
the feature distributions as expected (struct_1d bias flip, alignment_score
shift, etc.).

Run:
    python3 -m backtest.compare_parquets
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/rl_training")
BACKUP_DIR = DATA_DIR / "before_optfix"

CLASSES = ["crypto", "forex", "commodities", "stocks"]
KEY_FEATURES = [
    "struct_1d", "alignment_score", "adx_1h",
    "h4_confirms", "h1_confirms", "h1_choch",
    "fvg_bull_active", "ob_bull_active",
]


def compare_class(asset_class: str) -> None:
    new_path = DATA_DIR / f"{asset_class}_samples.parquet"
    old_path = BACKUP_DIR / f"{asset_class}_samples.parquet"

    if not new_path.exists():
        logger.warning("New %s parquet missing", asset_class)
        return
    if not old_path.exists():
        logger.warning("Old %s parquet missing", asset_class)
        return

    logger.info("=" * 70)
    logger.info("%s", asset_class.upper())
    logger.info("=" * 70)

    new_df = pd.read_parquet(new_path)
    old_df = pd.read_parquet(old_path)

    # Count summary
    n_new = len(new_df)
    n_old = len(old_df)
    ent_new = int((new_df["label_action"] > 0).sum())
    ent_old = int((old_df["label_action"] > 0).sum())
    logger.info("  Rows:    old=%d  new=%d  (%+.0f%%)",
                n_old, n_new, 100 * (n_new - n_old) / max(n_old, 1))
    logger.info("  Entries: old=%d  new=%d  (%+.0f%%)",
                ent_old, ent_new, 100 * (ent_new - ent_old) / max(ent_old, 1))

    # Entry-only subset
    new_ent = new_df[new_df["label_action"] > 0].copy()
    old_ent = old_df[old_df["label_action"] > 0].copy()

    logger.info("")
    logger.info("  Feature distribution (entry bars only):")
    logger.info("  %-20s %-20s %-20s %-10s", "feature", "old mean/std", "new mean/std", "Δ mean")
    for f in KEY_FEATURES:
        if f not in new_ent.columns or f not in old_ent.columns:
            continue
        o_mean = float(old_ent[f].mean())
        o_std = float(old_ent[f].std())
        n_mean = float(new_ent[f].mean())
        n_std = float(new_ent[f].std())
        d = n_mean - o_mean
        logger.info("  %-20s %+7.4f/%-7.4f %+7.4f/%-7.4f %+7.4f",
                    f, o_mean, o_std, n_mean, n_std, d)


def main() -> None:
    logger.info("Comparing parquets: %s vs %s", BACKUP_DIR, DATA_DIR)
    logger.info("")
    for ac in CLASSES:
        compare_class(ac)
        logger.info("")


if __name__ == "__main__":
    main()
