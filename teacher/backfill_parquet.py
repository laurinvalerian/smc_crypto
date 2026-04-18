"""
Backfill the 4 Teacher v2 labels into existing parquet training files.

Why in-place backfill instead of regeneration:
  Teacher v2 labels are pure functions of `label_max_favorable_rr` and
  `label_mae_rr`, both already present in every training parquet. A
  full regeneration of crypto_samples.parquet would take hours; a
  pandas/numpy vectorised pass is ~30 s per class.

Stocks_samples.parquet is corrupt and must be regenerated separately
via `backtest.generate_rl_data --class stocks` — this script skips it.

Usage
-----
    python3 -m teacher.backfill_parquet \
        --data-dir data/rl_training \
        [--classes crypto forex commodities] \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from teacher.teacher_v2 import compute_teacher_labels

logger = logging.getLogger("teacher.backfill")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


_DEFAULT_CLASSES = ["crypto", "forex", "stocks", "commodities"]
_NEW_COLUMNS = ["optimal_entry", "optimal_sl_rr", "optimal_tp_rr", "optimal_size"]
_REQUIRED_INPUTS = ["label_max_favorable_rr", "label_mae_rr"]


def backfill_one(path: Path, dry_run: bool = False) -> bool:
    """Backfill a single parquet in place.

    Returns True on success, False when skipped (missing inputs, corrupt,
    or already backfilled).
    """
    if not path.exists():
        logger.warning("Skip: %s does not exist", path)
        return False

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.error("Skip: failed to read %s: %s", path, exc)
        return False

    missing = [c for c in _REQUIRED_INPUTS if c not in df.columns]
    if missing:
        logger.error("Skip: %s missing required columns %s", path, missing)
        return False

    already_present = [c for c in _NEW_COLUMNS if c in df.columns]
    if already_present:
        logger.info("%s already has %s — will overwrite", path, already_present)

    outcome_arr = df["label_outcome"].values if "label_outcome" in df.columns else None

    t0 = time.perf_counter()
    labels = compute_teacher_labels(
        df["label_max_favorable_rr"].values,
        df["label_mae_rr"].values,
        outcome=outcome_arr,
    )
    for col, values in labels.items():
        df[col] = values
    elapsed = time.perf_counter() - t0

    pos_entries = int((df["optimal_entry"] == 1).sum())
    n_rows = len(df)
    logger.info(
        "%s: %d rows, optimal_entry positive rate %.2f%%, "
        "sl median %.2fR, tp median %.2fR (labelled in %.1fs)",
        path.name, n_rows,
        100 * pos_entries / max(n_rows, 1),
        float(df["optimal_sl_rr"].median()),
        float(df["optimal_tp_rr"].median()),
        elapsed,
    )

    if dry_run:
        logger.info("DRY RUN — not writing %s", path)
        return True

    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    logger.info("Wrote %s", path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rl_training"))
    parser.add_argument("--classes", nargs="+", default=_DEFAULT_CLASSES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_ok = 0
    n_skip = 0
    for cls in args.classes:
        path = args.data_dir / f"{cls}_samples.parquet"
        if backfill_one(path, dry_run=args.dry_run):
            n_ok += 1
        else:
            n_skip += 1

    logger.info("Backfill complete: %d ok, %d skipped", n_ok, n_skip)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
