"""Parameter-landscape heatmap + top-region spread gate.

Phase G of `.omc/plans/quality-upgrade-plan.md`.

Motivation
----------
PBO (Phase E) quantifies *selection* bias: how often the IS-best
strategy loses OOS. A PBO ≈ 0 run can still pick a point that sits on
a narrow ridge of the parameter landscape — small perturbations
collapse the metric. Region-Heatmap closes that hole: it aggregates the
full grid onto a 2-D ``(alignment_threshold × risk_reward)`` pivot of
median DSR and measures the *spread* of the top-10% cells.

  spread_q90_q10 on top-10% cells < 0.10  ⇒  plateau, deploy-safe.
  spread_q90_q10 on top-10% cells ≥ 0.10  ⇒  ridge/cliff, reject.

The DSR per grid row is computed from the row's mean window-Sharpe
against the full trial-Sharpe null distribution — identical to the
Phase-D CPCV / Phase-E PBO setup so the three gates share a common
statistic (DSR ∈ [0, 1]).

Reference (DSR formula used per cell)
-------------------------------------
Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe
Ratio." The Journal of Portfolio Management, 40 (5) 94-107.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.metrics import (
    deflated_sharpe_ratio as _deflated_sharpe,
    trial_sharpe_variance as _trial_sharpe_variance,
)


# ═══════════════════════════════════════════════════════════════════
#  build_region_grid
# ═══════════════════════════════════════════════════════════════════

def build_region_grid(
    grid_df: pd.DataFrame,
    trial_sharpes: list[float],
    observation_count: int = 60,
    window_sharpe_cols: list[str] | None = None,
    param_x: str = "alignment_threshold",
    param_y: str = "risk_reward",
    filter_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a ``(param_y × param_x)`` pivot of median DSR per cell.

    Per grid row, DSR is computed from the row's mean window-Sharpe
    against the variance of ``trial_sharpes``. Rows with all-NaN
    Sharpe cells are dropped; their cell contributes nothing. A pivot
    aggfunc of ``median`` makes the gate robust to intra-cell outliers.

    Parameters
    ----------
    grid_df
        Brute-force results with at least ``param_x``, ``param_y`` and
        one or more ``w{i}_sharpe`` columns.
    trial_sharpes
        Null-distribution sample (same vector used by Phase-D/E).
    observation_count
        Number of return observations backing each row's Sharpe (daily
        returns over the OOS horizon). A conservative default of 60
        (~3 months of daily returns) keeps DSR meaningful when the row
        itself has no ``n_obs_daily`` column.
    window_sharpe_cols
        Explicit window-Sharpe columns. If ``None`` we auto-detect
        ``w{i}_sharpe`` columns.
    param_x, param_y
        Pivot axes. Default to the SMC gate dimensions.
    filter_mask
        Optional boolean Series selecting a subset of ``grid_df`` rows
        (e.g., ``grid_df["is_evergreen"]``).
    """
    if grid_df.empty:
        return pd.DataFrame()

    df = grid_df
    if filter_mask is not None:
        df = df[filter_mask].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame()

    if window_sharpe_cols is None:
        window_sharpe_cols = [
            c for c in df.columns
            if c.startswith("w") and c.endswith("_sharpe")
        ]
    if not window_sharpe_cols:
        return pd.DataFrame()

    if param_x not in df.columns or param_y not in df.columns:
        return pd.DataFrame()

    clean_trials = [
        float(s) for s in trial_sharpes
        if s is not None and not (isinstance(s, float) and np.isnan(s))
    ]
    trial_var = _trial_sharpe_variance(clean_trials)
    n_trials = len(clean_trials)

    mean_sharpes = df[window_sharpe_cols].mean(axis=1).to_numpy(dtype=float)
    dsrs = np.full(len(df), np.nan, dtype=float)

    if trial_var > 0 and n_trials >= 2 and observation_count >= 2:
        for i, obs in enumerate(mean_sharpes):
            if np.isnan(obs):
                continue
            dsrs[i] = _deflated_sharpe(
                observed_sharpe=float(obs),
                sharpe_variance=trial_var,
                n_trials=n_trials,
                observation_count=int(observation_count),
                skewness=0.0,
                kurtosis=3.0,
            )

    out = df[[param_x, param_y]].copy()
    out["dsr"] = dsrs
    pivot = out.pivot_table(
        index=param_y, columns=param_x, values="dsr", aggfunc="median",
    )
    return pivot


# ═══════════════════════════════════════════════════════════════════
#  region_summary
# ═══════════════════════════════════════════════════════════════════

def region_summary(
    region_df: pd.DataFrame,
    top_pct: float = 0.10,
    gate_threshold: float = 0.10,
) -> dict[str, Any]:
    """Summarise a region pivot into a JSON-safe dict with a gate_pass flag.

    Gate: ``spread_q90_q10 < gate_threshold`` over the top ``top_pct``
    cells, where ``spread_q90_q10`` is the inter-decile spread of the
    top cells' DSRs. Using inter-decile (not max-min) keeps the gate
    robust to a single outlier cell.

    Empty or all-NaN regions are a conservative FAIL — no evidence
    cannot bless a deploy.
    """
    n_total = int(region_df.size)
    default: dict[str, Any] = {
        "n_cells_total": n_total,
        "n_cells_valid": 0,
        "n_cells_top": 0,
        "top_cells_max": 0.0,
        "top_cells_min": 0.0,
        "top_cells_q10": 0.0,
        "top_cells_q90": 0.0,
        "spread_q90_q10": 0.0,
        "spread_max_min": 0.0,
        "top_pct": float(top_pct),
        "gate_threshold": float(gate_threshold),
        "gate_pass": False,
    }
    if region_df.empty:
        return default

    values = region_df.to_numpy(dtype=float).flatten()
    mask = ~np.isnan(values)
    valid = values[mask]
    if valid.size == 0:
        return default

    n_top = max(1, int(np.ceil(top_pct * valid.size)))
    top_values = np.sort(valid)[::-1][:n_top]  # descending, take top-K

    q10 = float(np.quantile(top_values, 0.10))
    q90 = float(np.quantile(top_values, 0.90))
    spread_q = q90 - q10

    return {
        "n_cells_total": n_total,
        "n_cells_valid": int(valid.size),
        "n_cells_top": int(n_top),
        "top_cells_max": float(top_values.max()),
        "top_cells_min": float(top_values.min()),
        "top_cells_q10": q10,
        "top_cells_q90": q90,
        "spread_q90_q10": float(spread_q),
        "spread_max_min": float(top_values.max() - top_values.min()),
        "top_pct": float(top_pct),
        "gate_threshold": float(gate_threshold),
        "gate_pass": bool(spread_q < gate_threshold),
    }


# ═══════════════════════════════════════════════════════════════════
#  plot_region_heatmap
# ═══════════════════════════════════════════════════════════════════

def plot_region_heatmap(
    region_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "Parameter region — median DSR",
    cmap: str = "viridis",
) -> None:
    """Render ``region_df`` to a PNG file using matplotlib (Agg backend)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    if region_df.empty:
        ax.text(
            0.5, 0.5, "no data", ha="center", va="center",
            transform=ax.transAxes, fontsize=14,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        data = region_df.to_numpy(dtype=float)
        im = ax.imshow(
            data, cmap=cmap, aspect="auto",
            origin="lower", vmin=0.0, vmax=1.0,
        )
        ax.set_xticks(range(len(region_df.columns)))
        ax.set_xticklabels([f"{v:g}" for v in region_df.columns])
        ax.set_yticks(range(len(region_df.index)))
        ax.set_yticklabels([f"{v:g}" for v in region_df.index])
        ax.set_xlabel(region_df.columns.name or "param_x")
        ax.set_ylabel(region_df.index.name or "param_y")
        fig.colorbar(im, ax=ax, label="median DSR")
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=8,
                    )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


__all__ = [
    "build_region_grid",
    "region_summary",
    "plot_region_heatmap",
]
