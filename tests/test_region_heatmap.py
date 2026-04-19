"""Tests for backtest/region_heatmap.py — Phase G parameter-landscape gate.

Purpose
-------
The PBO gate (Phase E) asks *"does the IS-winner win OOS?"*. That is a
question about a single selection. Region-Heatmap asks a complementary
question: *is the selected (alignment, RR) cell part of a broad
high-quality plateau, or is it an isolated cliff?* A cliff means the
selection is unstable to small parameter perturbations — a deployment
risk even when PBO is low.

We aggregate the full grid into a 2-D pivot of median DSR per cell,
then measure the *spread* of the top-10% cells. Tight spread =
plateau-like landscape.

Gate (plan §2.6): ``spread_q90_q10 < 0.10`` over the top-10% region.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtest.region_heatmap import (
    build_region_grid,
    build_region_grid_sharpe,
    plot_region_heatmap,
    region_summary,
)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

def _make_grid_df(
    align_values: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90),
    rr_values: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    n_windows: int = 8,
    peak: tuple[float, float] = (0.80, 2.0),
    peak_sharpe: float = 1.5,
    noise: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic grid_df with a Gaussian-peaked Sharpe landscape around `peak`."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | bool]] = []
    for a in align_values:
        for rr in rr_values:
            dx = (a - peak[0]) / 0.1
            dy = (rr - peak[1]) / 1.0
            mu = peak_sharpe * np.exp(-(dx * dx + dy * dy) / 0.5)
            row: dict[str, float | bool] = {
                "alignment_threshold": float(a),
                "risk_reward": float(rr),
                "is_evergreen": True,
            }
            for w in range(n_windows):
                row[f"w{w}_sharpe"] = float(mu + rng.normal(0, noise))
            rows.append(row)
    return pd.DataFrame(rows)


def _trial_sharpes_from(df: pd.DataFrame) -> list[float]:
    sharpe_cols = [c for c in df.columns if c.startswith("w") and c.endswith("_sharpe")]
    return df[sharpe_cols].mean(axis=1).astype(float).tolist()


# ═══════════════════════════════════════════════════════════════════
#  build_region_grid
# ═══════════════════════════════════════════════════════════════════

class TestBuildRegionGrid:
    def test_pivot_shape(self):
        df = _make_grid_df()
        trial_sharpes = _trial_sharpes_from(df)
        pivot = build_region_grid(df, trial_sharpes=trial_sharpes, observation_count=60)
        assert pivot.shape == (4, 5)  # 4 RR × 5 alignment

    def test_empty_grid_df_returns_empty_pivot(self):
        pivot = build_region_grid(
            pd.DataFrame(),
            trial_sharpes=[1.0] * 10,
            observation_count=60,
        )
        assert pivot.empty

    def test_dsr_values_in_unit_interval(self):
        df = _make_grid_df()
        trial_sharpes = _trial_sharpes_from(df)
        pivot = build_region_grid(df, trial_sharpes=trial_sharpes, observation_count=60)
        values = pivot.values[~np.isnan(pivot.values)]
        assert (values >= 0.0).all() and (values <= 1.0).all()

    def test_missing_rr_row_excluded(self):
        df = _make_grid_df()
        df = df[df["risk_reward"] != 2.0].reset_index(drop=True)
        trial_sharpes = _trial_sharpes_from(df)
        pivot = build_region_grid(df, trial_sharpes=trial_sharpes, observation_count=60)
        assert 2.0 not in pivot.index
        assert pivot.shape == (3, 5)

    def test_peak_cell_dsr_gt_corner_dsr(self):
        df = _make_grid_df(seed=1)
        trial_sharpes = _trial_sharpes_from(df)
        pivot = build_region_grid(df, trial_sharpes=trial_sharpes, observation_count=60)
        assert pivot.loc[2.0, 0.80] > pivot.loc[1.5, 0.70]
        assert pivot.loc[2.0, 0.80] > pivot.loc[3.0, 0.90]

    def test_no_window_sharpe_cols_returns_empty(self):
        df = pd.DataFrame({
            "alignment_threshold": [0.70, 0.80],
            "risk_reward": [1.5, 2.0],
        })
        pivot = build_region_grid(
            df, trial_sharpes=[1.0] * 10, observation_count=60,
        )
        assert pivot.empty

    def test_custom_filter_mask_applied(self):
        df = _make_grid_df()
        trial_sharpes = _trial_sharpes_from(df)
        mask = df["risk_reward"] >= 2.0
        pivot = build_region_grid(
            df,
            trial_sharpes=trial_sharpes,
            observation_count=60,
            filter_mask=mask,
        )
        assert 1.5 not in pivot.index


# ═══════════════════════════════════════════════════════════════════
#  region_summary
# ═══════════════════════════════════════════════════════════════════

class TestRegionSummary:
    def test_gate_passes_when_top_region_flat(self):
        pivot = pd.DataFrame(
            np.full((4, 5), 0.90),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        s = region_summary(pivot, top_pct=0.10, gate_threshold=0.10)
        assert s["gate_pass"] is True
        assert s["spread_q90_q10"] < 0.01

    def test_gate_fails_when_top_region_dispersed(self):
        rng = np.random.default_rng(0)
        pivot = pd.DataFrame(
            rng.uniform(0.0, 1.0, size=(4, 5)),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        s = region_summary(pivot, top_pct=0.50, gate_threshold=0.10)
        assert s["gate_pass"] is False

    def test_top_pct_always_keeps_at_least_one_cell(self):
        pivot = pd.DataFrame([[0.5]], index=[1.5], columns=[0.70])
        s = region_summary(pivot, top_pct=0.01, gate_threshold=0.10)
        assert s["n_cells_top"] == 1

    def test_empty_pivot_is_gate_fail(self):
        s = region_summary(pd.DataFrame(), top_pct=0.10, gate_threshold=0.10)
        assert s["n_cells_total"] == 0
        assert s["n_cells_valid"] == 0
        assert s["gate_pass"] is False

    def test_nan_cells_excluded(self):
        pivot = pd.DataFrame(
            [[0.9, np.nan], [np.nan, 0.9]],
            index=[1.5, 2.0],
            columns=[0.70, 0.75],
        )
        s = region_summary(pivot, top_pct=1.0, gate_threshold=0.10)
        assert s["n_cells_total"] == 4
        assert s["n_cells_valid"] == 2
        assert s["gate_pass"] is True

    def test_top_pct_monotone_wider_pct_higher_or_equal_spread(self):
        pivot = pd.DataFrame(
            [[0.1, 0.2, 0.8, 0.9, 0.95]],
            index=[1.5],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        s_narrow = region_summary(pivot, top_pct=0.20, gate_threshold=0.10)
        s_wide = region_summary(pivot, top_pct=1.0, gate_threshold=0.10)
        assert s_wide["spread_max_min"] >= s_narrow["spread_max_min"]


# ═══════════════════════════════════════════════════════════════════
#  Output contract
# ═══════════════════════════════════════════════════════════════════

class TestOutput:
    def test_summary_output_keys(self):
        pivot = pd.DataFrame(
            [[0.5, 0.6], [0.7, 0.8]],
            index=[1.5, 2.0],
            columns=[0.70, 0.75],
        )
        s = region_summary(pivot, top_pct=0.25, gate_threshold=0.10)
        expected = {
            "n_cells_total", "n_cells_valid", "n_cells_top",
            "top_cells_max", "top_cells_min",
            "top_cells_q10", "top_cells_q90",
            "spread_q90_q10", "spread_max_min",
            "top_pct", "gate_threshold", "gate_pass",
        }
        assert expected.issubset(s.keys())

    def test_json_roundtrip(self):
        pivot = pd.DataFrame(
            [[0.5, 0.6], [0.7, 0.8]],
            index=[1.5, 2.0],
            columns=[0.70, 0.75],
        )
        s = region_summary(pivot, top_pct=0.25, gate_threshold=0.10)
        back = json.loads(json.dumps(s))
        assert back["gate_pass"] == s["gate_pass"]
        assert back["spread_q90_q10"] == s["spread_q90_q10"]


# ═══════════════════════════════════════════════════════════════════
#  plot_region_heatmap
# ═══════════════════════════════════════════════════════════════════

class TestPlot:
    def test_png_file_written(self, tmp_path):
        pivot = pd.DataFrame(
            [[0.5, 0.6], [0.7, 0.8]],
            index=[1.5, 2.0],
            columns=[0.70, 0.75],
        )
        out = tmp_path / "heatmap.png"
        plot_region_heatmap(pivot, output_path=out, title="Test")
        assert out.exists()
        assert out.stat().st_size > 100

    def test_empty_pivot_still_produces_png(self, tmp_path):
        out = tmp_path / "heatmap_empty.png"
        plot_region_heatmap(pd.DataFrame(), output_path=out, title="Empty")
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════
#  build_region_grid_sharpe — v1.11 saturation-fix companion
# ═══════════════════════════════════════════════════════════════════

class TestBuildRegionGridSharpe:
    def test_pivot_shape_matches_dsr_pivot(self):
        df = _make_grid_df()
        dsr_pivot = build_region_grid(
            df, trial_sharpes=_trial_sharpes_from(df), observation_count=60,
        )
        sharpe_pivot = build_region_grid_sharpe(df)
        assert sharpe_pivot.shape == dsr_pivot.shape
        assert list(sharpe_pivot.index) == list(dsr_pivot.index)
        assert list(sharpe_pivot.columns) == list(dsr_pivot.columns)

    def test_empty_grid_returns_empty(self):
        assert build_region_grid_sharpe(pd.DataFrame()).empty

    def test_no_window_sharpe_cols_returns_empty(self):
        df = pd.DataFrame(
            {"alignment_threshold": [0.70, 0.80], "risk_reward": [1.5, 2.0]},
        )
        assert build_region_grid_sharpe(df).empty

    def test_peak_cell_sharpe_gt_corner_sharpe(self):
        df = _make_grid_df(seed=1)
        pivot = build_region_grid_sharpe(df)
        assert pivot.loc[2.0, 0.80] > pivot.loc[1.5, 0.70]
        assert pivot.loc[2.0, 0.80] > pivot.loc[3.0, 0.90]

    def test_sharpe_not_clipped_to_unit_interval(self):
        """Sharpe can exceed 1.0 — that's the whole point (DSR cannot)."""
        df = _make_grid_df(peak_sharpe=5.0, noise=0.05)
        pivot = build_region_grid_sharpe(df)
        values = pivot.values[~np.isnan(pivot.values)]
        # peak ≈ 5.0 → allow cells above 1.0
        assert values.max() > 1.0

    def test_filter_mask_applied(self):
        df = _make_grid_df()
        mask = df["risk_reward"] >= 2.0
        pivot = build_region_grid_sharpe(df, filter_mask=mask)
        assert 1.5 not in pivot.index


# ═══════════════════════════════════════════════════════════════════
#  region_summary — saturation-fix secondary Sharpe gate (v1.11)
# ═══════════════════════════════════════════════════════════════════

class TestRegionSummarySaturationFix:
    def test_backward_compat_without_sharpe_pivot(self):
        """Without sharpe_region_df, behaviour matches the v1.10 single gate."""
        pivot = pd.DataFrame(
            np.full((4, 5), 0.90),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        s = region_summary(pivot, top_pct=0.10, gate_threshold=0.10)
        assert s["sharpe_gate_enabled"] is False
        assert s["dsr_gate_pass"] is True
        assert s["gate_pass"] is True

    def test_dsr_saturation_caught_by_sharpe_ridge(self):
        """DSR near-saturated (all ≈ 1.0) → DSR-spread ≈ 0 PASS.
        But Sharpe landscape has a ridge → Sharpe-rel-spread high → combined FAIL.
        This is exactly the v1.10 failure mode the fix addresses.

        Ridge cell given uniquely-max DSR (1.0 vs 0.9999 for the rest) so the
        top-K selection is deterministic across numpy/pandas versions — the
        saturation character is preserved since DSR-spread = 1e-4 ≪ gate.
        """
        dsr_vals = np.full((4, 5), 0.9999)
        dsr_vals[1, 2] = 1.0  # ridge cell uniquely max — deterministic top-1
        dsr_pivot = pd.DataFrame(
            dsr_vals,
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        # Ridge: one cell Sharpe=3.0, rest Sharpe=1.0 → rel_spread ≫ 0.15
        sharpe_vals = np.full((4, 5), 1.0)
        sharpe_vals[1, 2] = 3.0  # rr=2.0, align=0.80
        sharpe_pivot = pd.DataFrame(
            sharpe_vals,
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        s = region_summary(
            dsr_pivot, top_pct=0.50,  # top 50% to include the ridge cell
            gate_threshold=0.10, sharpe_region_df=sharpe_pivot,
            sharpe_rel_gate=0.15,
        )
        assert s["dsr_gate_pass"] is True
        assert s["sharpe_gate_pass"] is False
        assert s["gate_pass"] is False  # combined AND → FAIL

    def test_plateau_both_gates_pass(self):
        """DSR flat PLUS Sharpe flat → both gates PASS → combined PASS."""
        dsr_pivot = pd.DataFrame(
            np.full((4, 5), 0.90),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        sharpe_pivot = pd.DataFrame(
            np.full((4, 5), 2.0),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        s = region_summary(
            dsr_pivot, top_pct=0.10, gate_threshold=0.10,
            sharpe_region_df=sharpe_pivot, sharpe_rel_gate=0.15,
        )
        assert s["dsr_gate_pass"] is True
        assert s["sharpe_gate_pass"] is True
        assert s["gate_pass"] is True
        assert s["sharpe_rel_spread_q90_q10"] < 0.15

    def test_dsr_failing_but_sharpe_flat_still_fail(self):
        """DSR spread > threshold must FAIL regardless of Sharpe."""
        rng = np.random.default_rng(0)
        dsr_pivot = pd.DataFrame(
            rng.uniform(0.0, 1.0, size=(4, 5)),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        sharpe_pivot = pd.DataFrame(
            np.full((4, 5), 2.0),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        s = region_summary(
            dsr_pivot, top_pct=0.50, gate_threshold=0.10,
            sharpe_region_df=sharpe_pivot, sharpe_rel_gate=0.15,
        )
        # With uniform DSR on 20 cells and top 50% → spread ≈ 0.3+ → FAIL
        assert s["dsr_gate_pass"] is False
        assert s["gate_pass"] is False

    def test_shape_mismatch_conservative_fail(self):
        """If Sharpe pivot shape doesn't match DSR pivot, fail closed."""
        dsr_pivot = pd.DataFrame(
            np.full((4, 5), 1.0),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        sharpe_pivot = pd.DataFrame(
            np.full((2, 2), 2.0),  # wrong shape
            index=[1.5, 2.0],
            columns=[0.70, 0.75],
        )
        s = region_summary(
            dsr_pivot, top_pct=0.10, gate_threshold=0.10,
            sharpe_region_df=sharpe_pivot, sharpe_rel_gate=0.15,
        )
        assert s["dsr_gate_pass"] is True
        assert s["sharpe_gate_pass"] is False  # shape mismatch → closed
        assert s["gate_pass"] is False

    def test_sharpe_rel_spread_scale_invariant(self):
        """Same *relative* spread at scale 1.0 and scale 10.0 → same rel_spread."""
        dsr_pivot = pd.DataFrame(
            np.full((4, 5), 1.0),
            index=[1.5, 2.0, 2.5, 3.0],
            columns=[0.70, 0.75, 0.80, 0.85, 0.90],
        )
        sharpe_small = np.full((4, 5), 1.0)
        sharpe_small[1, 2] = 1.5
        sharpe_big = sharpe_small * 10.0

        s_small = region_summary(
            dsr_pivot, top_pct=0.50, gate_threshold=0.10,
            sharpe_region_df=pd.DataFrame(
                sharpe_small, index=dsr_pivot.index, columns=dsr_pivot.columns,
            ),
            sharpe_rel_gate=0.50,
        )
        s_big = region_summary(
            dsr_pivot, top_pct=0.50, gate_threshold=0.10,
            sharpe_region_df=pd.DataFrame(
                sharpe_big, index=dsr_pivot.index, columns=dsr_pivot.columns,
            ),
            sharpe_rel_gate=0.50,
        )
        assert abs(
            s_small["sharpe_rel_spread_q90_q10"]
            - s_big["sharpe_rel_spread_q90_q10"]
        ) < 1e-9

    def test_summary_output_keys_include_sharpe_fields(self):
        dsr_pivot = pd.DataFrame([[0.9, 0.8]], index=[1.5], columns=[0.70, 0.75])
        sharpe_pivot = pd.DataFrame([[2.0, 2.1]], index=[1.5], columns=[0.70, 0.75])
        s = region_summary(
            dsr_pivot, top_pct=0.50, gate_threshold=0.10,
            sharpe_region_df=sharpe_pivot, sharpe_rel_gate=0.15,
        )
        expected = {
            "sharpe_gate_enabled", "sharpe_rel_gate",
            "sharpe_top_cells_median", "sharpe_top_cells_q10",
            "sharpe_top_cells_q90", "sharpe_spread_q90_q10",
            "sharpe_rel_spread_q90_q10", "sharpe_gate_pass",
            "dsr_gate_pass",
        }
        assert expected.issubset(s.keys())

    def test_json_roundtrip_with_sharpe_gate(self):
        dsr_pivot = pd.DataFrame([[0.9, 0.8]], index=[1.5], columns=[0.70, 0.75])
        sharpe_pivot = pd.DataFrame([[2.0, 2.1]], index=[1.5], columns=[0.70, 0.75])
        s = region_summary(
            dsr_pivot, top_pct=0.50, gate_threshold=0.10,
            sharpe_region_df=sharpe_pivot, sharpe_rel_gate=0.15,
        )
        back = json.loads(json.dumps(s))
        assert back["gate_pass"] == s["gate_pass"]
        assert back["sharpe_gate_pass"] == s["sharpe_gate_pass"]
        assert back["sharpe_rel_spread_q90_q10"] == s["sharpe_rel_spread_q90_q10"]
