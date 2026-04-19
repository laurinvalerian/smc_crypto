"""Tests for backtest/monte_carlo.py — Phase G tail-risk (CVaR-DD) gate.

Rationale
---------
The pre-Phase-G ``monte_carlo_check`` returns ``worst_dd_95pct`` = the 5th
percentile of simulated max-DDs (a *VaR*). That hides the shape of the
tail: two strategies with identical 5th-percentile can have very
different mean-DDs beyond it.

Phase G replaces that with the **Conditional VaR** (CVaR / Expected
Shortfall): the mean of the worst ``1 - confidence`` fraction of
simulated max-DDs. Sub-additive, coherent, penalises tail concentration —
the correct funded-account gate metric.

Gate (plan §2.6): ``cvar_dd_95 > -0.20``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtest.monte_carlo import compute_mc_cvar_dd


def _make_trades(
    r_multiples: list[float],
    account_size: float = 100_000.0,
    risk_pct: float = 0.01,
) -> pd.DataFrame:
    """Build a tiny trades_df compatible with the MC R-multiple resampler."""
    rows: list[dict[str, float | str]] = []
    equity = account_size
    for r in r_multiples:
        risk_amt = equity * risk_pct
        pnl = risk_amt * r
        equity = max(equity + pnl, 0.0)
        outcome = "win" if r > 0 else ("loss" if r < 0 else "breakeven")
        rows.append({
            "actual_rr": float(r),
            "pnl": float(pnl),
            "risk_pct": float(risk_pct),
            "equity": float(equity),
            "outcome": outcome,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Defaults / guards
# ═══════════════════════════════════════════════════════════════════

class TestDefaults:
    def test_empty_df_returns_default(self):
        r = compute_mc_cvar_dd(pd.DataFrame(), seed=42)
        assert r["n_simulations"] == 0
        assert r["cvar_dd_95"] == 0.0
        assert r["reason"] == "not_enough_trades"

    def test_too_few_trades_returns_default(self):
        df = _make_trades([-1.0] * 5)
        r = compute_mc_cvar_dd(df, seed=42)
        assert r["n_simulations"] == 0
        assert r["reason"] == "not_enough_trades"


# ═══════════════════════════════════════════════════════════════════
#  CVaR semantics
# ═══════════════════════════════════════════════════════════════════

class TestCvarSemantics:
    def test_cvar_dd_le_var_dd(self):
        """CVaR ≤ VaR (DDs are negative — CVaR = mean of worst tail ≤ VaR quantile)."""
        df = _make_trades([1.5, -1.0] * 20)
        r = compute_mc_cvar_dd(df, n_simulations=500, seed=42)
        assert r["cvar_dd_95"] <= r["var_dd_95"] + 1e-12

    def test_cvar_dd_nonpositive(self):
        df = _make_trades([1.5] * 30)
        r = compute_mc_cvar_dd(df, n_simulations=300, seed=42)
        assert r["cvar_dd_95"] <= 0.0

    def test_all_winners_cvar_near_zero(self):
        df = _make_trades([1.5] * 30)
        r = compute_mc_cvar_dd(df, n_simulations=300, seed=42)
        assert r["cvar_dd_95"] >= -0.005

    def test_all_losers_cvar_deeply_negative(self):
        df = _make_trades([-1.0] * 30)
        r = compute_mc_cvar_dd(df, n_simulations=300, seed=42)
        # 30 consecutive -1R losses at 1% risk → DD around -0.26. CVaR must reflect that.
        assert r["cvar_dd_95"] <= -0.20

    def test_confidence_monotone_tighter_tail_is_worse(self):
        """CVaR@99 is a narrower/worse tail than CVaR@95 → more negative (or equal)."""
        df = _make_trades([1.5, -1.0] * 25)
        r_95 = compute_mc_cvar_dd(df, n_simulations=500, seed=42, confidence=0.95)
        r_99 = compute_mc_cvar_dd(df, n_simulations=500, seed=42, confidence=0.99)
        assert r_99["cvar_dd_95"] <= r_95["cvar_dd_95"] + 1e-9


# ═══════════════════════════════════════════════════════════════════
#  Determinism / reproducibility
# ═══════════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_same_seed_same_result(self):
        df = _make_trades([1.5, -1.0] * 10)
        r1 = compute_mc_cvar_dd(df, n_simulations=200, seed=42)
        r2 = compute_mc_cvar_dd(df, n_simulations=200, seed=42)
        assert r1["cvar_dd_95"] == r2["cvar_dd_95"]
        assert r1["var_dd_95"] == r2["var_dd_95"]
        assert r1["median_dd"] == r2["median_dd"]

    def test_n_simulations_respected(self):
        df = _make_trades([1.5, -1.0] * 10)
        r = compute_mc_cvar_dd(df, n_simulations=150, seed=42)
        assert r["n_simulations"] == 150


# ═══════════════════════════════════════════════════════════════════
#  Input variants
# ═══════════════════════════════════════════════════════════════════

class TestInputs:
    def test_fallback_when_actual_rr_missing(self):
        """If actual_rr is absent, R-multiples are recomputed from pnl/risk_pct/equity."""
        df = _make_trades([1.5, -1.0] * 10).drop(columns=["actual_rr"])
        r = compute_mc_cvar_dd(df, n_simulations=200, seed=42)
        assert r["n_simulations"] == 200
        assert r["cvar_dd_95"] <= 0.0

    def test_extreme_r_multiples_clipped(self):
        """An outlier R=+1000 must not blow up the CVaR sanity."""
        df = _make_trades([1000.0, -1.0] * 10)
        r = compute_mc_cvar_dd(df, n_simulations=200, seed=42)
        assert np.isfinite(r["cvar_dd_95"])
        assert r["cvar_dd_95"] <= 0.0

    def test_custom_account_size(self):
        df = _make_trades([1.5, -1.0] * 10)
        r_100k = compute_mc_cvar_dd(df, account_size=100_000, n_simulations=200, seed=42)
        r_50k = compute_mc_cvar_dd(df, account_size=50_000, n_simulations=200, seed=42)
        # DD is a *percentage* metric → should be near-invariant to account size.
        assert abs(r_100k["cvar_dd_95"] - r_50k["cvar_dd_95"]) < 0.01


# ═══════════════════════════════════════════════════════════════════
#  Output contract
# ═══════════════════════════════════════════════════════════════════

class TestOutput:
    def test_output_keys(self):
        df = _make_trades([1.5, -1.0] * 10)
        r = compute_mc_cvar_dd(df, n_simulations=200, seed=42)
        expected = {
            "cvar_dd_95", "var_dd_95", "median_dd", "mean_dd",
            "median_final_pnl", "ci_lower_pnl", "ci_upper_pnl",
            "pct_profitable", "n_simulations", "seed", "confidence", "reason",
        }
        assert expected.issubset(r.keys())

    def test_json_roundtrip_safe(self):
        df = _make_trades([1.5, -1.0] * 10)
        r = compute_mc_cvar_dd(df, n_simulations=200, seed=42)
        serialized = json.dumps(r)
        back = json.loads(serialized)
        assert back["cvar_dd_95"] == pytest.approx(r["cvar_dd_95"])
        assert back["n_simulations"] == r["n_simulations"]

    def test_values_are_native_python(self):
        """No numpy types leak to the dict (keeps downstream JSON dump safe)."""
        df = _make_trades([1.5, -1.0] * 10)
        r = compute_mc_cvar_dd(df, n_simulations=200, seed=42)
        for k, v in r.items():
            assert not isinstance(v, (np.floating, np.integer)), (
                f"{k} leaked numpy type {type(v)}"
            )
