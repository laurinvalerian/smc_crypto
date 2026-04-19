"""Tests for backtest/cost_stress.py — Phase H transaction-cost robustness gate.

Rationale
---------
A strategy passes CPCV / PBO / MC / Region on the backtest cost model
(0.04 % + 0.01 %) but may collapse on a pessimistic real-fill model
(0.06 % + 0.04 %). This gate shocks the cost assumption and demands the
core metrics (PF, Sharpe, DD) stay above a funded-safe floor.

We stub ``simulate_fn`` with a deterministic cost-sensitive generator
to isolate the gate's arithmetic from the heavy signal-generation /
trade-simulation machinery of ``optuna_backtester.py``.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from backtest.cost_stress import compute_cost_stress


# ═══════════════════════════════════════════════════════════════════
#  Stub simulate_fn — trades_df sensitive to commission + slippage
# ═══════════════════════════════════════════════════════════════════

def _make_stub_simulator(
    base_rr: float = 2.0,
    n_signals: int = 60,
    cost_sensitivity: float = 50.0,
):
    """Return a fake ``simulate_fn`` whose PnL shrinks with cost total.

    Each signal yields a trade with:
        pnl = account_size * 0.01 * (base_rr - cost_sensitivity * cost)
    i.e. per-trade PnL is a 1%-risk win at base RR, reduced linearly
    by the total cost (commission + slippage).

    Produces a 60-trade sample spaced 30 min apart so sharpe_daily has
    multiple daily observations.
    """
    def _sim(params, signals, cfg, sym2asset):
        bt = cfg.get("backtest", {})
        cost_total = bt.get("commission_pct", 0.0004) + bt.get("slippage_pct", 0.0001)
        acct = cfg.get("account", {}).get("size", 100_000.0)

        per_trade_r = base_rr - cost_sensitivity * cost_total
        # Alternate wins and minor losses so PF stays finite but shrinks
        # when cost eats into the wins.
        rows = []
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        # Grow over ~15 days — at 4 trades per day → 60 trades → enough
        # daily observations to compute sharpe_daily.
        for i in range(n_signals):
            is_win = i % 3 != 0  # 2 of 3 are wins
            if is_win:
                pnl = acct * 0.01 * per_trade_r
            else:
                pnl = -acct * 0.01  # flat -1R loss
            ts = start + pd.Timedelta(hours=6 * i)
            rows.append({
                "timestamp": ts,
                "exit_time": ts + pd.Timedelta(hours=2),
                "pnl": pnl,
                "actual_rr": per_trade_r if is_win else -1.0,
                "risk_pct": 0.01,
                "equity": acct + (i + 1) * pnl,  # very rough running equity
            })
        df = pd.DataFrame(rows)
        return df, {}

    return _sim


# ═══════════════════════════════════════════════════════════════════
#  Empty / degenerate inputs
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_signals_is_fail(self):
        r = compute_cost_stress(
            best_params={}, oos_signals_union=[], cfg={},
            symbol_to_asset=None,
            simulate_fn=lambda *a, **kw: (pd.DataFrame(), {}),
        )
        assert r["gate_pass"] is False
        assert r["reason"] == "empty_signals"
        assert r["stressed_n_trades"] == 0

    def test_default_gate_thresholds_from_defaults(self):
        """No `cost_stress` cfg section → module defaults kick in."""
        r = compute_cost_stress(
            best_params={}, oos_signals_union=[], cfg={},
            symbol_to_asset=None,
            simulate_fn=lambda *a, **kw: (pd.DataFrame(), {}),
        )
        assert r["stressed_commission_pct"] == pytest.approx(0.0006)
        assert r["stressed_slippage_pct"] == pytest.approx(0.0004)
        assert r["gate_pf"] == pytest.approx(1.5)
        assert r["gate_sharpe"] == pytest.approx(0.5)
        assert r["gate_dd"] == pytest.approx(-0.12)


# ═══════════════════════════════════════════════════════════════════
#  Cost-shock semantics
# ═══════════════════════════════════════════════════════════════════

class TestCostShock:
    def _cfg(self, baseline_commission=0.0004, baseline_slippage=0.0001):
        return {
            "backtest": {
                "commission_pct": baseline_commission,
                "slippage_pct": baseline_slippage,
            },
            "account": {"size": 100_000.0},
        }

    def test_baseline_cfg_not_mutated(self):
        """The stress test must not mutate the user's cfg dict."""
        cfg = self._cfg()
        before = json.dumps(cfg)
        compute_cost_stress(
            best_params={},
            oos_signals_union=[object()] * 60,
            cfg=cfg,
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(),
        )
        assert json.dumps(cfg) == before

    def test_stressed_cfg_sets_force_backtest_costs(self):
        """The stressed cfg must set force_backtest_costs=True so
        _simulate_with_params ignores ASSET_COMMISSION / ASSET_SLIPPAGE
        defaults and honours the elevated costs. Without this flag the
        v1.11 run silently reported baseline==stressed (uninformative).
        """
        seen_cfgs: list[dict] = []

        def _capturing_sim(params, signals, cfg, sym2asset):
            seen_cfgs.append(cfg)
            return _make_stub_simulator()(params, signals, cfg, sym2asset)

        compute_cost_stress(
            best_params={},
            oos_signals_union=[object()] * 60,
            cfg=self._cfg(),
            symbol_to_asset=None,
            simulate_fn=_capturing_sim,
        )
        # Two simulate_fn calls: [0] baseline, [1] stressed.
        assert len(seen_cfgs) == 2
        base_cfg, stressed_cfg = seen_cfgs
        assert base_cfg["backtest"].get("force_backtest_costs", False) is False
        assert stressed_cfg["backtest"]["force_backtest_costs"] is True
        # Stressed cfg carries the elevated numbers so a backtester that
        # honours the flag will see them.
        assert stressed_cfg["backtest"]["commission_pct"] == pytest.approx(0.0006)
        assert stressed_cfg["backtest"]["slippage_pct"] == pytest.approx(0.0004)

    def test_stressed_worse_than_baseline(self):
        """With a cost-sensitive simulator, stressed metrics must be
        <= baseline (cost eats into PnL)."""
        r = compute_cost_stress(
            best_params={},
            oos_signals_union=[object()] * 60,
            cfg=self._cfg(),
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(),
        )
        assert r["reason"] == "ok"
        assert r["baseline_n_trades"] == 60
        assert r["stressed_n_trades"] == 60
        assert r["stressed_pf"] <= r["baseline_pf"]
        assert r["stressed_sharpe"] <= r["baseline_sharpe"]
        # pf_delta must be negative (stress reduces PF)
        assert r["pf_delta"] <= 0.0

    def test_gate_passes_when_strategy_robust(self):
        """A high-edge stub (base_rr=3.0, low sensitivity) still clears
        gates at 6 bps + 4 bps."""
        r = compute_cost_stress(
            best_params={},
            oos_signals_union=[object()] * 60,
            cfg=self._cfg(),
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(
                base_rr=3.0, cost_sensitivity=20.0,
            ),
        )
        assert r["gate_pass"] is True
        assert r["gate_pf_pass"] is True
        assert r["gate_sharpe_pass"] is True
        assert r["gate_dd_pass"] is True

    def test_gate_fails_when_cost_collapses_pf(self):
        """Cost-fragile strategy → stressed PF drops below 1.5 → FAIL.

        Stub arithmetic at stressed (0.06 % + 0.04 %) = 0.001 total:
          per_trade_r = 0.7 − 500 × 0.001 = 0.2
          40 wins × 0.01 × 0.2 = 0.08  (wins in $-units of risk)
          20 losses × 0.01       = 0.20
          PF = 0.08 / 0.20 = 0.4  ≪ gate 1.5 → FAIL.
        """
        r = compute_cost_stress(
            best_params={},
            oos_signals_union=[object()] * 60,
            cfg=self._cfg(),
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(
                base_rr=0.7, cost_sensitivity=500.0,
            ),
        )
        assert r["gate_pass"] is False
        assert r["gate_pf_pass"] is False

    def test_deltas_sign_matches_direction(self):
        """pf_delta and sharpe_delta are stressed − baseline; both must
        be <= 0 with a cost-sensitive simulator."""
        r = compute_cost_stress(
            best_params={},
            oos_signals_union=[object()] * 60,
            cfg=self._cfg(),
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(
                base_rr=2.0, cost_sensitivity=80.0,
            ),
        )
        assert r["pf_delta"] <= 0.0
        assert r["sharpe_delta"] <= 0.0


# ═══════════════════════════════════════════════════════════════════
#  Custom thresholds + JSON roundtrip
# ═══════════════════════════════════════════════════════════════════

class TestOutputContract:
    def test_custom_gate_thresholds_honored(self):
        r = compute_cost_stress(
            best_params={},
            oos_signals_union=[object()] * 60,
            cfg={"backtest": {"commission_pct": 0.0004, "slippage_pct": 0.0001},
                 "account": {"size": 100_000.0}},
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(base_rr=3.0, cost_sensitivity=10.0),
            gate_pf=10.0,   # unreachable → FAIL even though strategy is strong
            gate_sharpe=0.5,
            gate_dd=-0.12,
        )
        assert r["gate_pf_pass"] is False
        assert r["gate_pass"] is False

    def test_config_section_thresholds_honored(self):
        cfg = {
            "backtest": {"commission_pct": 0.0004, "slippage_pct": 0.0001},
            "account": {"size": 100_000.0},
            "cost_stress": {
                "commission_pct": 0.0012,   # harsher than default 0.06%
                "slippage_pct": 0.0008,
                "gate_pf": 1.3,
                "gate_sharpe": 0.3,
                "gate_dd": -0.15,
            },
        }
        r = compute_cost_stress(
            best_params={}, oos_signals_union=[object()] * 60,
            cfg=cfg, symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(base_rr=3.0, cost_sensitivity=20.0),
        )
        assert r["stressed_commission_pct"] == pytest.approx(0.0012)
        assert r["stressed_slippage_pct"] == pytest.approx(0.0008)
        assert r["gate_pf"] == pytest.approx(1.3)
        assert r["gate_sharpe"] == pytest.approx(0.3)

    def test_json_roundtrip_safe(self):
        r = compute_cost_stress(
            best_params={}, oos_signals_union=[object()] * 60,
            cfg={"backtest": {"commission_pct": 0.0004, "slippage_pct": 0.0001},
                 "account": {"size": 100_000.0}},
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(base_rr=2.5, cost_sensitivity=20.0),
        )
        back = json.loads(json.dumps(r))
        assert back["gate_pass"] == r["gate_pass"]
        assert back["stressed_pf"] == pytest.approx(r["stressed_pf"])

    def test_output_keys_complete(self):
        r = compute_cost_stress(
            best_params={}, oos_signals_union=[object()] * 60,
            cfg={"backtest": {"commission_pct": 0.0004, "slippage_pct": 0.0001},
                 "account": {"size": 100_000.0}},
            symbol_to_asset=None,
            simulate_fn=_make_stub_simulator(),
        )
        expected = {
            "baseline_pf", "baseline_sharpe", "baseline_dd", "baseline_n_trades",
            "stressed_pf", "stressed_sharpe", "stressed_dd", "stressed_n_trades",
            "pf_delta", "sharpe_delta", "dd_delta",
            "stressed_commission_pct", "stressed_slippage_pct",
            "gate_pf", "gate_sharpe", "gate_dd",
            "gate_pf_pass", "gate_sharpe_pass", "gate_dd_pass",
            "gate_pass", "reason",
        }
        assert expected.issubset(r.keys())
