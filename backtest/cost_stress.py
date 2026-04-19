"""Transaction-cost stress test — robustness of the selected params under
conservative cost assumptions.

Phase H of `.omc/plans/quality-upgrade-plan.md` (v1.11, 2026-04-19).

Motivation
----------
The default backtest uses the live exchange's posted commission and a
thin slippage assumption (``0.04 % + 0.01 %`` on crypto). A funded
account must survive pessimistic fills too — spread blow-outs during
volatile regimes, taker-only liquidity, slower WebSocket fills, etc.

A strategy is *cost-robust* iff it remains profitable at a conservative
cost scenario. The stress test re-simulates the same OOS signals with
elevated ``commission_pct`` and ``slippage_pct`` and demands the
stressed daily Sharpe and PF exceed a funded-deploy floor.

Gate (this module, funded-safe defaults)
----------------------------------------
At 0.06 % commission + 0.04 % slippage:

  - stressed ``pf_real >= 1.5``       (kept profitable after cost shock)
  - stressed ``sharpe_daily >= 0.5``  (risk-adjusted edge not erased)
  - stressed ``max_drawdown >= -0.12``  (not worse than 12 % DD)

All three must pass — a cost-collapse on any one is a reject. The
baseline (non-stressed) result is captured alongside so the output
JSON shows how much the metrics degraded under the shock.

Design notes
------------
- No new simulator — we re-use ``_simulate_with_params`` via an injected
  callable so tests and callers can parametrise.
- The cost shock is applied by mutating a *copy* of the config with the
  stressed values, preserving the rest of the pipeline unchanged.
- Like MC/Region, empty inputs are conservative-FAIL.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from core.metrics import sharpe_daily


# Funded-safe defaults — can be overridden via config ``cost_stress``.
_DEFAULT_STRESS_COMMISSION_PCT: float = 0.0006
_DEFAULT_STRESS_SLIPPAGE_PCT: float = 0.0004
_DEFAULT_GATE_PF: float = 1.5
_DEFAULT_GATE_SHARPE: float = 0.5
_DEFAULT_GATE_DD: float = -0.12


# ═══════════════════════════════════════════════════════════════════
#  Internals
# ═══════════════════════════════════════════════════════════════════

def _stressed_config(
    base_cfg: dict[str, Any],
    stressed_commission: float,
    stressed_slippage: float,
) -> dict[str, Any]:
    """Return a deep copy of ``base_cfg`` with elevated transaction costs.

    Also sets ``backtest.force_backtest_costs = True`` so
    ``_simulate_with_params`` ignores the module-level
    ``ASSET_COMMISSION`` / ``ASSET_SLIPPAGE`` defaults (which would
    otherwise override ``backtest.commission_pct``). Without this flag
    the stressed simulation silently re-uses the Binance-taker baseline
    and returns ``stressed == baseline`` — an uninformative gate.
    """
    out = deepcopy(base_cfg)
    bt = out.setdefault("backtest", {})
    bt["commission_pct"] = float(stressed_commission)
    bt["slippage_pct"] = float(stressed_slippage)
    bt["force_backtest_costs"] = True
    return out


def _metrics_from_trades(
    trades_df: pd.DataFrame,
    account_size: float,
) -> dict[str, float]:
    """Extract the three cost-stress gate metrics from a trades DataFrame."""
    if trades_df.empty:
        return {"pf_real": 0.0, "sharpe_daily": 0.0, "max_drawdown": 0.0,
                "n_trades": 0}

    # PF: sum(wins) / abs(sum(losses)), using signed pnl directly.
    pnl = trades_df["pnl"].astype(float)
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    pf_real = float(wins / abs(losses)) if losses < 0 else float("inf")
    if not np.isfinite(pf_real):
        # All wins → cap to a large but finite number so gate logic works.
        pf_real = 10.0

    sharpe_d = float(sharpe_daily(trades_df, account_size=account_size))

    # Max drawdown on the equity curve.
    ordered = trades_df.sort_values(
        "exit_time" if "exit_time" in trades_df.columns else "timestamp",
    )
    equity = float(account_size) + ordered["pnl"].cumsum().to_numpy()
    if equity.size == 0:
        max_dd = 0.0
    else:
        running_max = np.maximum.accumulate(
            np.maximum(equity, float(account_size))
        )
        dd_series = np.where(
            running_max > 0,
            (equity - running_max) / running_max,
            0.0,
        )
        max_dd = float(dd_series.min()) if dd_series.size else 0.0

    return {
        "pf_real": pf_real,
        "sharpe_daily": sharpe_d,
        "max_drawdown": max_dd,
        "n_trades": int(len(trades_df)),
    }


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

def compute_cost_stress(
    best_params: dict[str, Any],
    oos_signals_union: Sequence[Any],
    cfg: dict[str, Any],
    symbol_to_asset: dict[str, str] | None,
    simulate_fn: Callable[..., tuple[pd.DataFrame, dict[str, float]]],
    account_size: float | None = None,
    stressed_commission_pct: float | None = None,
    stressed_slippage_pct: float | None = None,
    gate_pf: float | None = None,
    gate_sharpe: float | None = None,
    gate_dd: float | None = None,
) -> dict[str, Any]:
    """Re-simulate ``best_params`` under stressed transaction costs.

    Returns a JSON-safe dict containing the baseline and stressed
    metrics plus per-gate and combined pass flags. Empty / unusable
    inputs fall back to a conservative FAIL.

    Parameters
    ----------
    best_params
        Param dict identical to the one passed to ``simulate_fn``.
    oos_signals_union
        Concatenated OOS signals across all WF windows.
    cfg
        Full YAML config. A deep copy with elevated cost values is used
        for the stressed run; the original is not mutated.
    symbol_to_asset
        Mapping for ``simulate_fn``. May be ``None``.
    simulate_fn
        ``(params, signals, cfg, symbol_to_asset) -> (trades_df, metrics)``.
        In production this is ``_simulate_with_params``.
    account_size
        Starting equity. If ``None`` taken from ``cfg["account"]["size"]``.
    stressed_commission_pct, stressed_slippage_pct
        Conservative cost values. Defaults from
        ``cfg["cost_stress"]`` or module-level safe defaults.
    gate_pf, gate_sharpe, gate_dd
        Minimum thresholds for the stressed metrics. All must pass.
    """
    cs_cfg = (cfg or {}).get("cost_stress", {}) or {}
    stressed_commission = (
        stressed_commission_pct
        if stressed_commission_pct is not None
        else float(cs_cfg.get("commission_pct", _DEFAULT_STRESS_COMMISSION_PCT))
    )
    stressed_slippage = (
        stressed_slippage_pct
        if stressed_slippage_pct is not None
        else float(cs_cfg.get("slippage_pct", _DEFAULT_STRESS_SLIPPAGE_PCT))
    )
    g_pf = gate_pf if gate_pf is not None else float(
        cs_cfg.get("gate_pf", _DEFAULT_GATE_PF)
    )
    g_sh = gate_sharpe if gate_sharpe is not None else float(
        cs_cfg.get("gate_sharpe", _DEFAULT_GATE_SHARPE)
    )
    g_dd = gate_dd if gate_dd is not None else float(
        cs_cfg.get("gate_dd", _DEFAULT_GATE_DD)
    )

    if account_size is None:
        account_size = float(
            ((cfg or {}).get("account") or {}).get("size", 100_000.0)
        )

    empty_out: dict[str, Any] = {
        "baseline_pf": 0.0,
        "baseline_sharpe": 0.0,
        "baseline_dd": 0.0,
        "baseline_n_trades": 0,
        "stressed_pf": 0.0,
        "stressed_sharpe": 0.0,
        "stressed_dd": 0.0,
        "stressed_n_trades": 0,
        "pf_delta": 0.0,
        "sharpe_delta": 0.0,
        "dd_delta": 0.0,
        "stressed_commission_pct": float(stressed_commission),
        "stressed_slippage_pct": float(stressed_slippage),
        "gate_pf": float(g_pf),
        "gate_sharpe": float(g_sh),
        "gate_dd": float(g_dd),
        "gate_pf_pass": False,
        "gate_sharpe_pass": False,
        "gate_dd_pass": False,
        "gate_pass": False,
        "reason": "empty_signals",
    }

    if not oos_signals_union:
        return empty_out

    # ── Baseline simulation (original costs) ──
    try:
        base_trades, _base_m = simulate_fn(
            best_params, oos_signals_union, cfg, symbol_to_asset,
        )
    except Exception as exc:  # pragma: no cover — defensive
        out = dict(empty_out)
        out["reason"] = f"baseline_sim_failed: {exc}"
        return out

    base_metrics = _metrics_from_trades(base_trades, account_size)

    # ── Stressed simulation ──
    stressed_cfg = _stressed_config(cfg, stressed_commission, stressed_slippage)
    try:
        stress_trades, _stress_m = simulate_fn(
            best_params, oos_signals_union, stressed_cfg, symbol_to_asset,
        )
    except Exception as exc:  # pragma: no cover — defensive
        out = dict(empty_out)
        out["reason"] = f"stressed_sim_failed: {exc}"
        return out

    stress_metrics = _metrics_from_trades(stress_trades, account_size)

    gate_pf_pass = bool(stress_metrics["pf_real"] >= g_pf)
    gate_sharpe_pass = bool(stress_metrics["sharpe_daily"] >= g_sh)
    # Drawdown is a negative number — "worse than" means more negative.
    gate_dd_pass = bool(stress_metrics["max_drawdown"] >= g_dd)
    combined = gate_pf_pass and gate_sharpe_pass and gate_dd_pass

    return {
        "baseline_pf": float(base_metrics["pf_real"]),
        "baseline_sharpe": float(base_metrics["sharpe_daily"]),
        "baseline_dd": float(base_metrics["max_drawdown"]),
        "baseline_n_trades": int(base_metrics["n_trades"]),
        "stressed_pf": float(stress_metrics["pf_real"]),
        "stressed_sharpe": float(stress_metrics["sharpe_daily"]),
        "stressed_dd": float(stress_metrics["max_drawdown"]),
        "stressed_n_trades": int(stress_metrics["n_trades"]),
        "pf_delta": float(
            stress_metrics["pf_real"] - base_metrics["pf_real"]
        ),
        "sharpe_delta": float(
            stress_metrics["sharpe_daily"] - base_metrics["sharpe_daily"]
        ),
        "dd_delta": float(
            stress_metrics["max_drawdown"] - base_metrics["max_drawdown"]
        ),
        "stressed_commission_pct": float(stressed_commission),
        "stressed_slippage_pct": float(stressed_slippage),
        "gate_pf": float(g_pf),
        "gate_sharpe": float(g_sh),
        "gate_dd": float(g_dd),
        "gate_pf_pass": gate_pf_pass,
        "gate_sharpe_pass": gate_sharpe_pass,
        "gate_dd_pass": gate_dd_pass,
        "gate_pass": bool(combined),
        "reason": "ok",
    }


__all__ = ["compute_cost_stress"]
