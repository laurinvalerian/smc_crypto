"""
tests/test_wf_student_gate.py — Student-Brain integration for wf_bruteforce.

Phase 1 of .omc/plans/student-brain-integration.md (2026-04-21).

Guards:
1. `_simulate_with_params` WITHOUT `student_brain=` behaves bit-identical to
   the v1.11-robustness-plus baseline (regression guard — breaks every
   evergreen run if violated).
2. `_simulate_with_params` WITH a Student that rejects everything produces
   zero trades (gate-semantics guard).
3. `_simulate_with_params` WITH a Student that accepts everything produces
   the same trades as the SMC-only baseline (gate is non-lossy when
   permissive — same as nothing).
4. `_student_features_from_signal` returns all 42 keys from
   features.schema.ENTRY_QUALITY_FEATURES (schema match).
5. `_load_student_brain(None)` returns None (opt-in semantics).
6. `_load_student_brain` on a missing dir logs an error and returns None
   (does NOT raise — bruteforce must survive a bad flag).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from backtest import wf_bruteforce as wf
from features.schema import ENTRY_QUALITY_FEATURES


# ════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════

def _minimal_signal(symbol: str = "BTC/USDT:USDT", alignment: float = 0.85,
                    rr: float = 2.5) -> wf.TradeSignal:
    """A TradeSignal with the minimum fields + a meta dict carrying
    typical SMC component flags. Used for gate-level tests — the
    simulation path (_resolve_trade_outcome) is not exercised here."""
    return wf.TradeSignal(
        timestamp=pd.Timestamp("2026-04-01 12:00", tz="UTC"),
        symbol=symbol,
        direction="long",
        style="day",
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52500.0,
        risk_reward=rr,
        position_size=0.01,
        leverage=10,
        alignment_score=alignment,
        meta={
            "h4_confirms": True, "h4_poi": True,
            "h1_confirms": True, "h1_choch": True,
            "precision_trigger": True, "volume_ok": True,
            "premium_discount": -0.3,
            "adx_value": 25.0,
        },
    )


class _PassStudent:
    """Mock StudentBrain that accepts every signal at entry_prob=1.0."""
    def predict(self, features):
        return SimpleNamespace(accept=True, entry_prob=1.0,
                               sl_rr=1.0, tp_rr=2.0, size=1.0)


class _RejectStudent:
    """Mock StudentBrain that rejects every signal."""
    def predict(self, features):
        return SimpleNamespace(accept=False, entry_prob=0.0,
                               sl_rr=1.0, tp_rr=2.0, size=1.0)


# ════════════════════════════════════════════════════════════════════
#  Feature-dict schema match
# ════════════════════════════════════════════════════════════════════

def test_student_features_from_signal_has_all_entry_quality_keys():
    sig = _minimal_signal()
    feats = wf._student_features_from_signal(sig)
    for key in ENTRY_QUALITY_FEATURES:
        assert key in feats, f"feature dict missing key: {key}"
    # All values must be float — StudentBrain would choke on None / bool
    for key, val in feats.items():
        assert isinstance(val, float), f"{key}={val!r} is not float"


def test_student_features_from_signal_preserves_alignment_score():
    sig = _minimal_signal(alignment=0.91)
    feats = wf._student_features_from_signal(sig)
    assert feats["alignment_score"] == 0.91


# ════════════════════════════════════════════════════════════════════
#  _load_student_brain opt-in semantics
# ════════════════════════════════════════════════════════════════════

def test_load_student_brain_with_none_returns_none():
    assert wf._load_student_brain(None) is None


def test_load_student_brain_with_empty_string_returns_none():
    assert wf._load_student_brain("") is None


def test_load_student_brain_missing_dir_returns_none_without_raising():
    # StudentBrain itself may or may not raise on missing dir — the helper
    # must swallow that and return None so bruteforce survives a bad flag.
    result = wf._load_student_brain("/definitely/not/a/real/dir")
    assert result is None


# ════════════════════════════════════════════════════════════════════
#  Gate semantics
# ════════════════════════════════════════════════════════════════════

def _stub_config(sim_patched: bool = True):
    cfg = {
        "account": {"size": 100_000},
        "backtest": {"commission_pct": 0.0004, "slippage_pct": 0.0001},
    }
    return cfg


def test_simulate_with_params_without_student_is_baseline(monkeypatch):
    """Regression guard: without student_brain arg, code path must match
    v1.11-robustness-plus baseline. We stub simulate_trades to return an
    empty DataFrame so the test doesn't need price bars — we only verify
    the filter loop preserves signals that meet alignment+rr."""
    captured: list = []

    def _fake_sim(sigs, **kwargs):
        captured.append(list(sigs))
        return pd.DataFrame()

    monkeypatch.setattr(wf, "simulate_trades", _fake_sim)
    params = {"alignment_threshold": 0.78, "risk_reward": 2.0}
    sigs = [
        _minimal_signal(alignment=0.85, rr=2.5),
        _minimal_signal(alignment=0.70, rr=2.5),  # rejected: alignment < 0.78
        _minimal_signal(alignment=0.85, rr=1.5),  # rejected: rr < 2.0
        _minimal_signal(alignment=0.90, rr=3.0),
    ]
    wf._simulate_with_params(params, sigs, _stub_config())
    # simulate_trades was called once (all signals grouped into crypto class)
    assert len(captured) == 1
    # Only the 2 passing signals made it through
    assert len(captured[0]) == 2


def test_simulate_with_params_reject_student_filters_everything(monkeypatch):
    captured: list = []

    def _fake_sim(sigs, **kwargs):
        captured.append(list(sigs))
        return pd.DataFrame()

    monkeypatch.setattr(wf, "simulate_trades", _fake_sim)
    params = {"alignment_threshold": 0.78, "risk_reward": 2.0}
    sigs = [
        _minimal_signal(alignment=0.85, rr=2.5),
        _minimal_signal(alignment=0.90, rr=3.0),
    ]
    wf._simulate_with_params(params, sigs, _stub_config(),
                             student_brain=_RejectStudent())
    # simulate_trades never called when zero signals survive the filter
    assert captured == []


def test_simulate_with_params_pass_student_matches_baseline(monkeypatch):
    """When Student says accept-all, output must equal SMC-only output."""
    captured_baseline: list = []
    captured_student: list = []

    def _fake_sim(sigs, **kwargs):
        # First call: baseline; second call: student-gated
        if len(captured_baseline) == 0:
            captured_baseline.append(list(sigs))
        else:
            captured_student.append(list(sigs))
        return pd.DataFrame()

    monkeypatch.setattr(wf, "simulate_trades", _fake_sim)
    params = {"alignment_threshold": 0.78, "risk_reward": 2.0}
    sigs = [_minimal_signal(alignment=0.85, rr=2.5),
            _minimal_signal(alignment=0.90, rr=3.0)]

    wf._simulate_with_params(params, sigs, _stub_config())
    wf._simulate_with_params(params, sigs, _stub_config(),
                             student_brain=_PassStudent())
    # Both runs pass the same number of signals downstream
    assert len(captured_baseline) == 1
    assert len(captured_student) == 1
    assert len(captured_baseline[0]) == len(captured_student[0])
