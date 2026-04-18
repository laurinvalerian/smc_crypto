"""
SSOT constants tests.

Verifies that core.constants is the Single Source of Truth for
COMMISSION, SLIPPAGE, and ALIGNMENT_THRESHOLD, and that all downstream
modules import the same values (Phase 2.1 regression guard).
"""
from __future__ import annotations


def test_core_constants_have_expected_values():
    from core.constants import (
        COMMISSION,
        SLIPPAGE,
        ALIGNMENT_THRESHOLD,
        AAA_PLUS_PLUS_THRESHOLD,
        AAA_PLUS_THRESHOLD,
        LEVERAGE_CAP,
        DEFAULT_RISK_PER_TRADE,
        MAX_RISK_PER_TRADE,
        MAX_PORTFOLIO_HEAT,
    )
    assert COMMISSION == 0.0004
    assert SLIPPAGE == 0.0002
    assert ALIGNMENT_THRESHOLD == 0.78
    assert AAA_PLUS_PLUS_THRESHOLD == 0.88
    assert AAA_PLUS_THRESHOLD == ALIGNMENT_THRESHOLD
    assert LEVERAGE_CAP == 10
    assert DEFAULT_RISK_PER_TRADE == 0.005
    assert MAX_RISK_PER_TRADE == 0.015
    assert MAX_PORTFOLIO_HEAT == 0.06


def test_commission_matches_in_live_multi_bot():
    from core.constants import COMMISSION, SLIPPAGE
    import live_multi_bot as lm
    assert lm.ASSET_COMMISSION["crypto"] == COMMISSION
    assert lm._TRAIN_COMMISSION["crypto"] == COMMISSION
    assert lm._TRAIN_SLIPPAGE["crypto"] == SLIPPAGE


def test_commission_matches_in_generate_rl_data():
    from core.constants import COMMISSION, SLIPPAGE
    from backtest import generate_rl_data as gr
    assert gr.ASSET_COMMISSION["crypto"] == COMMISSION
    assert gr.ASSET_SLIPPAGE["crypto"] == SLIPPAGE


def test_commission_matches_in_optuna_backtester():
    from core.constants import COMMISSION, SLIPPAGE
    from backtest import optuna_backtester as ob
    assert ob.ASSET_COMMISSION["crypto"] == COMMISSION
    assert ob.ASSET_SLIPPAGE["crypto"] == SLIPPAGE


def test_commission_matches_in_paper_grid():
    from core.constants import COMMISSION, SLIPPAGE
    import paper_grid as pg
    assert pg.ASSET_COMMISSION["crypto"] == COMMISSION
    assert pg.SLIPPAGE_PCT == SLIPPAGE


def test_commission_matches_in_replay_adapter():
    from core.constants import COMMISSION
    from exchanges import replay_adapter as ra
    assert ra.DEFAULT_COMMISSION["crypto"] == COMMISSION


def test_asset_commission_is_crypto_only():
    """After Phase 1 strip, all ASSET_COMMISSION dicts contain only 'crypto'."""
    import live_multi_bot as lm
    from backtest import generate_rl_data as gr, optuna_backtester as ob
    import paper_grid as pg
    from exchanges import replay_adapter as ra

    for name, d in [
        ("live_multi_bot.ASSET_COMMISSION", lm.ASSET_COMMISSION),
        ("generate_rl_data.ASSET_COMMISSION", gr.ASSET_COMMISSION),
        ("optuna_backtester.ASSET_COMMISSION", ob.ASSET_COMMISSION),
        ("paper_grid.ASSET_COMMISSION", pg.ASSET_COMMISSION),
        ("replay_adapter.DEFAULT_COMMISSION", ra.DEFAULT_COMMISSION),
    ]:
        assert set(d.keys()) == {"crypto"}, f"{name} should be crypto-only, got {list(d.keys())}"
