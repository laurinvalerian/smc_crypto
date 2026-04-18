"""
Regression guards for Bug_006 and Bug_007.

Bug_006 (trade_journal.py): open_trade pre-populated MFE/MAE dicts with
0.0, defeating the .get(trade_id, fallback) call in close_trade. Trades
that closed before record_bar fired (fast TP fills, missed polls) wrote
max_favorable=0.0 even when the realized exit move was positive.

Bug_007 (generate_rl_data.py): post_tp_max_rr was initialised to tp_rr
(the planned TP level) instead of max(tp_rr, fav), so the TP-hitting
bar's MFE overshoot was never captured — silently re-introducing the
MFE-cap-at-TP that the post-TP continuation feature was added to fix.
"""
from __future__ import annotations

from pathlib import Path


class TestBug006TradeJournalMFEMAE:
    def test_open_trade_does_not_prepopulate_mfe_mae_dicts(self):
        """
        The fix removes the pre-population. This test verifies the source
        no longer contains the two assignments that defeated the fallback.
        """
        src = Path("trade_journal.py").read_text()
        # Neither assignment should appear in the open_trade method body.
        assert "self._max_favorable[trade_id] = 0.0" not in src, (
            "Bug_006 regression: open_trade pre-populates _max_favorable. "
            "This makes .get(trade_id, fallback) in close_trade always return 0.0."
        )
        assert "self._max_adverse[trade_id] = 0.0" not in src, (
            "Bug_006 regression: open_trade pre-populates _max_adverse."
        )

    def test_record_bar_still_uses_defensive_get(self):
        """record_bar must work even without pre-populated keys."""
        src = Path("trade_journal.py").read_text()
        assert "self._max_favorable.get(trade_id, 0.0)" in src, (
            "record_bar must use .get(trade_id, 0.0) so it works without pre-population."
        )
        assert "self._max_adverse.get(trade_id, 0.0)" in src


class TestBug007PostTpMaxRr:
    def test_post_tp_max_rr_init_captures_tp_bar_mfe(self):
        """
        The fix uses max(tp_rr, fav) so the TP-hitting bar's overshoot
        (which triggered TP in the first place) is included in
        post_tp_max_rr. `fav` is the MFE computed in the main pre-TP loop.
        """
        src = Path("backtest/generate_rl_data.py").read_text()
        assert "post_tp_max_rr = max(tp_rr, fav)" in src, (
            "Bug_007 regression: post_tp_max_rr should be init'd to "
            "max(tp_rr, fav) so spike-then-reverse TP fills capture the "
            "wick MFE, not just the planned TP level."
        )
        # The buggy form must not reappear.
        assert "post_tp_max_rr = tp_rr  # starts at TP level" not in src, (
            "Bug_007 regression: old init present again."
        )

    def test_fav_is_computed_before_tp_hit_check(self):
        """
        For the fix to work, `fav` must be in scope at line 1224. Verify
        by checking that the pre-TP MFE computation precedes the tp_hit
        branch in the function body.
        """
        src = Path("backtest/generate_rl_data.py").read_text()
        fav_long_idx = src.find("fav = (bar_high - entry_price) / sl_dist")
        tp_hit_idx = src.find("if tp_hit:")
        assert fav_long_idx != -1, "Expected long-direction fav computation"
        assert tp_hit_idx != -1, "Expected tp_hit branch"
        assert fav_long_idx < tp_hit_idx, (
            "`fav` must be computed before the tp_hit branch so post_tp_max_rr = max(tp_rr, fav) is valid."
        )
