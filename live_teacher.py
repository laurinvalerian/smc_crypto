"""
Live Teacher — Retroactive SMC analysis for closed trades.

Uses NON-CAUSAL SMC indicators (with lookahead) to evaluate
how optimal a trade was compared to perfect hindsight.

Live bot uses compute_smc_indicators_causal (no lookahead).
Teacher uses compute_smc_indicators (WITH lookahead by design).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.smc_multi_style import compute_smc_indicators

logger = logging.getLogger(__name__)

FEEDBACK_PATH = Path("live_results/teacher_feedback.jsonl")


# ═════════════════════════════════════════════════════════════════
#  Analyze a closed trade with perfect-hindsight SMC indicators
# ═════════════════════════════════════════════════════════════════

def analyze_closed_trade(
    trade: dict[str, Any],
    candle_data: list[list[float]],
    symbol: str,
    asset_class: str,
    exit_reason: str = "unknown",
) -> dict[str, Any]:
    """Retroactive SMC analysis on a closed trade (synchronous).

    Called via ``asyncio.to_thread`` from the live bot.

    Parameters
    ----------
    trade : dict
        Closed-trade record with at least ``entry_price``, ``sl``, ``tp``,
        ``direction``, and optionally ``entry_time``.
    candle_data : list[list[float]]
        OHLCV rows from the exchange adapter: ``[ts, o, h, l, c, v]``.
    symbol : str
        Instrument identifier (e.g. ``"BTC/USDT:USDT"``).
    asset_class : str
        One of ``"crypto"``, ``"forex"``, ``"stocks"``, ``"commodities"``.

    Returns
    -------
    dict
        Teacher feedback dict with ``actual`` and ``teacher`` sub-dicts.
    """
    entry_price = trade.get("entry_price", 0.0)
    sl_price = trade.get("sl", 0.0)
    tp_price = trade.get("tp", 0.0)
    direction = trade.get("direction", "long")

    # Degraded result returned when analysis cannot complete.
    fallback: dict[str, Any] = {
        "trade_id": trade.get("rl_trade_id", "unknown"),
        "symbol": symbol,
        "asset_class": asset_class,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actual": {
            "entry": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "direction": direction,
            "style": trade.get("style", "day"),
            "outcome": trade.get("outcome", "unknown"),
            "pnl_pct": trade.get("pnl_pct", 0.0),
            "exit_reason": exit_reason,
        },
        "teacher": {
            "optimal_entry": None,
            "entry_offset_pips": None,
            "optimal_sl": None,
            "sl_could_be_tighter": None,
            "optimal_tp": None,
            "tp_realistic": None,
            "confluences_at_entry": [],
            "missed_setups_in_window": 0,
            "grade": "N/A",
        },
    }

    try:
        # ── 1. Build DataFrame ────────────────────────────────────
        if not candle_data or len(candle_data) < 20:
            logger.warning("Teacher: insufficient candle data for %s (%d rows)", symbol, len(candle_data) if candle_data else 0)
            return fallback

        df = pd.DataFrame(candle_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < 20:
            return fallback

        # ── 2. Compute non-causal SMC indicators ──────────────────
        smc = compute_smc_indicators(
            df,
            swing_length=10,
            fvg_threshold=0.0,
            ob_lookback=20,
            liq_range_pct=50,
        )

        bos_choch = smc.get("bos_choch")
        fvg_df = smc.get("fvg")
        ob_df = smc.get("order_blocks")
        liq_df = smc.get("liquidity")
        swing_df = smc.get("swing_highs_lows")

        n = len(df)

        # ── 3. Locate entry candle ────────────────────────────────
        entry_time = trade.get("entry_time")
        if entry_time is not None and "timestamp" in df.columns:
            ts_series = pd.to_numeric(df["timestamp"], errors="coerce")
            entry_ts = float(entry_time) if not isinstance(entry_time, (int, float)) else entry_time
            entry_idx = int((ts_series - entry_ts).abs().idxmin())
        else:
            # Fallback: find candle closest to entry_price
            entry_idx = int((df["close"] - entry_price).abs().idxmin())

        # ── 4a. Entry quality — confluence check ──────────────────
        confluences: list[str] = []

        # BOS at or near entry
        if bos_choch is not None and not bos_choch.empty:
            window = range(max(0, entry_idx - 3), min(n, entry_idx + 2))
            for i in window:
                bos_val = bos_choch["BOS"].iat[i] if "BOS" in bos_choch.columns else 0
                choch_val = bos_choch["CHOCH"].iat[i] if "CHOCH" in bos_choch.columns else 0
                if pd.notna(bos_val) and bos_val != 0:
                    confluences.append(f"BOS_{'up' if bos_val > 0 else 'down'}")
                if pd.notna(choch_val) and choch_val != 0:
                    confluences.append(f"CHoCH_{'up' if choch_val > 0 else 'down'}")

        # FVG at or near entry
        if fvg_df is not None and not fvg_df.empty:
            fvg_col = "FVG" if "FVG" in fvg_df.columns else None
            if fvg_col:
                window = range(max(0, entry_idx - 3), min(n, entry_idx + 2))
                for i in window:
                    fval = fvg_df[fvg_col].iat[i]
                    if pd.notna(fval) and fval != 0:
                        confluences.append(f"FVG_{'bull' if fval > 0 else 'bear'}")

        # OB at or near entry
        if ob_df is not None and not ob_df.empty:
            ob_col = "OB" if "OB" in ob_df.columns else None
            if ob_col:
                window = range(max(0, entry_idx - 3), min(n, entry_idx + 2))
                for i in window:
                    oval = ob_df[ob_col].iat[i]
                    if pd.notna(oval) and oval != 0:
                        confluences.append(f"OB_{'bull' if oval > 0 else 'bear'}")

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_confluences: list[str] = []
        for c in confluences:
            if c not in seen:
                seen.add(c)
                unique_confluences.append(c)
        confluences = unique_confluences

        # ── 4b. Optimal entry — nearest FVG/OB to actual entry ────
        optimal_entry = entry_price  # default: actual was optimal
        if ob_df is not None and not ob_df.empty:
            for col_name in ("OBBottom", "OBTop"):
                if col_name in ob_df.columns:
                    ob_levels = ob_df[col_name].dropna()
                    if not ob_levels.empty:
                        if direction == "long" and col_name == "OBBottom":
                            nearby = ob_levels[(ob_levels > 0) & (ob_levels <= entry_price)]
                            if not nearby.empty:
                                optimal_entry = float(nearby.iloc[-1])
                        elif direction == "short" and col_name == "OBTop":
                            nearby = ob_levels[(ob_levels > 0) & (ob_levels >= entry_price)]
                            if not nearby.empty:
                                optimal_entry = float(nearby.iloc[-1])

        # Pip divisor: forex pairs ~0.0001, JPY ~0.01, crypto/stocks ~price-based
        pip_size = _pip_size(symbol, asset_class)
        offset_pips = round((entry_price - optimal_entry) / pip_size, 1) if pip_size > 0 else 0.0

        # ── 4c. SL analysis — compare to nearest structure level ──
        optimal_sl = sl_price
        sl_tighter = False
        if swing_df is not None and not swing_df.empty:
            lows = df["low"].values
            highs = df["high"].values
            hl_col = "HighLow" if "HighLow" in swing_df.columns else None
            if hl_col is not None:
                if direction == "long":
                    # Find nearest swing low below entry
                    mask = (swing_df[hl_col] == -1)
                    swing_lows = df.loc[mask.values, "low"] if mask.any() else pd.Series(dtype=float)
                    below_entry = swing_lows[swing_lows < entry_price]
                    if not below_entry.empty:
                        nearest_low = float(below_entry.iloc[-1])
                        optimal_sl = nearest_low
                        sl_tighter = abs(entry_price - optimal_sl) < abs(entry_price - sl_price)
                else:
                    mask = (swing_df[hl_col] == 1)
                    swing_highs = df.loc[mask.values, "high"] if mask.any() else pd.Series(dtype=float)
                    above_entry = swing_highs[swing_highs > entry_price]
                    if not above_entry.empty:
                        nearest_high = float(above_entry.iloc[-1])
                        optimal_sl = nearest_high
                        sl_tighter = abs(entry_price - optimal_sl) < abs(entry_price - sl_price)

        # ── 4d. TP analysis — next liquidity / OB target ──────────
        optimal_tp = tp_price
        tp_realistic = True
        if liq_df is not None and not liq_df.empty:
            liq_col = "Liquidity" if "Liquidity" in liq_df.columns else None
            liq_level_col = "Level" if "Level" in liq_df.columns else None
            target_col = liq_level_col or liq_col
            if target_col and target_col in liq_df.columns:
                levels = liq_df[target_col].dropna()
                levels = levels[levels > 0]
                if not levels.empty:
                    if direction == "long":
                        above = levels[levels > entry_price]
                        if not above.empty:
                            optimal_tp = float(above.iloc[0])
                    else:
                        below = levels[levels < entry_price]
                        if not below.empty:
                            optimal_tp = float(below.iloc[-1])

        # TP realistic check: was TP within the range seen in data?
        price_max = float(df["high"].max())
        price_min = float(df["low"].min())
        if direction == "long":
            tp_realistic = tp_price <= price_max * 1.02
        else:
            tp_realistic = tp_price >= price_min * 0.98

        # ── 4e. Missed setups — FVG+BOS confluences not taken ─────
        missed_count = 0
        if bos_choch is not None and fvg_df is not None and not bos_choch.empty and not fvg_df.empty:
            bos_col = "BOS" if "BOS" in bos_choch.columns else None
            fvg_col = "FVG" if "FVG" in fvg_df.columns else None
            if bos_col and fvg_col:
                for i in range(n):
                    bv = bos_choch[bos_col].iat[i]
                    fv = fvg_df[fvg_col].iat[i]
                    if pd.notna(bv) and bv != 0 and pd.notna(fv) and fv != 0:
                        # Same direction confluence
                        if (bv > 0 and fv > 0) or (bv < 0 and fv < 0):
                            missed_count += 1
            # The trade itself used one setup, so subtract 1
            if missed_count > 0:
                missed_count = max(0, missed_count - 1)

        # ── 4f. Grade ─────────────────────────────────────────────
        grade = _compute_grade(
            confluences=confluences,
            sl_tighter=sl_tighter,
            tp_realistic=tp_realistic,
            offset_pips=offset_pips,
            pnl_pct=trade.get("pnl_pct", 0.0),
            exit_reason=exit_reason,
        )

        return {
            "trade_id": trade.get("rl_trade_id", "unknown"),
            "symbol": symbol,
            "asset_class": asset_class,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actual": {
                "entry": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "direction": direction,
                "style": trade.get("style", "day"),
                "outcome": trade.get("outcome", "unknown"),
                "pnl_pct": trade.get("pnl_pct", 0.0),
                "exit_reason": exit_reason,
            },
            "teacher": {
                "optimal_entry": optimal_entry,
                "entry_offset_pips": offset_pips,
                "optimal_sl": optimal_sl,
                "sl_could_be_tighter": sl_tighter,
                "optimal_tp": optimal_tp,
                "tp_realistic": tp_realistic,
                "confluences_at_entry": confluences,
                "missed_setups_in_window": missed_count,
                "grade": grade,
            },
        }

    except Exception:
        logger.exception("Teacher analysis failed for %s %s", symbol, trade.get("rl_trade_id", "?"))
        return fallback


# ═════════════════════════════════════════════════════════════════
#  Persist feedback
# ═════════════════════════════════════════════════════════════════

def save_feedback(feedback: dict[str, Any], path: Path | None = None) -> None:
    """Append a single feedback record as a JSON line."""
    target = path or FEEDBACK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback, default=str) + "\n")


# ═════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════

def _pip_size(symbol: str, asset_class: str) -> float:
    """Return approximate pip size for offset calculation."""
    sym = symbol.upper()
    if asset_class == "forex":
        if "JPY" in sym:
            return 0.01
        return 0.0001
    if asset_class == "crypto":
        # Use 0.01 for most crypto; rough but serviceable
        return 0.01
    if asset_class == "commodities":
        if "XAU" in sym:
            return 0.1
        if "XAG" in sym:
            return 0.01
        return 0.01
    # stocks
    return 0.01


def _compute_grade(
    confluences: list[str],
    sl_tighter: bool,
    tp_realistic: bool,
    offset_pips: float,
    pnl_pct: float,
    exit_reason: str = "unknown",
) -> str:
    """Assign a letter grade based on teacher analysis."""
    score = 0

    # Confluence quality (0-3)
    has_bos = any("BOS" in c for c in confluences)
    has_fvg = any("FVG" in c for c in confluences)
    has_ob = any("OB" in c for c in confluences)
    score += int(has_bos) + int(has_fvg) + int(has_ob)

    # SL placement (0-1)
    if not sl_tighter:
        score += 1  # SL was already tight

    # TP realism (0-1)
    if tp_realistic:
        score += 1

    # Entry precision (0-1)
    if abs(offset_pips) < 5:
        score += 1

    # Outcome bonus (0-1)
    if pnl_pct > 0:
        score += 1

    # Exit-reason adjustments
    if exit_reason == "timeout":
        score -= 1  # Timeout exits are noisy / suboptimal
    if exit_reason == "tp_hit" and pnl_pct > 0:
        score += 1  # Clean TP exit bonus

    # Map score to grade (max 8)
    if score >= 7:
        return "A+"
    if score >= 6:
        return "A"
    if score >= 5:
        return "B+"
    if score >= 4:
        return "B"
    if score >= 2:
        return "C"
    return "D"
