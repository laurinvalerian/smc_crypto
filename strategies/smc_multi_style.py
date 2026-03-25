"""
═══════════════════════════════════════════════════════════════════
 strategies/smc_multi_style.py
 ─────────────────────────────
 Smart-Money-Concepts (SMC / ICT) Day-Trading-Only strategy,
 optimised for high-volatility Crypto Perpetuals (BTC, SOL, ETH …).

 Top-Down Flow (2025/2026 community best-practice):
   1. Daily Bias          → 1D BOS/CHoCH
   2. Structure Confirm   → 1H alignment
   3. Entry Zone          → 15m FVG + OB + Liquidity
   4. Decision & Trigger  → 5m bar-by-bar (BOS/CHoCH or FVG mitigation)

 All SMC indicators are temporally sliced to the current 5m bar
 (no future-peeking).

 Usage (from project root):
     from strategies.smc_multi_style import SMCMultiStyleStrategy
     strat = SMCMultiStyleStrategy(config, params)
     signals = strat.generate_signals(symbol)
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from smartmoneyconcepts.smc import smc as smc_lib

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TradeSignal:
    """A single trade signal produced by the strategy."""

    timestamp: pd.Timestamp
    symbol: str
    direction: str            # "long" | "short"
    style: str                # "day" (only style)
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    position_size: float      # In base asset
    leverage: int
    alignment_score: float
    meta: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
#  Timeframe helpers
# ═══════════════════════════════════════════════════════════════════

# Map config shorthand → pandas offset
TF_TO_OFFSET: dict[str, str] = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}


def resample_ohlcv(df_1m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 1 m OHLCV to *target_tf*."""
    offset = TF_TO_OFFSET.get(target_tf, target_tf)
    df = df_1m.set_index("timestamp").resample(offset).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"]).reset_index()
    return df


# ═══════════════════════════════════════════════════════════════════
#  SMC indicator wrappers
# ═══════════════════════════════════════════════════════════════════

def _to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLC columns are float and indexed properly for the SMC lib."""
    out = df[["open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = out[col].astype(float)
    out = out.reset_index(drop=True)
    return out


def compute_smc_indicators(
    df: pd.DataFrame,
    swing_length: int = 8,
    fvg_threshold: float = 0.0004,
    ob_lookback: int = 15,
    liq_range_pct: float = 0.005,
) -> dict[str, Any]:
    """
    Compute all SMC indicators on a single-timeframe OHLCV DataFrame
    using the smartmoneyconcepts library (smc class).

    Returns a dict with keys:
        swing_highs_lows, fvg, order_blocks, bos_choch, liquidity
    Each value is a pandas DataFrame aligned to the input index.
    """
    ohlc = _to_ohlc(df)
    results: dict[str, Any] = {}

    # ── Swing Highs / Lows (combined) ─────────────────────────────
    swing_hl = smc_lib.swing_highs_lows(ohlc, swing_length=swing_length)
    results["swing_highs_lows"] = swing_hl

    # ── Fair Value Gaps (FVG) ─────────────────────────────────────
    fvg_data = smc_lib.fvg(ohlc)
    results["fvg"] = fvg_data

    # ── Order Blocks (requires swing_highs_lows) ──────────────────
    ob_data = smc_lib.ob(ohlc, swing_hl)
    results["order_blocks"] = ob_data

    # ── Break of Structure / Change of Character ──────────────────
    bos_choch = smc_lib.bos_choch(ohlc, swing_hl)
    results["bos_choch"] = bos_choch

    # ── Liquidity (requires swing_highs_lows) ─────────────────────
    liquidity = smc_lib.liquidity(ohlc, swing_hl, range_percent=liq_range_pct)
    results["liquidity"] = liquidity

    return results


# ═══════════════════════════════════════════════════════════════════
#  Bias & structure helpers (temporally-safe slicing)
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  Precomputed running arrays for temporal slicing (no future peek)
# ═══════════════════════════════════════════════════════════════════

def _compute_ema_bias(df_1d: pd.DataFrame, period: int = 200) -> np.ndarray:
    """Fallback: EMA200 Trend auf 1D (bullish wenn close > EMA)."""
    if len(df_1d) < period:
        return np.zeros(len(df_1d), dtype=np.int8)
    ema = df_1d["close"].ewm(span=period, adjust=False).mean()
    bias = np.where(df_1d["close"] > ema, 1, -1)
    return bias.astype(np.int8)


def _precompute_running_bias(indicators: dict[str, Any], df_1d: pd.DataFrame) -> np.ndarray:
    """
    Primary: BOS/CHoCH (wie bisher)
    Fallback: EMA200 Trend, wenn BOS/CHoCH neutral bleibt.
    """
    bos_choch = indicators.get("bos_choch")
    n = len(df_1d)
    running = np.zeros(n, dtype=np.int8)
    last_sig = 0

    # 1. BOS/CHoCH (primary)
    if bos_choch is not None and not bos_choch.empty:
        for i in range(n):
            choch = bos_choch["CHOCH"].iat[i]
            bos = bos_choch["BOS"].iat[i]
            val = choch if (pd.notna(choch) and choch != 0) else bos
            if pd.notna(val) and val != 0:
                last_sig = 1 if val > 0 else -1
            running[i] = last_sig

    # 2. EMA-Fallback nur wo immer noch neutral
    ema_bias = _compute_ema_bias(df_1d)
    mask = running == 0
    running[mask] = ema_bias[mask]

    return running


def _precompute_bias_strong(indicators: dict[str, Any], df_1d: pd.DataFrame) -> np.ndarray:
    """
    Track whether the daily bias came from BOS/CHoCH (strong=True)
    or just from EMA fallback (strong=False).
    Returns a boolean array: True where bias is from BOS/CHoCH.
    """
    bos_choch = indicators.get("bos_choch")
    n = len(df_1d)
    strong = np.zeros(n, dtype=bool)
    has_bos_choch = False

    if bos_choch is not None and not bos_choch.empty:
        for i in range(n):
            choch = bos_choch["CHOCH"].iat[i]
            bos = bos_choch["BOS"].iat[i]
            val = choch if (pd.notna(choch) and choch != 0) else bos
            if pd.notna(val) and val != 0:
                has_bos_choch = True
            strong[i] = has_bos_choch

    return strong


def _precompute_h1_choch_mask(indicators: dict[str, Any]) -> np.ndarray:
    """
    Track whether there has been a recent 1H CHoCH (stronger than BOS).
    Returns a boolean running mask: True if the most recent 1H signal was a CHoCH.
    """
    bos_choch = indicators.get("bos_choch")
    if bos_choch is None or bos_choch.empty:
        return np.zeros(0, dtype=bool)

    n = len(bos_choch)
    mask = np.zeros(n, dtype=bool)
    last_was_choch = False

    for i in range(n):
        choch = bos_choch["CHOCH"].iat[i]
        bos = bos_choch["BOS"].iat[i]
        if pd.notna(choch) and choch != 0:
            last_was_choch = True
        elif pd.notna(bos) and bos != 0:
            last_was_choch = False
        mask[i] = last_was_choch

    return mask


def _check_h4_poi(indicators_4h: dict[str, Any], df_4h: pd.DataFrame,
                   bias: str, valid_len: int) -> bool:
    """Check if there's a recent 4H FVG or OB aligned with bias (last 6 bars)."""
    if valid_len <= 0:
        return False
    max_lookback = 6

    for data_key in ("fvg", "order_blocks"):
        data = indicators_4h.get(data_key)
        if data is None or data.empty:
            continue
        end = min(valid_len, len(data))
        scan_start = max(0, end - max_lookback)
        for idx in range(end - 1, scan_start - 1, -1):
            row = data.iloc[idx]
            dir_key = "FVG" if data_key == "fvg" else "OB"
            val = row.get(dir_key, 0)
            if pd.isna(val) or val == 0:
                continue
            if bias == "bullish" and val > 0:
                return True
            if bias == "bearish" and val < 0:
                return True
    return False


def _compute_discount_premium(
    ind_4h: dict[str, Any] | None,
    df_4h: pd.DataFrame,
    decision_df: pd.DataFrame,
    vlen_4h: np.ndarray,
) -> np.ndarray:
    """
    Classify each 5m bar as discount (-1), premium (+1), or neutral (0)
    based on the 4H swing range.

    Logic (no future peek):
    - Track last confirmed 4H swing high and swing low
    - Midpoint = (last_swing_high + last_swing_low) / 2
    - Close > midpoint → premium (+1), close < midpoint → discount (-1)
    - If no valid range yet → neutral (0)
    """
    n = len(decision_df)
    result = np.zeros(n, dtype=np.int8)

    if ind_4h is None or df_4h.empty:
        return result

    swing_hl = ind_4h.get("swing_highs_lows")
    if swing_hl is None or swing_hl.empty:
        return result

    # Build running last swing high / low from 4H data
    n_4h = len(swing_hl)
    last_sh = np.full(n_4h, np.nan)  # last swing high price
    last_sl = np.full(n_4h, np.nan)  # last swing low price
    running_sh = np.nan
    running_sl = np.nan

    for j in range(n_4h):
        hl = swing_hl["HighLow"].iat[j]
        lvl = swing_hl["Level"].iat[j]
        if pd.notna(hl) and pd.notna(lvl):
            if hl > 0:  # swing high
                running_sh = lvl
            elif hl < 0:  # swing low
                running_sl = lvl
        last_sh[j] = running_sh
        last_sl[j] = running_sl

    # Map to 5m bars using vlen_4h (temporal index)
    for i in range(n):
        vl = int(vlen_4h[i])
        if vl <= 0:
            continue
        idx = min(vl - 1, n_4h - 1)
        sh = last_sh[idx]
        sl = last_sl[idx]
        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            continue
        midpoint = (sh + sl) / 2.0
        close_price = float(decision_df["close"].iat[i])
        if close_price > midpoint:
            result[i] = 1   # premium
        elif close_price < midpoint:
            result[i] = -1  # discount

    return result


def _find_structure_tp_OLD(
    indicators: dict[str, dict[str, Any]],
    entry_price: float,
    bias: str,
    sl_dist: float,
    vlen_4h: int,
    vlen_1h: int,
    min_rr: float = 1.5,
) -> tuple[float, str]:
    """
    Find take-profit from market structure (Liquidity → FVG → OB).

    Searches 4H then 1H for the nearest opposing structure level.
    Like a real SMC trader: TP at the next liquidity pool, FVG, or OB.

    Returns (tp_price, tp_source) or (0.0, "none") if nothing found.
    """
    is_long = bias == "bullish"
    min_tp_dist = sl_dist * min_rr  # minimum TP distance for acceptable RR

    # Search order: Liquidity 4H → 1H → FVG 4H → 1H → OB 4H → 1H
    search_plan: list[tuple[str, str, str]] = [
        ("4h", "liquidity", "liq_4h"),
        ("1h", "liquidity", "liq_1h"),
        ("4h", "fvg", "fvg_4h"),
        ("1h", "fvg", "fvg_1h"),
        ("4h", "order_blocks", "ob_4h"),
        ("1h", "order_blocks", "ob_1h"),
    ]

    for tf, ind_key, source_label in search_plan:
        ind = indicators.get(tf)
        if ind is None:
            continue

        vlen = vlen_4h if tf == "4h" else vlen_1h
        if vlen <= 0:
            continue

        data = ind.get(ind_key)
        if data is None or data.empty:
            continue

        # Only look at bars we can see (no future peek)
        visible = data.iloc[:vlen]

        if ind_key == "liquidity":
            # Liquidity has: Liquidity (direction), Level (price), Swept
            levels = visible.dropna(subset=["Level"])
            if levels.empty:
                continue
            # Filter unswiped liquidity
            if "Swept" in levels.columns:
                levels = levels[levels["Swept"] != True]  # noqa: E712
            if levels.empty:
                continue

            if is_long:
                # Find nearest liquidity ABOVE entry
                candidates = levels[levels["Level"] > entry_price + min_tp_dist]
            else:
                candidates = levels[levels["Level"] < entry_price - min_tp_dist]

            if candidates.empty:
                continue

            # Nearest level
            if is_long:
                tp = float(candidates["Level"].min())
            else:
                tp = float(candidates["Level"].max())
            return tp, source_label

        elif ind_key == "fvg":
            # FVG has: FVG (+1 bullish, -1 bearish), Top, Bottom, MitigatedIndex
            fvgs = visible.dropna(subset=["FVG"])
            if fvgs.empty:
                continue

            # For long TP: find bearish FVG (-1) above entry (resistance)
            # For short TP: find bullish FVG (+1) below entry (support)
            if is_long:
                opposing = fvgs[fvgs["FVG"] == -1]  # bearish FVG = resistance
                opposing = opposing[opposing["Bottom"] > entry_price + min_tp_dist]
                # Prefer unmitigated
                if "MitigatedIndex" in opposing.columns:
                    unmitigated = opposing[opposing["MitigatedIndex"].isna()]
                    if not unmitigated.empty:
                        opposing = unmitigated
                if opposing.empty:
                    continue
                tp = float(opposing["Bottom"].min())  # nearest bottom of bearish FVG
            else:
                opposing = fvgs[fvgs["FVG"] == 1]  # bullish FVG = support
                opposing = opposing[opposing["Top"] < entry_price - min_tp_dist]
                if "MitigatedIndex" in opposing.columns:
                    unmitigated = opposing[opposing["MitigatedIndex"].isna()]
                    if not unmitigated.empty:
                        opposing = unmitigated
                if opposing.empty:
                    continue
                tp = float(opposing["Top"].max())  # nearest top of bullish FVG
            return tp, source_label

        elif ind_key == "order_blocks":
            # OB has: OB (+1 bullish, -1 bearish), Top, Bottom, MitigatedIndex
            obs = visible.dropna(subset=["OB"])
            if obs.empty:
                continue

            if is_long:
                opposing = obs[obs["OB"] == -1]  # bearish OB = resistance
                opposing = opposing[opposing["Bottom"] > entry_price + min_tp_dist]
                if "MitigatedIndex" in opposing.columns:
                    unmitigated = opposing[opposing["MitigatedIndex"].isna()]
                    if not unmitigated.empty:
                        opposing = unmitigated
                if opposing.empty:
                    continue
                tp = float(opposing["Bottom"].min())
            else:
                opposing = obs[obs["OB"] == 1]  # bullish OB = support
                opposing = opposing[opposing["Top"] < entry_price - min_tp_dist]
                if "MitigatedIndex" in opposing.columns:
                    unmitigated = opposing[opposing["MitigatedIndex"].isna()]
                    if not unmitigated.empty:
                        opposing = unmitigated
                if opposing.empty:
                    continue
                tp = float(opposing["Top"].max())
            return tp, source_label

    # Nothing found
    return 0.0, "none"


# ═══════════════════════════════════════════════════════════════════
#  Lookahead-safe structure TP helpers (computed from raw OHLC only)
# ═══════════════════════════════════════════════════════════════════


def _find_swing_highs_lows(
    high: np.ndarray, low: np.ndarray, swing_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized left-side-only swing detection. No future bars needed.

    A swing high at bar i: high[i] >= max(high[i-swing_length : i])
    A swing low  at bar i: low[i]  <= min(low[i-swing_length : i])

    Returns boolean arrays (same length as input) marking swing points.
    """
    n = len(high)
    is_swing_high = np.zeros(n, dtype=np.bool_)
    is_swing_low = np.zeros(n, dtype=np.bool_)

    if n <= swing_length:
        return is_swing_high, is_swing_low

    # Rolling max of high over the PREVIOUS swing_length bars (not including current)
    # Use pandas for efficient rolling, then compare.
    high_s = pd.Series(high)
    low_s = pd.Series(low)

    # rolling window=swing_length on SHIFTED series (so window covers [i-swing_length, i-1])
    roll_max = high_s.shift(1).rolling(window=swing_length, min_periods=swing_length).max()
    roll_min = low_s.shift(1).rolling(window=swing_length, min_periods=swing_length).min()

    is_swing_high = (high_s >= roll_max).values & ~np.isnan(roll_max.values)
    is_swing_low = (low_s <= roll_min).values & ~np.isnan(roll_min.values)

    return is_swing_high, is_swing_low


def _find_liquidity_tp(
    visible_high: np.ndarray,
    visible_low: np.ndarray,
    entry_price: float,
    is_long: bool,
    min_tp_dist: float,
    swing_highs: np.ndarray,
    swing_lows: np.ndarray,
) -> float:
    """
    Find TP at nearest unswept swing high (long) or swing low (short).

    Uses pre-computed swing arrays. "Unswept" means no subsequent bar has
    traded through the swing level.

    Returns TP price, or 0.0 if nothing found.
    """
    n = len(visible_high)

    if is_long:
        # Find swing highs above entry + min_tp_dist
        sh_indices = np.where(swing_highs)[0]
        if len(sh_indices) == 0:
            return 0.0

        sh_levels = visible_high[sh_indices]
        min_price = entry_price + min_tp_dist

        # Filter: above min price
        mask = sh_levels > min_price
        sh_indices = sh_indices[mask]
        sh_levels = sh_levels[mask]
        if len(sh_indices) == 0:
            return 0.0

        # Check unswept: no bar AFTER the swing high has high > swing level
        best_tp = 0.0
        best_dist = np.inf
        for idx, level in zip(sh_indices, sh_levels):
            dist = level - entry_price
            if dist >= best_dist:
                continue  # already have a closer one
            # Check if swept: any bar after idx has high > level
            if idx + 1 < n:
                if np.any(visible_high[idx + 1:] > level):
                    continue  # swept
            best_tp = level
            best_dist = dist

        return best_tp

    else:
        # Find swing lows below entry - min_tp_dist
        sl_indices = np.where(swing_lows)[0]
        if len(sl_indices) == 0:
            return 0.0

        sl_levels = visible_low[sl_indices]
        max_price = entry_price - min_tp_dist

        mask = sl_levels < max_price
        sl_indices = sl_indices[mask]
        sl_levels = sl_levels[mask]
        if len(sl_indices) == 0:
            return 0.0

        best_tp = 0.0
        best_dist = np.inf
        for idx, level in zip(sl_indices, sl_levels):
            dist = entry_price - level
            if dist >= best_dist:
                continue
            if idx + 1 < n:
                if np.any(visible_low[idx + 1:] < level):
                    continue  # swept
            best_tp = level
            best_dist = dist

        return best_tp


def _find_fvg_tp(
    visible_high: np.ndarray,
    visible_low: np.ndarray,
    visible_close: np.ndarray,
    entry_price: float,
    is_long: bool,
    min_tp_dist: float,
) -> float:
    """
    Find TP at nearest unmitigated FVG.

    Bullish FVG: bar[i].low > bar[i-2].high  (gap up)
    Bearish FVG: bar[i].high < bar[i-2].low  (gap down)

    For long TP: nearest bearish FVG bottom above entry (resistance).
    For short TP: nearest bullish FVG top below entry (support).

    Returns TP price, or 0.0 if nothing found.
    """
    n = len(visible_high)
    if n < 3:
        return 0.0

    if is_long:
        # Bearish FVG: bar[i].high < bar[i-2].low => gap down = resistance zone
        # The gap zone is [bar[i].high, bar[i-2].low]
        # TP at the bottom of the gap = bar[i].high (nearest edge for long approaching)
        # Actually for a long approaching from below, the FVG bottom is the first level hit.
        gap_top = visible_low[:-2]      # bar[i-2].low  (indices 0..n-3)
        gap_bottom = visible_high[2:]   # bar[i].high   (indices 2..n-1)
        is_bearish_fvg = gap_bottom < gap_top  # gap down

        if not np.any(is_bearish_fvg):
            return 0.0

        fvg_indices = np.where(is_bearish_fvg)[0]  # index into 0-based (maps to bar index + 2)
        fvg_bottoms = gap_bottom[fvg_indices]  # bar[i].high = bottom of gap
        fvg_tops = gap_top[fvg_indices]        # bar[i-2].low = top of gap

        min_price = entry_price + min_tp_dist
        price_mask = fvg_bottoms > min_price

        fvg_indices = fvg_indices[price_mask]
        fvg_bottoms = fvg_bottoms[price_mask]
        fvg_tops = fvg_tops[price_mask]

        if len(fvg_indices) == 0:
            return 0.0

        # Check unmitigated: price hasn't entered the gap zone after formation
        # The actual bar index of the FVG is fvg_idx + 2
        best_tp = 0.0
        best_dist = np.inf
        for k in range(len(fvg_indices)):
            bar_idx = fvg_indices[k] + 2
            bottom = fvg_bottoms[k]
            top = fvg_tops[k]
            dist = bottom - entry_price
            if dist >= best_dist:
                continue
            # Check mitigation: any bar after formation with low <= top of gap
            if bar_idx + 1 < n:
                if np.any(visible_low[bar_idx + 1:] <= top):
                    continue  # mitigated
            best_tp = bottom
            best_dist = dist

        return best_tp

    else:
        # Bullish FVG: bar[i].low > bar[i-2].high => gap up = support zone
        # Gap zone is [bar[i-2].high, bar[i].low]
        # TP at the top of the gap = bar[i].low (nearest edge for short approaching)
        gap_bottom = visible_high[:-2]  # bar[i-2].high (indices 0..n-3)
        gap_top = visible_low[2:]       # bar[i].low    (indices 2..n-1)
        is_bullish_fvg = gap_top > gap_bottom  # gap up

        if not np.any(is_bullish_fvg):
            return 0.0

        fvg_indices = np.where(is_bullish_fvg)[0]
        fvg_tops = gap_top[fvg_indices]       # bar[i].low = top of gap
        fvg_bottoms = gap_bottom[fvg_indices]  # bar[i-2].high = bottom of gap

        max_price = entry_price - min_tp_dist
        price_mask = fvg_tops < max_price

        fvg_indices = fvg_indices[price_mask]
        fvg_tops = fvg_tops[price_mask]
        fvg_bottoms = fvg_bottoms[price_mask]

        if len(fvg_indices) == 0:
            return 0.0

        best_tp = 0.0
        best_dist = np.inf
        for k in range(len(fvg_indices)):
            bar_idx = fvg_indices[k] + 2
            top = fvg_tops[k]
            bottom = fvg_bottoms[k]
            dist = entry_price - top
            if dist >= best_dist:
                continue
            # Mitigated if any bar after formation has high >= bottom of gap
            if bar_idx + 1 < n:
                if np.any(visible_high[bar_idx + 1:] >= bottom):
                    continue  # mitigated
            best_tp = top
            best_dist = dist

        return best_tp


def _find_ob_tp(
    visible_open: np.ndarray,
    visible_high: np.ndarray,
    visible_low: np.ndarray,
    visible_close: np.ndarray,
    entry_price: float,
    is_long: bool,
    min_tp_dist: float,
    swing_highs: np.ndarray,
    swing_lows: np.ndarray,
) -> float:
    """
    Find TP at nearest order block (last opposite candle before a swing break).

    Bearish OB (resistance for long TP): last red candle before a swing high break.
    Bullish OB (support for short TP): last green candle before a swing low break.

    Returns TP price, or 0.0 if nothing found.
    """
    n = len(visible_high)
    is_red = visible_close < visible_open
    is_green = visible_close > visible_open

    if is_long:
        # Find bearish OBs = resistance. These form before swing high breaks.
        # For each swing high, find the last red candle before it.
        sh_indices = np.where(swing_highs)[0]
        if len(sh_indices) == 0:
            return 0.0

        min_price = entry_price + min_tp_dist
        best_tp = 0.0
        best_dist = np.inf

        for sh_idx in sh_indices:
            # Look backwards from the swing high for the last red candle
            search_start = max(0, sh_idx - 10)
            red_before = np.where(is_red[search_start:sh_idx])[0]
            if len(red_before) == 0:
                continue
            ob_idx = search_start + red_before[-1]  # last red candle before swing
            ob_low = visible_low[ob_idx]
            ob_high = visible_high[ob_idx]

            # OB zone is [ob_low, ob_high]. For long TP, use ob_low (first level hit)
            if ob_low <= min_price:
                continue
            dist = ob_low - entry_price
            if dist >= best_dist:
                continue
            # Check unmitigated: no bar after OB formation has traded through the OB zone
            if ob_idx + 1 < n:
                if np.any(visible_low[ob_idx + 1:] <= ob_low):
                    continue  # mitigated
            best_tp = ob_low
            best_dist = dist

        return best_tp

    else:
        # Find bullish OBs = support. These form before swing low breaks.
        sl_indices = np.where(swing_lows)[0]
        if len(sl_indices) == 0:
            return 0.0

        max_price = entry_price - min_tp_dist
        best_tp = 0.0
        best_dist = np.inf

        for sl_idx in sl_indices:
            search_start = max(0, sl_idx - 10)
            green_before = np.where(is_green[search_start:sl_idx])[0]
            if len(green_before) == 0:
                continue
            ob_idx = search_start + green_before[-1]
            ob_high = visible_high[ob_idx]
            ob_low = visible_low[ob_idx]

            # For short TP, use ob_high (first level hit from above)
            if ob_high >= max_price:
                continue
            dist = entry_price - ob_high
            if dist >= best_dist:
                continue
            if ob_idx + 1 < n:
                if np.any(visible_high[ob_idx + 1:] >= ob_high):
                    continue  # mitigated
            best_tp = ob_high
            best_dist = dist

        return best_tp


@dataclass
class _HTFArrays:
    """Pre-extracted numpy arrays for a higher-timeframe DataFrame.
    Computed once per generate_signals() call, reused for every signal bar."""
    high: np.ndarray
    low: np.ndarray
    open: np.ndarray
    close: np.ndarray
    swing_highs: np.ndarray
    swing_lows: np.ndarray
    length: int


def _precompute_htf_arrays(
    df: pd.DataFrame, swing_length: int,
) -> _HTFArrays | None:
    """Extract full arrays and swing detection for an HTF DataFrame."""
    if df.empty:
        return None
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    sh, sl_ = _find_swing_highs_lows(h, l, swing_length)
    return _HTFArrays(
        high=h, low=l, open=o, close=c,
        swing_highs=sh, swing_lows=sl_, length=len(h),
    )


def _find_structure_tp_safe(
    htf_4h: "_HTFArrays | None",
    htf_1h: "_HTFArrays | None",
    vlen_4h: int,
    vlen_1h: int,
    entry_price: float,
    bias: str,
    sl_dist: float,
    min_rr: float = 2.0,
) -> tuple[float, str]:
    """
    Lookahead-safe structure TP — uses pre-sliced raw OHLC arrays only.

    Unlike _find_structure_tp_OLD, this does NOT use pre-computed SMC
    indicators (which may contain future information). Swings, FVGs, and OBs
    are detected from the visible portion of pre-extracted numpy arrays.

    The _HTFArrays are computed ONCE per generate_signals() call via
    _precompute_htf_arrays(). Each call here only slices to vlen and
    searches — no pandas overhead.

    Search order: liq_4h -> liq_1h -> fvg_4h -> fvg_1h -> ob_4h -> ob_1h -> fallback.
    Returns (tp_price, source_label).
    """
    is_long = bias == "bullish"
    min_tp_dist = sl_dist * min_rr

    # Build search list: (arrays, visible_length, label)
    search: list[tuple[_HTFArrays, int, str]] = []
    if htf_4h is not None and vlen_4h > 0:
        search.append((htf_4h, min(vlen_4h, htf_4h.length), "4h"))
    if htf_1h is not None and vlen_1h > 0:
        search.append((htf_1h, min(vlen_1h, htf_1h.length), "1h"))

    # Search: liq -> fvg -> ob across timeframes
    for arr, vl, tf_label in search:
        tp = _find_liquidity_tp(
            arr.high[:vl], arr.low[:vl], entry_price, is_long, min_tp_dist,
            arr.swing_highs[:vl], arr.swing_lows[:vl],
        )
        if tp > 0:
            return tp, f"liq_{tf_label}"

    for arr, vl, tf_label in search:
        tp = _find_fvg_tp(
            arr.high[:vl], arr.low[:vl], arr.close[:vl],
            entry_price, is_long, min_tp_dist,
        )
        if tp > 0:
            return tp, f"fvg_{tf_label}"

    for arr, vl, tf_label in search:
        tp = _find_ob_tp(
            arr.open[:vl], arr.high[:vl], arr.low[:vl], arr.close[:vl],
            entry_price, is_long, min_tp_dist,
            arr.swing_highs[:vl], arr.swing_lows[:vl],
        )
        if tp > 0:
            return tp, f"ob_{tf_label}"

    # Fallback: 3.0 RR
    if is_long:
        return entry_price + sl_dist * 3.0, "fallback"
    else:
        return entry_price - sl_dist * 3.0, "fallback"


def _check_volume_ok(decision_df: pd.DataFrame, bar_idx: int,
                     lookback: int = 20, min_ratio: float = 1.2) -> bool:
    """Check if current volume is above average (simple volume confirmation)."""
    start = max(0, bar_idx - lookback)
    vol_slice = decision_df["volume"].iloc[start:bar_idx + 1]
    if len(vol_slice) < 5:
        return False
    avg_vol = vol_slice.iloc[:-1].mean()
    if avg_vol <= 0:
        return False
    return float(vol_slice.iloc[-1]) >= avg_vol * min_ratio


def _precompute_running_structure(indicators: dict[str, Any]) -> np.ndarray:
    """
    For each bar index i, compute the latest BOS/CHoCH direction (+1/-1/0).
    Used for 1H structure confirmation (no EMA fallback needed).
    """
    bos_choch = indicators.get("bos_choch")
    if bos_choch is None or bos_choch.empty:
        return np.zeros(0, dtype=np.int8)

    n = len(bos_choch)
    running = np.zeros(n, dtype=np.int8)
    last_sig = 0

    for i in range(n):
        choch = bos_choch["CHOCH"].iat[i]
        bos = bos_choch["BOS"].iat[i]
        val = choch if (pd.notna(choch) and choch != 0) else bos
        if pd.notna(val) and val != 0:
            last_sig = 1 if val > 0 else -1
        running[i] = last_sig

    return running


def _bias_from_running(running: np.ndarray, valid_len: int) -> str:
    """Look up precomputed running bias."""
    if valid_len <= 0 or len(running) == 0:
        return "neutral"
    idx = min(valid_len - 1, len(running) - 1)
    val = running[idx]
    if val > 0:
        return "bullish"
    if val < 0:
        return "bearish"
    return "neutral"


def _structure_confirms_from_running(running: np.ndarray, bias: str, valid_len: int) -> bool:
    """Check if running structure matches bias."""
    if valid_len <= 0 or len(running) == 0:
        return False
    idx = min(valid_len - 1, len(running) - 1)
    val = running[idx]
    if bias == "bullish" and val > 0:
        return True
    if bias == "bearish" and val < 0:
        return True
    return False


def _find_entry_zone_at(
    indicators_15m: dict[str, Any],
    df_15m: pd.DataFrame,
    bias: str,
    fvg_threshold: float,
    valid_len: int,
    max_zone_bars: int = 6,
) -> dict[str, Any] | None:
    """
    On the 15m timeframe, locate the most recent FVG or OB that
    aligns with the daily bias, only considering first *valid_len* rows.

    Strict version: only zones from the last *max_zone_bars* 15m-bars.
    Default 6 (1.5h for crypto/stocks), forex uses 12 (3h — slower structure).
    Additionally, price must be at least 30 % into the FVG/OB zone.
    """
    if valid_len <= 0:
        return None

    current_price = float(df_15m["close"].iloc[valid_len - 1])

    # Check FVGs
    fvg_data = indicators_15m.get("fvg")
    if fvg_data is not None and not fvg_data.empty:
        end = min(valid_len, len(fvg_data))
        scan_start = max(0, end - max_zone_bars)
        for idx in range(end - 1, scan_start - 1, -1):
            row = fvg_data.iloc[idx]
            fvg_dir = row.get("FVG", 0)
            top_val = row.get("Top", np.nan)
            bottom_val = row.get("Bottom", np.nan)

            if pd.isna(top_val) or pd.isna(bottom_val) or pd.isna(fvg_dir) or fvg_dir == 0:
                continue

            gap_size = abs(top_val - bottom_val) / current_price
            if gap_size < fvg_threshold:
                continue

            zone_range = abs(top_val - bottom_val)
            if zone_range == 0:
                continue

            if bias == "bullish" and fvg_dir > 0 and current_price >= bottom_val:
                penetration = (current_price - bottom_val) / zone_range
                if penetration >= 0.30:
                    return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val), "direction": "bullish"}
            if bias == "bearish" and fvg_dir < 0 and current_price <= top_val:
                penetration = (top_val - current_price) / zone_range
                if penetration >= 0.30:
                    return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val), "direction": "bearish"}

    # Fallback: check Order Blocks
    ob_data = indicators_15m.get("order_blocks")
    if ob_data is not None and not ob_data.empty:
        end = min(valid_len, len(ob_data))
        scan_start = max(0, end - max_zone_bars)
        for idx in range(end - 1, scan_start - 1, -1):
            row = ob_data.iloc[idx]
            ob_dir = row.get("OB", 0)
            ob_top = row.get("Top", np.nan)
            ob_bottom = row.get("Bottom", np.nan)

            if pd.isna(ob_top) or pd.isna(ob_bottom) or pd.isna(ob_dir) or ob_dir == 0:
                continue

            zone_range = abs(ob_top - ob_bottom)
            if zone_range == 0:
                continue

            if bias == "bullish" and ob_dir > 0 and current_price >= ob_bottom:
                penetration = (current_price - ob_bottom) / zone_range
                if penetration >= 0.30:
                    return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom), "direction": "bullish"}
            if bias == "bearish" and ob_dir < 0 and current_price <= ob_top:
                penetration = (ob_top - current_price) / zone_range
                if penetration >= 0.30:
                    return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom), "direction": "bearish"}

    return None


def _precompute_5m_trigger_mask(
    indicators_5m: dict[str, Any],
    lookback_bars: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute boolean arrays for bullish/bearish triggers on 5m.

    Only **real BOS or CHoCH** within *lookback_bars* of the current bar.
    Default 1 (crypto/stocks: current + previous bar).
    Forex uses 3 (15 min window — tick volume produces noisier swings).
    FVG is excluded from the trigger (too noisy for 5m).

    Returns (bullish_trigger, bearish_trigger) arrays.
    """
    bos_choch = indicators_5m.get("bos_choch")

    if bos_choch is None or bos_choch.empty:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

    n = len(bos_choch)
    bull_raw = np.zeros(n, dtype=bool)
    bear_raw = np.zeros(n, dtype=bool)

    for i in range(n):
        choch = bos_choch["CHOCH"].iat[i]
        bos = bos_choch["BOS"].iat[i]
        val = choch if (pd.notna(choch) and choch != 0) else bos
        if pd.notna(val) and val > 0:
            bull_raw[i] = True
        elif pd.notna(val) and val < 0:
            bear_raw[i] = True

    # Allow current bar + lookback_bars preceding bars
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    for i in range(n):
        start = max(0, i - lookback_bars)
        bull[i] = any(bull_raw[start:i + 1])
        bear[i] = any(bear_raw[start:i + 1])

    return bull, bear


# ═══════════════════════════════════════════════════════════════════
#  Position sizing
# ═══════════════════════════════════════════════════════════════════

def compute_position_size(
    account_size: float,
    risk_pct: float,
    leverage: int,
    entry_price: float,
    stop_loss: float,
) -> float:
    """
    Exact position-size calculation.
        risk_amount  = account_size × risk_pct
        sl_distance  = |entry_price − stop_loss|
        position_usd = risk_amount / (sl_distance / entry_price) × leverage
        quantity     = position_usd / entry_price

    Returns the quantity in base asset (e.g. BTC).
    """
    if entry_price <= 0 or stop_loss <= 0:
        return 0.0

    risk_amount = account_size * risk_pct
    sl_distance_pct = abs(entry_price - stop_loss) / entry_price
    if sl_distance_pct == 0:
        return 0.0

    # Notional exposure limited by risk
    notional = risk_amount / sl_distance_pct
    # Margin required given leverage
    margin_required = notional / leverage
    # Cap notional so margin does not exceed account size
    if margin_required > account_size:
        notional = account_size * leverage

    quantity = notional / entry_price
    return quantity


# ═══════════════════════════════════════════════════════════════════
#  Alignment score (4-step, 0.25 per step)
# ═══════════════════════════════════════════════════════════════════

def _compute_alignment_score(
    daily_bias: str,
    h1_confirms: bool,
    entry_zone: dict | None,
    precision_trigger: bool,
    style_weight: float = 1.0,
    *,
    bias_strong: bool = False,
    h4_confirms: bool = False,
    h4_poi: bool = False,
    h1_choch: bool = False,
    volume_ok: bool = False,
    asset_class: str | None = None,
) -> float:
    """
    Granular top-down alignment score (0–1).

    Default scoring (crypto/stocks/commodities):
      • Daily bias (1D) +0.12, strong +0.08, 4H +0.08, 4H POI +0.08,
        1H +0.08, CHoCH +0.06, entry_zone +0.15, trigger +0.15, volume +0.10
      Max = 0.90

    Forex scoring (redistributed — entry_zone/trigger unreliable with tick volume):
      • Daily bias +0.12, strong +0.12, 4H +0.12, 4H POI +0.08,
        1H +0.10, CHoCH +0.06, entry_zone +0.08, trigger +0.08, volume +0.14
      Max = 0.90

    Clamped to [0, 1].
    """
    # Forex-specific weights: less on entry_zone/trigger, more on HTF structure
    if asset_class == "forex":
        w_bias, w_strong, w_h4, w_h4poi = 0.12, 0.12, 0.12, 0.08
        w_h1, w_choch, w_zone, w_trigger, w_vol = 0.10, 0.06, 0.08, 0.08, 0.14
    else:
        w_bias, w_strong, w_h4, w_h4poi = 0.12, 0.08, 0.08, 0.08
        w_h1, w_choch, w_zone, w_trigger, w_vol = 0.08, 0.06, 0.15, 0.15, 0.10

    score = 0.0

    if daily_bias in ("bullish", "bearish"):
        score += w_bias
        if bias_strong:
            score += w_strong
    if h4_confirms:
        score += w_h4
    if h4_poi:
        score += w_h4poi
    if h1_confirms:
        score += w_h1
        if h1_choch:
            score += w_choch
    if entry_zone is not None:
        score += w_zone
    if precision_trigger:
        score += w_trigger
    if volume_ok:
        score += w_vol

    # Backward compat: if using old 4-arg call and no new flags,
    # fall back to roughly equivalent old scoring
    old_style = (
        not bias_strong and not h4_confirms and not h4_poi
        and not h1_choch and not volume_ok
    )
    if old_style and score > 0:
        # Boost to roughly match old 0.25-per-step scale
        score = min(score * 1.3, 1.0)

    return min(score * style_weight, 1.0)


# ═══════════════════════════════════════════════════════════════════
#  Strategy class
# ═══════════════════════════════════════════════════════════════════

class SMCMultiStyleStrategy:
    """
    Day-trading-only SMC / ICT strategy for Crypto Perpetuals.

    Parameters
    ----------
    config : dict
        Full config from default_config.yaml.
    params : dict
        Tunable parameters for this trial (from Optuna or manual).
        Expected keys: leverage, risk_per_trade, risk_reward,
                        swing_length, fvg_threshold, alignment_threshold.
    """

    def __init__(self, config: dict[str, Any], params: dict[str, Any]) -> None:
        self.cfg = config
        self.params = params

        # Unpack key params with defaults from config
        self.account_size: float = config["account"]["size"]
        self.leverage: int = int(params.get("leverage", config["leverage"]["min"]))
        self.risk_pct: float = params.get("risk_per_trade", config["risk_per_trade"]["min"])
        self.rr_ratio: float = params.get("risk_reward", 3.0)
        self.swing_length: int = int(params.get("swing_length", config["smc"]["swing_length"]))
        self.fvg_threshold: float = params.get("fvg_threshold", config["smc"]["fvg_threshold"])
        self.ob_lookback: int = int(params.get("order_block_lookback", config["smc"]["order_block_lookback"]))
        self.liq_range_pct: float = params.get("liquidity_range_percent", config["smc"]["liquidity_range_percent"])
        self.alignment_threshold: float = params.get(
            "alignment_threshold", config["top_down"]["alignment_threshold"]
        )
        self.asset_class: str | None = params.get("asset_class")

        # Day style weight (only style)
        sw = params.get("style_weights", {})
        self.style_weight: float = sw.get("day", config["styles"]["day"]["weight"])

        self.data_dir = Path(config["data"]["data_dir"])
        self.data_dirs: list[Path] = [self.data_dir]
        # Multi-asset: add all per-class data directories
        for key in ("crypto_dir", "forex_dir", "stocks_dir", "commodities_dir"):
            d = config["data"].get(key)
            if d:
                p = Path(d)
                if p.exists() and p not in self.data_dirs:
                    self.data_dirs.append(p)
        self.commission_pct: float = config["backtest"].get("commission_pct", 0.0004)

    # ── Data loading ──────────────────────────────────────────────

    def _load_tf(self, symbol: str, tf: str) -> pd.DataFrame:
        """Load a timeframe Parquet file for *symbol*, searching all data dirs."""
        safe = symbol.replace("/", "_").replace(":", "_")
        for d in self.data_dirs:
            path = d / f"{safe}_{tf}.parquet"
            if path.exists():
                return pd.read_parquet(path)
        raise FileNotFoundError(f"Data file not found for {safe}_{tf} in {self.data_dirs}")

    def _load_all_timeframes(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Load or resample all required timeframes from 1m (or 5m) base."""
        tfs_needed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        frames: dict[str, pd.DataFrame] = {}

        # Try loading pre-saved Parquet for each TF
        for tf in tfs_needed:
            try:
                frames[tf] = self._load_tf(symbol, tf)
            except FileNotFoundError:
                pass

        # If only 1m is present, resample the rest
        if "1m" in frames and len(frames) < len(tfs_needed):
            for tf in tfs_needed - set(frames.keys()):
                frames[tf] = resample_ohlcv(frames["1m"], tf)
        # Stocks: no 1m available, use 5m as base for higher TFs
        elif "5m" in frames and "1m" not in frames and len(frames) < len(tfs_needed):
            for tf in tfs_needed - set(frames.keys()) - {"1m"}:
                frames[tf] = resample_ohlcv(frames["5m"], tf)

        return frames

    # ── Signal generation ─────────────────────────────────────────

    def generate_signals(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> list[TradeSignal]:
        """
        Walk bar-by-bar through 5m data and generate day-trade signals.

        Top-down flow per bar:
          1. 1D daily bias (BOS/CHoCH)
          2. 1H structure confirmation
          3. 15m entry zone (FVG or OB)
          4. 5m precision trigger (BOS/CHoCH or FVG mitigation)

        Parameters
        ----------
        symbol : str   e.g. "BTC/USDT:USDT"
        start, end : optional timestamps to restrict the window
        """
        # Normalize start/end to tz-aware UTC (data is always tz-aware)
        if start is not None and not isinstance(start, pd.Timestamp):
            start = pd.Timestamp(start)
        if start is not None and start.tzinfo is None:
            start = start.tz_localize("UTC")
        if end is not None and not isinstance(end, pd.Timestamp):
            end = pd.Timestamp(end)
        if end is not None and end.tzinfo is None:
            end = end.tz_localize("UTC")

        frames = self._load_all_timeframes(symbol)
        if not frames:
            logger.warning("[%s] No data available – skipping", symbol)
            return []

        # Slice to window — keep lookback buffer for higher TFs so indicators
        # have enough history.  Only the decision TF (5m) is hard-sliced later
        # when emitting signals; higher TFs keep extra bars for warmup.
        lookback_bars = {
            "1m": 0, "5m": 0, "15m": 50, "1h": 100, "4h": 100, "1d": 250,
        }
        for tf in list(frames.keys()):
            df = frames[tf]
            if end is not None:
                df = df[df["timestamp"] <= end]
            if start is not None:
                buf = lookback_bars.get(tf, 0)
                if buf > 0:
                    # Keep extra bars before start for indicator warmup
                    mask = df["timestamp"] >= start
                    first_idx = mask.idxmax() if mask.any() else len(df)
                    keep_from = max(0, first_idx - buf)
                    df = df.iloc[keep_from:]
                else:
                    df = df[df["timestamp"] >= start]
            frames[tf] = df.reset_index(drop=True)
        # Remember the actual signal window start for filtering output
        _signal_window_start = start  # already normalized to tz-aware UTC above

        signals: list[TradeSignal] = []

        # Pre-compute SMC indicators for each timeframe (full range)
        indicators: dict[str, dict[str, Any]] = {}
        for tf, df in frames.items():
            if df.empty or len(df) < self.swing_length * 2:
                continue
            try:
                indicators[tf] = compute_smc_indicators(
                    df,
                    swing_length=self.swing_length,
                    fvg_threshold=self.fvg_threshold,
                    ob_lookback=self.ob_lookback,
                    liq_range_pct=self.liq_range_pct,
                )
            except Exception as exc:
                logger.debug("[%s] SMC computation failed for %s: %s", symbol, tf, exc)

        # Decision timeframe: 5m (the crypto sweet spot)
        decision_tf = "5m"
        decision_df = frames.get(decision_tf)
        if decision_df is None or decision_df.empty:
            logger.info("[%s] No 5m data – 0 signals", symbol)
            return signals

        # Required higher-TF indicators
        ind_1d = indicators.get("1d")
        ind_1h = indicators.get("1h")
        ind_4h = indicators.get("4h")
        ind_15m = indicators.get("15m")
        ind_5m = indicators.get("5m")

        if ind_1d is None or ind_1h is None:
            logger.info("[%s] Missing 1D or 1H indicators – 0 signals", symbol)
            return signals

        df_1d = frames.get("1d", pd.DataFrame())
        df_4h = frames.get("4h", pd.DataFrame())
        df_1h = frames.get("1h", pd.DataFrame())
        df_15m = frames.get("15m", pd.DataFrame())
        df_5m = decision_df

        # ── Precompute running arrays (O(n) once) ────────────────
        running_bias_1d = _precompute_running_bias(ind_1d, df_1d)
        running_bias_strong = _precompute_bias_strong(ind_1d, df_1d)
        running_struct_1h = _precompute_running_structure(ind_1h)
        running_h1_choch = _precompute_h1_choch_mask(ind_1h)
        running_struct_4h = (
            _precompute_running_structure(ind_4h)
            if ind_4h is not None else np.zeros(0, dtype=np.int8)
        )
        _trigger_lookback = 3 if self.asset_class == "forex" else 1
        bull_trigger_5m, bear_trigger_5m = (
            _precompute_5m_trigger_mask(ind_5m, lookback_bars=_trigger_lookback)
            if ind_5m is not None else (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
        )

        # ── Precompute temporal index maps (searchsorted, O(n log m)) ─
        ts_5m = decision_df["timestamp"].values

        def _build_valid_len_map(htf_df: pd.DataFrame) -> np.ndarray:
            """Return an array where each entry is the count of HTF rows
            with timestamp <= the corresponding 5m timestamp."""
            if htf_df.empty:
                return np.zeros(len(ts_5m), dtype=np.int64)
            htf_ts = htf_df["timestamp"].values
            return np.searchsorted(htf_ts, ts_5m, side="right").astype(np.int64)

        vlen_1d = _build_valid_len_map(df_1d)
        vlen_4h = _build_valid_len_map(df_4h)
        vlen_1h = _build_valid_len_map(df_1h)
        vlen_15m = _build_valid_len_map(df_15m)

        # ── Precompute HTF arrays for lookahead-safe structure TP ──
        _htf_4h = _precompute_htf_arrays(df_4h, self.swing_length)
        _htf_1h = _precompute_htf_arrays(df_1h, self.swing_length)

        # ── Precompute discount/premium zones (4H swing range) ──
        discount_premium = _compute_discount_premium(
            ind_4h, df_4h, decision_df, vlen_4h,
        )

        # Debug counters
        n_neutral = 0
        n_no_confirm = 0
        n_no_zone = 0
        n_no_trigger = 0
        n_low_score = 0
        n_sl_too_small = 0
        n_dp_rejected = 0
        n_emitted = 0
        tp_source_counts: dict[str, int] = {}

        min_start = self.swing_length * 2
        total_bars = len(decision_df)

        # Iterate over 5m bars
        for i in range(min_start, total_bars):
            bar = decision_df.iloc[i]
            ts = pd.Timestamp(bar["timestamp"])

            # ── Step 1: Daily bias (1D) ───────────────────────────
            bias = _bias_from_running(running_bias_1d, int(vlen_1d[i]))
            if bias == "neutral":
                n_neutral += 1
                continue

            # ── Step 1a: Discount/Premium filter ─────────────────
            dp_zone = discount_premium[i]  # +1=premium, -1=discount, 0=neutral
            if bias == "bullish" and dp_zone == 1:
                n_dp_rejected += 1
                continue  # Long in premium → REJECT
            if bias == "bearish" and dp_zone == -1:
                n_dp_rejected += 1
                continue  # Short in discount → REJECT

            # ── Step 1b: Bias strength (BOS/CHoCH vs EMA fallback) ─
            vl_1d = int(vlen_1d[i])
            bias_strong = bool(
                running_bias_strong[min(vl_1d - 1, len(running_bias_strong) - 1)]
            ) if vl_1d > 0 and len(running_bias_strong) > 0 else False

            # ── Step 2: 4H structure confirmation ─────────────────
            h4_ok = _structure_confirms_from_running(
                running_struct_4h, bias, int(vlen_4h[i])
            )
            h4_poi = (
                _check_h4_poi(ind_4h, df_4h, bias, int(vlen_4h[i]))
                if ind_4h is not None and not df_4h.empty else False
            )

            # ── Step 3: 1H structure confirmation ─────────────────
            vl_1h = int(vlen_1h[i])
            h1_ok = _structure_confirms_from_running(
                running_struct_1h, bias, vl_1h
            )
            h1_choch = bool(
                running_h1_choch[min(vl_1h - 1, len(running_h1_choch) - 1)]
            ) if vl_1h > 0 and len(running_h1_choch) > 0 else False

            # ── Step 4: 15m entry zone (FVG / OB) ────────────────
            entry_zone = None
            _zone_bars = 12 if self.asset_class == "forex" else 6
            if ind_15m is not None and not df_15m.empty:
                entry_zone = _find_entry_zone_at(
                    ind_15m, df_15m, bias, self.fvg_threshold, int(vlen_15m[i]),
                    max_zone_bars=_zone_bars,
                )

            # ── Step 5: 5m precision trigger ──────────────────────
            precision_ok = False
            if i < len(bull_trigger_5m):
                if bias == "bullish":
                    precision_ok = bool(bull_trigger_5m[i])
                elif bias == "bearish":
                    precision_ok = bool(bear_trigger_5m[i])

            # ── Step 6: Volume confirmation ───────────────────────
            volume_ok = _check_volume_ok(decision_df, i)

            # ── Alignment score (full 13-component) ──────────────
            score = _compute_alignment_score(
                bias, h1_ok, entry_zone, precision_ok, self.style_weight,
                bias_strong=bias_strong,
                h4_confirms=h4_ok,
                h4_poi=h4_poi,
                h1_choch=h1_choch,
                volume_ok=volume_ok,
                asset_class=self.asset_class,
            )

            if score < self.alignment_threshold:
                if not h1_ok:
                    n_no_confirm += 1
                elif entry_zone is None:
                    n_no_zone += 1
                elif not precision_ok:
                    n_no_trigger += 1
                else:
                    n_low_score += 1
                continue

            # ── Entry, SL, TP ─────────────────────────────────────
            entry_price = float(bar["close"])
            if entry_zone is not None:
                if bias == "bullish":
                    stop_loss = entry_zone["bottom"] * (1 - self.liq_range_pct)
                else:
                    stop_loss = entry_zone["top"] * (1 + self.liq_range_pct)
            else:
                # Fallback SL: use recent 5m swing low/high (20 bars ≈ 100 min)
                _sl_lookback = 20
                if bias == "bullish":
                    recent_lows = decision_df["low"].iloc[max(0, i - _sl_lookback): i + 1]
                    stop_loss = float(recent_lows.min()) * (1 - self.liq_range_pct)
                else:
                    recent_highs = decision_df["high"].iloc[max(0, i - _sl_lookback): i + 1]
                    stop_loss = float(recent_highs.max()) * (1 + self.liq_range_pct)

            sl_dist = abs(entry_price - stop_loss)
            if sl_dist == 0:
                continue

            sl_dist_pct = sl_dist / entry_price

            # ── SL distance filter (crypto daytrading minimum) ────
            if sl_dist_pct < 0.0035:
                n_sl_too_small += 1
                continue

            direction = "long" if bias == "bullish" else "short"

            # ── Structure-based TP (lookahead-safe, from raw OHLC) ──
            tp_price, tp_source = _find_structure_tp_safe(
                _htf_4h, _htf_1h,
                vlen_4h=int(vlen_4h[i]),
                vlen_1h=int(vlen_1h[i]),
                entry_price=entry_price,
                bias=bias,
                sl_dist=sl_dist,
                min_rr=2.0,
            )
            take_profit = tp_price

            # Derive actual RR from structure levels
            tp_dist = abs(take_profit - entry_price)
            actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
            if actual_rr < 2.0:
                continue  # RR too low even with structure TP

            # ── Position sizing ───────────────────────────────────
            qty = compute_position_size(
                self.account_size,
                self.risk_pct,
                self.leverage,
                entry_price,
                stop_loss,
            )
            if qty <= 0:
                continue

            # Only emit signals within the requested window (not from lookback buffer)
            if _signal_window_start is not None and ts < _signal_window_start:
                continue

            n_emitted += 1
            tp_source_counts[tp_source] = tp_source_counts.get(tp_source, 0) + 1
            signals.append(
                TradeSignal(
                    timestamp=ts,
                    symbol=symbol,
                    direction=direction,
                    style="day",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=actual_rr,
                    position_size=qty,
                    leverage=self.leverage,
                    alignment_score=score,
                    meta={
                        "bias": bias,
                        "bias_strong": bias_strong,
                        "h4_confirms": h4_ok,
                        "h4_poi": h4_poi,
                        "h1_confirm": h1_ok,
                        "h1_choch": h1_choch,
                        "entry_zone": entry_zone,
                        "precision_trigger": precision_ok,
                        "volume_ok": volume_ok,
                        "dp_zone": int(dp_zone),
                        "tp_source": tp_source,
                    },
                )
            )

        # ── Summary statistics ────────────────────────────────────
        avg_score = np.mean([s.alignment_score for s in signals]) if signals else 0
        avg_rr = np.mean([s.risk_reward for s in signals]) if signals else 0
        logger.info(
            "[%s] FINAL SIGNALS: %d | avg_score=%.2f | avg_rr=%.1f | "
            "dp_rejected=%d | tp_sources=%s",
            symbol, len(signals), avg_score, avg_rr, n_dp_rejected,
            tp_source_counts,
        )

        # Save signals for debugging
        if signals:
            results_dir = Path("backtest/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            sig_df = pd.DataFrame([asdict(s) for s in signals])
            sig_df.to_csv(results_dir / f"signals_{symbol.replace('/', '_')}.csv", index=False)

        return signals