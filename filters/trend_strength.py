"""
Trend Strength Filter
=====================
ADX, momentum confluence, and multi-TF trend agreement scoring.

Used to ensure trades are only taken in clearly trending markets
with accelerating momentum and full timeframe alignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  ADX (Average Directional Index)
# ═══════════════════════════════════════════════════════════════════

def compute_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[float, float, float]:
    """
    Compute ADX, +DI, -DI from price arrays.

    Returns (adx, plus_di, minus_di).
    All values in range [0, 100].
    Returns (0, 0, 0) if insufficient data.
    """
    n = len(closes)
    if n < period + 2:
        return 0.0, 0.0, 0.0

    # True Range, +DM, -DM
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, n):
        h, l, pc = float(highs[i]), float(lows[i]), float(closes[i - 1])
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

        up_move = float(highs[i]) - float(highs[i - 1])
        down_move = float(lows[i - 1]) - float(lows[i])

        plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0.0)

    tr_arr = np.array(tr_list, dtype=np.float64)
    plus_dm = np.array(plus_dm_list, dtype=np.float64)
    minus_dm = np.array(minus_dm_list, dtype=np.float64)

    if len(tr_arr) < period:
        return 0.0, 0.0, 0.0

    # Wilder smoothing (initial SMA then EMA-like)
    atr = np.mean(tr_arr[:period])
    smooth_plus = np.mean(plus_dm[:period])
    smooth_minus = np.mean(minus_dm[:period])

    dx_values = []

    for i in range(period, len(tr_arr)):
        atr = atr - (atr / period) + tr_arr[i]
        smooth_plus = smooth_plus - (smooth_plus / period) + plus_dm[i]
        smooth_minus = smooth_minus - (smooth_minus / period) + minus_dm[i]

        if atr > 0:
            pdi = 100.0 * smooth_plus / atr
            mdi = 100.0 * smooth_minus / atr
        else:
            pdi = mdi = 0.0

        di_sum = pdi + mdi
        if di_sum > 0:
            dx_values.append(100.0 * abs(pdi - mdi) / di_sum)
        else:
            dx_values.append(0.0)

    if len(dx_values) < period:
        return 0.0, 0.0, 0.0

    # ADX = Wilder-smoothed DX (proper average, not sum)
    adx = np.mean(dx_values[:period])
    for i in range(period, len(dx_values)):
        adx = (adx * (period - 1) + dx_values[i]) / period

    # Final +DI, -DI
    if atr > 0:
        final_pdi = 100.0 * smooth_plus / atr
        final_mdi = 100.0 * smooth_minus / atr
    else:
        final_pdi = final_mdi = 0.0

    return float(np.clip(adx, 0, 100)), float(np.clip(final_pdi, 0, 100)), float(np.clip(final_mdi, 0, 100))


def _ema_val(data: np.ndarray, span: int) -> float:
    """Compute last EMA value."""
    if len(data) < span:
        return float(data[-1]) if len(data) > 0 else 0.0
    alpha = 2.0 / (span + 1)
    ema = float(data[0])
    for v in data[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    return ema


# ═══════════════════════════════════════════════════════════════════
#  Momentum Confluence (RSI + MACD)
# ═══════════════════════════════════════════════════════════════════

def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI over the last *period* bars."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):].astype(np.float64))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd_histogram(closes: np.ndarray) -> tuple[float, float]:
    """
    Compute MACD histogram (current and previous) for momentum direction.
    MACD = EMA12 - EMA26, Signal = EMA9(MACD).
    Returns (current_hist, previous_hist).
    """
    if len(closes) < 35:
        return 0.0, 0.0

    ema12_vals = []
    ema26_vals = []
    alpha12 = 2.0 / 13.0
    alpha26 = 2.0 / 27.0

    e12 = float(closes[0])
    e26 = float(closes[0])

    for c in closes:
        c_f = float(c)
        e12 = alpha12 * c_f + (1.0 - alpha12) * e12
        e26 = alpha26 * c_f + (1.0 - alpha26) * e26
        ema12_vals.append(e12)
        ema26_vals.append(e26)

    macd_line = [e12 - e26 for e12, e26 in zip(ema12_vals, ema26_vals)]

    # Signal line (EMA9 of MACD)
    alpha9 = 2.0 / 10.0
    signal = macd_line[0]
    hist_vals = []
    for m in macd_line:
        signal = alpha9 * m + (1.0 - alpha9) * signal
        hist_vals.append(m - signal)

    if len(hist_vals) < 2:
        return 0.0, 0.0

    return hist_vals[-1], hist_vals[-2]


def check_momentum_confluence(
    closes: np.ndarray,
    direction: str,
) -> tuple[bool, float]:
    """
    Check if RSI and MACD histogram are aligned for the trade direction.

    For longs:
      - RSI between 40-65 (not overbought, room to move up)
      - MACD histogram positive AND increasing (momentum accelerating)

    For shorts:
      - RSI between 35-60 (not oversold, room to move down)
      - MACD histogram negative AND decreasing (momentum accelerating)

    Returns (is_confluent, momentum_score 0-1).
    """
    rsi = _compute_rsi(closes, 14)
    hist_curr, hist_prev = _compute_macd_histogram(closes)

    rsi_ok = False
    macd_ok = False
    score = 0.0

    if direction == "long":
        rsi_ok = 40.0 <= rsi <= 65.0
        macd_ok = hist_curr > 0 and hist_curr > hist_prev  # positive & accelerating
    else:
        rsi_ok = 35.0 <= rsi <= 60.0
        macd_ok = hist_curr < 0 and hist_curr < hist_prev  # negative & accelerating

    if rsi_ok:
        score += 0.5
    if macd_ok:
        score += 0.5

    return (rsi_ok and macd_ok), score


# ═══════════════════════════════════════════════════════════════════
#  Multi-TF Trend Agreement
# ═══════════════════════════════════════════════════════════════════

def multi_tf_trend_agreement(
    closes_1d: np.ndarray | None,
    closes_4h: np.ndarray | None,
    closes_1h: np.ndarray | None,
    closes_15m: np.ndarray | None,
    direction: str,
) -> tuple[int, float]:
    """
    Check EMA20/50 alignment across 4 timeframes.

    For longs: EMA20 > EMA50 means bullish.
    For shorts: EMA20 < EMA50 means bearish.

    Returns (aligned_count out of 4, score 0-1).
    """
    datasets = [closes_1d, closes_4h, closes_1h, closes_15m]
    aligned = 0
    total = 0

    for closes in datasets:
        if closes is None or len(closes) < 50:
            continue
        total += 1
        ema20 = _ema_val(closes, 20)
        ema50 = _ema_val(closes, 50)
        if direction == "long" and ema20 > ema50:
            aligned += 1
        elif direction == "short" and ema20 < ema50:
            aligned += 1

    if total == 0:
        return 0, 0.0

    return aligned, aligned / total


# ═══════════════════════════════════════════════════════════════════
#  Combined Trend Strength Score
# ═══════════════════════════════════════════════════════════════════

def compute_trend_strength_score(
    highs_1h: np.ndarray,
    lows_1h: np.ndarray,
    closes_1h: np.ndarray,
    direction: str,
    closes_1d: np.ndarray | None = None,
    closes_4h: np.ndarray | None = None,
    closes_15m: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute all trend strength metrics.

    Returns dict with:
      - adx: float (0-100)
      - adx_strong: bool (ADX > 25)
      - plus_di: float
      - minus_di: float
      - momentum_confluent: bool
      - momentum_score: float (0-1)
      - tf_aligned_count: int (0-4)
      - tf_agreement_score: float (0-1)
      - trend_score: float (0-1, composite)
    """
    from typing import Any

    # ADX on 1H
    adx, plus_di, minus_di = compute_adx(highs_1h, lows_1h, closes_1h, period=14)
    adx_strong = adx > 25.0

    # DI direction check: +DI > -DI for longs, -DI > +DI for shorts
    di_aligned = (direction == "long" and plus_di > minus_di) or \
                 (direction == "short" and minus_di > plus_di)

    # Momentum confluence on 1H
    momentum_ok, momentum_score = check_momentum_confluence(closes_1h, direction)

    # Multi-TF agreement
    tf_count, tf_score = multi_tf_trend_agreement(
        closes_1d, closes_4h, closes_1h, closes_15m, direction,
    )

    # Composite trend score
    adx_component = min(adx / 50.0, 1.0) * 0.35  # ADX 50+ = full score
    momentum_component = momentum_score * 0.30
    tf_component = tf_score * 0.25
    di_component = 0.10 if di_aligned else 0.0

    trend_score = float(np.clip(
        adx_component + momentum_component + tf_component + di_component,
        0.0, 1.0,
    ))

    return {
        "adx": adx,
        "adx_strong": adx_strong,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "di_aligned": di_aligned,
        "momentum_confluent": momentum_ok,
        "momentum_score": momentum_score,
        "tf_aligned_count": tf_count,
        "tf_agreement_score": tf_score,
        "trend_score": trend_score,
    }
