"""
Shared indicator functions used by BOTH training (generate_rl_data.py) and
live inference (live_multi_bot.py).

CRITICAL: Any change here affects model training ↔ serving parity.
Do NOT modify without regenerating training data or verifying feature parity.
"""
from __future__ import annotations

import numpy as np


def compute_rsi_wilders(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI with Wilder's smoothing (NOT EWM).

    Returns array of RSI values in [0, 100] range, same length as input.
    Default fill = 50.0 for bars before enough data.
    """
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_atr_wilders(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """ATR with Wilder's smoothing (NOT EWM).

    Returns array of ATR values, same length as input.
    """
    n = len(high)
    atr = np.zeros(n)
    if n < 2:
        return atr
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]),
                   np.abs(low[1:] - close[:-1])),
    )
    tr = np.concatenate([[high[0] - low[0]], tr])
    if n <= period:
        atr[:] = np.mean(tr)
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr
