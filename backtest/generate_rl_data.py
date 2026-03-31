"""
═══════════════════════════════════════════════════════════════════
 backtest/generate_rl_data.py
 ─────────────────────────────
 Generate training data for RL Brain V2 (Teacher-Student approach).

 Features: Extracted from CAUSAL SMC indicators (what model sees in real-time)
 Labels:   Generated from LOOKAHEAD SMC indicators (perfect hindsight)
           + forward price simulation (actual outcomes)

 Usage:
     python3 -m backtest.generate_rl_data --all-classes
     python3 -m backtest.generate_rl_data --classes crypto stocks
     python3 -m backtest.generate_rl_data --classes crypto --symbols BTC/USDT:USDT
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.indicators import compute_rsi_wilders, compute_atr_wilders

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.smc_multi_style import (
    SMCMultiStyleStrategy,
    compute_smc_indicators,       # LOOKAHEAD (teacher)
    compute_smc_indicators_causal, # CAUSAL (student features)
    resample_ohlcv,
    _precompute_running_bias,
    _precompute_bias_strong,
    _precompute_running_structure,
    _precompute_h1_choch_mask,
    _precompute_5m_trigger_mask,
    _check_h4_poi,
    _check_volume_ok,
    _compute_alignment_score,
    _compute_discount_premium,
    _find_entry_zone_at,
    _find_structure_tp_safe,
    _precompute_htf_arrays,
    _to_ohlc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/results/rl_data_gen.log"),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("data/rl_training")

# Walk-forward windows — 3 years covering bull/bear/sideways regimes
# Each window: 3 months of data, for diverse market conditions
WINDOWS = [
    ("2023-04-01", "2023-07-01"),   # W0  — post-FTX recovery
    ("2023-07-01", "2023-10-01"),   # W1  — summer sideways
    ("2023-10-01", "2024-01-01"),   # W2  — Q4 2023 rally
    ("2024-01-01", "2024-04-01"),   # W3  — BTC ETF launch
    ("2024-04-01", "2024-07-01"),   # W4  — halving period
    ("2024-07-01", "2024-10-01"),   # W5  — summer consolidation
    ("2024-10-01", "2025-01-01"),   # W6  — Q4 2024 bull
    ("2025-01-01", "2025-04-01"),   # W7  — Q1 2025
    ("2025-04-01", "2025-07-01"),   # W8  — Q2 2025
    ("2025-07-01", "2025-10-01"),   # W9  — Q3 2025
    ("2025-10-01", "2026-01-01"),   # W10 — Q4 2025
    ("2026-01-01", "2026-03-26"),   # W11 — Q1 2026 (partial)
]

# SMC profiles per asset class (from config)
ASSET_SMC_PARAMS: dict[str, dict[str, Any]] = {
    "crypto":      {"swing_length": 8,  "fvg_threshold": 0.0006, "order_block_lookback": 20, "liquidity_range_percent": 0.01},
    "forex":       {"swing_length": 20, "fvg_threshold": 0.001,  "order_block_lookback": 30, "liquidity_range_percent": 0.008},
    "stocks":      {"swing_length": 10, "fvg_threshold": 0.0003, "order_block_lookback": 20, "liquidity_range_percent": 0.005},
    "commodities": {"swing_length": 10, "fvg_threshold": 0.0004, "order_block_lookback": 20, "liquidity_range_percent": 0.005},
}

ASSET_COMMISSION: dict[str, float] = {
    "crypto": 0.0004, "forex": 0.0003, "stocks": 0.0001, "commodities": 0.0003,
}

ASSET_SLIPPAGE: dict[str, float] = {
    "crypto": 0.0002, "forex": 0.0001, "stocks": 0.0001, "commodities": 0.0002,
}

# Max bars to simulate forward for outcome (48h of 5m bars)
MAX_FORWARD_BARS = 576

# Feature names
FEATURE_NAMES: list[str] = []  # Populated at module level below

# Module-level cache for symbol ranks (computed once per training run)
_symbol_ranks_cache: dict[str, dict[str, dict[str, float]]] = {}


# ═══════════════════════════════════════════════════════════════════
#  Symbol Ranks (per-class percentile features)
# ═══════════════════════════════════════════════════════════════════

def compute_symbol_ranks(asset_class: str) -> dict[str, dict[str, float]]:
    """Compute per-symbol percentile ranks within an asset class.

    Computes ATR, volume, and spread statistics for all instruments in the
    given asset class, then returns percentile ranks [0, 1] for each.

    Results are cached so they are computed once per training run.

    Returns:
        Dict of symbol -> {"volatility_rank": float, "liquidity_rank": float,
                           "spread_rank": float}
    """
    if asset_class in _symbol_ranks_cache:
        return _symbol_ranks_cache[asset_class]

    data_dir = Path(f"data/{asset_class}")
    if not data_dir.exists():
        _symbol_ranks_cache[asset_class] = {}
        return {}

    parquets = sorted(data_dir.glob("*_5m.parquet"))
    if not parquets:
        _symbol_ranks_cache[asset_class] = {}
        return {}

    # Compute raw stats per instrument
    stats: list[tuple[str, float, float, float]] = []  # (sym, atr_pct, vol_usd, spread_pct)

    for p in parquets:
        sym = p.stem.replace("_5m", "")
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if len(df) < 100:
            continue

        close = pd.to_numeric(df["close"], errors="coerce").values.astype(float)
        high = pd.to_numeric(df["high"], errors="coerce").values.astype(float)
        low = pd.to_numeric(df["low"], errors="coerce").values.astype(float)

        # ATR % (14-period average true range as fraction of price)
        if len(close) > 14:
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]),
                           np.abs(low[1:] - close[:-1])),
            )
            atr_mean = float(np.mean(tr[:min(len(tr), 5000)]))
            price_mean = float(np.mean(close[1:min(len(close), 5001)]))
            atr_pct = atr_mean / price_mean if price_mean > 0 else 0.0
        else:
            atr_pct = 0.0

        # Volume USD (0 for forex/commodities tick volume)
        vol_usd = 0.0
        if "volume" in df.columns and asset_class not in ("forex", "commodities"):
            vol = pd.to_numeric(df["volume"], errors="coerce").values.astype(float)
            vol_usd = float(np.mean(vol[:min(len(vol), 5000)] * close[:min(len(close), 5000)]))

        # Spread % (average bar range as fraction of close)
        ranges = high - low
        mask = close > 0
        spread_pct = float(np.mean(ranges[mask] / close[mask])) if mask.any() else 0.0

        stats.append((sym, atr_pct, vol_usd, spread_pct))

    if not stats:
        _symbol_ranks_cache[asset_class] = {}
        return {}

    # Compute percentile ranks
    symbols = [s[0] for s in stats]
    atr_vals = np.array([s[1] for s in stats])
    vol_vals = np.array([s[2] for s in stats])
    spread_vals = np.array([s[3] for s in stats])

    def _percentile_rank(values: np.ndarray) -> np.ndarray:
        """Compute percentile rank [0, 1] for each value."""
        n = len(values)
        if n <= 1:
            return np.full(n, 0.5)
        order = values.argsort().argsort()  # rank positions
        return order.astype(float) / (n - 1)

    atr_ranks = _percentile_rank(atr_vals)
    vol_ranks = _percentile_rank(vol_vals)
    spread_ranks = _percentile_rank(spread_vals)

    result: dict[str, dict[str, float]] = {}
    for i, sym in enumerate(symbols):
        result[sym] = {
            "volatility_rank": float(atr_ranks[i]),
            "liquidity_rank": float(vol_ranks[i]),
            "spread_rank": float(spread_ranks[i]),
        }

    _symbol_ranks_cache[asset_class] = result
    return result


# ═══════════════════════════════════════════════════════════════════
#  Symbol Discovery
# ═══════════════════════════════════════════════════════════════════

def get_symbols_for_class(asset_class: str, max_crypto: int = 30) -> list[str]:
    """Discover available symbols for an asset class from data directory.

    Returns symbols in the same format as the backtester:
    - Crypto: "BTCUSDT" (no slash/colon)
    - Forex: "EUR_USD" (underscore)
    - Stocks: "AAPL" (plain)
    - Commodities: "XAU_USD" (underscore)
    """
    data_dirs = {
        "crypto": Path("data/crypto"),
        "forex": Path("data/forex"),
        "stocks": Path("data/stocks"),
        "commodities": Path("data/commodities"),
    }
    d = data_dirs.get(asset_class)
    if d is None or not d.exists():
        return []

    # Crypto: rank by 1m file size (same as backtester), use 5m as fallback
    if asset_class == "crypto":
        parquets = list(d.glob("*_1m.parquet"))
        if not parquets:
            parquets = list(d.glob("*_5m.parquet"))
            suffix = "_5m"
        else:
            suffix = "_1m"
        sized = []
        for p in parquets:
            raw = p.stem.replace(suffix, "")
            if "USDT" in raw:
                sized.append((raw, p.stat().st_size))
        sized.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in sized[:max_crypto]]

    # Stocks: 5m base (no 1m available)
    if asset_class == "stocks":
        parquets = sorted(d.glob("*_5m.parquet"))
        return [p.stem.replace("_5m", "") for p in parquets]

    # Forex/Commodities: 5m base (1m files no longer needed)
    parquets = sorted(d.glob("*_5m.parquet"))
    return [p.stem.replace("_5m", "") for p in parquets]


# ═══════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_multi_tf_data(
    symbol: str, asset_class: str,
    start: str | None = None, end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all timeframes for a symbol, with optional time slicing."""
    data_dir = Path(f"data/{asset_class}")
    # Symbols are already in file-safe format (BTCUSDT, EUR_USD, AAPL)
    safe = symbol.replace("/", "_").replace(":", "_")

    tfs_needed = ["5m", "15m", "1h", "4h", "1d"]
    frames: dict[str, pd.DataFrame] = {}

    for tf in tfs_needed:
        p = data_dir / f"{safe}_{tf}.parquet"
        if p.exists():
            frames[tf] = pd.read_parquet(p)

    # Resample missing TFs from 5m
    if "5m" in frames:
        for tf in tfs_needed:
            if tf not in frames and tf != "5m":
                frames[tf] = resample_ohlcv(frames["5m"], tf)

    # Time slice with lookback buffer for higher TFs
    lookback_bars = {"5m": 0, "15m": 50, "1h": 100, "4h": 100, "1d": 250}
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    for tf in list(frames.keys()):
        df = frames[tf]
        if "timestamp" not in df.columns:
            continue
        # Ensure tz-aware
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        if end_ts is not None:
            df = df[df["timestamp"] <= end_ts]
        if start_ts is not None:
            buf = lookback_bars.get(tf, 0)
            if buf > 0:
                mask = df["timestamp"] >= start_ts
                first_idx = mask.idxmax() if mask.any() else len(df)
                keep_from = max(0, first_idx - buf)
                df = df.iloc[keep_from:]
            else:
                df = df[df["timestamp"] >= start_ts]
        frames[tf] = df.reset_index(drop=True)

    return frames


# ═══════════════════════════════════════════════════════════════════
#  Feature Extraction (CAUSAL — what model sees in real-time)
# ═══════════════════════════════════════════════════════════════════

def _compute_ema(close: np.ndarray, span: int) -> np.ndarray:
    """EMA with exponential moving average."""
    alpha = 2.0 / (span + 1)
    ema = np.empty_like(close)
    ema[0] = close[0]
    for i in range(1, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
    return ema


# RSI and ATR use shared implementations from utils.indicators
# to guarantee train/serve parity with live_multi_bot.py
_compute_rsi = compute_rsi_wilders
_compute_atr = compute_atr_wilders


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """ADX indicator."""
    n = len(high)
    adx = np.full(n, 25.0)  # default neutral
    if n < period * 2 + 1:
        return adx
    # +DM / -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _compute_atr(high, low, close, period)
    # Smoothed DM
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    # Simple init
    if atr[period] > 0:
        plus_di[period] = np.mean(plus_dm[:period]) / atr[period] * 100
        minus_di[period] = np.mean(minus_dm[:period]) / atr[period] * 100
    for i in range(period + 1, n):
        if atr[i] > 0:
            plus_di[i] = min((plus_di[i-1] * (period-1) + plus_dm[i-1]) / period / atr[i] * 100, 100.0)
            minus_di[i] = min((minus_di[i-1] * (period-1) + minus_dm[i-1]) / period / atr[i] * 100, 100.0)
    # DX
    dx = np.zeros(n)
    for i in range(period, n):
        s = plus_di[i] + minus_di[i]
        dx[i] = abs(plus_di[i] - minus_di[i]) / s * 100 if s > 0 else 0
    # ADX = smoothed DX
    if n > period * 2:
        adx[period * 2] = np.mean(dx[period:period * 2 + 1])
        for i in range(period * 2 + 1, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return adx


def _nearest_zone_distances(
    zone_df: pd.DataFrame,
    close_prices: np.ndarray,
    atr: np.ndarray,
    type_col: str,  # "FVG", "OB", or "Liquidity"
    mitigated_col: str = "MitigatedIndex",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bar distance to nearest bullish/bearish zone (ATR-normalized).
    Returns (above_dist, below_dist, active_count) arrays.
    """
    n = len(close_prices)
    above_dist = np.full(n, 5.0)   # cap at 5 ATR
    below_dist = np.full(n, 5.0)
    active_count = np.zeros(n)

    if zone_df is None or zone_df.empty or type_col not in zone_df.columns:
        return above_dist, below_dist, active_count

    types = zone_df[type_col].values
    tops = zone_df["Top"].values if "Top" in zone_df.columns else np.full(n, np.nan)
    bottoms = zone_df["Bottom"].values if "Bottom" in zone_df.columns else np.full(n, np.nan)
    levels = zone_df["Level"].values if "Level" in zone_df.columns else np.full(n, np.nan)
    mitigated = zone_df[mitigated_col].values if mitigated_col in zone_df.columns else np.full(n, np.nan)

    # Track active zones: (creation_bar, type, mid_price)
    active_zones: list[tuple[int, float, float]] = []

    for i in range(n):
        # Register new zone at this bar
        if not np.isnan(types[i]):
            if type_col == "Liquidity":
                mid = float(levels[i]) if not np.isnan(levels[i]) else 0
            else:
                t = float(tops[i]) if not np.isnan(tops[i]) else 0
                b = float(bottoms[i]) if not np.isnan(bottoms[i]) else 0
                mid = (t + b) / 2 if t > 0 and b > 0 else 0
            if mid > 0:
                active_zones.append((i, float(types[i]), mid))

        # Remove mitigated zones
        if not np.isnan(types[i]) and not np.isnan(mitigated[i]):
            # Zone at this bar was already mitigated — handled by sweep check below
            pass

        # Compute distances to active zones
        best_above = 5.0
        best_below = 5.0
        count = 0
        still_active = []

        for (zbar, ztype, zmid) in active_zones:
            # Check if zone was mitigated by current bar
            if zbar < len(mitigated) and not np.isnan(mitigated[zbar]) and mitigated[zbar] <= i:
                continue  # mitigated
            still_active.append((zbar, ztype, zmid))
            count += 1

            if atr[i] > 0:
                dist = (zmid - close_prices[i]) / atr[i]
            else:
                dist = 0

            if dist > 0:  # zone is above
                best_above = min(best_above, dist)
            elif dist < 0:  # zone is below
                best_below = min(best_below, abs(dist))

        active_zones = still_active
        above_dist[i] = min(best_above, 5.0)
        below_dist[i] = min(best_below, 5.0)
        active_count[i] = min(count, 20) / 20.0  # normalize

    return above_dist, below_dist, active_count


def extract_features_for_instrument(
    frames: dict[str, pd.DataFrame],
    causal_indicators: dict[str, dict[str, Any]],
    asset_class: str,
    smc_params: dict[str, Any],
    signal_window_start: pd.Timestamp | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Extract RL feature matrix for all 5m bars of one instrument.

    Returns DataFrame with 43 feature columns + timestamp column.
    Features use ONLY causal (past) data — safe for inference.

    The 3 symbol-level features (volatility_rank, liquidity_rank,
    spread_rank) are percentile ranks [0, 1] within the asset class,
    computed once per training run and cached.
    """
    df_5m = frames.get("5m")
    if df_5m is None or df_5m.empty or len(df_5m) < 50:
        return pd.DataFrame()

    df_1d = frames.get("1d", pd.DataFrame())
    df_4h = frames.get("4h", pd.DataFrame())
    df_1h = frames.get("1h", pd.DataFrame())
    df_15m = frames.get("15m", pd.DataFrame())

    ind_1d = causal_indicators.get("1d")
    ind_4h = causal_indicators.get("4h")
    ind_1h = causal_indicators.get("1h")
    ind_15m = causal_indicators.get("15m")
    ind_5m = causal_indicators.get("5m")

    n = len(df_5m)
    close_5m = df_5m["close"].values.astype(float)
    high_5m = df_5m["high"].values.astype(float)
    low_5m = df_5m["low"].values.astype(float)
    vol_5m = df_5m["volume"].values.astype(float)
    ts_5m = df_5m["timestamp"].values

    # ── Temporal index maps ──────────────────────────────────────
    def _vlen(htf_df: pd.DataFrame) -> np.ndarray:
        if htf_df.empty:
            return np.zeros(n, dtype=np.int64)
        return np.searchsorted(htf_df["timestamp"].values, ts_5m, side="right").astype(np.int64)

    vlen_1d = _vlen(df_1d)
    vlen_4h = _vlen(df_4h)
    vlen_1h = _vlen(df_1h)
    vlen_15m = _vlen(df_15m)

    # ── Structure Direction per TF ───────────────────────────────
    def _struct_dir_mapped(indicators, vlen_arr):
        if indicators is None:
            return np.zeros(n, dtype=np.float32)
        running = _precompute_running_structure(indicators)
        if len(running) == 0:
            return np.zeros(n, dtype=np.float32)
        result = np.zeros(n, dtype=np.float32)
        for i in range(n):
            idx = int(vlen_arr[i]) - 1
            if idx >= 0 and idx < len(running):
                result[i] = float(running[idx])
        return result

    struct_1d = np.zeros(n, dtype=np.float32)
    if ind_1d is not None:
        bias_running = _precompute_running_bias(ind_1d, df_1d)
        for i in range(n):
            idx = int(vlen_1d[i]) - 1
            if 0 <= idx < len(bias_running):
                struct_1d[i] = float(bias_running[idx])

    struct_4h = _struct_dir_mapped(ind_4h, vlen_4h)
    struct_1h = _struct_dir_mapped(ind_1h, vlen_1h)
    struct_15m = _struct_dir_mapped(ind_15m, vlen_15m)
    struct_5m = _struct_dir_mapped(ind_5m, np.arange(n))

    # ── Bars Since Break (exponential decay) ─────────────────────
    def _break_decay(indicators):
        bos_choch = indicators.get("bos_choch") if indicators else None
        if bos_choch is None or bos_choch.empty:
            return np.zeros(0)
        n_tf = len(bos_choch)
        decay = np.zeros(n_tf, dtype=np.float32)
        last_break = -100
        for i in range(n_tf):
            choch = bos_choch["CHOCH"].iat[i]
            bos = bos_choch["BOS"].iat[i]
            if (pd.notna(choch) and choch != 0) or (pd.notna(bos) and bos != 0):
                last_break = i
            bars = i - last_break
            decay[i] = np.exp(-0.05 * bars)  # slow decay
        return decay

    def _map_decay(indicators, vlen_arr):
        d = _break_decay(indicators)
        if len(d) == 0:
            return np.zeros(n, dtype=np.float32)
        result = np.zeros(n, dtype=np.float32)
        for i in range(n):
            idx = int(vlen_arr[i]) - 1
            if 0 <= idx < len(d):
                result[i] = d[idx]
        return result

    decay_1d = _map_decay(ind_1d, vlen_1d)
    decay_4h = _map_decay(ind_4h, vlen_4h)
    decay_1h = _map_decay(ind_1h, vlen_1h)
    decay_15m = _map_decay(ind_15m, vlen_15m)
    decay_5m = _map_decay(ind_5m, np.arange(n))

    # ── Bias Features ────────────────────────────────────────────
    bias_strong = np.zeros(n, dtype=np.float32)
    if ind_1d is not None:
        bs = _precompute_bias_strong(ind_1d, df_1d)
        for i in range(n):
            idx = int(vlen_1d[i]) - 1
            if 0 <= idx < len(bs):
                bias_strong[i] = float(bs[idx])

    # Premium/Discount
    dp = _compute_discount_premium(ind_4h, df_4h, df_5m, vlen_4h)
    premium_discount = dp.astype(np.float32)

    # ── Component Flags ──────────────────────────────────────────
    # H4 confirms
    h4_confirms = np.zeros(n, dtype=np.float32)
    if ind_4h is not None:
        rs4 = _precompute_running_structure(ind_4h)
        for i in range(n):
            idx = int(vlen_4h[i]) - 1
            if 0 <= idx < len(rs4):
                # Confirms if structure direction matches daily bias
                if struct_1d[i] > 0 and rs4[idx] > 0:
                    h4_confirms[i] = 1.0
                elif struct_1d[i] < 0 and rs4[idx] < 0:
                    h4_confirms[i] = 1.0

    # H4 POI — vectorized: check if recent 4H FVG/OB exists aligned with bias
    h4_poi = np.zeros(n, dtype=np.float32)
    if ind_4h is not None and not df_4h.empty:
        # Precompute running "has recent FVG/OB" mask on 4H
        fvg4 = ind_4h.get("fvg")
        ob4 = ind_4h.get("order_blocks")
        n4 = len(df_4h)
        has_bull_poi = np.zeros(n4, dtype=bool)
        has_bear_poi = np.zeros(n4, dtype=bool)
        for data, col in [(fvg4, "FVG"), (ob4, "OB")]:
            if data is None or data.empty or col not in data.columns:
                continue
            vals = data[col].values
            for j in range(n4):
                if pd.notna(vals[j]) and vals[j] != 0:
                    end = min(j + 7, n4)  # active for ~6 bars
                    if vals[j] > 0:
                        has_bull_poi[j:end] = True
                    else:
                        has_bear_poi[j:end] = True
        for i in range(n):
            idx = int(vlen_4h[i]) - 1
            if 0 <= idx < n4:
                if struct_1d[i] > 0 and has_bull_poi[idx]:
                    h4_poi[i] = 1.0
                elif struct_1d[i] < 0 and has_bear_poi[idx]:
                    h4_poi[i] = 1.0

    # H1 confirms
    h1_confirms = np.zeros(n, dtype=np.float32)
    if ind_1h is not None:
        rs1 = _precompute_running_structure(ind_1h)
        for i in range(n):
            idx = int(vlen_1h[i]) - 1
            if 0 <= idx < len(rs1):
                if struct_1d[i] > 0 and rs1[idx] > 0:
                    h1_confirms[i] = 1.0
                elif struct_1d[i] < 0 and rs1[idx] < 0:
                    h1_confirms[i] = 1.0

    # H1 CHoCH
    h1_choch = np.zeros(n, dtype=np.float32)
    if ind_1h is not None:
        cm = _precompute_h1_choch_mask(ind_1h)
        for i in range(n):
            idx = int(vlen_1h[i]) - 1
            if 0 <= idx < len(cm):
                h1_choch[i] = float(cm[idx])

    # Entry Zone (15m) — vectorized: check if recent 15m FVG/OB exists
    has_entry_zone = np.zeros(n, dtype=np.float32)
    if ind_15m is not None and not df_15m.empty:
        _zone_bars = 12 if asset_class == "forex" else 6
        fvg15 = ind_15m.get("fvg")
        ob15 = ind_15m.get("order_blocks")
        n15 = len(df_15m)
        has_bull_zone = np.zeros(n15, dtype=bool)
        has_bear_zone = np.zeros(n15, dtype=bool)
        for data, col in [(fvg15, "FVG"), (ob15, "OB")]:
            if data is None or data.empty or col not in data.columns:
                continue
            vals = data[col].values
            mit = data["MitigatedIndex"].values if "MitigatedIndex" in data.columns else np.full(n15, np.nan)
            for j in range(n15):
                if pd.notna(vals[j]) and vals[j] != 0 and np.isnan(mit[j]):
                    end = min(j + _zone_bars + 1, n15)
                    if vals[j] > 0:
                        has_bull_zone[j:end] = True
                    else:
                        has_bear_zone[j:end] = True
        for i in range(n):
            idx = int(vlen_15m[i]) - 1
            if 0 <= idx < n15:
                if struct_1d[i] > 0 and has_bull_zone[idx]:
                    has_entry_zone[i] = 1.0
                elif struct_1d[i] < 0 and has_bear_zone[idx]:
                    has_entry_zone[i] = 1.0

    # Precision Trigger (5m)
    precision_trigger = np.zeros(n, dtype=np.float32)
    if ind_5m is not None:
        _tlb = 3 if asset_class == "forex" else 1
        bt, brt = _precompute_5m_trigger_mask(ind_5m, lookback_bars=_tlb)
        for i in range(min(n, len(bt))):
            if struct_1d[i] > 0 and bt[i]:
                precision_trigger[i] = 1.0
            elif struct_1d[i] < 0 and brt[i]:
                precision_trigger[i] = 1.0

    # Volume OK
    volume_ok = np.zeros(n, dtype=np.float32)
    for i in range(20, n):
        volume_ok[i] = float(_check_volume_ok(df_5m, i))

    # ── Market Context ───────────────────────────────────────────
    ema20_5m = _compute_ema(close_5m, 20)
    ema50_5m = _compute_ema(close_5m, 50)
    ema20_dist_5m = np.where(close_5m > 0, (close_5m - ema20_5m) / close_5m, 0).astype(np.float32)
    ema50_dist_5m = np.where(close_5m > 0, (close_5m - ema50_5m) / close_5m, 0).astype(np.float32)

    # 1H context
    ema20_dist_1h = np.zeros(n, dtype=np.float32)
    ema50_dist_1h = np.zeros(n, dtype=np.float32)
    if not df_1h.empty and len(df_1h) > 50:
        c1h = df_1h["close"].values.astype(float)
        e20_1h = _compute_ema(c1h, 20)
        e50_1h = _compute_ema(c1h, 50)
        d20 = np.where(c1h > 0, (c1h - e20_1h) / c1h, 0)
        d50 = np.where(c1h > 0, (c1h - e50_1h) / c1h, 0)
        for i in range(n):
            idx = int(vlen_1h[i]) - 1
            if 0 <= idx < len(d20):
                ema20_dist_1h[i] = d20[idx]
                ema50_dist_1h[i] = d50[idx]

    # ATR normalized
    atr_5m = _compute_atr(high_5m, low_5m, close_5m, 14)
    atr_5m_norm = np.where(close_5m > 0, atr_5m / close_5m, 0).astype(np.float32)

    atr_1h_norm = np.zeros(n, dtype=np.float32)
    if not df_1h.empty:
        h1h = df_1h["high"].values.astype(float)
        l1h = df_1h["low"].values.astype(float)
        c1h = df_1h["close"].values.astype(float)
        atr1h = _compute_atr(h1h, l1h, c1h, 14)
        atr1h_n = np.where(c1h > 0, atr1h / c1h, 0)
        for i in range(n):
            idx = int(vlen_1h[i]) - 1
            if 0 <= idx < len(atr1h_n):
                atr_1h_norm[i] = atr1h_n[idx]

    atr_daily_norm = np.zeros(n, dtype=np.float32)
    if not df_1d.empty:
        h1d = df_1d["high"].values.astype(float)
        l1d = df_1d["low"].values.astype(float)
        c1d = df_1d["close"].values.astype(float)
        atr1d = _compute_atr(h1d, l1d, c1d, 14)
        atr1d_n = np.where(c1d > 0, atr1d / c1d, 0)
        for i in range(n):
            idx = int(vlen_1d[i]) - 1
            if 0 <= idx < len(atr1d_n):
                atr_daily_norm[i] = atr1d_n[idx]

    # RSI
    rsi_5m = (_compute_rsi(close_5m, 14) / 100.0).astype(np.float32)

    rsi_1h = np.full(n, 0.5, dtype=np.float32)
    if not df_1h.empty:
        c1h = df_1h["close"].values.astype(float)
        r1h = _compute_rsi(c1h, 14) / 100.0
        for i in range(n):
            idx = int(vlen_1h[i]) - 1
            if 0 <= idx < len(r1h):
                rsi_1h[i] = r1h[idx]

    # Volume ratio
    vol_ratio = np.ones(n, dtype=np.float32)
    vol_ma = pd.Series(vol_5m).rolling(20, min_periods=5).mean().values
    mask = vol_ma > 0
    vol_ratio[mask] = np.minimum(vol_5m[mask] / vol_ma[mask], 5.0).astype(np.float32)

    # ADX on 1H
    adx_1h = np.full(n, 0.5, dtype=np.float32)
    if not df_1h.empty and len(df_1h) > 30:
        h1h = df_1h["high"].values.astype(float)
        l1h = df_1h["low"].values.astype(float)
        c1h = df_1h["close"].values.astype(float)
        adx_vals = _compute_adx(h1h, l1h, c1h, 14) / 50.0
        for i in range(n):
            idx = int(vlen_1h[i]) - 1
            if 0 <= idx < len(adx_vals):
                adx_1h[i] = min(adx_vals[idx], 2.0)

    # Alignment Score — vectorized computation (matching 13-component logic)
    if asset_class == "forex":
        w_bias, w_strong, w_h4, w_h4poi = 0.12, 0.12, 0.12, 0.08
        w_h1, w_choch, w_zone, w_trigger, w_vol = 0.10, 0.06, 0.08, 0.08, 0.14
    else:
        w_bias, w_strong, w_h4, w_h4poi = 0.12, 0.08, 0.08, 0.08
        w_h1, w_choch, w_zone, w_trigger, w_vol = 0.08, 0.06, 0.15, 0.15, 0.10

    has_bias = (struct_1d != 0).astype(np.float32)
    alignment = (
        has_bias * w_bias
        + has_bias * bias_strong * w_strong
        + h4_confirms * w_h4
        + h4_poi * w_h4poi
        + h1_confirms * w_h1
        + h1_confirms * h1_choch * w_choch
        + has_entry_zone * w_zone
        + precision_trigger * w_trigger
        + volume_ok * w_vol
    ).astype(np.float32)
    alignment = np.minimum(alignment, 1.0)

    # Time encoding
    if hasattr(ts_5m[0], 'hour'):
        hours = np.array([pd.Timestamp(t).hour for t in ts_5m], dtype=np.float32)
    else:
        hours = np.array([pd.Timestamp(t).hour for t in ts_5m], dtype=np.float32)
    hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)

    # ── Zone Proximity — simplified (count-based, not distance) ───
    # Count active FVG/OB/Liq zones on 15m (faster than per-bar distance calc)
    fvg_bull_active = np.zeros(n, dtype=np.float32)
    fvg_bear_active = np.zeros(n, dtype=np.float32)
    ob_bull_active = np.zeros(n, dtype=np.float32)
    ob_bear_active = np.zeros(n, dtype=np.float32)
    liq_above_count = np.zeros(n, dtype=np.float32)
    liq_below_count = np.zeros(n, dtype=np.float32)

    # FVG active counts on 15m
    if ind_15m is not None:
        fvg_df = ind_15m.get("fvg")
        if fvg_df is not None and not fvg_df.empty:
            n15 = len(fvg_df)
            bull_c = np.zeros(n15, dtype=np.float32)
            bear_c = np.zeros(n15, dtype=np.float32)
            # Running count of active FVGs
            active: list[tuple[int, float]] = []  # (idx, type)
            fvg_vals = fvg_df["FVG"].values
            mit_vals = fvg_df["MitigatedIndex"].values
            for j in range(n15):
                if pd.notna(fvg_vals[j]) and fvg_vals[j] != 0:
                    active.append((j, float(fvg_vals[j])))
                active = [(a, t) for a, t in active
                          if np.isnan(mit_vals[a]) or mit_vals[a] > j]
                bull_c[j] = sum(1 for _, t in active if t > 0) / 5.0
                bear_c[j] = sum(1 for _, t in active if t < 0) / 5.0
            for i in range(n):
                idx = int(vlen_15m[i]) - 1
                if 0 <= idx < n15:
                    fvg_bull_active[i] = min(bull_c[idx], 1.0)
                    fvg_bear_active[i] = min(bear_c[idx], 1.0)

    # OB active counts on 15m
    if ind_15m is not None:
        ob_df = ind_15m.get("order_blocks")
        if ob_df is not None and not ob_df.empty:
            n15 = len(ob_df)
            bull_c = np.zeros(n15, dtype=np.float32)
            bear_c = np.zeros(n15, dtype=np.float32)
            active = []
            ob_vals = ob_df["OB"].values
            mit_vals = ob_df["MitigatedIndex"].values
            for j in range(n15):
                if pd.notna(ob_vals[j]) and ob_vals[j] != 0:
                    active.append((j, float(ob_vals[j])))
                active = [(a, t) for a, t in active
                          if np.isnan(mit_vals[a]) or mit_vals[a] > j]
                bull_c[j] = sum(1 for _, t in active if t > 0) / 5.0
                bear_c[j] = sum(1 for _, t in active if t < 0) / 5.0
            for i in range(n):
                idx = int(vlen_15m[i]) - 1
                if 0 <= idx < n15:
                    ob_bull_active[i] = min(bull_c[idx], 1.0)
                    ob_bear_active[i] = min(bear_c[idx], 1.0)

    # Liquidity levels on 1H (above/below count)
    if ind_1h is not None:
        liq_df = ind_1h.get("liquidity")
        if liq_df is not None and not liq_df.empty:
            n1h = len(liq_df)
            above_c = np.zeros(n1h, dtype=np.float32)
            below_c = np.zeros(n1h, dtype=np.float32)
            active = []
            liq_vals = liq_df["Liquidity"].values
            swept_vals = liq_df["Swept"].values if "Swept" in liq_df.columns else np.full(n1h, np.nan)
            for j in range(n1h):
                if pd.notna(liq_vals[j]) and liq_vals[j] != 0:
                    active.append((j, float(liq_vals[j])))
                active = [(a, t) for a, t in active
                          if np.isnan(swept_vals[a]) or swept_vals[a] > j]
                above_c[j] = sum(1 for _, t in active if t > 0) / 5.0
                below_c[j] = sum(1 for _, t in active if t < 0) / 5.0
            for i in range(n):
                idx = int(vlen_1h[i]) - 1
                if 0 <= idx < n1h:
                    liq_above_count[i] = min(above_c[idx], 1.0)
                    liq_below_count[i] = min(below_c[idx], 1.0)

    # ── Assemble Feature Matrix ──────────────────────────────────
    features = {
        # Structure (10)
        "struct_1d": struct_1d,
        "struct_4h": struct_4h,
        "struct_1h": struct_1h,
        "struct_15m": struct_15m,
        "struct_5m": struct_5m,
        "decay_1d": decay_1d,
        "decay_4h": decay_4h,
        "decay_1h": decay_1h,
        "decay_15m": decay_15m,
        "decay_5m": decay_5m,
        # Bias (3)
        "bias_strong": bias_strong,
        "premium_discount": premium_discount,
        "daily_bias": struct_1d,  # alias
        # Components (7)
        "h4_confirms": h4_confirms,
        "h4_poi": h4_poi,
        "h1_confirms": h1_confirms,
        "h1_choch": h1_choch,
        "has_entry_zone": has_entry_zone,
        "precision_trigger": precision_trigger,
        "volume_ok": volume_ok,
        # Market Context (14)
        "ema20_dist_5m": ema20_dist_5m,
        "ema50_dist_5m": ema50_dist_5m,
        "ema20_dist_1h": ema20_dist_1h,
        "ema50_dist_1h": ema50_dist_1h,
        "atr_5m_norm": atr_5m_norm,
        "atr_1h_norm": atr_1h_norm,
        "atr_daily_norm": atr_daily_norm,
        "rsi_5m": rsi_5m,
        "rsi_1h": rsi_1h,
        "volume_ratio": vol_ratio,
        "adx_1h": adx_1h,
        "alignment_score": alignment,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        # Zone Activity (6)
        "fvg_bull_active": fvg_bull_active,
        "fvg_bear_active": fvg_bear_active,
        "ob_bull_active": ob_bull_active,
        "ob_bear_active": ob_bear_active,
        "liq_above_count": liq_above_count,
        "liq_below_count": liq_below_count,
    }

    # ── Symbol-Level Features (3) ───────────────────────────────
    # Percentile ranks within asset class — constant per instrument
    if symbol is not None:
        ranks = compute_symbol_ranks(asset_class)
        sym_ranks = ranks.get(symbol, {})
    else:
        sym_ranks = {}
    features["symbol_volatility_rank"] = np.full(
        n, sym_ranks.get("volatility_rank", 0.5), dtype=np.float32,
    )
    features["symbol_liquidity_rank"] = np.full(
        n, sym_ranks.get("liquidity_rank", 0.5), dtype=np.float32,
    )
    features["symbol_spread_rank"] = np.full(
        n, sym_ranks.get("spread_rank", 0.5), dtype=np.float32,
    )

    df = pd.DataFrame(features)
    df.insert(0, "timestamp", df_5m["timestamp"].values)

    # Filter to signal window
    if signal_window_start is not None:
        # Handle tz-naive timestamps in the assembled DataFrame
        ts_col = df["timestamp"]
        if ts_col.dt.tz is None and signal_window_start.tzinfo is not None:
            signal_window_start = signal_window_start.tz_localize(None)
        elif ts_col.dt.tz is not None and signal_window_start.tzinfo is None:
            signal_window_start = signal_window_start.tz_localize("UTC")
        df = df[df["timestamp"] >= signal_window_start].reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════
#  Label Generation (LOOKAHEAD — teacher knowledge)
# ═══════════════════════════════════════════════════════════════════

def _funding_cost_rr(asset_class: str, bars_held: int,
                     entry_price: float, sl_dist: float) -> float:
    """Crypto perpetual funding rate cost expressed in R-multiples."""
    if asset_class != "crypto" or bars_held <= 0:
        return 0.0
    hours_held = bars_held * 5 / 60  # 5m bars → hours
    funding_cost = entry_price * 0.0001 * (hours_held / 8)  # ~0.01% per 8h
    return funding_cost / sl_dist


def _simulate_forward(
    df_5m: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    commission_pct: float = 0.0004,
    slippage_pct: float = 0.0002,
    be_ratchet_r: float = 1.5,
    asset_class: str = "crypto",
    emit_bar_rows: bool = False,
    rsi_5m_arr: np.ndarray | None = None,
    adx_1h_arr: np.ndarray | None = None,
    atr_5m_arr: np.ndarray | None = None,
    asset_class_id: float = 0.0,
) -> tuple[str, float, int, float, int] | tuple[str, float, int, float, int, list[dict]]:
    """
    Simulate a trade forward from entry_idx.

    Returns (outcome, net_rr, exit_bar_offset, max_favorable_rr, exit_mechanism)
    when emit_bar_rows=False (default, backward compatible).

    When emit_bar_rows=True, returns a 6-tuple with an additional list of
    per-bar dicts.  Each dict represents one 5m bar during the trade with
    causal features only (no future data).  After the simulation loop, the
    label ``label_hold_better`` is backfilled on every bar using the known
    final net_rr (two-pass: forward to exit, backward to fill labels).

      exit_mechanism: 0=skip, 1=TP, 2=SL, 3=BE, 4=timeout
      net_rr: realized RR after commission + slippage + funding
      label_hold_better: 1 if continuing to hold from this bar improved outcome
    """
    is_long = direction == "long"
    sl_dist = abs(entry_price - stop_loss)
    if sl_dist == 0:
        if emit_bar_rows:
            return "skip", 0.0, 0, 0.0, 0, []
        return "skip", 0.0, 0, 0.0, 0

    high = df_5m["high"].values.astype(float)
    low = df_5m["low"].values.astype(float)
    close = df_5m["close"].values.astype(float)
    n = len(df_5m)

    # Cost model: commission + slippage on entry AND exit
    cost_pct = (commission_pct + slippage_pct) * 2
    cost_rr = (entry_price * cost_pct) / sl_dist

    current_sl = stop_loss
    be_triggered = False
    fee_buffer = entry_price * (commission_pct * 2 + slippage_pct * 2)
    best_favorable_rr = 0.0

    # Per-bar state for exit episode emission (populated when emit_bar_rows=True)
    bar_rows: list[dict] = []

    for i in range(entry_idx + 2, min(entry_idx + MAX_FORWARD_BARS + 1, n)):
        bar_high = high[i]
        bar_low = low[i]
        bar_close = close[i]
        bars_held = i - entry_idx

        # Track max favorable excursion (in R-multiples)
        if is_long:
            fav = (bar_high - entry_price) / sl_dist
            unrealized_rr = (bar_close - entry_price) / sl_dist
            sl_dist_pct = (bar_close - current_sl) / bar_close if bar_close > 0 else 0.0
        else:
            fav = (entry_price - bar_low) / sl_dist
            unrealized_rr = (entry_price - bar_close) / sl_dist
            sl_dist_pct = (current_sl - bar_close) / bar_close if bar_close > 0 else 0.0
        if fav > best_favorable_rr:
            best_favorable_rr = fav

        # Collect bar state for exit episodes (causal: only data known at this bar)
        if emit_bar_rows:
            # -- derived features for 15-col schema --
            rsi_val = float(rsi_5m_arr[i]) if rsi_5m_arr is not None and i < len(rsi_5m_arr) else 50.0
            adx_val = float(adx_1h_arr[i]) if adx_1h_arr is not None and i < len(adx_1h_arr) else 25.0
            atr_val = float(atr_5m_arr[i]) if atr_5m_arr is not None and i < len(atr_5m_arr) else 0.0
            atr_norm = atr_val / bar_close if bar_close > 0 and atr_val > 0 else 0.005
            prev_rr = bar_rows[-1]["bar_unrealized_rr"] if bar_rows else 0.0
            pnl_velocity = unrealized_rr - prev_rr
            mfe_dd = (best_favorable_rr - unrealized_rr) / best_favorable_rr if best_favorable_rr > 0 else 0.0
            bars_in_profit = sum(1 for r in bar_rows if r["bar_unrealized_rr"] > 0) + (1 if unrealized_rr > 0 else 0)
            time_in_profit_ratio = bars_in_profit / max(bars_held, 1)
            sl_dist_atr = max(sl_dist_pct, 0.0) / max(atr_norm, 1e-6)

            bar_rows.append({
                "bar_index": bars_held,
                "bars_held": bars_held,
                "bar_unrealized_rr": unrealized_rr,
                "sl_distance_pct": max(sl_dist_pct, 0.0),
                "max_favorable_seen": best_favorable_rr,
                "be_triggered": int(be_triggered),
                "asset_class_id": asset_class_id,
                "rsi_5m": rsi_val / 100.0,
                "adx_1h": adx_val / 50.0,
                "bars_held_normalized": bars_held / MAX_FORWARD_BARS,
                "pnl_velocity": float(np.clip(pnl_velocity, -1.0, 1.0)),
                "mfe_drawdown": float(np.clip(mfe_dd, 0.0, 1.0)),
                "time_in_profit_ratio": float(np.clip(time_in_profit_ratio, 0.0, 1.0)),
                "sl_distance_atr": float(np.clip(sl_dist_atr, 0.0, 5.0)),
                "regime_volatility": 1.0,
                "opposing_structure_count": 0,
                # label_hold_better backfilled after exit (two-pass)
                "label_hold_better": None,
            })

        # Check BE ratchet
        if not be_triggered:
            if is_long:
                max_favorable = bar_high - entry_price
            else:
                max_favorable = entry_price - bar_low
            if max_favorable >= sl_dist * be_ratchet_r:
                be_triggered = True
                if is_long:
                    current_sl = entry_price + fee_buffer
                else:
                    current_sl = entry_price - fee_buffer

        # Check SL hit
        if is_long and bar_low <= current_sl:
            raw_rr = (current_sl - entry_price) / sl_dist
            net_rr = raw_rr - cost_rr - _funding_cost_rr(asset_class, bars_held, entry_price, sl_dist)
            if be_triggered and abs(raw_rr) < 0.5:
                outcome, mech = "breakeven", 3
            else:
                outcome, mech = ("win" if net_rr > 0 else "loss"), 2
            if emit_bar_rows:
                _backfill_hold_labels(bar_rows, net_rr)
                return outcome, net_rr, bars_held, best_favorable_rr, mech, bar_rows
            return outcome, net_rr, bars_held, best_favorable_rr, mech

        if not is_long and bar_high >= current_sl:
            raw_rr = (entry_price - current_sl) / sl_dist
            net_rr = raw_rr - cost_rr - _funding_cost_rr(asset_class, bars_held, entry_price, sl_dist)
            if be_triggered and abs(raw_rr) < 0.5:
                outcome, mech = "breakeven", 3
            else:
                outcome, mech = ("win" if net_rr > 0 else "loss"), 2
            if emit_bar_rows:
                _backfill_hold_labels(bar_rows, net_rr)
                return outcome, net_rr, bars_held, best_favorable_rr, mech, bar_rows
            return outcome, net_rr, bars_held, best_favorable_rr, mech

        # Check TP hit
        if is_long and bar_high >= take_profit:
            raw_rr = (take_profit - entry_price) / sl_dist
            net_rr = raw_rr - cost_rr - _funding_cost_rr(asset_class, bars_held, entry_price, sl_dist)
            if emit_bar_rows:
                _backfill_hold_labels(bar_rows, net_rr)
                return "win", net_rr, bars_held, best_favorable_rr, 1, bar_rows
            return "win", net_rr, bars_held, best_favorable_rr, 1

        if not is_long and bar_low <= take_profit:
            raw_rr = (entry_price - take_profit) / sl_dist
            net_rr = raw_rr - cost_rr - _funding_cost_rr(asset_class, bars_held, entry_price, sl_dist)
            if emit_bar_rows:
                _backfill_hold_labels(bar_rows, net_rr)
                return "win", net_rr, bars_held, best_favorable_rr, 1, bar_rows
            return "win", net_rr, bars_held, best_favorable_rr, 1

    # Timeout — classify by current position
    bars_held = min(MAX_FORWARD_BARS, n - 1 - entry_idx)
    exit_price = close[min(entry_idx + MAX_FORWARD_BARS, n - 1)]
    if is_long:
        raw_rr = (exit_price - entry_price) / sl_dist
    else:
        raw_rr = (entry_price - exit_price) / sl_dist
    net_rr = raw_rr - cost_rr - _funding_cost_rr(asset_class, bars_held, entry_price, sl_dist)

    if net_rr >= 0.5:
        outcome, mech = "win", 4
    elif net_rr <= -0.5:
        outcome, mech = "loss", 4
    else:
        outcome, mech = "breakeven", 4

    if emit_bar_rows:
        _backfill_hold_labels(bar_rows, net_rr)
        return outcome, net_rr, bars_held, best_favorable_rr, mech, bar_rows
    return outcome, net_rr, bars_held, best_favorable_rr, mech


def _backfill_hold_labels(bar_rows: list[dict], final_net_rr: float) -> None:
    """
    Two-pass label backfill: after final exit is known, label each bar.

    label_hold_better = 1 if (final_net_rr - bar_unrealized_rr) > 0
      → holding from this bar improved the final outcome
    label_hold_better = 0 if exit at this bar was already optimal

    This is causal: we use the known final RR of a closed trade (offline batch),
    not any future bar price beyond the actual exit.
    """
    for row in bar_rows:
        remaining_rr = final_net_rr - row["bar_unrealized_rr"]
        row["label_hold_better"] = int(remaining_rr > 0)


def generate_teacher_labels(
    frames: dict[str, pd.DataFrame],
    lookahead_indicators: dict[str, dict[str, Any]],
    asset_class: str,
    smc_params: dict[str, Any],
    signal_window_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Generate per-bar labels using LOOKAHEAD indicators (teacher).

    At each 5m bar, checks if the lookahead teacher identifies a good entry.
    If yes, simulates forward to determine outcome.

    Returns DataFrame with columns:
        timestamp, label_action (0/1/2), label_rr, label_outcome, label_profitable
    """
    df_5m = frames.get("5m")
    if df_5m is None or df_5m.empty:
        return pd.DataFrame()

    df_1d = frames.get("1d", pd.DataFrame())
    df_4h = frames.get("4h", pd.DataFrame())
    df_1h = frames.get("1h", pd.DataFrame())
    df_15m = frames.get("15m", pd.DataFrame())

    ind_1d = lookahead_indicators.get("1d")
    ind_4h = lookahead_indicators.get("4h")
    ind_1h = lookahead_indicators.get("1h")
    ind_15m = lookahead_indicators.get("15m")
    ind_5m = lookahead_indicators.get("5m")

    n = len(df_5m)
    ts_5m = df_5m["timestamp"].values
    close = df_5m["close"].values.astype(float)
    open_prices = df_5m["open"].values.astype(float)

    # Temporal maps
    def _vlen(htf_df):
        if htf_df.empty:
            return np.zeros(n, dtype=np.int64)
        return np.searchsorted(htf_df["timestamp"].values, ts_5m, side="right").astype(np.int64)

    vlen_1d = _vlen(df_1d)
    vlen_4h = _vlen(df_4h)
    vlen_1h = _vlen(df_1h)
    vlen_15m = _vlen(df_15m)

    # Precompute running arrays (LOOKAHEAD version)
    running_bias = _precompute_running_bias(ind_1d, df_1d) if ind_1d else np.zeros(0)
    running_bias_strong = _precompute_bias_strong(ind_1d, df_1d) if ind_1d else np.zeros(0)
    running_struct_1h = _precompute_running_structure(ind_1h) if ind_1h else np.zeros(0)
    running_h1_choch = _precompute_h1_choch_mask(ind_1h) if ind_1h else np.zeros(0)
    running_struct_4h = _precompute_running_structure(ind_4h) if ind_4h else np.zeros(0)

    _trigger_lookback = 3 if asset_class == "forex" else 1
    bull_trigger, bear_trigger = (
        _precompute_5m_trigger_mask(ind_5m, lookback_bars=_trigger_lookback)
        if ind_5m else (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
    )

    # HTF arrays for structure TP
    swing_len = smc_params.get("swing_length", 8)
    _htf_4h = _precompute_htf_arrays(df_4h, swing_len) if not df_4h.empty else None
    _htf_1h = _precompute_htf_arrays(df_1h, swing_len) if not df_1h.empty else None

    # Output arrays
    label_action = np.zeros(n, dtype=np.int8)     # 0=no, 1=long, 2=short
    label_rr = np.zeros(n, dtype=np.float32)
    label_outcome = np.zeros(n, dtype=np.int8)     # 0=no, 1=win, 2=loss, 3=be
    label_profitable = np.zeros(n, dtype=np.int8)
    # Extended labels for Phase 2 models
    label_exit_mechanism = np.zeros(n, dtype=np.int8)   # 0=none, 1=TP, 2=SL, 3=BE, 4=timeout
    label_exit_bar = np.zeros(n, dtype=np.int16)         # bars held
    label_max_favorable_rr = np.zeros(n, dtype=np.float32)  # best unrealized RR
    label_tp_rr = np.zeros(n, dtype=np.float32)          # planned TP in R-multiples
    label_cost_rr = np.zeros(n, dtype=np.float32)        # total fees+slippage in R

    min_start = swing_len * 2
    commission = ASSET_COMMISSION.get(asset_class, 0.0004)
    slippage = ASSET_SLIPPAGE.get(asset_class, 0.0002)
    liq_range = smc_params.get("liquidity_range_percent", 0.005)
    _zone_bars = 12 if asset_class == "forex" else 6
    fvg_thresh = smc_params.get("fvg_threshold", 0.0004)

    # Normalize signal_window_start timezone
    if signal_window_start is not None:
        if len(ts_5m) > 0:
            sample_ts = pd.Timestamp(ts_5m[0])
            if sample_ts.tzinfo is None and signal_window_start.tzinfo is not None:
                signal_window_start = signal_window_start.tz_localize(None)
            elif sample_ts.tzinfo is not None and signal_window_start.tzinfo is None:
                signal_window_start = signal_window_start.tz_localize("UTC")

    n_entries = 0

    for i in range(min_start, n - 10):
        ts = ts_5m[i]
        if signal_window_start is not None and pd.Timestamp(ts) < signal_window_start:
            continue

        # ── Teacher Signal Detection (using LOOKAHEAD indicators) ──
        # Step 1: Daily bias
        vl_1d = int(vlen_1d[i])
        if vl_1d <= 0 or len(running_bias) == 0:
            continue
        bias_val = running_bias[min(vl_1d - 1, len(running_bias) - 1)]
        if bias_val == 0:
            continue
        bias = "bullish" if bias_val > 0 else "bearish"

        # Step 2: 1H structure (soft gate — contributes to quality, not hard requirement)
        vl_1h = int(vlen_1h[i])
        h1_ok = False
        if len(running_struct_1h) > 0 and vl_1h > 0:
            idx = min(vl_1h - 1, len(running_struct_1h) - 1)
            if (bias == "bullish" and running_struct_1h[idx] > 0) or \
               (bias == "bearish" and running_struct_1h[idx] < 0):
                h1_ok = True
        # NOTE: h1_ok is NO LONGER a hard gate — brain learns from h1_ok as a feature

        # Step 3: Entry zone (15m) — lookahead finds more zones
        entry_zone = None
        if ind_15m is not None and not df_15m.empty:
            entry_zone = _find_entry_zone_at(
                ind_15m, df_15m, bias, fvg_thresh,
                int(vlen_15m[i]), max_zone_bars=_zone_bars,
            )

        # Step 4: Precision trigger
        precision_ok = False
        if i < len(bull_trigger):
            if bias == "bullish":
                precision_ok = bool(bull_trigger[i])
            elif bias == "bearish":
                precision_ok = bool(bear_trigger[i])

        # Widened gate: only skip if we have NOTHING (no zone, no trigger, no H1)
        # Previously required zone OR trigger. Now only requires daily bias (already checked above).
        # Swing-based SL fallback handles entries without zones.
        if not h1_ok and entry_zone is None and not precision_ok:
            continue

        # ── Entry/SL/TP ──────────────────────────────────────────
        # Realistic fill: signal at close[i], execution at next bar open
        if i + 1 >= n - 10:
            continue
        entry_price = float(open_prices[i + 1])
        if entry_zone is not None:
            if bias == "bullish":
                stop_loss = entry_zone["bottom"] * (1 - liq_range)
            else:
                stop_loss = entry_zone["top"] * (1 + liq_range)
        else:
            _lb = 20
            if bias == "bullish":
                recent_lows = df_5m["low"].iloc[max(0, i - _lb):i + 1]
                stop_loss = float(recent_lows.min()) * (1 - liq_range)
            else:
                recent_highs = df_5m["high"].iloc[max(0, i - _lb):i + 1]
                stop_loss = float(recent_highs.max()) * (1 + liq_range)

        sl_dist = abs(entry_price - stop_loss)
        if sl_dist == 0 or sl_dist / entry_price < 0.001:
            continue

        direction = "long" if bias == "bullish" else "short"

        # Structure TP
        tp_price, tp_source = _find_structure_tp_safe(
            _htf_4h, _htf_1h,
            vlen_4h=int(vlen_4h[i]),
            vlen_1h=int(vlen_1h[i]),
            entry_price=entry_price,
            bias=bias,
            sl_dist=sl_dist,
            min_rr=1.0,  # Widened to capture more diverse training examples
        )

        actual_rr = abs(tp_price - entry_price) / sl_dist
        if actual_rr < 1.0:  # lowered from 1.5 — brain learns to filter bad RR
            continue

        # ── Simulate forward ─────────────────────────────────────
        planned_tp_rr = abs(tp_price - entry_price) / sl_dist
        outcome, realized_rr, exit_bars, max_fav_rr, exit_mech = _simulate_forward(
            df_5m, i, entry_price, stop_loss, tp_price,
            direction, commission_pct=commission, slippage_pct=slippage,
            asset_class=asset_class,
        )
        if outcome == "skip":
            continue

        # Cost in R-multiples (for reference)
        _cost_pct = (commission + slippage) * 2
        _cost_rr = (entry_price * _cost_pct) / sl_dist
        _fund_rr = _funding_cost_rr(asset_class, exit_bars, entry_price, sl_dist)

        # ── Label ────────────────────────────────────────────────
        label_action[i] = 1 if direction == "long" else 2
        label_rr[i] = realized_rr
        if outcome == "win":
            label_outcome[i] = 1
            label_profitable[i] = 1
        elif outcome == "loss":
            label_outcome[i] = 2
        elif outcome == "breakeven":
            label_outcome[i] = 3
        # Extended labels
        label_exit_mechanism[i] = exit_mech
        label_exit_bar[i] = exit_bars
        label_max_favorable_rr[i] = max_fav_rr
        label_tp_rr[i] = planned_tp_rr
        label_cost_rr[i] = _cost_rr + _fund_rr
        n_entries += 1

    logger.info("  Teacher labels: %d entries out of %d bars (%.1f%%)",
                n_entries, n, 100 * n_entries / max(n, 1))

    result = pd.DataFrame({
        "timestamp": ts_5m,
        "label_action": label_action,
        "label_rr": label_rr,
        "label_outcome": label_outcome,
        "label_profitable": label_profitable,
        "label_exit_mechanism": label_exit_mechanism,
        "label_exit_bar": label_exit_bar,
        "label_max_favorable_rr": label_max_favorable_rr,
        "label_tp_rr": label_tp_rr,
        "label_cost_rr": label_cost_rr,
    })

    if signal_window_start is not None:
        ts_col = result["timestamp"]
        sw = signal_window_start
        if hasattr(ts_col.dtype, 'tz') and ts_col.dt.tz is None and sw.tzinfo is not None:
            sw = sw.tz_localize(None)
        elif hasattr(ts_col.dtype, 'tz') and ts_col.dt.tz is not None and sw.tzinfo is None:
            sw = sw.tz_localize("UTC")
        result = result[result["timestamp"] >= sw].reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def process_instrument(
    symbol: str,
    asset_class: str,
    config: dict[str, Any],
    window_start: str,
    window_end: str,
) -> pd.DataFrame | None:
    """Process one instrument: extract features + generate labels."""
    smc_params = ASSET_SMC_PARAMS.get(asset_class, ASSET_SMC_PARAMS["crypto"])

    # Load data with lookback buffer (6 months for EMA200 warmup on daily)
    history_start = (pd.Timestamp(window_start, tz="UTC") - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    frames = load_multi_tf_data(symbol, asset_class, start=history_start, end=window_end)

    if "5m" not in frames or len(frames["5m"]) < 100:
        logger.warning("  %s: Not enough 5m data — skipping", symbol)
        return None

    # Compute CAUSAL indicators (student features)
    causal_indicators: dict[str, dict[str, Any]] = {}
    for tf, df in frames.items():
        if df.empty or len(df) < smc_params["swing_length"] * 2:
            continue
        try:
            causal_indicators[tf] = compute_smc_indicators_causal(
                df,
                swing_length=smc_params["swing_length"],
                fvg_threshold=smc_params["fvg_threshold"],
                ob_lookback=smc_params.get("order_block_lookback", 20),
                liq_range_pct=smc_params["liquidity_range_percent"],
            )
        except Exception as e:
            logger.debug("  %s causal %s failed: %s", symbol, tf, e)

    # Compute LOOKAHEAD indicators (teacher labels)
    lookahead_indicators: dict[str, dict[str, Any]] = {}
    for tf, df in frames.items():
        if df.empty or len(df) < smc_params["swing_length"] * 2:
            continue
        try:
            lookahead_indicators[tf] = compute_smc_indicators(
                df,
                swing_length=smc_params["swing_length"],
                fvg_threshold=smc_params["fvg_threshold"],
                ob_lookback=smc_params.get("order_block_lookback", 20),
                liq_range_pct=smc_params["liquidity_range_percent"],
            )
        except Exception as e:
            logger.debug("  %s lookahead %s failed: %s", symbol, tf, e)

    window_start_ts = pd.Timestamp(window_start, tz="UTC")

    # Extract CAUSAL features
    features_df = extract_features_for_instrument(
        frames, causal_indicators, asset_class, smc_params,
        signal_window_start=window_start_ts,
        symbol=symbol,
    )
    if features_df.empty:
        logger.warning("  %s: No features extracted — skipping", symbol)
        return None

    # Generate LOOKAHEAD labels
    labels_df = generate_teacher_labels(
        frames, lookahead_indicators, asset_class, smc_params,
        signal_window_start=window_start_ts,
    )
    if labels_df.empty:
        logger.warning("  %s: No labels generated — skipping", symbol)
        return None

    # Merge on timestamp
    merged = features_df.merge(labels_df, on="timestamp", how="inner")

    # Add metadata
    merged["symbol"] = symbol
    merged["asset_class"] = asset_class

    return merged


def _process_instrument_worker(args: tuple) -> pd.DataFrame | None:
    """Worker function for multiprocessing — processes one (symbol, window) pair."""
    sym, ac, config, w_start, w_end, wi = args
    try:
        result = process_instrument(sym, ac, config, w_start, w_end)
        if result is not None and len(result) > 0:
            result["window"] = wi
            return result
    except Exception as e:
        logger.error("    %s W%d FAILED: %s", sym, wi, e)
    return None


def run_data_generation(
    classes: list[str],
    symbols_override: list[str] | None = None,
    max_crypto: int = 30,
    n_workers: int = 4,
) -> None:
    """Main entry point: generate RL training data for specified asset classes."""
    import yaml
    from multiprocessing import Pool

    config_path = Path("config/default_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("backtest/results").mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_entries = 0

    for ac in classes:
        logger.info("═══ GENERATING RL DATA: %s ═══", ac.upper())

        if symbols_override:
            symbols = symbols_override
        else:
            symbols = get_symbols_for_class(ac, max_crypto=max_crypto)
        logger.info("  %d instruments, %d windows, %d workers", len(symbols), len(WINDOWS), n_workers)

        if not symbols:
            logger.warning("  No symbols found for %s — skipping", ac)
            continue

        all_data: list[pd.DataFrame] = []

        # Build all (symbol, window) work items
        work_items = []
        for wi, (w_start, w_end) in enumerate(WINDOWS):
            for sym in symbols:
                work_items.append((sym, ac, config, w_start, w_end, wi))

        logger.info("  Total work items: %d (parallel with %d workers)", len(work_items), n_workers)

        # Process in parallel
        completed = 0
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_process_instrument_worker, work_items, chunksize=1):
                completed += 1
                if completed % 20 == 0 or completed == len(work_items):
                    logger.info("  Progress: %d/%d (%.0f%%)", completed, len(work_items),
                                100 * completed / len(work_items))
                if result is not None and len(result) > 0:
                    all_data.append(result)

        # Log per-window stats
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            for wi in range(len(WINDOWS)):
                wdf = combined[combined["window"] == wi]
                if len(wdf) > 0:
                    n_entries = int((wdf["label_action"] > 0).sum())
                    logger.info("  Window %d: %d samples, %d entries (%.1f%%)",
                                wi, len(wdf), n_entries,
                                100 * n_entries / max(len(wdf), 1))

        gc.collect()

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True).reset_index(drop=True)
            out_path = OUTPUT_DIR / f"{ac}_samples.parquet"
            final_df.to_parquet(out_path, index=False)

            n_entries = int((final_df["label_action"] > 0).sum())
            total_samples += len(final_df)
            total_entries += n_entries

            # Diagnostics
            logger.info("═══ %s SUMMARY ═══", ac.upper())
            logger.info("  Total samples: %d", len(final_df))
            logger.info("  Total entries:  %d (%.1f%%)", n_entries,
                        100 * n_entries / max(len(final_df), 1))
            logger.info("  Long entries:   %d", int((final_df["label_action"] == 1).sum()))
            logger.info("  Short entries:  %d", int((final_df["label_action"] == 2).sum()))
            logger.info("  Win:  %d", int((final_df["label_outcome"] == 1).sum()))
            logger.info("  Loss: %d", int((final_df["label_outcome"] == 2).sum()))
            logger.info("  BE:   %d", int((final_df["label_outcome"] == 3).sum()))
            if n_entries > 0:
                wins = final_df[final_df["label_outcome"] == 1]["label_rr"]
                losses = final_df[final_df["label_outcome"] == 2]["label_rr"]
                logger.info("  Avg Win RR:  %.2f", wins.mean() if len(wins) > 0 else 0)
                logger.info("  Avg Loss RR: %.2f", losses.mean() if len(losses) > 0 else 0)
                logger.info("  Win Rate:    %.1f%%",
                            100 * len(wins) / max(len(wins) + len(losses), 1))

            # Per-symbol breakdown
            sym_stats = final_df[final_df["label_action"] > 0].groupby("symbol").agg(
                entries=("label_action", "count"),
                wins=("label_profitable", "sum"),
                avg_rr=("label_rr", "mean"),
            ).sort_values("entries", ascending=False)
            logger.info("  Per-symbol breakdown (top 10):\n%s",
                        sym_stats.head(10).to_string())

            logger.info("  Saved: %s", out_path)
        else:
            logger.warning("  %s: No data generated!", ac)

        gc.collect()

    logger.info("═══ TOTAL ═══")
    logger.info("  Samples: %d", total_samples)
    logger.info("  Entries:  %d (%.2f%%)", total_entries,
                100 * total_entries / max(total_samples, 1))


# ═══════════════════════════════════════════════════════════════════
#  Exit Episode Generation
# ═══════════════════════════════════════════════════════════════════

def _process_exit_episodes_worker(args: tuple) -> "pd.DataFrame | None":
    """Worker: generate bar-by-bar exit episodes for one (symbol, window) pair."""
    sym, ac, config, w_start, w_end, wi = args
    try:
        smc_params = ASSET_SMC_PARAMS.get(ac, ASSET_SMC_PARAMS["crypto"])
        commission = ASSET_COMMISSION.get(ac, 0.0004)
        slippage = ASSET_SLIPPAGE.get(ac, 0.0002)

        # Lookahead buffer same as process_instrument
        history_start = (
            pd.Timestamp(w_start, tz="UTC") - pd.Timedelta(days=180)
        ).strftime("%Y-%m-%d")
        frames = load_multi_tf_data(sym, ac, start=history_start, end=w_end)
        if frames is None or "5m" not in frames or len(frames["5m"]) < 100:
            return None
        df_5m = frames["5m"]

        # Guard: must use causal indicators for bar-level features
        assert compute_smc_indicators_causal.__name__ == "compute_smc_indicators_causal", (
            "lookahead leakage prevention: must use causal indicators"
        )

        # Compute LOOKAHEAD indicators (for teacher entry detection only)
        lookahead_inds: dict[str, dict[str, Any]] = {}
        for tf, df in frames.items():
            if df.empty or len(df) < smc_params["swing_length"] * 2:
                continue
            try:
                lookahead_inds[tf] = compute_smc_indicators(
                    df,
                    swing_length=smc_params["swing_length"],
                    fvg_threshold=smc_params["fvg_threshold"],
                    ob_lookback=smc_params.get("order_block_lookback", 20),
                    liq_range_pct=smc_params["liquidity_range_percent"],
                )
            except Exception:
                pass

        if "5m" not in lookahead_inds:
            return None

        window_start_ts = pd.Timestamp(w_start, tz="UTC")

        # Pre-compute indicator arrays for bar-level exit features (causal)
        from features.feature_extractor import ASSET_CLASS_MAP as _EXIT_AC_MAP
        close_5m = df_5m["close"].values.astype(float)
        high_5m = df_5m["high"].values.astype(float)
        low_5m = df_5m["low"].values.astype(float)
        rsi_5m_arr = _compute_rsi(close_5m, period=14)
        atr_5m_arr = _compute_atr(high_5m, low_5m, close_5m, period=14)

        # ADX on 1h if available, else default array
        df_1h = frames.get("1h")
        if df_1h is not None and len(df_1h) >= 30:
            adx_1h_full = _compute_adx(
                df_1h["high"].values.astype(float),
                df_1h["low"].values.astype(float),
                df_1h["close"].values.astype(float),
                period=14,
            )
            # Upsample 1h ADX to 5m resolution via forward-fill index mapping
            ts_1h = df_1h["timestamp"].values
            ts_5m_vals = df_5m["timestamp"].values
            adx_1h_arr = np.full(len(df_5m), 25.0)
            j = 0
            for k in range(len(ts_5m_vals)):
                while j + 1 < len(ts_1h) and ts_1h[j + 1] <= ts_5m_vals[k]:
                    j += 1
                adx_1h_arr[k] = adx_1h_full[j]
        else:
            adx_1h_arr = np.full(len(df_5m), 25.0)

        ac_id = _EXIT_AC_MAP.get(ac, 0.0)

        # Get teacher-labeled entries (lookahead for detection, not for bar features)
        label_df = generate_teacher_labels(
            frames, lookahead_inds, ac, smc_params,
            signal_window_start=window_start_ts,
        )

        entry_mask = label_df["label_action"] > 0
        if not entry_mask.any():
            return None

        # Map label timestamps → df_5m integer indices for fast lookup
        ts_5m = df_5m["timestamp"].values
        ts_map: dict = {ts: idx for idx, ts in enumerate(ts_5m)}

        all_episode_rows: list[dict] = []

        for _, label_row in label_df[entry_mask].iterrows():
            label_ts = label_row["timestamp"]
            entry_idx = ts_map.get(label_ts)
            if entry_idx is None or entry_idx + 10 >= len(df_5m):
                continue

            direction = "long" if int(label_row["label_action"]) == 1 else "short"
            entry_price = float(df_5m["open"].iloc[entry_idx + 1])  # next bar open
            if entry_price <= 0:
                continue

            tp_rr = float(label_row.get("label_tp_rr", 3.0))
            if tp_rr <= 0.5:
                continue

            # Reconstruct SL/TP from a local ATR estimate (causal: slice up to entry)
            atr_slice = df_5m["close"].iloc[max(0, entry_idx - 14):entry_idx + 1]
            atr_est = float(atr_slice.std()) if len(atr_slice) > 1 else entry_price * 0.005
            atr_est = max(atr_est, entry_price * 0.002)

            if direction == "long":
                stop_loss = entry_price - atr_est
                take_profit = entry_price + atr_est * tp_rr
            else:
                stop_loss = entry_price + atr_est
                take_profit = entry_price - atr_est * tp_rr

            result = _simulate_forward(
                df_5m, entry_idx, entry_price, stop_loss, take_profit,
                direction, commission_pct=commission, slippage_pct=slippage,
                asset_class=ac, emit_bar_rows=True,
                rsi_5m_arr=rsi_5m_arr, adx_1h_arr=adx_1h_arr,
                atr_5m_arr=atr_5m_arr, asset_class_id=ac_id,
            )
            if len(result) != 6:
                continue
            outcome, net_rr, exit_bars, max_fav_rr, exit_mech, bar_rows = result

            if outcome == "skip" or not bar_rows:
                continue

            # Attach trade-level metadata to every bar row
            for row in bar_rows:
                row.update({
                    "symbol": sym,
                    "asset_class": ac,
                    "direction": direction,
                    "window": wi,
                    "entry_price": entry_price,
                    "final_net_rr": net_rr,
                    "outcome": outcome,
                    "exit_mechanism": exit_mech,
                    "max_favorable_rr": max_fav_rr,
                    "tp_rr": tp_rr,
                })
                all_episode_rows.append(row)

        if not all_episode_rows:
            return None

        return pd.DataFrame(all_episode_rows)

    except Exception as exc:
        logger.warning("Exit episode worker failed for %s/%s: %s", sym, ac, exc)
        return None


def generate_exit_training_data(
    classes: list[str],
    symbols_override: list[str] | None = None,
    max_crypto: int = 30,
    n_workers: int = 4,
) -> None:
    """
    Generate bar-by-bar exit episode data from historical simulations.

    Extends _simulate_forward() with emit_bar_rows=True to replay every
    historical trade bar-by-bar, computing the label_hold_better signal
    using a two-pass backfill (final net_rr known at episode end).

    Output: data/rl_training/{ac}_exit_episodes.parquet
      One row per 5m bar during each simulated trade.
      label_hold_better = 1 if holding improved final outcome, 0 if exit was optimal.

    Causal guarantee: only compute_smc_indicators_causal is used for bar features.
    """
    import yaml
    from multiprocessing import Pool

    config_path = Path("config/default_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ac in classes:
        logger.info("═══ GENERATING EXIT EPISODES: %s ═══", ac.upper())

        symbols = symbols_override or get_symbols_for_class(ac, max_crypto=max_crypto)
        if not symbols:
            logger.warning("  No symbols for %s — skipping", ac)
            continue

        work_items = [
            (sym, ac, config, w_start, w_end, wi)
            for wi, (w_start, w_end) in enumerate(WINDOWS)
            for sym in symbols
        ]
        logger.info("  %d work items (%d symbols × %d windows), %d workers",
                    len(work_items), len(symbols), len(WINDOWS), n_workers)

        all_frames: list[pd.DataFrame] = []
        completed = 0
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(
                _process_exit_episodes_worker, work_items, chunksize=1
            ):
                completed += 1
                if completed % 20 == 0 or completed == len(work_items):
                    logger.info("  Progress: %d/%d", completed, len(work_items))
                if result is not None and len(result) > 0:
                    all_frames.append(result)

        gc.collect()

        if not all_frames:
            logger.warning("  %s: No exit episodes generated", ac)
            continue

        combined = pd.concat(all_frames, ignore_index=True)
        out_path = OUTPUT_DIR / f"{ac}_exit_episodes.parquet"
        combined.to_parquet(out_path, index=False)

        n_hold = int((combined["label_hold_better"] == 1).sum())
        n_exit = int((combined["label_hold_better"] == 0).sum())
        n_trades_approx = combined.groupby(["symbol", "direction", "entry_price"]).ngroups
        logger.info(
            "  %s: %d bar rows from ~%d trades | HOLD=%d (%.1f%%) EXIT=%d (%.1f%%)",
            ac, len(combined), n_trades_approx,
            n_hold, 100 * n_hold / max(len(combined), 1),
            n_exit, 100 * n_exit / max(len(combined), 1),
        )
        logger.info("  Saved: %s", out_path)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate RL training data")
    parser.add_argument("--all-classes", action="store_true",
                        help="Generate for all 4 asset classes")
    parser.add_argument("--classes", nargs="+",
                        choices=["crypto", "forex", "stocks", "commodities"],
                        help="Specific classes to generate for")
    parser.add_argument("--symbols", nargs="+", help="Override symbols")
    parser.add_argument("--max-crypto", type=int, default=30,
                        help="Max crypto symbols (default 30)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4)")
    parser.add_argument("--exit-episodes", action="store_true",
                        help="Generate bar-by-bar exit episodes (for exit classifier training)")
    args = parser.parse_args()

    if args.all_classes:
        classes = ["crypto", "forex", "stocks", "commodities"]
    elif args.classes:
        classes = args.classes
    else:
        classes = ["crypto", "stocks"]  # default: most reliable

    if args.exit_episodes:
        generate_exit_training_data(classes, symbols_override=args.symbols,
                                    max_crypto=args.max_crypto, n_workers=args.workers)
    else:
        run_data_generation(classes, symbols_override=args.symbols,
                            max_crypto=args.max_crypto, n_workers=args.workers)


if __name__ == "__main__":
    main()
