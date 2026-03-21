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
) -> dict[str, Any] | None:
    """
    On the 15m timeframe, locate the most recent FVG or OB that
    aligns with the daily bias, only considering first *valid_len* rows.

    Strict version: only zones from the last **6** 15m-bars (max 1.5 hours).
    Additionally, price must be at least 30 % into the FVG/OB zone.
    """
    if valid_len <= 0:
        return None

    current_price = float(df_15m["close"].iloc[valid_len - 1])
    max_zone_bars = 6  # 6 × 15m = 1.5 hours

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


def _precompute_5m_trigger_mask(indicators_5m: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute boolean arrays for bullish/bearish triggers on 5m.

    Strict version: only **real BOS or CHoCH** on exactly the current bar
    or the immediately preceding bar (max 1 bar lookback, no rolling window).
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

    # Only allow current bar or previous bar (max 1 bar lookback)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    for i in range(n):
        bull[i] = bull_raw[i] or (i > 0 and bull_raw[i - 1])
        bear[i] = bear_raw[i] or (i > 0 and bear_raw[i - 1])

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
) -> float:
    """
    Granular top-down alignment score (0–1).

    Scoring breakdown (adds up to 1.0 maximum):
      • Daily bias present (1D)       → +0.12
      • Daily bias from BOS/CHoCH     → +0.08  (bonus for strong bias)
      • 4H structure confirms          → +0.08
      • 4H POI (OB/FVG) active        → +0.08
      • 1H structure confirms          → +0.08
      • 1H CHoCH detected             → +0.06  (stronger than BOS)
      • 15m entry zone exists          → +0.15
      • 5m precision trigger active    → +0.15
      • 5m volume above average       → +0.10
      • Style weight multiplier        → ×weight

    Clamped to [0, 1].
    """
    score = 0.0

    if daily_bias in ("bullish", "bearish"):
        score += 0.12
        if bias_strong:
            score += 0.08
    if h4_confirms:
        score += 0.08
    if h4_poi:
        score += 0.08
    if h1_confirms:
        score += 0.08
        if h1_choch:
            score += 0.06
    if entry_zone is not None:
        score += 0.15
    if precision_trigger:
        score += 0.15
    if volume_ok:
        score += 0.10

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

        # Day style weight (only style)
        sw = params.get("style_weights", {})
        self.style_weight: float = sw.get("day", config["styles"]["day"]["weight"])

        self.data_dir = Path(config["data"]["data_dir"])
        self.commission_pct: float = config["backtest"].get("commission_pct", 0.0004)

    # ── Data loading ──────────────────────────────────────────────

    def _load_tf(self, symbol: str, tf: str) -> pd.DataFrame:
        """Load a timeframe Parquet file for *symbol*."""
        safe = symbol.replace("/", "_").replace(":", "_")
        path = self.data_dir / f"{safe}_{tf}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_parquet(path)

    def _load_all_timeframes(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Load or resample all required timeframes from 1 m base."""
        tfs_needed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        frames: dict[str, pd.DataFrame] = {}

        # Try loading pre-saved Parquet for each TF
        for tf in tfs_needed:
            try:
                frames[tf] = self._load_tf(symbol, tf)
            except FileNotFoundError:
                pass

        # If only 1 m is present, resample the rest
        if "1m" in frames and len(frames) < len(tfs_needed):
            for tf in tfs_needed - set(frames.keys()):
                frames[tf] = resample_ohlcv(frames["1m"], tf)

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
        frames = self._load_all_timeframes(symbol)
        if not frames:
            logger.warning("[%s] No data available – skipping", symbol)
            return []

        # Slice to window
        for tf in list(frames.keys()):
            df = frames[tf]
            if start is not None:
                df = df[df["timestamp"] >= start]
            if end is not None:
                df = df[df["timestamp"] <= end]
            frames[tf] = df.reset_index(drop=True)

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
        ind_15m = indicators.get("15m")
        ind_5m = indicators.get("5m")

        if ind_1d is None or ind_1h is None:
            logger.info("[%s] Missing 1D or 1H indicators – 0 signals", symbol)
            return signals

        df_1d = frames.get("1d", pd.DataFrame())
        df_1h = frames.get("1h", pd.DataFrame())
        df_15m = frames.get("15m", pd.DataFrame())
        df_5m = decision_df

        # ── Precompute running arrays (O(n) once) ────────────────
        running_bias_1d = _precompute_running_bias(ind_1d, df_1d)
        running_struct_1h = _precompute_running_structure(ind_1h)
        bull_trigger_5m, bear_trigger_5m = (
            _precompute_5m_trigger_mask(ind_5m)
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
        vlen_1h = _build_valid_len_map(df_1h)
        vlen_15m = _build_valid_len_map(df_15m)

        # Debug counters
        n_neutral = 0
        n_no_confirm = 0
        n_no_zone = 0
        n_no_trigger = 0
        n_low_score = 0
        n_sl_too_small = 0
        n_emitted = 0

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

            # ── Step 2: 1H structure confirmation ─────────────────
            h1_ok = _structure_confirms_from_running(
                running_struct_1h, bias, int(vlen_1h[i])
            )

            # ── Step 3: 15m entry zone (FVG / OB) ────────────────
            entry_zone = None
            if ind_15m is not None and not df_15m.empty:
                entry_zone = _find_entry_zone_at(
                    ind_15m, df_15m, bias, self.fvg_threshold, int(vlen_15m[i]),
                )

            # ── Step 4: 5m precision trigger ──────────────────────
            precision_ok = False
            if i < len(bull_trigger_5m):
                if bias == "bullish":
                    precision_ok = bool(bull_trigger_5m[i])
                elif bias == "bearish":
                    precision_ok = bool(bear_trigger_5m[i])

            # ── Alignment score ───────────────────────────────────
            score = _compute_alignment_score(
                bias, h1_ok, entry_zone, precision_ok, self.style_weight,
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

            if bias == "bullish":
                take_profit = entry_price + sl_dist * self.rr_ratio
                direction = "long"
            else:
                take_profit = entry_price - sl_dist * self.rr_ratio
                direction = "short"

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

            n_emitted += 1
            signals.append(
                TradeSignal(
                    timestamp=ts,
                    symbol=symbol,
                    direction=direction,
                    style="day",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=self.rr_ratio,
                    position_size=qty,
                    leverage=self.leverage,
                    alignment_score=score,
                    meta={
                        "bias": bias,
                        "h1_confirm": h1_ok,
                        "entry_zone": entry_zone,
                        "precision_trigger": precision_ok,
                    },
                )
            )

        # ── Summary statistics ────────────────────────────────────
        avg_score = np.mean([s.alignment_score for s in signals]) if signals else 0
        logger.info(
            "[%s] FINAL SIGNALS: %d | avg_score=%.2f",
            symbol, len(signals), avg_score,
        )

        # Save signals for debugging
        if signals:
            results_dir = Path("backtest/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            sig_df = pd.DataFrame([asdict(s) for s in signals])
            sig_df.to_csv(results_dir / f"signals_{symbol.replace('/', '_')}.csv", index=False)

        return signals