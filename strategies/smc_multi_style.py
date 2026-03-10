"""
═══════════════════════════════════════════════════════════════════
 strategies/smc_multi_style.py
 ─────────────────────────────
 Smart-Money-Concepts (SMC / ICT) multi-style strategy.

 Responsibilities:
   • Load & resample OHLCV data for every required timeframe
   • Compute all SMC indicators via the 'smartmoneyconcepts' package:
       FVG, Order Blocks, BOS / CHoCH, Liquidity Pools,
       Swing High / Low, Previous High / Low, Sessions
   • Top-Down analysis: Daily Bias → 4 h Structure → 15 m Entry
     Zone → 5 m / 1 m precision entry
   • Automatic style decision (Scalp / Day / Swing) based on an
     alignment score
   • Exact position-size calculation (risk % × leverage × SL distance)
   • Return a list of trade signals for the backtester

 Usage (from project root):
     from strategies.smc_multi_style import SMCMultiStyleStrategy
     strat = SMCMultiStyleStrategy(config, params)
     signals = strat.generate_signals(symbol)
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    style: str                # "scalp" | "day" | "swing"
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
    swing_length: int = 10,
    fvg_threshold: float = 0.001,
    ob_lookback: int = 20,
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
#  Bias & structure helpers
# ═══════════════════════════════════════════════════════════════════

def _daily_bias(indicators: dict[str, Any], df: pd.DataFrame) -> str:
    """
    Determine bullish / bearish / neutral daily bias from the latest
    BOS/CHoCH and swing structure.
    """
    bos_choch = indicators["bos_choch"]
    if bos_choch is None or bos_choch.empty:
        return "neutral"

    # The SMC lib returns columns BOS and CHOCH with +1 (bullish) / -1 (bearish) / 0
    last_valid_idx = bos_choch[["BOS", "CHOCH"]].replace(0, np.nan).last_valid_index()
    if last_valid_idx is None:
        return "neutral"

    row = bos_choch.loc[last_valid_idx]
    value = row["CHOCH"] if row["CHOCH"] != 0 else row["BOS"]
    if value > 0:
        return "bullish"
    if value < 0:
        return "bearish"
    return "neutral"


def _structure_confirms(indicators: dict[str, Any], bias: str) -> bool:
    """Check whether the 4 h structure aligns with the daily bias."""
    bos_choch = indicators["bos_choch"]
    if bos_choch is None or bos_choch.empty:
        return False

    last_idx = bos_choch[["BOS", "CHOCH"]].replace(0, np.nan).last_valid_index()
    if last_idx is None:
        return False

    row = bos_choch.loc[last_idx]
    val = row["CHOCH"] if row["CHOCH"] != 0 else row["BOS"]
    if bias == "bullish" and val > 0:
        return True
    if bias == "bearish" and val < 0:
        return True
    return False


def _find_entry_zone(
    indicators: dict[str, Any],
    df: pd.DataFrame,
    bias: str,
    fvg_threshold: float,
) -> dict[str, Any] | None:
    """
    On the 15 m timeframe, locate the most recent FVG or OB that
    aligns with the daily bias.  Returns a dict with zone boundaries
    or None if no valid zone exists.
    """
    fvg_data = indicators["fvg"]
    ob_data = indicators["order_blocks"]
    current_price = float(df["close"].iloc[-1])

    # Check FVGs
    if fvg_data is not None and not fvg_data.empty:
        for idx in range(len(fvg_data) - 1, max(len(fvg_data) - 30, -1), -1):
            row = fvg_data.iloc[idx]
            fvg_dir = row.get("FVG", 0)
            top_val = row.get("Top", np.nan)
            bottom_val = row.get("Bottom", np.nan)

            if pd.isna(top_val) or pd.isna(bottom_val):
                continue
            if fvg_dir == 0:
                continue

            gap_size = abs(top_val - bottom_val) / current_price
            if gap_size < fvg_threshold:
                continue

            if bias == "bullish" and fvg_dir > 0 and current_price >= bottom_val:
                return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val), "direction": "bullish"}
            if bias == "bearish" and fvg_dir < 0 and current_price <= top_val:
                return {"type": "fvg", "top": float(top_val), "bottom": float(bottom_val), "direction": "bearish"}

    # Fallback: check Order Blocks
    if ob_data is not None and not ob_data.empty:
        for idx in range(len(ob_data) - 1, max(len(ob_data) - 20, -1), -1):
            row = ob_data.iloc[idx]
            ob_dir = row.get("OB", 0)
            ob_top = row.get("Top", np.nan)
            ob_bottom = row.get("Bottom", np.nan)

            if pd.isna(ob_top) or pd.isna(ob_bottom) or ob_dir == 0:
                continue

            if bias == "bullish" and ob_dir > 0 and current_price >= ob_bottom:
                return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom), "direction": "bullish"}
            if bias == "bearish" and ob_dir < 0 and current_price <= ob_top:
                return {"type": "ob", "top": float(ob_top), "bottom": float(ob_bottom), "direction": "bearish"}

    return None


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
#  Alignment score
# ═══════════════════════════════════════════════════════════════════

def _compute_alignment_score(
    daily_bias: str,
    h4_confirms: bool,
    entry_zone: dict | None,
    precision_indicators: dict[str, Any] | None,
    style_weight: float,
) -> float:
    """
    Combine top-down alignment signals into a 0–1 score.
      • Daily bias present          → +0.25
      • 4 h structure confirms      → +0.25
      • 15 m entry zone exists      → +0.25
      • 5 m / 1 m precision signal  → +0.25
    Each component is multiplied by *style_weight* and clamped to [0, 1].
    """
    score = 0.0
    if daily_bias in ("bullish", "bearish"):
        score += 0.25
    if h4_confirms:
        score += 0.25
    if entry_zone is not None:
        score += 0.25
    if precision_indicators is not None:
        # Check for recent BOS on precision TF
        bos = precision_indicators.get("bos_choch")
        if bos is not None and not bos.empty:
            last_idx = bos[["BOS", "CHOCH"]].replace(0, np.nan).last_valid_index()
            if last_idx is not None:
                score += 0.25

    return min(score * style_weight, 1.0)


# ═══════════════════════════════════════════════════════════════════
#  Strategy class
# ═══════════════════════════════════════════════════════════════════

class SMCMultiStyleStrategy:
    """
    Multi-style SMC / ICT strategy.

    Parameters
    ----------
    config : dict
        Full config from default_config.yaml.
    params : dict
        Tunable parameters for this trial (from Optuna or manual).
        Expected keys: leverage, risk_per_trade, risk_reward,
                        style_weights (dict), swing_length,
                        fvg_threshold, alignment_threshold.
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

        # Style weights
        sw = params.get("style_weights", {})
        self.style_weights: dict[str, float] = {
            "scalp": sw.get("scalp", config["styles"]["scalp"]["weight"]),
            "day": sw.get("day", config["styles"]["day"]["weight"]),
            "swing": sw.get("swing", config["styles"]["swing"]["weight"]),
        }

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
        Walk bar-by-bar through the data and generate trade signals.

        Parameters
        ----------
        symbol : str   e.g. "BTC/USDT:USDT"
        start, end : optional timestamps to restrict the window
        """
        frames = self._load_all_timeframes(symbol)
        if not frames:
            logger.warning("No data for %s", symbol)
            return []

        # Slice to window
        for tf in frames:
            df = frames[tf]
            if start is not None:
                df = df[df["timestamp"] >= start]
            if end is not None:
                df = df[df["timestamp"] <= end]
            frames[tf] = df.reset_index(drop=True)

        signals: list[TradeSignal] = []

        # Pre-compute SMC indicators for each timeframe
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
                logger.debug("SMC computation failed for %s %s: %s", symbol, tf, exc)

        # Reference frame for iteration: use 15 m as the "decision" cadence
        decision_tf = "15m"
        decision_df = frames.get(decision_tf)
        if decision_df is None or decision_df.empty:
            return signals

        # Iterate over decision-TF bars
        for i in range(self.swing_length * 2, len(decision_df)):
            bar = decision_df.iloc[i]
            ts = bar["timestamp"]

            # ── Step 1: Daily bias ────────────────────────────────
            daily_ind = indicators.get("1d")
            if daily_ind is None:
                continue
            bias = _daily_bias(daily_ind, frames["1d"])

            if bias == "neutral":
                continue

            # ── Step 2: 4 h structure confirmation ────────────────
            h4_ind = indicators.get("4h")
            if h4_ind is None:
                continue
            h4_ok = _structure_confirms(h4_ind, bias)

            # ── Step 3: 15 m entry zone ───────────────────────────
            m15_ind = indicators.get("15m")
            if m15_ind is None:
                continue
            # Slice indicators to current bar index
            entry_zone = _find_entry_zone(
                m15_ind,
                decision_df.iloc[: i + 1],
                bias,
                self.fvg_threshold,
            )

            # ── Step 4: Precision TF check ────────────────────────
            precision_ind = indicators.get("5m") or indicators.get("1m")

            # ── Alignment score & style decision ──────────────────
            best_style = None
            best_score = 0.0
            for style_name, weight in self.style_weights.items():
                score = _compute_alignment_score(
                    bias, h4_ok, entry_zone, precision_ind, weight
                )
                if score > best_score:
                    best_score = score
                    best_style = style_name

            if best_score < self.alignment_threshold or best_style is None:
                continue

            # ── Entry, SL, TP ─────────────────────────────────────
            entry_price = float(bar["close"])
            if entry_zone is not None:
                # Place SL outside the entry zone
                if bias == "bullish":
                    stop_loss = entry_zone["bottom"] * (1 - self.liq_range_pct)
                else:
                    stop_loss = entry_zone["top"] * (1 + self.liq_range_pct)
            else:
                # Fallback: use recent swing low/high
                if bias == "bullish":
                    recent_lows = decision_df["low"].iloc[max(0, i - 20) : i + 1]
                    stop_loss = float(recent_lows.min()) * (1 - self.liq_range_pct)
                else:
                    recent_highs = decision_df["high"].iloc[max(0, i - 20) : i + 1]
                    stop_loss = float(recent_highs.max()) * (1 + self.liq_range_pct)

            sl_dist = abs(entry_price - stop_loss)
            if sl_dist == 0:
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

            signals.append(
                TradeSignal(
                    timestamp=ts,
                    symbol=symbol,
                    direction=direction,
                    style=best_style,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=self.rr_ratio,
                    position_size=qty,
                    leverage=self.leverage,
                    alignment_score=best_score,
                    meta={
                        "bias": bias,
                        "h4_confirm": h4_ok,
                        "entry_zone": entry_zone,
                    },
                )
            )

        logger.info(
            "Generated %d signals for %s (styles: %s)",
            len(signals),
            symbol,
            {s.style for s in signals} if signals else "–",
        )
        return signals
