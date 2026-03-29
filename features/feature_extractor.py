"""
Unified feature extraction for the RL exit model.

Schema versioning: every trained model stores the SCHEMA_VERSION it was built
with.  At inference time the FeatureExtractor checks the current version against
the model's expected version and logs a warning on mismatch.  This prevents
silent feature-drift between training and serving.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version -- bump whenever the feature vector changes
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Ordered feature names (positions matter for the model)
# ---------------------------------------------------------------------------
EXIT_BAR_FEATURE_NAMES: list[str] = [
    "bars_held",
    "bar_unrealized_rr",
    "sl_distance_pct",
    "max_favorable_seen",
    "be_triggered",
    "asset_class_id",
    "rsi_5m",
    "bars_held_normalized",
    "pnl_velocity",
    "mfe_drawdown",
    "time_in_profit_ratio",
    "sl_distance_atr",
    "regime_volatility",
    "adx_1h",
    "opposing_structure_count",
]

# ---------------------------------------------------------------------------
# Asset-class encoding (float for the model, matches rl_brain_v2 semantics)
# ---------------------------------------------------------------------------
ASSET_CLASS_MAP: Dict[str, float] = {
    "crypto": 0.0,
    "forex": 0.33,
    "stocks": 0.66,
    "commodities": 1.0,
}

# ---------------------------------------------------------------------------
# Expected value ranges per feature (for validate_features)
# ---------------------------------------------------------------------------
_FEATURE_RANGES: Dict[str, tuple[float, float]] = {
    "bars_held":                 (0.0, 576.0),
    "bar_unrealized_rr":         (-3.0, 10.0),
    "sl_distance_pct":           (0.0, 0.1),
    "max_favorable_seen":        (0.0, 10.0),
    "be_triggered":              (0.0, 1.0),
    "asset_class_id":            (0.0, 1.0),
    "rsi_5m":                    (0.0, 1.0),
    "bars_held_normalized":      (0.0, 1.0),
    "pnl_velocity":              (-1.0, 1.0),
    "mfe_drawdown":              (0.0, 1.0),
    "time_in_profit_ratio":      (0.0, 1.0),
    "sl_distance_atr":           (0.0, 5.0),
    "regime_volatility":         (0.5, 2.0),
    "adx_1h":                    (0.0, 1.0),
    "opposing_structure_count":  (0.0, 10.0),
}


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Stateless feature builder shared by training and live inference.

    Usage::

        fe = FeatureExtractor()
        feats = fe.extract_exit_bar_features(bars_held=12, ...)
        fe.validate_features(feats)            # optional sanity check
        vec = fe.to_numpy(feats)               # ordered float32 array
    """

    # ---- schema helpers ---------------------------------------------------

    @staticmethod
    def schema_version() -> str:
        return SCHEMA_VERSION

    @staticmethod
    def check_schema(expected_version: str) -> bool:
        """Return True if versions match; log warning otherwise."""
        if expected_version != SCHEMA_VERSION:
            logger.warning(
                "Feature schema mismatch: model expects %s, current is %s",
                expected_version,
                SCHEMA_VERSION,
            )
            return False
        return True

    # ---- main extraction --------------------------------------------------

    @staticmethod
    def extract_exit_bar_features(
        bars_held: int,
        unrealized_pnl_pct: float,
        risk_pct: float,
        sl_distance_pct: float,
        max_favorable_seen: float,
        be_triggered: bool,
        asset_class: str,
        rsi_5m: float,
        adx_1h: float,
        atr_5m: float,
        prev_unrealized_pnl_pct: float,
        bars_in_profit: int,
        std_returns_50: float,
        std_returns_200: float,
        structure_breaks_against: int,
        max_hold_bars: int = 576,
    ) -> dict[str, float]:
        """Build the 15-element exit-bar feature dict.

        All heavy normalisation / clamping happens here so callers pass raw
        values and always get a model-ready dict back.
        """
        safe_risk = max(risk_pct, 1e-6)

        # 1. bars_held (raw, integer-valued float)
        f_bars_held = float(np.clip(bars_held, 0, max_hold_bars))

        # 2. bar_unrealized_rr
        f_bar_unrealized_rr = float(np.clip(
            unrealized_pnl_pct / safe_risk, -3.0, 10.0,
        ))

        # 3. sl_distance_pct
        f_sl_distance_pct = float(np.clip(sl_distance_pct, 0.0, 0.1))

        # 4. max_favorable_seen (in RR multiples)
        f_max_favorable_seen = float(np.clip(
            max_favorable_seen / safe_risk, 0.0, 10.0,
        ))

        # 5. be_triggered
        f_be_triggered = 1.0 if be_triggered else 0.0

        # 6. asset_class_id
        f_asset_class_id = ASSET_CLASS_MAP.get(asset_class, 0.0)

        # 7. rsi_5m  (raw [0,100] -> [0,1])
        f_rsi_5m = float(np.clip(rsi_5m / 100.0, 0.0, 1.0))

        # 8. bars_held_normalized
        f_bars_held_normalized = float(np.clip(
            bars_held / max(max_hold_bars, 1), 0.0, 1.0,
        ))

        # 9. pnl_velocity
        f_pnl_velocity = float(np.clip(
            (unrealized_pnl_pct - prev_unrealized_pnl_pct) / safe_risk,
            -1.0,
            1.0,
        ))

        # 10. mfe_drawdown
        if max_favorable_seen > 0.0:
            f_mfe_drawdown = float(np.clip(
                (max_favorable_seen - unrealized_pnl_pct) / max_favorable_seen,
                0.0,
                1.0,
            ))
        else:
            f_mfe_drawdown = 0.0

        # 11. time_in_profit_ratio
        f_time_in_profit_ratio = float(np.clip(
            bars_in_profit / max(bars_held, 1), 0.0, 1.0,
        ))

        # 12. sl_distance_atr
        f_sl_distance_atr = float(np.clip(
            sl_distance_pct / max(atr_5m, 1e-6), 0.0, 5.0,
        ))

        # 13. regime_volatility
        if std_returns_200 > 1e-8:
            f_regime_volatility = float(np.clip(
                std_returns_50 / std_returns_200, 0.5, 2.0,
            ))
        else:
            f_regime_volatility = 1.0

        # 14. adx_1h (normalised)
        f_adx_1h = float(np.clip(adx_1h / 50.0, 0.0, 1.0))

        # 15. opposing_structure_count
        f_opposing_structure_count = float(min(structure_breaks_against, 10))

        return {
            "bars_held":                 f_bars_held,
            "bar_unrealized_rr":         f_bar_unrealized_rr,
            "sl_distance_pct":           f_sl_distance_pct,
            "max_favorable_seen":        f_max_favorable_seen,
            "be_triggered":              f_be_triggered,
            "asset_class_id":            f_asset_class_id,
            "rsi_5m":                    f_rsi_5m,
            "bars_held_normalized":      f_bars_held_normalized,
            "pnl_velocity":              f_pnl_velocity,
            "mfe_drawdown":              f_mfe_drawdown,
            "time_in_profit_ratio":      f_time_in_profit_ratio,
            "sl_distance_atr":           f_sl_distance_atr,
            "regime_volatility":         f_regime_volatility,
            "adx_1h":                    f_adx_1h,
            "opposing_structure_count":  f_opposing_structure_count,
        }

    # ---- conversion helpers -----------------------------------------------

    @staticmethod
    def to_numpy(features: dict[str, float]) -> np.ndarray:
        """Return a float32 array ordered by EXIT_BAR_FEATURE_NAMES."""
        return np.array(
            [features[k] for k in EXIT_BAR_FEATURE_NAMES], dtype=np.float32,
        )

    # ---- validation -------------------------------------------------------

    @staticmethod
    def validate_features(features: dict[str, float]) -> list[str]:
        """Check all 15 keys are present and values fall in expected ranges.

        Returns a list of human-readable violation strings (empty == all OK).
        """
        violations: list[str] = []

        # missing / extra keys
        expected = set(EXIT_BAR_FEATURE_NAMES)
        actual = set(features.keys())
        for k in expected - actual:
            violations.append(f"missing key: {k}")
        for k in actual - expected:
            violations.append(f"unexpected key: {k}")

        # range checks
        for k in expected & actual:
            lo, hi = _FEATURE_RANGES[k]
            v = features[k]
            if not (lo <= v <= hi):
                violations.append(
                    f"{k}={v:.6f} out of range [{lo}, {hi}]"
                )

        if violations:
            logger.warning("Feature validation issues: %s", violations)

        return violations
