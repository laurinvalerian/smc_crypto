"""
Shared Feature Schema — Single Source of Truth for XGBoost feature names.

Both training (backtest/generate_rl_data.py) and live (live_multi_bot.py)
MUST import feature names from here. Never define feature names inline.

The canonical list was extracted from the trained model's feat_names
(models/rl_entry_filter.pkl, 40 features, 2026-04-01).
"""
from __future__ import annotations


# The 40 features the entry_quality XGBoost model expects, in order.
ENTRY_QUALITY_FEATURES: list[str] = [
    # Structure direction per TF (causal SMC indicators)
    "struct_1d", "struct_4h", "struct_1h", "struct_15m", "struct_5m",
    # Break decay per TF
    "decay_1d", "decay_4h", "decay_1h", "decay_15m", "decay_5m",
    # Premium / Discount zone
    "premium_discount",
    # Boolean component flags
    "h4_confirms", "h4_poi", "h1_confirms", "h1_choch",
    "precision_trigger", "volume_ok",
    # EMA distances (normalized)
    "ema20_dist_5m", "ema50_dist_5m", "ema20_dist_1h", "ema50_dist_1h",
    # ATR (normalized)
    "atr_5m_norm", "atr_1h_norm", "atr_daily_norm",
    # Oscillators
    "rsi_5m", "rsi_1h",
    # Volume & trend
    "volume_ratio", "adx_1h",
    # Time encoding (cyclical)
    "hour_sin", "hour_cos",
    # SMC indicator counts (causal)
    "fvg_bull_active", "fvg_bear_active",
    "ob_bull_active", "ob_bear_active",
    "liq_above_count", "liq_below_count",
    # Symbol characteristics (pre-computed, shipped as symbol_ranks.json)
    "symbol_volatility_rank", "symbol_liquidity_rank", "symbol_spread_rank",
    # Asset class encoding
    "asset_class_id",
]

# Features excluded from entry_quality model (data leaks in training context)
# These ARE still computed and passed — used by TP/BE/sizing models.
ENTRY_QUALITY_EXCLUDE: frozenset[str] = frozenset({
    "has_entry_zone",
    "alignment_score",
})

# All features including excluded ones (for models that need them)
ALL_FEATURES: list[str] = ENTRY_QUALITY_FEATURES + sorted(ENTRY_QUALITY_EXCLUDE)


def validate_against_model(model_feat_names: list[str]) -> tuple[set[str], set[str]]:
    """Compare schema features against a model's feat_names.

    Returns (missing_from_schema, extra_in_schema) relative to the model.
    Both sets should be empty for a valid match.
    """
    schema_set = set(ENTRY_QUALITY_FEATURES)
    model_set = set(model_feat_names)
    return model_set - schema_set, schema_set - model_set
