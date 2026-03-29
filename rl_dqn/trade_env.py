"""
TradeManagementEnv -- Gymnasium environment for in-trade exit management.

Replays one historical trade episode per reset(). The agent observes
15-dim bar features and chooses actions: HOLD, EXIT, MOVE_SL_TO_BE, PARTIAL_EXIT.

State space: Box(15,) float32 -- the 15 features from FeatureExtractor
Action space: Discrete(4) -- HOLD=0, EXIT=1, MOVE_SL_TO_BE=2, PARTIAL_EXIT=3
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from features.feature_extractor import EXIT_BAR_FEATURE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ACTION_HOLD = 0
ACTION_EXIT = 1
ACTION_MOVE_SL_TO_BE = 2
ACTION_PARTIAL_EXIT = 3

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
TIME_PENALTY = -0.001
BE_BONUS = 0.05
PARTIAL_FRACTION = 0.5


def _generate_dummy_episodes(n_episodes: int = 200,
                              max_bars: int = 60) -> list[np.ndarray]:
    """Create synthetic random-walk episodes for testing when no parquet data
    exists. Each episode is an (n_bars, 15) float32 array."""
    rng = np.random.default_rng(42)
    episodes: list[np.ndarray] = []
    for _ in range(n_episodes):
        n_bars = rng.integers(10, max_bars + 1)
        feats = np.zeros((n_bars, len(EXIT_BAR_FEATURE_NAMES)), dtype=np.float32)

        # Random walk for bar_unrealized_rr (index 1)
        pnl = 0.0
        max_fav = 0.0
        for b in range(n_bars):
            pnl += rng.normal(0.05, 0.3)
            pnl = float(np.clip(pnl, -3.0, 10.0))
            max_fav = max(max_fav, pnl)

            feats[b, 0] = float(b)                          # bars_held
            feats[b, 1] = pnl                                # bar_unrealized_rr
            feats[b, 2] = rng.uniform(0.001, 0.05)          # sl_distance_pct
            feats[b, 3] = max_fav                            # max_favorable_seen
            feats[b, 4] = 0.0                                # be_triggered
            feats[b, 5] = rng.choice([0.0, 0.33, 0.66, 1.0])  # asset_class_id
            feats[b, 6] = rng.uniform(0.2, 0.8)             # rsi_5m
            feats[b, 7] = float(b) / max_bars                # bars_held_normalized
            feats[b, 8] = rng.normal(0.0, 0.2)              # pnl_velocity
            feats[b, 9] = max(0.0, (max_fav - pnl) / max(max_fav, 1e-6))  # mfe_drawdown
            feats[b, 10] = rng.uniform(0.3, 0.8)            # time_in_profit_ratio
            feats[b, 11] = rng.uniform(0.5, 3.0)            # sl_distance_atr
            feats[b, 12] = rng.uniform(0.7, 1.3)            # regime_volatility
            feats[b, 13] = rng.uniform(0.1, 0.8)            # adx_1h
            feats[b, 14] = float(rng.integers(0, 5))        # opposing_structure_count

        episodes.append(feats)
    logger.info("Generated %d dummy episodes for testing", n_episodes)
    return episodes


def _load_episodes_from_parquet(
    episodes_dir: str,
    asset_classes: Optional[list[str]],
    walk_forward_start: Optional[str],
    walk_forward_end: Optional[str],
) -> list[np.ndarray]:
    """Load exit-episode parquet files grouped by trade_id.

    Returns a list of (n_bars, 15) float32 arrays, one per trade episode.
    Falls back to dummy episodes if no data is available.
    """
    if not PANDAS_AVAILABLE:
        logger.warning("pandas not available -- using dummy episodes")
        return _generate_dummy_episodes()

    ep_dir = Path(episodes_dir)
    if not ep_dir.exists():
        logger.warning("Episodes dir %s not found -- using dummy episodes", ep_dir)
        return _generate_dummy_episodes()

    # Discover parquet files matching *_exit_episodes.parquet
    parquet_files: list[Path] = sorted(ep_dir.glob("*_exit_episodes.parquet"))

    # Filter by asset class if requested
    if asset_classes:
        cls_set = {c.lower() for c in asset_classes}
        parquet_files = [
            p for p in parquet_files
            if any(c in p.stem.lower() for c in cls_set)
        ]

    if not parquet_files:
        logger.warning("No exit episode parquets found in %s -- using dummy episodes", ep_dir)
        return _generate_dummy_episodes()

    frames: list[pd.DataFrame] = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            frames.append(df)
            logger.info("Loaded %d rows from %s", len(df), pf.name)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pf, exc)

    if not frames:
        logger.warning("All parquet reads failed -- using dummy episodes")
        return _generate_dummy_episodes()

    data = pd.concat(frames, ignore_index=True)

    # Walk-forward time filter
    ts_col = None
    for candidate in ("timestamp", "bar_timestamp", "time", "date"):
        if candidate in data.columns:
            ts_col = candidate
            break

    if ts_col and (walk_forward_start or walk_forward_end):
        data[ts_col] = pd.to_datetime(data[ts_col], errors="coerce")
        if walk_forward_start:
            data = data[data[ts_col] >= pd.Timestamp(walk_forward_start)]
        if walk_forward_end:
            data = data[data[ts_col] <= pd.Timestamp(walk_forward_end)]

    # Synthesize trade_id if missing: detect episode boundaries where bars_held resets
    if "trade_id" not in data.columns:
        if "bars_held" in data.columns:
            # Episode boundary = where bars_held decreases (new trade starts)
            bh = data["bars_held"].values
            boundaries = np.where(np.diff(bh) < 0)[0] + 1
            trade_ids = np.zeros(len(data), dtype=np.int64)
            tid = 0
            prev = 0
            for b in boundaries:
                trade_ids[prev:b] = tid
                tid += 1
                prev = b
            trade_ids[prev:] = tid
            data = data.copy()
            data["trade_id"] = trade_ids
            logger.info("Synthesized %d trade_ids from bars_held boundaries", tid + 1)
        else:
            logger.warning("No trade_id or bars_held column -- using dummy episodes")
            return _generate_dummy_episodes()

    # Determine which feature columns are present
    available_feats = [f for f in EXIT_BAR_FEATURE_NAMES if f in data.columns]
    missing_feats = [f for f in EXIT_BAR_FEATURE_NAMES if f not in data.columns]
    if missing_feats:
        logger.info("Missing features (filled with defaults): %s", missing_feats)

    # Default values for missing features
    defaults: dict[str, float] = {
        "bars_held": 0.0,
        "bar_unrealized_rr": 0.0,
        "sl_distance_pct": 0.01,
        "max_favorable_seen": 0.0,
        "be_triggered": 0.0,
        "asset_class_id": 0.0,
        "rsi_5m": 0.5,
        "bars_held_normalized": 0.0,
        "pnl_velocity": 0.0,
        "mfe_drawdown": 0.0,
        "time_in_profit_ratio": 0.5,
        "sl_distance_atr": 1.0,
        "regime_volatility": 1.0,
        "adx_1h": 0.5,
        "opposing_structure_count": 0.0,
    }

    for feat in missing_feats:
        data[feat] = defaults.get(feat, 0.0)

    # Group by trade_id and build episode arrays
    episodes: list[np.ndarray] = []
    for _tid, group in data.groupby("trade_id"):
        group = group.sort_values("bars_held" if "bars_held" in group.columns else group.columns[0])
        arr = group[EXIT_BAR_FEATURE_NAMES].values.astype(np.float32)
        if len(arr) >= 2:
            episodes.append(arr)

    if not episodes:
        logger.warning("No valid episodes after grouping -- using dummy episodes")
        return _generate_dummy_episodes()

    logger.info("Loaded %d trade episodes from %d parquet files", len(episodes), len(parquet_files))
    return episodes


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

if GYM_AVAILABLE:
    _base_class = gym.Env
else:
    _base_class = object  # type: ignore[misc]


class TradeManagementEnv(_base_class):  # type: ignore[valid-type]
    """Gymnasium environment that replays historical trade episodes.

    Observation: Box(15,) -- the 15 EXIT_BAR_FEATURE_NAMES
    Actions: Discrete(4) -- HOLD, EXIT, MOVE_SL_TO_BE, PARTIAL_EXIT
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes_dir: str = "data/rl_training",
        asset_classes: Optional[list[str]] = None,
        walk_forward_start: Optional[str] = None,
        walk_forward_end: Optional[str] = None,
        target_rr: float = 3.0,
    ):
        if not GYM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for TradeManagementEnv. "
                "Install with: pip install gymnasium"
            )
        super().__init__()

        self.target_rr = target_rr
        self.episodes = _load_episodes_from_parquet(
            episodes_dir, asset_classes, walk_forward_start, walk_forward_end,
        )

        n_features = len(EXIT_BAR_FEATURE_NAMES)
        self.observation_space = spaces.Box(
            low=-5.0, high=15.0, shape=(n_features,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # Episode state
        self._current_episode: Optional[np.ndarray] = None
        self._step_idx: int = 0
        self._be_active: bool = False
        self._position_fraction: float = 1.0
        self._cumulative_partial_reward: float = 0.0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ep_idx = int(self._rng.integers(0, len(self.episodes)))
        self._current_episode = self.episodes[ep_idx]
        self._step_idx = 0
        self._be_active = False
        self._position_fraction = 1.0
        self._cumulative_partial_reward = 0.0

        obs = self._get_obs()
        info: dict[str, Any] = {
            "episode_length": len(self._current_episode),
            "episode_idx": ep_idx,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._current_episode is not None, "Call reset() before step()"

        ep = self._current_episode
        bar = ep[self._step_idx]
        unrealized_rr = float(bar[1])  # bar_unrealized_rr is index 1

        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"action": action}

        if action == ACTION_HOLD:
            reward = TIME_PENALTY
            self._step_idx += 1

        elif action == ACTION_EXIT:
            reward = unrealized_rr * self._position_fraction
            reward += self._cumulative_partial_reward
            terminated = True
            info["exit_reason"] = "agent_exit"

        elif action == ACTION_MOVE_SL_TO_BE:
            if not self._be_active:
                self._be_active = True
                reward = BE_BONUS
            else:
                reward = TIME_PENALTY  # already active, treat as hold
            self._step_idx += 1

        elif action == ACTION_PARTIAL_EXIT:
            if self._position_fraction > 0.25:
                partial_reward = PARTIAL_FRACTION * unrealized_rr * self._position_fraction
                self._cumulative_partial_reward += partial_reward
                reward = partial_reward
                self._position_fraction *= (1.0 - PARTIAL_FRACTION)
            else:
                reward = TIME_PENALTY  # position too small, treat as hold
            self._step_idx += 1

        # Check termination conditions (if not already terminated)
        if not terminated:
            # SL hit: unrealized RR drops below -1
            if unrealized_rr <= -1.0:
                if self._be_active and unrealized_rr >= -0.05:
                    # BE is active and price near entry -- SL at breakeven
                    reward = 0.0
                else:
                    reward = unrealized_rr * self._position_fraction
                reward += self._cumulative_partial_reward
                terminated = True
                info["exit_reason"] = "sl_hit"

            # TP hit
            elif unrealized_rr >= self.target_rr:
                reward = unrealized_rr * self._position_fraction
                reward += self._cumulative_partial_reward
                terminated = True
                info["exit_reason"] = "tp_hit"

            # Max bars reached
            elif self._step_idx >= len(ep):
                reward = unrealized_rr * self._position_fraction
                reward += self._cumulative_partial_reward
                truncated = True
                info["exit_reason"] = "max_bars"

        obs = self._get_obs()
        info["unrealized_rr"] = unrealized_rr
        info["position_fraction"] = self._position_fraction
        info["be_active"] = self._be_active
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Return the current bar's feature vector, clamped to obs space."""
        assert self._current_episode is not None
        idx = min(self._step_idx, len(self._current_episode) - 1)
        obs = self._current_episode[idx].copy()
        return np.clip(obs, -5.0, 15.0).astype(np.float32)
