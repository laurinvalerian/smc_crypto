"""
DQN Exit Manager -- lightweight inference wrapper for live bot integration.

Usage in live_multi_bot.py::

    dqn = DQNExitManager("models/dqn_exit_manager.zip")
    action, confidence = dqn.predict(bar_features_dict)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from features.feature_extractor import EXIT_BAR_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Action IDs
ACTION_HOLD = 0
ACTION_EXIT = 1
ACTION_MOVE_SL = 2
ACTION_PARTIAL = 3

ACTION_NAMES = {
    ACTION_HOLD: "HOLD",
    ACTION_EXIT: "EXIT",
    ACTION_MOVE_SL: "MOVE_SL_TO_BE",
    ACTION_PARTIAL: "PARTIAL_EXIT",
}

try:
    from stable_baselines3 import DQN

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class DQNExitManager:
    """Wraps a trained DQN model for live exit predictions.

    Handles the case where stable-baselines3 is not installed (returns
    HOLD with zero confidence) so the live bot can always instantiate
    this class without crashing.
    """

    def __init__(self, model_path: str):
        self._model = None
        self._model_path = model_path

        if not SB3_AVAILABLE:
            logger.warning(
                "stable-baselines3 not installed -- DQN exit manager disabled"
            )
            return

        path = Path(model_path)
        # SB3 auto-appends .zip if needed
        if not path.exists() and not path.with_suffix(".zip").exists():
            logger.warning("DQN model not found at %s -- disabled", model_path)
            return

        try:
            self._model = DQN.load(str(path))
            logger.info("DQN exit manager loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load DQN model from %s: %s", model_path, exc)
            self._model = None

    def predict(self, features: dict[str, float]) -> tuple[int, float]:
        """Predict exit action from a 15-feature dict.

        Args:
            features: Dict with keys matching EXIT_BAR_FEATURE_NAMES.

        Returns:
            (action_id, confidence) where:
                action_id: 0=HOLD, 1=EXIT, 2=MOVE_SL, 3=PARTIAL
                confidence: max Q-value normalized to [0, 1]
        """
        if self._model is None:
            return ACTION_HOLD, 0.0

        # Build observation vector in canonical order
        obs = np.array(
            [features.get(k, 0.0) for k in EXIT_BAR_FEATURE_NAMES],
            dtype=np.float32,
        )
        obs = np.clip(obs, -5.0, 15.0)

        # Get action (deterministic)
        action, _ = self._model.predict(obs, deterministic=True)
        action_id = int(action)

        # Compute confidence from Q-values
        try:
            obs_tensor = self._model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            q_values = self._model.policy.q_net(obs_tensor)
            q_np = q_values.detach().cpu().numpy().flatten()
            # Normalize: sigmoid of max Q-value for [0, 1] range
            max_q = float(q_np.max())
            confidence = 1.0 / (1.0 + np.exp(-max_q))
        except Exception:
            confidence = 0.5

        return action_id, float(confidence)

    def predict_with_name(self, features: dict[str, float]) -> tuple[str, float]:
        """Like predict() but returns the action name instead of ID."""
        action_id, confidence = self.predict(features)
        return ACTION_NAMES.get(action_id, "HOLD"), confidence

    def is_available(self) -> bool:
        """Check if the model is loaded and ready for predictions."""
        return self._model is not None
