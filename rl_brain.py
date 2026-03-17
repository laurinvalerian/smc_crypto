"""
═══════════════════════════════════════════════════════════════════
 rl_brain.py  –  Central PPO Brain (shared across all coins)
 ──────────────────────────────────────────────────────────────
 A single PPO agent learns a yes/no trade filter across all 100
 coins. Each decision includes the coin identifier as an extra
 feature so the shared model can specialise per asset.

 • Observation  : 12-dim market features + 1 coin-id feature
 • Action       : Binary  –  0 = skip, 1 = take the trade
 • Reward       : Pure PnL change in %  (no shaping, no R:R bonus)
 • Update       : Mini-batch PPO with clipped surrogate objective

 The brain is called only when the rule-based SMC strategy already
 signals a potential entry. It acts as a gating filter.
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

BASE_OBS_DIM = 12      # market feature length (from extract_features)
OBS_DIM = BASE_OBS_DIM + 1  # +1 coin-id feature
DEFAULT_COIN_FEATURE_DENOM = 100  # expected number of coins for normalisation
HIDDEN_DIM = 64        # hidden layer size
GAMMA = 0.99           # discount factor
CLIP_EPS = 0.2         # PPO clipping epsilon
LR = 3e-4              # learning rate
PPO_EPOCHS = 4         # epochs per update
MINI_BATCH_SIZE = 32   # mini-batch size
BUFFER_CAPACITY = 256  # transitions before PPO update
VALUE_COEFF = 0.5      # value loss weight
ENTROPY_COEFF = 0.01   # entropy bonus weight
GRAD_CLIP_MAX = 0.5    # max gradient norm for clipping


# ═══════════════════════════════════════════════════════════════════
#  Feature extraction (from candle buffer)
# ═══════════════════════════════════════════════════════════════════

def extract_features(
    candles: list[dict[str, Any]],
    alignment_score: float,
    direction: str,
) -> np.ndarray:
    """
    Build a 12-dim observation vector from the candle buffer and the
    SMC alignment score / direction.

    Features:
      0  alignment_score           [0, 1]
      1  direction_sign            -1 (short) or +1 (long)
      2  atr_14_normalised         ATR(14) / close
      3  ema20_distance            (close - EMA20) / close
      4  ema50_distance            (close - EMA50) / close
      5  ema_cross                 1 if EMA20 > EMA50, else 0
      6  volume_ratio              current_vol / avg_vol(20)
      7  close_return_1            1-bar return
      8  close_return_5            5-bar return
      9  close_return_20           20-bar return
     10  high_low_range_norm       (H-L) / close  (current bar)
     11  rsi_14_normalised         RSI(14) scaled to [0, 1]
    """
    n = len(candles)
    if n < 50:
        return np.zeros(BASE_OBS_DIM, dtype=np.float32)

    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    highs = np.array([c["high"] for c in candles], dtype=np.float64)
    lows = np.array([c["low"] for c in candles], dtype=np.float64)
    volumes = np.array([c["volume"] for c in candles], dtype=np.float64)

    price = closes[-1]
    if price <= 0:
        return np.zeros(BASE_OBS_DIM, dtype=np.float32)

    # ATR(14)
    trs: list[float] = []
    for i in range(-14, 0):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr14 = float(np.mean(trs))

    # EMAs
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)

    # Volume
    avg_vol_20 = float(np.mean(volumes[-20:])) if np.mean(volumes[-20:]) > 0 else 1.0
    vol_ratio = float(volumes[-1]) / avg_vol_20

    # Returns
    ret1 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0
    ret5 = (closes[-1] - closes[-6]) / closes[-6] if n >= 6 and closes[-6] > 0 else 0.0
    ret20 = (closes[-1] - closes[-21]) / closes[-21] if n >= 21 and closes[-21] > 0 else 0.0

    # RSI(14)
    rsi = _rsi(closes, 14)

    obs = np.array([
        alignment_score,
        1.0 if direction == "long" else -1.0,
        atr14 / price,
        (price - ema20) / price,
        (price - ema50) / price,
        1.0 if ema20 > ema50 else 0.0,
        min(vol_ratio, 5.0),           # clamp extreme volume spikes
        ret1,
        ret5,
        ret20,
        (highs[-1] - lows[-1]) / price,
        rsi / 100.0,
    ], dtype=np.float32)

    return obs


def _ema(data: np.ndarray, span: int) -> float:
    """Simple EMA of the last value."""
    alpha = 2.0 / (span + 1)
    ema = float(data[0])
    for v in data[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    return ema


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index over the last *period* bars."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ═══════════════════════════════════════════════════════════════════
#  PPO Actor-Critic Network
# ═══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class ActorCritic(nn.Module):
        """Small MLP with shared trunk, separate policy and value heads."""

        def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN_DIM) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden, 2)   # 2 actions: skip / trade
            self.value_head = nn.Linear(hidden, 1)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.shared(x)
            logits = self.policy_head(h)
            value = self.value_head(h).squeeze(-1)
            return logits, value


# ═══════════════════════════════════════════════════════════════════
#  CentralRLBrain – shared PPO agent
# ═══════════════════════════════════════════════════════════════════

class CentralRLBrain:
    """
    Shared PPO agent that gates trade entries across all coins.

    Usage
    -----
    1. Call ``should_trade(obs, coin_id)`` before each potential entry.
       Returns True/False.

    2. Call ``record_outcome(reward, coin_id)`` when a trade (or skip)
       resolves. *reward* = PnL change in % (positive or negative).

    3. The brain auto-updates its policy every BUFFER_CAPACITY steps.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        coin_ids: list[str] | None = None,
    ) -> None:
        self._model_dir = model_dir or Path("rl_models")
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = self._model_dir / "central_rl.pt"

        # Stable mapping coin_id -> index (persisted with the model)
        self._coin_to_idx: dict[str, int] = {}
        if coin_ids:
            self._coin_to_idx = {cid: i for i, cid in enumerate(coin_ids)}
        self._next_coin_idx: int = len(self._coin_to_idx)
        coin_count_or_default = len(self._coin_to_idx) if self._coin_to_idx else DEFAULT_COIN_FEATURE_DENOM
        self._feature_denom: float = float(max(1, coin_count_or_default))

        self._enabled = TORCH_AVAILABLE
        if not self._enabled:
            logger.warning(
                "CentralRLBrain: PyTorch not available – RL brain disabled (pass-through)",
            )
            return

        self._net = ActorCritic()
        self._optimiser = optim.Adam(self._net.parameters(), lr=LR)

        # Rollout buffer
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[int] = []
        self._logp_buf: list[float] = []
        self._val_buf: list[float] = []
        self._rew_buf: list[float] = []
        self._done_buf: list[bool] = []

        self._total_updates: int = 0
        self._load_model()

    # ── Public API ────────────────────────────────────────────────

    def should_trade(self, obs: np.ndarray, coin_id: Any) -> bool:
        """
        Given an observation vector, sample an action from the policy.
        Returns True (take trade) or False (skip).
        Stores the transition internally for later PPO update.
        """
        if not self._enabled:
            return True  # pass-through if no PyTorch

        obs_with_coin = self._augment_obs(obs, coin_id)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_with_coin).float().unsqueeze(0)
            logits, value = self._net(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        act = int(action.item())
        self._obs_buf.append(obs_with_coin.copy())
        self._act_buf.append(act)
        self._logp_buf.append(float(log_prob.item()))
        self._val_buf.append(float(value.item()))

        return act == 1  # 1 = trade, 0 = skip

    def record_outcome(
        self,
        reward: float,
        coin_id: Any | None = None,
        done: bool = True,
    ) -> None:
        """
        Record the reward for the most recent decision.
        *reward* should be pure PnL change in % (e.g. +1.5 or -0.8).
        *done* marks end of episode (True for each closed trade or skip).
        """
        if not self._enabled:
            return

        if coin_id is not None:
            self._encode_coin_id(coin_id)

        self._rew_buf.append(reward)
        self._done_buf.append(done)

        if len(self._rew_buf) >= BUFFER_CAPACITY:
            self._ppo_update()

    # ── PPO update ────────────────────────────────────────────────

    def _ppo_update(self) -> None:
        """Run PPO update on the collected buffer, then clear it."""
        n = len(self._rew_buf)
        if n == 0:
            return

        obs_t = torch.from_numpy(np.stack(self._obs_buf[:n])).float()
        act_t = torch.tensor(self._act_buf[:n], dtype=torch.long)
        old_logp_t = torch.tensor(self._logp_buf[:n], dtype=torch.float32)
        old_val_t = torch.tensor(self._val_buf[:n], dtype=torch.float32)

        # Compute returns and advantages (GAE-lambda = 1 for simplicity)
        rewards = np.array(self._rew_buf[:n], dtype=np.float64)
        dones = np.array(self._done_buf[:n], dtype=np.float64)
        returns = np.zeros(n, dtype=np.float64)
        running = 0.0
        for i in reversed(range(n)):
            if dones[i]:
                running = 0.0
            running = rewards[i] + GAMMA * running
            returns[i] = running

        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages = returns_t - old_val_t
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        # PPO epochs
        for _ in range(PPO_EPOCHS):
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, MINI_BATCH_SIZE):
                end = min(start + MINI_BATCH_SIZE, n)
                mb_idx = indices[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_ret = returns_t[mb_idx]
                mb_adv = advantages[mb_idx]

                logits, values = self._net(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_ret)
                loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

                self._optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), GRAD_CLIP_MAX)
                self._optimiser.step()

        self._total_updates += 1
        logger.debug(
            "CentralRLBrain: PPO update #%d (buffer=%d)",
            self._total_updates, n,
        )

        # Clear buffer
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._val_buf.clear()
        self._rew_buf.clear()
        self._done_buf.clear()

        # Auto-save
        self._save_model()

    # ── Persistence ───────────────────────────────────────────────

    def _save_model(self) -> None:
        if not self._enabled:
            return
        try:
            torch.save(
                {
                    "net": self._net.state_dict(),
                    "optimiser": self._optimiser.state_dict(),
                    "updates": self._total_updates,
                    "coin_to_idx": self._coin_to_idx,
                },
                self._model_path,
            )
        except Exception as exc:
            logger.warning("CentralRLBrain: model save failed: %s", exc)

    def _load_model(self) -> None:
        if not self._enabled:
            return
        if not self._model_path.exists():
            return
        try:
            ckpt = torch.load(self._model_path, weights_only=True)
            self._net.load_state_dict(ckpt["net"])
            self._optimiser.load_state_dict(ckpt["optimiser"])
            self._total_updates = ckpt.get("updates", 0)
            if isinstance(ckpt.get("coin_to_idx"), dict):
                self._coin_to_idx = {}
                for k, v in ckpt["coin_to_idx"].items():
                    idx_val = int(v)
                    str_key = str(k)
                    self._coin_to_idx[str_key] = idx_val
                self._next_coin_idx = len(self._coin_to_idx)
                if self._coin_to_idx:
                    self._feature_denom = float(max(1, len(self._coin_to_idx)))
            logger.info(
                "CentralRLBrain: loaded model (%d prior updates, %d coins)",
                self._total_updates,
                len(self._coin_to_idx),
            )
        except Exception as exc:
            logger.warning("CentralRLBrain: model load failed, starting fresh: %s", exc)

    # ── Flush remaining buffer on shutdown ────────────────────────

    def flush(self) -> None:
        """Run a final PPO update with whatever remains in the buffer."""
        if self._enabled and len(self._rew_buf) > 0:
            self._ppo_update()

    # ── Coin handling ─────────────────────────────────────────────

    def _encode_coin_id(self, coin_id: Any) -> int:
        """
        Map *coin_id* (str/int) to a stable integer index.
        Keys are normalised to strings for checkpoint stability.
        Unknown coins are appended to the mapping.
        """
        if isinstance(coin_id, int) and coin_id < 0:
            raise ValueError("coin_id must be non-negative")
        key = str(coin_id)
        if key not in self._coin_to_idx:
            self._coin_to_idx[key] = self._next_coin_idx
            self._next_coin_idx += 1
            self._feature_denom = max(self._feature_denom, float(self._next_coin_idx))
        return self._coin_to_idx[key]

    def _coin_feature(self, coin_id: Any) -> float:
        """Normalise coin index to [0, 1]."""
        idx = self._encode_coin_id(coin_id)
        feature = float(idx) / self._feature_denom
        return min(feature, 1.0)

    def _augment_obs(self, obs: np.ndarray, coin_id: Any) -> np.ndarray:
        """Append coin-id feature to the observation vector."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.shape[0] != BASE_OBS_DIM:
            raise ValueError(
                f"Expected obs of length {BASE_OBS_DIM}, got {obs_arr.shape[0]}"
            )
        coin_feat = self._coin_feature(coin_id)
        return np.append(obs_arr, np.float32(coin_feat))


# Backwards compatibility alias
RLBrain = CentralRLBrain
