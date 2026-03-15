"""
═══════════════════════════════════════════════════════════════════
 rl_brain.py  –  Per-Bot PPO Brain (Reinforcement Learning)
 ──────────────────────────────────────────────────────────────
 Each of the 100 bots has its own lightweight PPO agent that
 learns a yes/no trade filter based on market features.

 • Observation  : 12-dim vector (alignment, ATR-norm, EMAs, vol, …)
 • Action       : Binary  –  0 = skip, 1 = take the trade
 • Reward       : Pure PnL change in %  (no shaping, no R:R bonus)
 • Update       : Mini-batch PPO with clipped surrogate objective

 The brain is called only when the rule-based SMC strategy already
 signals a potential entry.  It acts as a gating filter.
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import math
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

OBS_DIM = 12           # observation vector length
HIDDEN_DIM = 64        # hidden layer size
GAMMA = 0.99           # discount factor
CLIP_EPS = 0.2         # PPO clipping epsilon
LR = 3e-4              # learning rate
PPO_EPOCHS = 4         # epochs per update
MINI_BATCH_SIZE = 32   # mini-batch size
BUFFER_CAPACITY = 256  # transitions before PPO update
VALUE_COEFF = 0.5      # value loss weight
ENTROPY_COEFF = 0.01   # entropy bonus weight


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
        return np.zeros(OBS_DIM, dtype=np.float32)

    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    highs = np.array([c["high"] for c in candles], dtype=np.float64)
    lows = np.array([c["low"] for c in candles], dtype=np.float64)
    volumes = np.array([c["volume"] for c in candles], dtype=np.float64)

    price = closes[-1]
    if price <= 0:
        return np.zeros(OBS_DIM, dtype=np.float32)

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
#  RLBrain – per-bot PPO agent
# ═══════════════════════════════════════════════════════════════════

class RLBrain:
    """
    Lightweight PPO agent that gates trade entries.

    Usage
    -----
    1. Call ``should_trade(obs)`` before each potential entry.
       Returns True/False.

    2. Call ``record_outcome(reward)`` when a trade (or skip) resolves.
       *reward* = PnL change in % (positive or negative).

    3. The brain auto-updates its policy every BUFFER_CAPACITY steps.
    """

    def __init__(self, bot_tag: str, model_dir: Path | None = None) -> None:
        self.bot_tag = bot_tag
        self._model_dir = model_dir or Path("rl_models")
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = self._model_dir / f"{bot_tag}_ppo.pt"

        self._enabled = TORCH_AVAILABLE
        if not self._enabled:
            logger.warning(
                "%s: PyTorch not available – RL brain disabled (random pass-through)",
                bot_tag,
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

    def should_trade(self, obs: np.ndarray) -> bool:
        """
        Given an observation vector, sample an action from the policy.
        Returns True (take trade) or False (skip).
        Stores the transition internally for later PPO update.
        """
        if not self._enabled:
            return True  # pass-through if no PyTorch

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            logits, value = self._net(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        act = int(action.item())
        self._obs_buf.append(obs.copy())
        self._act_buf.append(act)
        self._logp_buf.append(float(log_prob.item()))
        self._val_buf.append(float(value.item()))

        return act == 1  # 1 = trade, 0 = skip

    def record_outcome(self, reward: float, done: bool = True) -> None:
        """
        Record the reward for the most recent decision.
        *reward* should be pure PnL change in % (e.g. +1.5 or -0.8).
        *done* marks end of episode (True for each closed trade or skip).
        """
        if not self._enabled:
            return

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
                nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
                self._optimiser.step()

        self._total_updates += 1
        logger.debug(
            "%s: PPO update #%d (buffer=%d)",
            self.bot_tag, self._total_updates, n,
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
                },
                self._model_path,
            )
        except Exception as exc:
            logger.warning("%s: model save failed: %s", self.bot_tag, exc)

    def _load_model(self) -> None:
        if not self._enabled:
            return
        if not self._model_path.exists():
            return
        try:
            ckpt = torch.load(self._model_path, weights_only=False)
            self._net.load_state_dict(ckpt["net"])
            self._optimiser.load_state_dict(ckpt["optimiser"])
            self._total_updates = ckpt.get("updates", 0)
            logger.info(
                "%s: loaded model (%d prior updates)",
                self.bot_tag, self._total_updates,
            )
        except Exception as exc:
            logger.warning("%s: model load failed, starting fresh: %s", self.bot_tag, exc)

    # ── Flush remaining buffer on shutdown ────────────────────────

    def flush(self) -> None:
        """Run a final PPO update with whatever remains in the buffer."""
        if self._enabled and len(self._rew_buf) > 0:
            self._ppo_update()
