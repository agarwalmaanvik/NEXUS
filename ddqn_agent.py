"""
ddqn_agent.py — Session-adaptive exploit layer for NEXUS v2 (Pillar 5).

Double DQN that adapts to a specific opponent's patterns during a single session.
Starts with a random policy, improves over ~30 hands.

Key design decisions:
  - Decoupled action selection (online net) from value estimation (target net)
    → reduces overestimation bias (same problem Double DQN was designed for)
  - Small capacity replay buffer (500 transitions) — session-level only, not training
  - Target net synced every TARGET_SYNC_EVERY steps
  - CFR strategy gates how much weight this gets (via exploit_signal from TellDetector)
    Low exploit → DDQN weight ≈ 0.  High exploit → DDQN dominates.

This is NOT a replacement for CFR. It is a bolt-on exploit module.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ---------------------------------------------------------------------------
N_ACTIONS          = 7
REPLAY_CAPACITY    = 500
BATCH_SIZE         = 64
GAMMA              = 0.95        # discount — hands are short, lower gamma correct
LR                 = 5e-4
TARGET_SYNC_EVERY  = 100         # sync target net every N update() calls
HIDDEN             = 256
# ---------------------------------------------------------------------------


class _QNet(nn.Module):
    """Lightweight Q-network: INPUT_DIM → 256 → 128 → 7."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, HIDDEN),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN, 128),
            nn.LeakyReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DDQNAgent:
    """
    Session-level Double DQN exploit agent.

    Usage:
        agent = DDQNAgent(input_dim=355)

        # At decision time:
        q_vals = agent.q_values(state_vec)            # [7] numpy
        strategy = softmax_masked(q_vals, legal_mask)

        # After each hand:
        agent.record(s, a, reward, s_next, done)
        agent.update()
    """

    def __init__(self, input_dim: int = 355, device: str = "cpu"):
        self.device      = device
        self.input_dim   = input_dim
        self._step_count = 0

        # Online network (action selection) and target network (Q-value estimation)
        self.online = _QNet(input_dim).to(device)
        self.target = _QNet(input_dim).to(device)
        self._sync_target()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)
        self.memory    = deque(maxlen=REPLAY_CAPACITY)

        # Epsilon for exploration (decays over session)
        self.epsilon     = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98  # per update() call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def q_values(self, state_vec: np.ndarray) -> np.ndarray:
        """
        Returns raw Q-values [7] for all actions.
        Caller applies legal mask and blends with CFR strategy.
        """
        self.online.eval()
        with torch.no_grad():
            t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
            q = self.online(t).squeeze(0).cpu().numpy()
        return q

    def strategy(self, state_vec: np.ndarray,
                 legal_mask: np.ndarray) -> np.ndarray:
        """
        Returns softmax probability distribution [7] over legal actions.
        Uses temperature scaling to convert Q-values to probabilities.
        """
        q = self.q_values(state_vec)
        q_masked = q.copy()
        q_masked[legal_mask == 0] = -1e9
        # Temperature softmax (T=0.5 → more decisive than uniform)
        q_shifted = q_masked - q_masked.max()
        exp_q = np.exp(q_shifted / 0.5)
        exp_q[legal_mask == 0] = 0.0
        total = exp_q.sum()
        return exp_q / total if total > 1e-8 else legal_mask / legal_mask.sum()

    def record(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the replay buffer."""
        self.memory.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def update(self) -> float | None:
        """
        Run one Double DQN gradient step.
        Returns loss value, or None if buffer not yet large enough.
        """
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        s  = torch.tensor(np.array(states),      dtype=torch.float32).to(self.device)
        a  = torch.tensor(actions,               dtype=torch.long).to(self.device)
        r  = torch.tensor(rewards,               dtype=torch.float32).to(self.device)
        s2 = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        d  = torch.tensor(dones,                 dtype=torch.float32).to(self.device)

        # Double DQN: online net selects action, target net evaluates it
        self.online.train()
        with torch.no_grad():
            # Action selection with online net
            online_next_q = self.online(s2)
            best_actions  = online_next_q.argmax(dim=1)
            # Value estimation with target net (decoupled)
            target_next_q = self.target(s2)
            target_vals   = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            td_targets    = r + GAMMA * target_vals * (1.0 - d)

        current_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(current_q, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % TARGET_SYNC_EVERY == 0:
            self._sync_target()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def reset_session(self) -> None:
        """Call at the start of a new session to forget opponent patterns."""
        self.memory.clear()
        self.epsilon = 1.0
        self._step_count = 0
        self._sync_target()

    def hands_observed(self) -> int:
        """How many transitions have been recorded this session."""
        return len(self.memory)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """Hard-copy online weights to target network."""
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
