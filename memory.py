import numpy as np
import torch
import random
from collections import deque


class ReplayBuffer:
    """Legacy FIFO buffer. Kept for compatibility."""
    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, advantages, value):
        self.buffer.append((state, advantages, value))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, advs, vals = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(advs),   dtype=torch.float32),
            torch.tensor(np.array(vals),   dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class ReservoirBuffer:
    """
    Reservoir sampling buffer — uniform distribution over all items seen,
    not just recent ones. Critical for unbiased CFR advantage training.

    Bug fixed: capacity parameter is now correctly respected.
    """
    def __init__(self, capacity: int = 1_000_000):
        self.buffer:    list = []
        self.capacity:  int  = capacity  # Fixed: was previously hardcoded to 2M
        self.total_seen: int = 0

    def add(self, state, advantages, value):
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, advantages, value))
        else:
            # Reservoir sampling: item n is kept with probability capacity / n
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (state, advantages, value)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, advs, vals = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(advs),   dtype=torch.float32),
            torch.tensor(np.array(vals),   dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class PerPlayerReservoirBuffer:
    """
    Separate ReservoirBuffers for each player in CFR.

    Deep CFR requires independent advantage memories per player:
    advantages for P0 and P1 have different counterfactual semantics and
    must NOT be mixed. Mixing is a common bug that breaks CFR convergence.

    Usage:
        buf = PerPlayerReservoirBuffer(n_players=2, capacity=500_000)
        buf.add(player=0, state=..., advantages=..., value=...)
        states, advs, vals = buf.sample(player=0, batch_size=512)
    """

    def __init__(self, n_players: int = 2, capacity: int = 500_000):
        self.n_players = n_players
        self.buffers   = {p: ReservoirBuffer(capacity) for p in range(n_players)}

    def add(self, player: int, state, advantages, value):
        self.buffers[player].add(state, advantages, value)

    def sample(self, player: int, batch_size: int):
        return self.buffers[player].sample(batch_size)

    def __len__(self) -> dict:
        return {p: len(buf) for p, buf in self.buffers.items()}

    def total(self) -> int:
        return sum(len(buf) for buf in self.buffers.values())

    def min_size(self) -> int:
        return min(len(buf) for buf in self.buffers.values())
