"""
NEXUS CFR Agent — Dual-Network Architecture

Deep CFR requires two distinct networks:
  1. advantage_net  : learns regret/advantage estimates per iteration
                      (trained fresh each iteration on advantage buffer)
  2. strategy_net   : accumulates the average strategy across iterations
                      (average strategy converges to Nash; current doesn't)

At inference, ALWAYS use strategy_net. advantage_net is a training artefact.

Input to BOTH nets: 348-dim vector (171 game + 169 range + 8 tell features)
"""

import os
import copy
import torch
import torch.optim as optim
import numpy as np

from networks import NEXUS_GTO_Net, INPUT_DIM, N_ACTIONS

CHECKPOINT_DIR = "checkpoints"


class CFRAgent:
    """
    Manages the dual-network CFR agent.

    advantage_net  : Computes action regrets during CFR traversal.
    strategy_net   : Maintains running average strategy (used at inference).
    historical_nets: List of past strategy_net snapshots for FSP.
    """

    def __init__(self, device: str = "cpu", hidden_dim: int = 512,
                 num_blocks: int = 6, lr: float = 1e-4, input_dim: int = INPUT_DIM):
        self.device     = device
        self.input_dim  = input_dim

        # Primary advantage network (trained per CFR iteration)
        self.net = NEXUS_GTO_Net(input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_blocks=num_blocks).to(device)

        # Average strategy network (converges to Nash equilibrium)
        self.strategy_net = NEXUS_GTO_Net(input_dim=input_dim,
                                           hidden_dim=hidden_dim,
                                           num_blocks=num_blocks).to(device)

        # Separate optimisers; strategy_net uses smaller LR (smoother averaging)
        self.optimizer          = optim.Adam(self.net.parameters(), lr=lr)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=lr * 0.5)

        # FSP historical pool
        self.historical_nets: list[NEXUS_GTO_Net] = []

    # ------------------------------------------------------------------
    # Batch inference (used in parallel training data collection)
    # ------------------------------------------------------------------

    def get_batch_strategy(self, states: np.ndarray,
                           legal_masks: np.ndarray | None = None,
                           rag_priors: np.ndarray | None = None,
                           use_strategy_net: bool = False) -> np.ndarray:
        """
        Returns strategy matrix [batch, 7] for a batch of states.

        Args:
            states       : [N, 348] float32 numpy array.
            legal_masks  : [N, 7]   boolean numpy array (or None = all legal).
            rag_priors   : [N, 7]   retrieved RAG priors (or None).
            use_strategy_net: If True, use strategy_net (inference). Else advantage_net.
        """
        net = self.strategy_net if use_strategy_net else self.net
        net.eval()
        batch_size = states.shape[0]

        with torch.no_grad():
            t = torch.from_numpy(states).float().to(self.device)
            adv, _, _ = net(t)
            adv_np = adv.cpu().numpy()  # [N, 7]

        # Blend with RAG prior if provided
        if rag_priors is not None:
            alpha = net.get_alpha()
            # rag_priors: [N, 7]
            adv_np = alpha * adv_np + (1.0 - alpha) * rag_priors

        # Regret matching + legal mask
        strategies = np.maximum(adv_np, 0.0)  # clip negatives

        if legal_masks is not None:
            strategies *= legal_masks.astype(np.float32)

        row_sums = strategies.sum(axis=1, keepdims=True)
        # Where sum=0, fall back to uniform over legal moves
        zero_rows = (row_sums < 1e-8).squeeze(1)
        strategies[~zero_rows] /= row_sums[~zero_rows]

        if zero_rows.any():
            if legal_masks is not None:
                fallback = legal_masks[zero_rows].astype(np.float32)
                fallback_sums = fallback.sum(axis=1, keepdims=True)
                fallback /= np.where(fallback_sums > 0, fallback_sums, 1.0)
                strategies[zero_rows] = fallback
            else:
                strategies[zero_rows] = 1.0 / N_ACTIONS

        # Sample actions
        actions = torch.multinomial(
            torch.tensor(strategies, dtype=torch.float32), 1
        ).squeeze(1).numpy()

        return actions

    def get_single_strategy(self, state_vec: np.ndarray,
                            legal_moves: list,
                            rag_prior: np.ndarray | None = None,
                            use_strategy_net: bool = True) -> np.ndarray:
        """
        Returns strategy [7] for a single state (at inference time).
        Uses strategy_net by default.
        """
        net = self.strategy_net if use_strategy_net else self.net
        net.eval()

        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for m in legal_moves:
            legal_mask[m] = 1.0

        t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        rag_t = torch.from_numpy(rag_prior).float().unsqueeze(0) if rag_prior is not None else None

        with torch.no_grad():
            strategy = net.get_strategy(
                t,
                rag_prior=rag_t,
                legal_mask=torch.from_numpy(legal_mask).unsqueeze(0)
            )
        return strategy.numpy()

    # ------------------------------------------------------------------
    # Historical ghost management (FSP)
    # ------------------------------------------------------------------

    def save_snapshot(self, path: str | None = None) -> str:
        """Save current strategy_net snapshot for FSP ghost pool."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        if path is None:
            path = os.path.join(CHECKPOINT_DIR,
                                f"ghost_{len(self.historical_nets):04d}.pt")
        torch.save({
            'advantage_net': self.net.state_dict(),
            'strategy_net':  self.strategy_net.state_dict(),
        }, path)
        return path

    def load_historical(self, path: str) -> None:
        """Load a historical net as the VILLAIN ghost for FSP."""
        ghost = NEXUS_GTO_Net(input_dim=self.input_dim).to(self.device)
        data  = torch.load(path, map_location=self.device)
        key   = 'strategy_net' if 'strategy_net' in data else 'advantage_net'
        ghost.load_state_dict(data[key])
        ghost.eval()
        self.historical_nets.append(ghost)
        if len(self.historical_nets) > 10:  # Keep pool bounded
            self.historical_nets.pop(0)

    def get_ghost_strategy(self, state_vec: np.ndarray,
                           legal_moves: list) -> np.ndarray | None:
        """Returns strategy from a random historical ghost (or None if pool empty)."""
        if not self.historical_nets:
            return None
        import random
        ghost = random.choice(self.historical_nets)
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for m in legal_moves:
            legal_mask[m] = 1.0
        t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            adv, _, _ = ghost(t)
        regrets = torch.clamp(adv.squeeze(0), min=0.0)
        regrets *= torch.from_numpy(legal_mask)
        total = regrets.sum()
        if total > 1e-8:
            return (regrets / total).numpy()
        return legal_mask / (legal_mask.sum() + 1e-8)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path = path or os.path.join(CHECKPOINT_DIR, "nexus_latest.pt")
        torch.save({
            'advantage_net':          self.net.state_dict(),
            'strategy_net':           self.strategy_net.state_dict(),
            'optimizer':              self.optimizer.state_dict(),
            'strategy_optimizer':     self.strategy_optimizer.state_dict(),
        }, path)
        return path

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data = torch.load(path, map_location=self.device)
        if 'advantage_net' in data:
            self.net.load_state_dict(data['advantage_net'])
        elif 'model_state_dict' in data:  # Backward compat with old format
            self.net.load_state_dict(data['model_state_dict'])
        if 'strategy_net' in data:
            self.strategy_net.load_state_dict(data['strategy_net'])
        else:
            # Bootstrap strategy_net from advantage_net
            self.strategy_net.load_state_dict(self.net.state_dict())
        if 'optimizer' in data:
            try:
                self.optimizer.load_state_dict(data['optimizer'])
            except Exception:
                pass
        return True
