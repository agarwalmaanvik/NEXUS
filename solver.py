"""
NEXUS Solver — External Sampling MCCFR (v2 — correct GameState interface)

Interface corrections applied after auditing engine_core.py:
  - GameState has no `raise_count` attribute; compute from history
  - state.step(action_idx, amount=0) — amount ignored for buckets 0-6
  - Deep copy via state.get_state() / state.set_state() (not copy.deepcopy)
  - resolve_hand() returns list[float] indexed by seat_id
  - legal_moves is a list of ints (not a set)
  - Terminal: active_count <= 1 OR stage > 3
"""

import numpy as np
import torch
from typing import Optional

from vectorizer import PokerVectorizer
from networks import NEXUS_GTO_Net, INPUT_DIM

N_ACTIONS     = 7
STARTING_STACK = 2000.0   # Must match train_master.py STARTING_STACK


class ExternalSamplingMCCFR:
    """
    External Sampling Monte Carlo CFR solver.

    Used in two modes:
      1. TRAINING: run_traversal() — populates P0/P1 advantage buffers + RAG index
      2. INFERENCE: solve() — runs n_traversals from the current state to produce
         an improved strategy (called from poker_bot_api.py at decision time)
    """

    def __init__(self, net: NEXUS_GTO_Net, device: str = "cpu",
                 depth_limit: int = 4, n_traversals: int = 100,
                 rag=None):
        self.net          = net
        self.device       = device
        self.depth_limit  = depth_limit
        self.n_traversals = n_traversals
        self.rag          = rag

    # ------------------------------------------------------------------
    # Training-time traversal
    # ------------------------------------------------------------------

    def run_traversal(self, root_state, hero_seat: int,
                      p0_buffer=None, p1_buffer=None,
                      range_encoder=None) -> float:
        """
        Runs one External Sampling MCCFR traversal from root_state.
        Returns hero's estimated EV.
        """
        saved = root_state.get_state()
        try:
            ev = self._traverse(root_state, hero_seat, depth=0,
                                p0_buffer=p0_buffer, p1_buffer=p1_buffer,
                                range_encoder=range_encoder)
        finally:
            root_state.set_state(saved)  # Restore after traversal
        return ev

    # ------------------------------------------------------------------
    # Inference-time solver
    # ------------------------------------------------------------------

    def solve(self, root_state, hero_seat: int,
              range_encoder=None) -> np.ndarray:
        """
        Runs n_traversals from root_state to produce improved strategy.
        Returns np.ndarray [7] probability distribution.
        """
        legal = list(root_state.legal_moves)
        if len(legal) == 0:
            return np.ones(N_ACTIONS) / N_ACTIONS
        if len(legal) == 1:
            s = np.zeros(N_ACTIONS); s[legal[0]] = 1.0; return s

        root_regrets = np.zeros(N_ACTIONS, dtype=np.float64)
        saved = root_state.get_state()

        for _ in range(self.n_traversals):
            # Restore root state for each traversal
            root_state.set_state(saved)
            action_values: dict[int, float] = {}

            for action in legal:
                root_state.set_state(saved)
                done = root_state.step(action, 0)
                if done:
                    action_values[action] = float(root_state.resolve_hand()[hero_seat])
                else:
                    action_values[action] = self._rollout(
                        root_state, hero_seat, depth=1, range_encoder=range_encoder)
                root_state.set_state(saved)

            cur_strat = _regret_match(root_regrets, legal)
            state_val = sum(cur_strat[a] * action_values.get(a, 0.0) for a in legal)

            for a in legal:
                root_regrets[a] += action_values.get(a, 0.0) - state_val

        root_state.set_state(saved)
        return _regret_match_array(root_regrets, legal)

    # ------------------------------------------------------------------
    # Core recursive traversal
    # ------------------------------------------------------------------

    def _traverse(self, state, hero_seat: int, depth: int,
                  p0_buffer=None, p1_buffer=None,
                  range_encoder=None) -> float:
        """
        Recursive External Sampling MCCFR.
        Returns hero's estimated value at this node.
        """
        # Terminal: only one active player, or past river
        active_count = sum(1 for p in state.players if p.active)
        if active_count <= 1 or state.stage > 3:
            payouts = state.resolve_hand()
            return float(payouts[hero_seat])

        if depth >= self.depth_limit:
            return self._leaf_value(state, hero_seat, range_encoder)

        cur_player = int(state.current_player)
        legal = list(state.legal_moves)
        if not legal:
            return 0.0

        state_vec = self._full_vectorize(state, cur_player, range_encoder)
        strategy  = self._get_strategy_dict(state_vec, legal)

        if cur_player == hero_seat:
            # ── HERO NODE: traverse ALL actions ──────────────────────────
            action_values: dict[int, float] = {}
            saved = state.get_state()

            for a in legal:
                state.set_state(saved)
                done = state.step(a, 0)
                if done:
                    raw = float(state.resolve_hand()[hero_seat])
                    # Normalize to [-1, 1] (same scale as parallel env)
                    action_values[a] = np.clip(raw / STARTING_STACK, -2.0, 2.0)
                else:
                    action_values[a] = self._traverse(
                        state, hero_seat, depth + 1,
                        p0_buffer, p1_buffer, range_encoder)

            state.set_state(saved)

            node_value = sum(strategy[a] * action_values[a] for a in legal)

            target_adv = np.zeros(N_ACTIONS, dtype=np.float32)
            for a in legal:
                target_adv[a] = float(action_values[a] - node_value)

            buf = p0_buffer if cur_player == 0 else p1_buffer
            if buf is not None:
                buf.add(state_vec, target_adv, float(node_value))

            # Add to RAG (novelty-filtered internally)
            if self.rag is not None:
                try:
                    opp = 1 - cur_player
                    rb = range_encoder.to_numpy(opp) if range_encoder else None
                    emb = self.rag.embed_state(state, cur_player, range_belief=rb)
                    strat_arr = _regret_match_array(
                        np.array([strategy.get(a, 0.0) for a in range(N_ACTIONS)]),
                        legal)
                    self.rag.add(emb, strat_arr)
                except Exception:
                    pass

            return node_value

        else:
            # ── VILLAIN NODE: sample ONE action ──────────────────────────
            probs = np.array([strategy[a] for a in legal], dtype=np.float32)
            probs /= probs.sum() + 1e-8
            chosen = legal[np.random.choice(len(legal), p=probs)]

            # Update range encoder with opponent's observed action
            if range_encoder is not None:
                try:
                    range_encoder.update(
                        player=cur_player, action=chosen,
                        amount=0, board=list(state.board),
                        pot=float(state.pot), stage=int(state.stage))
                except Exception:
                    pass

            done = state.step(chosen, 0)
            if done:
                raw = float(state.resolve_hand()[hero_seat])
                result = np.clip(raw / STARTING_STACK, -2.0, 2.0)
            else:
                result = self._traverse(state, hero_seat, depth + 1,
                                         p0_buffer, p1_buffer, range_encoder)

            # External Sampling MCCFR: villain nodes do NOT write to buffer.
            # Only hero-node regrets are unbiased. Writing villain strategy-as-regret
            # corrupts both players' buffers with incorrect gradient targets.

            return result

    # ------------------------------------------------------------------
    # Rollout (inference only)
    # ------------------------------------------------------------------

    def _rollout(self, state, hero_seat: int, depth: int,
                 range_encoder=None) -> float:
        """Fast strategy rollout — no buffer writes."""
        active_count = sum(1 for p in state.players if p.active)
        if active_count <= 1 or state.stage > 3:
            raw = float(state.resolve_hand()[hero_seat])
            return np.clip(raw / STARTING_STACK, -2.0, 2.0)
        if depth >= self.depth_limit:
            return self._leaf_value(state, hero_seat, range_encoder)

        cur_player = int(state.current_player)
        legal = list(state.legal_moves)
        if not legal:
            return 0.0

        state_vec = self._full_vectorize(state, cur_player, range_encoder)
        strategy  = self._get_strategy_dict(state_vec, legal)

        probs  = np.array([strategy[a] for a in legal], dtype=np.float32)
        probs /= probs.sum() + 1e-8
        chosen = legal[np.random.choice(len(legal), p=probs)]

        done = state.step(chosen, 0)
        if done:
            raw = float(state.resolve_hand()[hero_seat])
            return np.clip(raw / STARTING_STACK, -2.0, 2.0)
        return self._rollout(state, hero_seat, depth + 1, range_encoder)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _full_vectorize(self, state, seat: int,
                        range_encoder=None) -> np.ndarray:
        """Build 348-dim vector: 171 (game) + 169 (range belief) + 8 (tell=zeros)."""
        game_vec = PokerVectorizer.state_to_tensor(state, seat)  # [171]

        if range_encoder is not None:
            opp = 1 - seat
            range_vec = range_encoder.to_numpy(opp)
        else:
            range_vec = np.ones(169, dtype=np.float32) / 169.0

        tell_vec = np.zeros(8, dtype=np.float32)
        return np.concatenate([game_vec, range_vec, tell_vec]).astype(np.float32)

    def _get_strategy_dict(self, state_vec: np.ndarray,
                           legal: list) -> dict[int, float]:
        """Regret-matched strategy dict from network output."""
        t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            adv, _, _ = self.net(t)
        regrets = adv.cpu().numpy()[0]
        return _regret_match(regrets, legal)

    def _leaf_value(self, state, hero_seat: int, range_encoder=None) -> float:
        """Value head estimate at depth limit."""
        state_vec = self._full_vectorize(state, hero_seat, range_encoder)
        t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, val, _ = self.net(t)
        return float(val.item())


# ------------------------------------------------------------------
# Regret matching utilities
# ------------------------------------------------------------------

def _regret_match(regrets: np.ndarray, legal: list) -> dict[int, float]:
    """Standard CFR regret matching → probability dict."""
    clipped   = np.maximum(regrets, 0.0)
    legal_sum = sum(clipped[a] for a in legal)
    if legal_sum > 1e-8:
        return {a: float(clipped[a] / legal_sum) for a in legal}
    p = 1.0 / max(len(legal), 1)
    return {a: p for a in legal}


def _regret_match_array(regrets: np.ndarray, legal: list) -> np.ndarray:
    """Returns full [N_ACTIONS] probability array."""
    d = _regret_match(regrets, legal)
    out = np.zeros(N_ACTIONS, dtype=np.float32)
    for a, p in d.items():
        out[a] = p
    return out


# Backward-compat alias
MCCFR_Solver = ExternalSamplingMCCFR
