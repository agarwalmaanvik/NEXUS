"""
NEXUS Subgame Retriever — Pillar 4: RAG Strategy Memory

Two optimisations that prevent the classic FAISS blow-up:
  1. Suit Isomorphism: canonicalise cards before embedding (24× space reduction)
  2. Novelty Filter: query before insert; skip if L2 < ε (dedup near-identical nodes)

Falls back to sklearn NearestNeighbors if faiss-cpu is not installed.
"""

import numpy as np
import os
import pickle
from typing import Optional

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors  # type: ignore

EMBED_DIM       = 48
NOVELTY_EPS     = 0.05   # L2 threshold; below this = duplicate, skip insert
N_ACTIONS       = 7


class SubgameRetriever:
    """
    FAISS-based (or sklearn fallback) retrieval of pre-solved subgame strategies.

    The index is populated during CFR traversal. At inference, we embed the
    current state, retrieve k=3 nearest solved situations and return their
    distance-weighted average strategy as a "RAG prior".

    Usage:
        rag = SubgameRetriever()
        emb = rag.embed_state(game_state, hero_seat)
        rag.add(emb, strategy_vector)          # during training
        prior = rag.retrieve(emb, k=3)         # at inference
        rag.save("checkpoints/rag.pkl")
    """

    def __init__(self, index_path: str = "checkpoints/rag.pkl",
                 novelty_eps: float = NOVELTY_EPS,
                 max_size: int = 200_000,
                 rebuild_every: int = 500):
        self.novelty_eps   = novelty_eps
        self.index_path    = index_path
        self.max_size      = max_size
        self._rebuild_every = rebuild_every
        self._since_rebuild = 0          # inserts since last index rebuild
        self._embeddings: list[np.ndarray] = []
        self._strategies: list[np.ndarray] = []
        self._index      = None   # FAISS or sklearn index (rebuilt on demand)
        self._dirty      = False  # True when index needs rebuild

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def embed_state(self, game_state, hero_seat: int,
                    range_belief: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encodes a game situation into a 48-dim float32 embedding.
        Applies suit isomorphism BEFORE computing any features.
        """
        board        = list(getattr(game_state, 'board', []))
        hero         = game_state.players[hero_seat]
        hand         = list(getattr(hero, 'hand', []))
        stage        = int(getattr(game_state, 'stage', 0))
        pot          = float(getattr(game_state, 'pot', 0))
        n_players    = int(getattr(game_state, 'n_players', 2))
        current_player = int(getattr(game_state, 'current_player', 0))

        # Stack depth
        stacks       = [float(p.stack) for p in game_state.players]
        effective_stack = float(np.min(stacks)) if stacks else 100.0
        bb_amt       = float(getattr(game_state, 'bb_amt', 20))
        stack_bb     = effective_stack / max(bb_amt, 1)

        # Canonicalise via suit isomorphism
        all_cards    = [int(c) for c in hand + board]
        canon_cards  = _canonicalise_suits(all_cards)
        canon_hand   = canon_cards[:len(hand)]
        canon_board  = canon_cards[len(hand):]

        # --- Feature groups (total = 48) ---

        # 1. Hand strength features (4 dims)
        hand_strength = _hand_percentile(canon_hand, canon_board)
        equity_vs_random = _equity_estimate(canon_hand, canon_board)
        hand_feats = np.array([hand_strength, equity_vs_random,
                                hand_strength * equity_vs_random,
                                abs(hand_strength - equity_vs_random)],
                               dtype=np.float32)

        # 2. Board texture (8 dims)
        board_feats = _board_texture(canon_board)  # 8

        # 3. Position (6-dim one-hot; pad if fewer players)
        pos_feat = np.zeros(6, dtype=np.float32)
        rel_pos  = (hero_seat - current_player) % max(n_players, 1)
        pos_feat[min(rel_pos, 5)] = 1.0

        # 4. Pot / stack ratios (4 dims)
        pot_to_stack   = float(np.clip(pot / max(effective_stack, 1), 0, 5)) / 5
        pot_to_bb      = float(np.clip(pot / max(bb_amt, 1), 0, 50)) / 50
        stack_bb_norm  = float(np.clip(stack_bb / 200.0, 0, 1))
        spr            = float(np.clip(effective_stack / max(pot, 1), 0, 20)) / 20
        pot_feats = np.array([pot_to_stack, pot_to_bb, stack_bb_norm, spr],
                              dtype=np.float32)

        # 5. Range width estimate from belief entropy (4 dims)
        if range_belief is not None and len(range_belief) == 169:
            p           = range_belief.astype(np.float32)
            entropy     = float(-np.sum(p * np.log(p + 1e-12)) / np.log(169))
            top20       = float(np.sort(p)[-34:].sum())   # top 20% mass
            range_feats = np.array([entropy, top20,
                                     entropy * top20, 1.0 - entropy],
                                    dtype=np.float32)
        else:
            range_feats = np.array([0.9, 0.12, 0.1, 0.1], dtype=np.float32)

        # 6. Action history summary (8 dims)
        history = list(getattr(game_state, 'action_history', [[]]))
        action_feats = _action_history_features(history, stage)  # 8

        # 7. Stage one-hot (4 dims)
        stage_feat = np.zeros(4, dtype=np.float32)
        stage_feat[min(stage, 3)] = 1.0

        # 8. Padding to reach exactly 48 dims
        # hand(4) + board(8) + pos(6) + pot(4) + range(4) + action(8) + stage(4) = 38
        # Padding: 10 zeros
        padding = np.zeros(10, dtype=np.float32)

        emb = np.concatenate([hand_feats, board_feats, pos_feat, pot_feats,
                               range_feats, action_feats, stage_feat, padding])
        assert emb.shape[0] == EMBED_DIM, f"Embed dim mismatch: {emb.shape[0]}"
        return emb.astype(np.float32)

    def add(self, embedding: np.ndarray, strategy: np.ndarray) -> bool:
        """
        Adds a solved subgame to the index IF it passes the novelty filter.
        Returns True if inserted, False if duplicate.
        """
        # Hard cap — prevents unbounded memory growth
        if len(self._embeddings) >= self.max_size:
            return False

        # Add tiny epsilon so near-zero preflop embeddings are never perfectly 0
        emb = embedding.astype(np.float32).flatten() + 1e-8

        # Novelty filter: only query when index is current AND we have entries
        # Use batched rebuild (every rebuild_every inserts) to avoid O(n) rebuild
        # on every call — at 45k entries this was causing 85-second stalls.
        if len(self._embeddings) >= 1 and self._index is not None:
            d = self._query_raw(emb, k=1)
            if d is not None:
                dist = d[0]
                if np.isnan(dist) or dist < self.novelty_eps:
                    return False  # Not novel enough

        self._embeddings.append(emb.copy())
        self._strategies.append(strategy.astype(np.float32).copy())
        self._since_rebuild += 1

        # Batched rebuild: only rebuild every N inserts
        if self._since_rebuild >= self._rebuild_every or self._index is None:
            self._dirty = True
            self._rebuild_if_dirty()
            self._since_rebuild = 0

        return True

    def retrieve(self, embedding: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Returns distance-weighted average strategy from k nearest neighbours.
        Falls back to uniform strategy if index is empty.
        """
        if len(self._embeddings) < k:
            return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS

        emb = embedding.astype(np.float32).flatten()
        self._rebuild_if_dirty()

        distances, indices = self._query_k(emb, k=min(k, len(self._embeddings)))
        if indices is None:
            return np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS

        # Distance-weighted average: closer = more weight
        weights = 1.0 / (distances + 1e-6)
        weights /= weights.sum()

        blended = np.zeros(N_ACTIONS, dtype=np.float32)
        for w, idx in zip(weights, indices):
            blended += w * self._strategies[int(idx)]

        total = blended.sum()
        if total > 0:
            blended /= total
        else:
            blended = np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS
        return blended

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.index_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'embeddings': self._embeddings,
                         'strategies': self._strategies}, f)

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.index_path
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._embeddings = data.get('embeddings', [])
            self._strategies = data.get('strategies', [])
            self._dirty = True
            return True
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self._embeddings)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_if_dirty(self) -> None:
        if not self._dirty or len(self._embeddings) == 0:
            return
        mat = np.vstack(self._embeddings).astype(np.float32)
        if _HAS_FAISS:
            index = faiss.IndexFlatL2(EMBED_DIM)
            index.add(mat)
            self._index = index
        else:
            # Use brute-force euclidean — ball_tree normalises internally
            # which produces NaN distances for near-zero preflop embeddings.
            index = NearestNeighbors(
                n_neighbors=min(10, len(self._embeddings)),
                metric='euclidean', algorithm='brute')
            index.fit(mat)
            self._index = index
        self._dirty = False

    def _query_raw(self, emb: np.ndarray, k: int = 1):
        """Returns distances array or None."""
        self._rebuild_if_dirty()
        if self._index is None:
            return None
        emb2 = emb.reshape(1, -1).astype(np.float32)
        try:
            if _HAS_FAISS:
                D, _ = self._index.search(emb2, k)
                return D[0]
            else:
                D, _ = self._index.kneighbors(emb2, n_neighbors=k)
                return D[0]
        except Exception:
            return None

    def _query_k(self, emb: np.ndarray, k: int = 3):
        """Returns (distances, indices) or (None, None)."""
        self._rebuild_if_dirty()
        if self._index is None:
            return None, None
        emb2 = emb.reshape(1, -1).astype(np.float32)
        try:
            if _HAS_FAISS:
                D, I = self._index.search(emb2, k)
                return D[0], I[0]
            else:
                D, I = self._index.kneighbors(emb2, n_neighbors=k)
                return D[0], I[0]
        except Exception:
            return None, None


# ------------------------------------------------------------------
# Suit Isomorphism
# ------------------------------------------------------------------

def _canonicalise_suits(cards: list[int]) -> list[int]:
    """
    Maps suits to canonical labels based on frequency in the card list.
    Most-frequent suit on the board → Suit 0, next → Suit 1, etc.
    A♠K♠ on 2♠7♥9♣  ≡  A♥K♥ on 2♥7♠9♣  (same canonical hand).

    Reduces FAISS state space by up to 24× without losing strategic info.
    """
    if not cards:
        return cards

    # Count suit frequencies
    freq: dict[int, int] = {}
    for c in cards:
        s = int(c) // 13
        freq[s] = freq.get(s, 0) + 1

    # Build canonical mapping: most frequent → 0, then 1, 2, 3
    sorted_suits = sorted(freq.keys(), key=lambda s: (-freq[s], s))
    suit_map: dict[int, int] = {}
    next_canon = 0
    for s in sorted_suits:
        if s not in suit_map:
            suit_map[s] = next_canon
            next_canon += 1
    # Any suit not in freq (never seen) gets sequential assignment
    for s in range(4):
        if s not in suit_map:
            suit_map[s] = next_canon
            next_canon += 1

    return [int(c) % 13 + suit_map[int(c) // 13] * 13 for c in cards]


# ------------------------------------------------------------------
# Feature helpers
# ------------------------------------------------------------------

def _hand_percentile(hand: list[int], board: list[int]) -> float:
    """Approximate hand strength percentile [0,1] based on rank."""
    if not hand:
        return 0.5
    ranks = sorted([int(c) % 13 for c in hand], reverse=True)
    # Simple proxy: highest card + pair bonus + suited bonus
    score = ranks[0] / 12.0
    if len(ranks) > 1:
        score += 0.1 * (ranks[1] / 12.0)
        if ranks[0] == ranks[1]:
            score += 0.2  # Pair bonus
    return float(np.clip(score / 1.3, 0.0, 1.0))


def _equity_estimate(hand: list[int], board: list[int]) -> float:
    """Very fast equity proxy (avoids full MC simulation in the embed path)."""
    if not hand:
        return 0.5
    from range_encoder import _precompute_quality, hand_to_class
    if len(hand) >= 2:
        cls = hand_to_class(hand[0], hand[1])
        q = _precompute_quality()
        return float(q[cls])
    return 0.45


def _board_texture(board: list[int]) -> np.ndarray:
    """8-dim float32 board texture features."""
    feats = np.zeros(8, dtype=np.float32)
    if not board:
        return feats

    ranks = [int(c) % 13 for c in board]
    suits = [int(c) // 13 for c in board]

    # [0] Paired board (max count of any rank)
    from collections import Counter
    rank_counts = Counter(ranks)
    max_rank_cnt = max(rank_counts.values())
    feats[0] = (max_rank_cnt - 1) / 2.0  # 0=no pair, 0.5=pair, 1=trips

    # [1] Monotone / suited fraction
    suit_counts = Counter(suits)
    feats[1] = max(suit_counts.values()) / len(board)

    # [2] Connectedness: fraction of cards within 4 of each other
    sorted_ranks = sorted(set(ranks))
    gaps = [sorted_ranks[i+1] - sorted_ranks[i] for i in range(len(sorted_ranks)-1)]
    feats[2] = len([g for g in gaps if g <= 4]) / max(len(gaps), 1)

    # [3] High-card presence (any broadway: T,J,Q,K,A = ranks 8-12)
    broadway = [r for r in ranks if r >= 8]
    feats[3] = min(len(broadway) / 3.0, 1.0)

    # [4] Low board (all ranks < 6)
    feats[4] = 1.0 if all(r < 6 for r in ranks) else 0.0

    # [5] Board contains Ace
    feats[5] = 1.0 if 12 in ranks else 0.0

    # [6] Number of cards (0-5, normalised)
    feats[6] = len(board) / 5.0

    # [7] Draw-heaviness: # possible straight draws
    feats[7] = min(len(gaps) / 4.0, 1.0) if gaps else 0.0

    return feats


def _action_history_features(history, stage: int) -> np.ndarray:
    """
    8-dim summary of action history.
    history is the flat list of Action namedtuples from GameState.history.
    Action = namedtuple('Action', ['player_id', 'action_type', 'amount', 'stage'])
    """
    feats = np.zeros(8, dtype=np.float32)
    if not history:
        return feats

    # Flatten: handle both flat Action namedtuples and nested list formats
    all_actions = []
    for item in history:
        if hasattr(item, 'action_type'):  # Action namedtuple
            all_actions.append(item)
        elif isinstance(item, list):       # Legacy nested format
            all_actions.extend(item)

    if not all_actions:
        return feats

    total = len(all_actions)
    feats[0] = min(total / 20.0, 1.0)  # [0] normalised total actions

    raises = [a for a in all_actions if hasattr(a, 'action_type') and a.action_type >= 2]
    feats[1] = len(raises) / max(total, 1)   # [1] aggression fraction
    feats[2] = 1.0 if any(hasattr(a, 'action_type') and a.action_type == 0
                           for a in all_actions) else 0.0  # [2] fold seen
    feats[3] = stage / 3.0                  # [3] current street

    # [4-7] per-street raise counts (streets 0-3)
    for s in range(4):
        street_raises = [a for a in all_actions
                         if hasattr(a, 'stage') and a.stage == s
                         and hasattr(a, 'action_type') and a.action_type >= 2]
        feats[4 + s] = min(len(street_raises) / 3.0, 1.0)

    return feats
