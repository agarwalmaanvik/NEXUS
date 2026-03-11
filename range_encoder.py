import numpy as np
import torch

class RangeEncoder:
    """
    Maintains a Bayesian probability distribution over the 169 canonical
    preflop hand classes for each opponent. Updated after every action.

    The 169 classes are:
      Pairs (13):    AA, KK, QQ, ..., 22  → indices 0-12
      Suited (78):   AKs, AQs, ..., 32s   → indices 13-90
      Offsuit (78):  AKo, AQo, ..., 32o   → indices 91-168

    Usage:
        enc = RangeEncoder(n_players=2)
        enc.update(opponent_seat=1, action=2, amount=60, board=[], pot=30, stage=0)
        tensor = enc.to_tensor(player=1)  # shape [169]
    """

    N_CLASSES = 169
    # Precomputed hand-class quality scores (0=worst, 1=best).
    # These are equity estimates vs. a random hand (Monte Carlo, lookup).
    # Index = hand class (see _class_to_quality).
    _QUALITY: np.ndarray | None = None

    def __init__(self, n_players: int = 2):
        self.n_players = n_players
        # Start with uniform prior over all non-blocking hand classes
        self.beliefs = {p: np.ones(self.N_CLASSES, dtype=np.float32)
                        for p in range(n_players)}
        self._normalise_all()
        if RangeEncoder._QUALITY is None:
            RangeEncoder._QUALITY = _precompute_quality()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, player: int | None = None) -> None:
        """Reset beliefs to uniform (call at start of each hand)."""
        targets = [player] if player is not None else list(self.beliefs.keys())
        for p in targets:
            self.beliefs[p] = np.ones(self.N_CLASSES, dtype=np.float32)
            self.beliefs[p] /= self.N_CLASSES

    def update(self, player: int, action: int, amount: float,
               board: list, pot: float, stage: int) -> None:
        """
        Bayesian update: P(hand|obs) ∝ P(obs|hand) × P(hand)

        action: 0=fold, 1=call/check, 2+=raise
        We compute a likelihood for each hand class based on how likely that
        hand type would take this action in this situation.
        """
        likelihoods = self._action_likelihood(action, amount, board, pot, stage)
        # Fold eliminates the player — reset to zero (dead range)
        if action == 0:
            self.beliefs[player] = np.zeros(self.N_CLASSES, dtype=np.float32)
            return
        self.beliefs[player] *= likelihoods
        self._normalise(player)

    def remove_blocker(self, player: int, known_cards: list[int]) -> None:
        """
        Remove impossible hands given observed cards (blockers).
        known_cards: list of card ints (0-51) — engine encoding: rank*4+suit.
        """
        # Engine encoding: card = rank*4 + suit  →  rank = card//4, suit = card%4
        blocked_ranks = set(int(c) // 4 for c in known_cards)

        for cls_idx in range(self.N_CLASSES):
            r1, r2, suited = _class_decompose(cls_idx)
            # If both ranks are blocked → zero probability
            if r1 in blocked_ranks and r2 in blocked_ranks:
                self.beliefs[player][cls_idx] = 0.0

        self._normalise(player)

    def to_tensor(self, player: int) -> torch.Tensor:
        """Returns belief distribution as [169] float32 tensor."""
        return torch.tensor(self.beliefs[player], dtype=torch.float32)

    def to_numpy(self, player: int) -> np.ndarray:
        return self.beliefs[player].copy()

    def entropy(self, player: int) -> float:
        """Shannon entropy of belief distribution. Low → tighter range."""
        p = self.beliefs[player]
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _action_likelihood(self, action: int, amount: float,
                           board: list, pot: float, stage: int) -> np.ndarray:
        """
        Returns a per-class likelihood vector for a given action.
        Higher quality hands are more likely to raise; lower likely to call/check.
        """
        q = RangeEncoder._QUALITY  # [169] quality scores in [0,1]
        L = np.ones(self.N_CLASSES, dtype=np.float32)

        # Pot odds normalised bet fraction
        pot_frac = (amount / max(pot, 1.0)) if pot > 0 else 0.0
        pot_frac = float(np.clip(pot_frac, 0.0, 3.0)) / 3.0  # [0,1]

        if action == 1:  # call / check
            # Weak-to-medium hands more likely to call;
            # Very strong hands might raise, very weak might fold.
            # Flat distribution slightly weighted toward medium quality.
            L = 0.3 + 0.7 * (1.0 - (q - 0.5) ** 2 * 2)
            L = np.clip(L, 0.05, 1.0)

        elif action >= 2:  # raise / bet
            # Strong hands exponentially more likely.
            # Larger bet = more polarised (strong OR air bluff).
            if pot_frac > 0.5:  # Large bet: polarised
                bluff_bias = q < 0.25  # Bottom of range = bluffs
                value_bias = q > 0.6   # Top of range = value
                L = np.where(bluff_bias | value_bias, 1.0, 0.2)
            else:  # Small bet: mostly strong hands, less polarised
                L = 0.1 + 0.9 * q

        # Apply stage discount: preflop ranges are wider
        if stage == 0:
            L = 0.4 + 0.6 * L  # Dampen signal preflop (wider range)

        return L.astype(np.float32)

    def _normalise(self, player: int) -> None:
        total = self.beliefs[player].sum()
        if total > 0:
            self.beliefs[player] /= total
        else:
            # Flat reset if all zeroed (shouldn't happen outside of fold)
            self.beliefs[player] = np.ones(self.N_CLASSES, dtype=np.float32) / self.N_CLASSES

    def _normalise_all(self) -> None:
        for p in self.beliefs:
            self._normalise(p)


# ------------------------------------------------------------------
# Hand class helpers (module-level for speed)
# ------------------------------------------------------------------

def _class_decompose(cls_idx: int) -> tuple[int, int, bool]:
    """Returns (rank1, rank2, suited) for a class index 0-168."""
    if cls_idx < 13:          # Pairs: AA=0, KK=1, ..., 22=12
        r = 12 - cls_idx
        return r, r, False
    elif cls_idx < 91:        # Suited: AKs=13 ... 32s=90
        idx = cls_idx - 13
        # Enumerate: AKs, AQs, ..., A2s (12 combos), KQs, ..., K2s (11), ...
        r1 = 12
        while idx >= r1:
            idx -= r1
            r1 -= 1
        r2 = r1 - 1 - idx
        return r1, r2, True
    else:                     # Offsuit: AKo=91 ... 32o=168
        idx = cls_idx - 91
        r1 = 12
        while idx >= r1:
            idx -= r1
            r1 -= 1
        r2 = r1 - 1 - idx
        return r1, r2, False


def hand_to_class(card1: int, card2: int) -> int:
    """Convert two card integers (0-51) to a class index (0-168).
    Engine encoding: card = rank*4 + suit  →  rank = card//4, suit = card%4.
    """
    r1, r2 = int(card1) // 4, int(card2) // 4
    s1, s2 = int(card1) % 4,  int(card2) % 4
    if r1 < r2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    suited = (s1 == s2)

    if r1 == r2:  # Pair
        return 12 - r1

    if suited:    # Suited
        # Count combos before this pair
        offset = 13
        for hi in range(12, r1, -1):
            offset += hi
        offset += (r1 - 1 - r2)
        return offset

    else:         # Offsuit
        offset = 91
        for hi in range(12, r1, -1):
            offset += hi
        offset += (r1 - 1 - r2)
        return offset


def _precompute_quality() -> np.ndarray:
    """
    Returns a [169] array of preflop hand qualities in [0, 1].
    Computed from well-known preflop equity estimates vs. random hand.
    Higher = stronger hand.
    """
    quality = np.zeros(169, dtype=np.float32)
    # Base equity values for key hands (vs. 1 random opponent)
    # Source: standard poker equity tables
    BASE = {
        # Pairs: (rank, equity)
        (12, 12): 0.85,  # AA
        (11, 11): 0.73,  # KK
        (10, 10): 0.70,  # QQ
        (9, 9):   0.66,  # JJ
        (8, 8):   0.63,  # TT
        (7, 7):   0.58,  # 99
        (6, 6):   0.55,  # 88
        (5, 5):   0.53,  # 77
        (4, 4):   0.52,  # 66
        (3, 3):   0.50,  # 55
        (2, 2):   0.49,  # 44
        (1, 1):   0.50,  # 33
        (0, 0):   0.49,  # 22
    }

    for cls_idx in range(169):
        r1, r2, suited = _class_decompose(cls_idx)

        if r1 == r2:  # Pair
            key = (r2, r2)
            if key in BASE:
                quality[cls_idx] = BASE[key]
            else:
                quality[cls_idx] = 0.49 + 0.005 * r1
        else:
            # Suited bonus +0.03, high card bonus
            high_bonus  = (r1 + r2) / 48.0   # 0→0, 23→0.48
            suited_bonus = 0.03 if suited else 0.0
            connectedness = max(0, 4 - abs(r1 - r2)) / 4.0 * 0.02
            quality[cls_idx] = float(np.clip(
                0.38 + high_bonus + suited_bonus + connectedness, 0.0, 1.0))

    # Normalise to [0, 1]
    mn, mx = quality.min(), quality.max()
    if mx > mn:
        quality = (quality - mn) / (mx - mn)
    return quality
