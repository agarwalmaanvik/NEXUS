"""
equity_calc.py — Monte Carlo equity vs opponent's Bayesian range.

Unlike the value head (which estimates a scalar win_prob against an unknown opponent),
this module samples opponent hands weighted by the RangeEncoder's belief distribution
and runs each to completion, giving a calibrated win/tie/lose breakdown.

Usage:
    from equity_calc import EquityCalc
    calc = EquityCalc()
    win_pct, tie_pct, lose_pct = calc.equity_vs_range(
        hero_hand=[c1, c2],
        board=[b1, b2, b3],
        range_belief=range_encoder.to_numpy(opp_seat),
        n_samples=500
    )
"""

import numpy as np
import random
from fast_evaluator import FastEvaluator
from range_encoder import _class_decompose

# Full 52-card deck as indices (engine encoding: rank*4+suit)
_FULL_DECK = list(range(52))


class EquityCalc:
    """Monte Carlo equity calculator vs Bayesian opponent range."""

    def __init__(self):
        self._ev = FastEvaluator()

    def equity_vs_range(self,
                        hero_hand: list[int],
                        board: list[int],
                        range_belief: np.ndarray,
                        n_samples: int = 500
                        ) -> tuple[float, float, float]:
        """
        Estimate hero's win/tie/lose percentages vs opponent's range.

        Args:
            hero_hand:     Hero's two hole cards (engine-encoded ints).
            board:         Community cards dealt so far (0–5 cards).
            range_belief:  [169] probability distribution from RangeEncoder.
            n_samples:     Monte Carlo samples. 500 = ~50ms on CPU.

        Returns:
            (win_pct, tie_pct, lose_pct) — all in [0,1], summing to 1.0.
        """
        if len(hero_hand) < 2:
            return 0.5, 0.0, 0.5

        dead = set(int(c) for c in hero_hand) | set(int(c) for c in board)
        available_deck = [c for c in _FULL_DECK if c not in dead]

        # Build weighted list of opponent hand classes (for sampling)
        classes, weights = self._build_sample_pool(range_belief, dead)
        if not classes:
            return 0.5, 0.0, 0.5

        wins = ties = losses = 0

        for _ in range(n_samples):
            try:
                # Sample opponent hand from range-weighted pool
                opp_class = random.choices(classes, weights=weights, k=1)[0]
                opp_hand  = self._sample_hand_from_class(opp_class, dead, available_deck)
                if opp_hand is None:
                    continue

                # Complete the board if needed
                needed = 5 - len(board)
                runout_pool = [c for c in available_deck
                               if c not in set(opp_hand) and c not in dead]
                if len(runout_pool) < needed:
                    continue
                runout = random.sample(runout_pool, needed)
                full_board = list(board) + runout

                # Evaluate both hands
                hero_rank = self._ev.evaluate(
                    [int(c) for c in hero_hand] + [int(c) for c in full_board])
                opp_rank  = self._ev.evaluate(
                    [int(c) for c in opp_hand]  + [int(c) for c in full_board])

                if hero_rank > opp_rank:
                    wins += 1
                elif hero_rank == opp_rank:
                    ties += 1
                else:
                    losses += 1
            except Exception:
                continue

        total = wins + ties + losses
        if total == 0:
            return 0.5, 0.0, 0.5

        return wins/total, ties/total, losses/total

    def equity_preflop(self, hero_hand: list[int], n_samples: int = 1000) -> float:
        """Quick preflop equity vs random opponent hand."""
        if len(hero_hand) < 2:
            return 0.5
        dead = set(int(c) for c in hero_hand)
        deck = [c for c in _FULL_DECK if c not in dead]
        wins = ties = total = 0
        for _ in range(n_samples):
            try:
                opp = random.sample(deck, 2)
                runout = random.sample([c for c in deck if c not in set(opp)], 5)
                hr = self._ev.evaluate([int(c) for c in hero_hand] + runout)
                or_ = self._ev.evaluate([int(c) for c in opp] + runout)
                if hr > or_: wins += 1
                elif hr == or_: ties += 1
                total += 1
            except Exception:
                continue
        return (wins + 0.5*ties) / max(total, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sample_pool(self, range_belief: np.ndarray,
                           dead: set) -> tuple[list, list]:
        """Build list of (class_idx, weight) pairs excluding blocked hands."""
        classes, weights = [], []
        for cls_idx, prob in enumerate(range_belief):
            if prob < 1e-6:
                continue
            r1, r2, suited = _class_decompose(cls_idx)
            # Check if this class has at least one unblocked combo
            # (simplified: just check if both ranks are not fully dead)
            dead_ranks = {int(c) // 4 for c in dead}
            if r1 in dead_ranks and r2 in dead_ranks and r1 == r2:
                continue  # Pair fully blocked
            classes.append(cls_idx)
            weights.append(float(prob))
        return classes, weights

    def _sample_hand_from_class(self, cls_idx: int, dead: set,
                                available: list) -> list[int] | None:
        """Sample a specific two-card hand consistent with class cls_idx."""
        r1, r2, suited = _class_decompose(cls_idx)
        # Find available cards with matching ranks
        c1_pool = [c for c in available if c // 4 == r1]
        c2_pool = [c for c in available if c // 4 == r2 and c not in c1_pool]

        if not c1_pool or not c2_pool:
            return None

        c1 = random.choice(c1_pool)
        if suited:
            s1 = c1 % 4
            c2_pool_s = [c for c in c2_pool if c % 4 == s1]
            if not c2_pool_s:
                return None
            c2 = random.choice(c2_pool_s)
        else:
            c2_pool_o = [c for c in c2_pool if c % 4 != c1 % 4]
            if not c2_pool_o:
                c2 = random.choice(c2_pool)  # fallback: any suit
            else:
                c2 = random.choice(c2_pool_o)

        return [c1, c2]
