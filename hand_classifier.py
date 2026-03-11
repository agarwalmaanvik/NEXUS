"""
hand_classifier.py — Human-readable hand descriptions for NEXUS v2.

Converts a raw evaluator rank (1–7462, higher=better) to:
  - A hand category name  ("Full House", "Flush", etc.)
  - A hand description    ("Kings full of Aces")
  - A percentile in [0,1] where 1.0 = Royal Flush

Usage:
    from hand_classifier import HandClassifier
    hc = HandClassifier()
    name, desc, pct = hc.classify(rank)
    # e.g. ("Full House", "Kings full of Aces", 0.92)
"""

from fast_evaluator import FastEvaluator

# Rank boundaries (FastEvaluator: 1=worst, 7462=Royal Flush)
# These thresholds separate the 9 hand categories.
# Derived from standard 5-card hand frequency analysis.
_RANK_BOUNDS = [
    (7462, 7462, "Royal Flush"),
    (7453, 7461, "Straight Flush"),
    (7297, 7452, "Four of a Kind"),
    (7141, 7296, "Full House"),
    (5864, 7140, "Flush"),
    (5854, 5863, "Straight"),
    (4996, 5853, "Three of a Kind"),
    (4138, 4995, "Two Pair"),
    (1278, 4137, "Pair"),
    (   1, 1277, "High Card"),
]

# Cleaner category ranges (non-overlapping)
_CATEGORIES = [
    (7462, "Royal Flush"),
    (7453, "Straight Flush"),
    (7297, "Four of a Kind"),
    (7141, "Full House"),
    (5864, "Flush"),
    (5854, "Straight"),
    (4996, "Three of a Kind"),
    (4138, "Two Pair"),
    (1278, "Pair"),
    (1,    "High Card"),
    (0,    "Invalid"),
]

_RANK_NAMES = {
    12: "Ace", 11: "King", 10: "Queen", 9: "Jack", 8: "Ten",
     7: "Nine",  6: "Eight",  5: "Seven",  4: "Six",   3: "Five",
     2: "Four",  1: "Three",  0: "Two",
}

_RANK_NAMES_PLURAL = {k: (v + "s" if v[-1] != "e" else v[:-1] + "ves"
                          if v == "Five" else v + "s")
                     for k, v in _RANK_NAMES.items()}
_RANK_NAMES_PLURAL[5] = "Fives"
_RANK_NAMES_PLURAL[8] = "Tens"


class HandClassifier:
    """Converts evaluator rank to human-readable hand info."""

    TOTAL_RANKS = 7462.0

    def classify(self, rank: int) -> tuple[str, float]:
        """
        Args:
            rank: Evaluator output (1=worst High Card, 7462=Royal Flush).

        Returns:
            (category_name, percentile)
            percentile: 0.0 = worst possible hand, 1.0 = Royal Flush.
        """
        category = self._rank_to_category(rank)
        percentile = max(0.0, min(1.0, (rank - 1) / (self.TOTAL_RANKS - 1)))
        return category, round(percentile, 4)

    def classify_full(self, hero_cards: list[int],
                      board: list[int]) -> tuple[str, float]:
        """
        Evaluate and classify a hand from raw card ints.

        Args:
            hero_cards: 2-card hero hand (engine encoding).
            board:      3–5 community cards.

        Returns:
            (category_name, percentile)
        """
        ev = FastEvaluator()
        if len(board) < 3 or len(hero_cards) < 2:
            return "Preflop", 0.0
        rank = ev.evaluate([int(c) for c in hero_cards] +
                           [int(c) for c in board])
        return self.classify(rank)

    def percentile_label(self, percentile: float) -> str:
        """Returns 'top X%' label for display."""
        top_pct = (1.0 - percentile) * 100
        if top_pct < 1:
            return "top 1%"
        return f"top {top_pct:.0f}%"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rank_to_category(self, rank: int) -> str:
        for threshold, name in _CATEGORIES:
            if rank >= threshold:
                return name
        return "High Card"
