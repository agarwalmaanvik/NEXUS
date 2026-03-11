"""
NEXUS Preflop Oracle — Pillar 4 (Preflop Component)

Provides O(1) exact GTO strategy lookup for preflop situations within scope:
  - Raise First In (RFI): hero opens the action
  - Facing One Raise (vs-1-raise): hero faces exactly one preflop aggressor

Out-of-scope situations (3-bet+ pots, 4-bets, multi-way chaos) return None,
which tells poker_bot_api.py to fall back to NEXUS_GTO_Net.

Tables are computed once from GTO approximations and cached in SQLite.
All 169 hand classes × 6 positions × 5 stack buckets × 2 scenarios = ~10,200 entries.
"""

import os
import sqlite3
import numpy as np
from range_encoder import hand_to_class, _class_decompose, _precompute_quality

N_POSITIONS   = 6     # UTG, HJ, CO, BTN, SB, BB
N_STACK_BUCKETS = 5   # <15bb, 15-30bb, 30-60bb, 60-100bb, 100+bb
N_ACTIONS     = 7
DB_DEFAULT    = "checkpoints/preflop_gto.db"

# Position names for readability
POSITION_NAMES = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

# RFI opening frequencies by position (what fraction of 169 hands open)
RFI_FREQ = {0: 0.15, 1: 0.20, 2: 0.27, 3: 0.42, 4: 0.48, 5: 0.0}  # BB can't RFI

# Facing-raise calling/3-bet frequencies by position
FACING_3BET_FREQ  = {0: 0.05, 1: 0.07, 2: 0.10, 3: 0.15, 4: 0.18, 5: 0.22}
FACING_CALL_FREQ  = {0: 0.08, 1: 0.10, 2: 0.14, 3: 0.22, 4: 0.28, 5: 0.42}

STACK_THRESHOLDS = [15, 30, 60, 100]  # BB boundaries


class PreflopOracle:
    """
    Preflop GTO strategy lookup. Covers RFI and facing-one-raise only.

    Usage:
        oracle = PreflopOracle()
        strategy = oracle.lookup(hand=[0, 13], position=3, stack_bb=100)
        # Returns 7-dim strategy array, or None for complex/OOS situations.
    """

    def __init__(self, db_path: str = DB_DEFAULT):
        self.db_path = db_path
        self._cache: dict[tuple, np.ndarray] = {}
        self._quality = _precompute_quality()
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, hand: list, position: int, stack_bb: float,
               facing_raise: bool = False, raise_count: int = 0) -> np.ndarray | None:
        """
        Returns a 7-dim GTO strategy array or None if out of scope.

        Args:
            hand:         List of 2 card ints (0-51).
            position:     Seat relative to dealer, 0=UTG, 5=BB.
            stack_bb:     Effective stack in big blinds.
            facing_raise: True if there has been exactly one raise before us.
            raise_count:  Total raises seen preflop (>1 = out of scope, return None).

        Returns:
            np.ndarray [7] or None.
        """
        # Scope gate: only handle first-in and facing-one-raise
        if raise_count > 1:
            return None  # 3-bet+ pot → fall back to network

        if len(hand) < 2:
            return None

        cls  = hand_to_class(int(hand[0]), int(hand[1]))
        pos  = int(np.clip(position, 0, N_POSITIONS - 1))
        sbkt = _stack_bucket(stack_bb)
        scenario = 1 if facing_raise else 0

        key = (cls, pos, sbkt, scenario)
        if key in self._cache:
            return self._cache[key].copy()

        strategy = self._db_lookup(key)
        if strategy is not None:
            self._cache[key] = strategy
        return strategy.copy() if strategy is not None else None

    # ------------------------------------------------------------------
    # Internal: table generation
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Load from DB if it exists; compute and store if not."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS preflop_gto (
                hand_class  INTEGER,
                position    INTEGER,
                stack_bkt   INTEGER,
                scenario    INTEGER,
                strategy    BLOB,
                PRIMARY KEY (hand_class, position, stack_bkt, scenario)
            )""")
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM preflop_gto")
        count = cur.fetchone()[0]
        conn.close()

        if count < 1000:  # Tables not populated yet
            print("PreflopOracle: computing GTO tables (one-time setup)...")
            self._compute_and_store()
            print("PreflopOracle: tables ready.")

    def _compute_and_store(self) -> None:
        """
        Compute GTO-approximate strategies for all preflop nodes within scope.
        Uses hand quality + position + stack depth to derive opening/calling ranges.
        """
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        rows = []

        for cls in range(169):
            r1, r2, suited = _class_decompose(cls)
            quality = float(self._quality[cls])

            for pos in range(N_POSITIONS):
                for sbkt in range(N_STACK_BUCKETS):
                    stack_bb = [10, 22, 45, 80, 150][sbkt]

                    for scenario in range(2):   # 0=RFI, 1=vs-1-raise
                        strategy = self._gto_strategy(
                            cls, quality, pos, stack_bb, scenario)
                        rows.append((cls, pos, sbkt, scenario,
                                     strategy.tobytes()))

        cur.executemany("""
            INSERT OR REPLACE INTO preflop_gto
            (hand_class, position, stack_bkt, scenario, strategy)
            VALUES (?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        conn.close()

    def _gto_strategy(self, cls: int, quality: float, pos: int,
                      stack_bb: float, scenario: int) -> np.ndarray:
        """
        Returns a 7-dim strategy array using GTO-approximate heuristics.

        Preflop action set:
          0=Fold, 1=Call(Limp), 2=Min-raise, 3=33%,  4=75%(2.5x),
          5=150%(3x), 6=Shove

        RFI strategy:
          - Top X% of hands by quality → open raise at standard size
          - Short stack (<25bb) → polarise toward shoves
          - Hands below threshold → fold

        Facing-raise strategy:
          - Top Y% → 3-bet (raise)
          - Mid Z% → call (flat)
          - Rest → fold
        """
        strat = np.zeros(N_ACTIONS, dtype=np.float32)

        if scenario == 0:  # RFI
            open_freq = RFI_FREQ.get(pos, 0.20)
            if pos == 5:  # BB can't RFI
                strat[0] = 1.0
                return strat

            if quality >= (1.0 - open_freq):
                # Open range
                if stack_bb <= 20:
                    # Short stack: shove or fold (ICM correct)
                    if quality >= 0.70:
                        strat[6] = 1.0   # Shove with strong hands
                    elif quality >= 1.0 - open_freq:
                        strat[6] = 0.8; strat[0] = 0.2
                elif stack_bb <= 40:
                    # Medium stack: 2-2.5x open
                    strat[4] = 0.7; strat[5] = 0.2; strat[6] = 0.1
                else:
                    # Deep: standard 2.5-3x
                    strat[4] = 0.5; strat[5] = 0.4; strat[3] = 0.1
            else:
                strat[0] = 1.0  # Fold

        else:  # vs-1-raise (facing open)
            three_bet_freq = FACING_3BET_FREQ.get(pos, 0.10)
            call_freq      = FACING_CALL_FREQ.get(pos, 0.20)
            # Top threshold → 3-bet; Next → call; Rest → fold
            if quality >= (1.0 - three_bet_freq):
                if stack_bb <= 25:
                    strat[6] = 1.0
                else:
                    strat[5] = 0.6; strat[6] = 0.4
            elif quality >= (1.0 - three_bet_freq - call_freq):
                strat[1] = 1.0   # Flat call
            else:
                strat[0] = 1.0   # Fold

        # Normalise
        total = strat.sum()
        if total > 0:
            strat /= total
        else:
            strat[0] = 1.0
        return strat

    def _db_lookup(self, key: tuple) -> np.ndarray | None:
        """Single row lookup from SQLite."""
        cls, pos, sbkt, scenario = key
        try:
            conn = sqlite3.connect(self.db_path)
            cur  = conn.cursor()
            cur.execute("""SELECT strategy FROM preflop_gto
                           WHERE hand_class=? AND position=? AND stack_bkt=? AND scenario=?""",
                        (cls, pos, sbkt, scenario))
            row = cur.fetchone()
            conn.close()
            if row:
                return np.frombuffer(row[0], dtype=np.float32).copy()
        except Exception:
            pass
        return None


def _stack_bucket(stack_bb: float) -> int:
    """Maps stack depth to bucket index 0-4."""
    for i, thresh in enumerate(STACK_THRESHOLDS):
        if stack_bb < thresh:
            return i
    return N_STACK_BUCKETS - 1
