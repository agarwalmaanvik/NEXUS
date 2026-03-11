import numpy as np
from collections import Counter

class PokerVectorizer:
    # Constants
    HISTORY_LENGTH = 10
    ACTION_DIM = 4 # Player (Rel), Action Type, Amount (BB), Stage

    @staticmethod
    def state_to_tensor(game_state, hero_seat_id):
        """
        Converts GameState to a fixed-size tensor.

        Exact dimension breakdown:
          52  hero cards (13-rank × 4-suit grid)
          52  board cards (13-rank × 4-suit grid)
           6  position one-hot
           6  opponent active flags
          11  geometric + hand-strength + board-texture context
          40  action history (10 × 4)
           7  legal moves mask
           4  opponent profiler stats
        ─────
         178  GAME_STATE_DIM total
        """
        hero = game_state.players[hero_seat_id]
        bb   = game_state.bb_amt if game_state.bb_amt > 0 else 1.0

        # 1. CARDS (104 bits — rank*4+suit layout)
        hero_bits  = np.zeros(52, dtype=np.float32)
        board_bits = np.zeros(52, dtype=np.float32)
        for c in hero.hand:
            hero_bits[int(c)] = 1.0
        for c in game_state.board:
            board_bits[int(c)] = 1.0

        # 2. POSITION (6 bits)
        n_pos   = 6
        rel_pos = (hero_seat_id - game_state.button) % game_state.n_players
        pos_bits = np.zeros(n_pos, dtype=np.float32)
        if rel_pos < n_pos:
            pos_bits[rel_pos] = 1.0

        # 3. OPPONENT ACTIVE MAP (6 bits)
        opp_bits = np.zeros(n_pos, dtype=np.float32)
        for p in game_state.players:
            if p.seat_id != hero_seat_id and p.active and not p.all_in:
                r = (p.seat_id - game_state.button) % game_state.n_players
                if r < n_pos:
                    opp_bits[r] = 1.0

        # 4. GEOMETRIC CONTEXT (4 floats)
        pot_bb     = game_state.pot / bb
        curr_stack = hero.stack / bb
        spr        = curr_stack / pot_bb if pot_bb > 0 else 0.0
        to_call    = max(p.bet for p in game_state.players) - hero.bet
        pot_odds   = to_call / (game_state.pot + to_call) if to_call > 0 else 0.0

        # 5. HAND FEATURES (3 floats)
        hand_strength = 0.0
        is_suited     = 0.0
        connectedness = 0.0
        if len(hero.hand) == 2:
            r1 = int(hero.hand[0]) // 4
            r2 = int(hero.hand[1]) // 4
            s1 = int(hero.hand[0]) % 4
            s2 = int(hero.hand[1]) % 4
            is_suited     = float(s1 == s2)
            connectedness = max(0.0, 1.0 - abs(r1 - r2) / 12.0)
            if len(game_state.board) >= 3:
                try:
                    from fast_evaluator import FastEvaluator
                    _ev = FastEvaluator()
                    rank = _ev.evaluate(
                        [int(c) for c in hero.hand] +
                        [int(c) for c in game_state.board])
                    hand_strength = rank / 7462.0
                except Exception:
                    hand_strength = 0.0

        # 6. BOARD TEXTURE (4 floats)
        board = list(game_state.board)
        flush_possible  = 0.0
        straight_poss   = 0.0
        board_paired    = 0.0
        is_monotone     = 0.0
        if board:
            suits = [c % 4 for c in board]
            ranks = sorted([c // 4 for c in board])
            suit_counts = Counter(suits)

            # Flush / monotone
            max_suit = max(suit_counts.values())
            flush_possible = max_suit / max(len(board), 1)
            is_monotone    = float(len(suit_counts) == 1 and len(board) >= 3)

            # Board paired
            rank_counts  = Counter(ranks)
            max_of_a_kind = max(rank_counts.values(), default=0)
            board_paired = min(max_of_a_kind - 1, 2) / 2.0

            # Straight draw score
            straight_poss = _straight_draw_score(ranks)

        context = np.array([
            curr_stack, pot_bb, spr, pot_odds,          # 4 geometric
            hand_strength, is_suited, connectedness,     # 3 hand features
            flush_possible, straight_poss,               # 2 board texture
            board_paired, is_monotone,                   # 2 board texture
        ], dtype=np.float32)                             # total: 11

        # 7. ACTION HISTORY (40 floats)
        hist_tensor = np.zeros(
            PokerVectorizer.HISTORY_LENGTH * PokerVectorizer.ACTION_DIM,
            dtype=np.float32)
        relevant_history = game_state.history[-PokerVectorizer.HISTORY_LENGTH:]
        for i, action in enumerate(relevant_history):
            p_rel    = (action.player_id - game_state.button) % game_state.n_players
            base_idx = i * PokerVectorizer.ACTION_DIM
            hist_tensor[base_idx]   = p_rel / 6.0
            hist_tensor[base_idx+1] = action.action_type / 3.0
            hist_tensor[base_idx+2] = min(action.amount / bb, 50.0)
            hist_tensor[base_idx+3] = action.stage / 3.0

        # 8. LEGAL MOVES (7 bits)
        legal = np.zeros(7, dtype=np.float32)
        for m in game_state.legal_moves:
            legal[m] = 1.0

        # 9. OPPONENT PROFILE (4 floats)
        opp_id   = (hero_seat_id + 1) % game_state.n_players
        opp_stats = game_state.profiler.get_stats(opp_id)

        # Concatenate → 52+52+6+6+11+40+7+4 = 178
        return np.concatenate([
            hero_bits,   # 52
            board_bits,  # 52
            pos_bits,    # 6
            opp_bits,    # 6
            context,     # 11
            hist_tensor, # 40
            legal,       # 7
            opp_stats    # 4
        ])


def _straight_draw_score(ranks: list[int]) -> float:
    """
    Score how connected the board is for straights.
    Returns 0.0 (no straight possibility) → 1.0 (straight on board).
    """
    if len(ranks) < 3:
        return 0.0
    rank_set = set(ranks)
    # Also consider Ace as low (rank 0 → virtual rank -1, i.e., treat 12 as -1)
    if 12 in rank_set:
        rank_set.add(-1)
    best = 0
    for low in range(-1, 9):   # low card of a 5-card straight window
        window = set(range(low, low + 5))
        hits   = len(rank_set & window)
        best   = max(best, hits)
    # 5 = made straight, 4 = open-ended/gutshot draw, 3 = weak draw
    if best >= 5: return 1.0
    if best == 4: return 0.75
    if best == 3: return 0.35
    return 0.0

