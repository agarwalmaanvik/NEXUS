import random
import numpy as np
from fast_evaluator import FastEvaluator
from collections import namedtuple
from opponent_model import OpponentProfiler

# Data structure for history
Action = namedtuple('Action', ['player_id', 'action_type', 'amount', 'stage'])

class Player:
    def __init__(self, seat_id, stack):
        self.seat_id = seat_id
        self.stack = float(stack)
        self.starting_stack = float(stack)
        self.bet = 0.0 
        self.total_bet = 0.0 
        self.hand = [] # Ints 0-51
        self.active = True 
        self.all_in = False
    
    def __repr__(self):
        return f"Player(id={self.seat_id}, stack={self.stack}, bet={self.bet}, active={self.active})"

    def to_dict(self):
        return {
            'seat_id': self.seat_id,
            'stack': self.stack,
            'starting_stack': self.starting_stack,
            'bet': self.bet,
            'total_bet': self.total_bet,
            'hand': self.hand[:],
            'active': self.active,
            'all_in': self.all_in
        }
    
    @staticmethod
    def from_dict(data):
        p = Player(data['seat_id'], data['stack'])
        p.starting_stack = data['starting_stack']
        p.bet = data['bet']
        p.total_bet = data['total_bet']
        p.hand = data['hand'][:]
        p.active = data['active']
        p.all_in = data['all_in']
        return p

class GameState:
    def __init__(self, n_players=6, small_blind=10, big_blind=20, training_mode=False):
        self.n_players = n_players
        self.sb_amt = small_blind
        self.bb_amt = big_blind
        self.deck = np.arange(52, dtype=np.int8) 
        self.board = []
        self.pot = 0.0
        self.stage = 0 # 0=Preflop, 1=Flop, 2=Turn, 3=River
        self.history = [] 
        self.players = [Player(i, 2000) for i in range(n_players)]
        self.button = 0
        self.current_player = 0
        self.min_raise = big_blind
        self.legal_moves = [] 
        self.evaluator = FastEvaluator()
        self.deck_idx = 0
        self.players_acted = set()
        self.profiler = OpponentProfiler(n_players, training_mode=training_mode)
        self.undo_stack = []  # State history for live advisor undo

    def reset(self, reset_stacks=True):
        self.profiler.start_hand() # <--- New Hand
        np.random.shuffle(self.deck)
        self.deck_idx = 0
        self.board = []
        self.pot = 0.0
        self.stage = 0
        self.history = []
        self.players_acted = set()
        self.button = (self.button + 1) % self.n_players
        
        # --- GUI FIX: CONDITIONAL CHIP RESET ---
        for p in self.players:
            if reset_stacks:
                p.stack = p.starting_stack # Used during training
            elif p.stack <= 0:
                p.stack = p.starting_stack # Auto-rebuy if you bust in the GUI!
                
            p.bet = 0
            p.total_bet = 0
            p.active = True
            p.all_in = False
            
            p.hand = [self.deck[self.deck_idx], self.deck[self.deck_idx+1]]
            self.deck_idx += 2

        # Post Blinds
        sb_pos = (self.button + 1) % self.n_players
        bb_pos = (self.button + 2) % self.n_players
        
        self.history.append(Action(sb_pos, 3, self.sb_amt, 0)) # 3=Blind/Post
        self._post_bet(self.players[sb_pos], self.sb_amt)
        self.history.append(Action(bb_pos, 3, self.bb_amt, 0))
        self._post_bet(self.players[bb_pos], self.bb_amt)
        
        self.current_player = (bb_pos + 1) % self.n_players
        self.min_raise = self.bb_amt
        self._update_legal_moves()
        self._find_next_active()
        return self

    def get_state(self):
        """Returns a deep copy of the current state as a dict."""
        return {
            'n_players': self.n_players,
            'sb_amt': self.sb_amt,
            'bb_amt': self.bb_amt,
            'deck': self.deck.copy(),
            'deck_idx': self.deck_idx,
            'board': self.board[:],
            'pot': self.pot,
            'stage': self.stage,
            'history': self.history[:],
            'players': [p.to_dict() for p in self.players],
            'button': self.button,
            'current_player': self.current_player,
            'min_raise': self.min_raise,
            'legal_moves': self.legal_moves[:],
            'players_acted': list(self.players_acted)
        }

    def set_state(self, state_dict):
        """Restores the game state from a dict."""
        self.n_players = state_dict['n_players']
        self.sb_amt = state_dict['sb_amt']
        self.bb_amt = state_dict['bb_amt']
        self.deck = state_dict['deck']
        self.deck_idx = state_dict['deck_idx']
        self.board = state_dict['board']
        self.pot = state_dict['pot']
        self.stage = state_dict['stage']
        self.history = state_dict['history']
        self.players = [Player.from_dict(d) for d in state_dict['players']]
        self.button = state_dict['button']
        self.current_player = state_dict['current_player']
        self.min_raise = state_dict['min_raise']
        self.legal_moves = state_dict['legal_moves']
        self.players_acted = set(state_dict.get('players_acted', []))

    def step(self, action_idx, amount=0):
        """
        Executes an action.
        action_idx: 0=FOLD, 1=CALL/CHECK, 2=RAISE
        amount: Only used for Raise. Total amount to raise TO (or by? Standard is TO in some engines, BY in others. 
                Let's stick to 'amount' as the 'add' amount for simplicity or 'raise_to' amount. 
                Logic below assumes 'amount' is what we want to add, subject to verification).
                Actually, simpler: 
                If RAISE: amount is the *total wager* for the round? No, usually 'raise amount' is 'amount ON TOP'.
                Let's define standard:
                RAISE X: The player puts in X more than the current high bet?
                Standard Poker Engine: Raise TO X.
        """
        p = self.players[self.current_player]
        stage = self.stage
        
        # Record that player acted
        self.players_acted.add(self.current_player)
        
        # Profile Action
        self.profiler.record_action(self.current_player, action_idx, stage)

        if action_idx == 0: # FOLD
            p.active = False
            self.history.append(Action(self.current_player, 0, 0, stage))
            
        elif action_idx == 1: # CALL / CHECK
            current_high = max(pl.bet for pl in self.players)
            to_call = current_high - p.bet
            amt = min(to_call, p.stack)
            self._post_bet(p, amt)
            self.history.append(Action(self.current_player, 1, amt, stage))
            
        elif action_idx >= 2: # RAISE (Buckets 2-6)
            current_high = max(pl.bet for pl in self.players)
            to_call = current_high - p.bet
            
            # Calculate Pot Base (Pot + Chips in current round if we called)
            # self.pot includes previous rounds. We need current round bets.
            # actually self.pot allows includes current round bets in this engine (`_post_bet` adds to self.pot).
            # So Pot Base = self.pot + (total_active_bets_if_we_called)?
            # Wait, `self.pot` is total money in middle.
            # If I call, I add `to_call` to `self.pot`.
            # So `pot_after_call = self.pot + to_call`.
            
            pot_after_call = self.pot + to_call
            
            # Calculate Raise Size (Amount ON TOP of call)
            raise_size = 0
            
            if action_idx == 2: # Min-Raise
                raise_size = self.min_raise
                
            elif action_idx == 3: # 33% Pot
                raise_size = 0.33 * pot_after_call
                
            elif action_idx == 4: # 75% Pot
                raise_size = 0.75 * pot_after_call
                
            elif action_idx == 5: # 150% Pot
                raise_size = 1.5 * pot_after_call
                
            elif action_idx == 6: # All-In
                # Raise size is everything we have left AFTER calling
                raise_size = p.stack - to_call
            
            elif action_idx == 7: # Human Custom Raise (Raise TO X)
                # Interpret 'amount' as the Target Total Bet
                # Raise Size = Target Total - Current High
                current_high = max(pl.bet for pl in self.players)
                target_total = amount
                raise_size = target_total - current_high
                
                # Check Min Raise
                if raise_size < self.min_raise:
                    raise_size = self.min_raise
            
            # Enforce Min-Raise constraints (except All-In)
            if action_idx != 6 and action_idx != 7: # 7 Checks itself
                raise_size = max(raise_size, self.min_raise)
            
            # Enforce Stack Cap (All-In)
            if raise_size + to_call >= p.stack:
                raise_size = p.stack - to_call
                
            # Create Action
            amt = to_call + raise_size
            self._post_bet(p, amt)
            
            self.history.append(Action(self.current_player, 2, amt, stage))
            
            # Update Min Raise
            if raise_size > self.min_raise:
                self.min_raise = raise_size

        # Move to next
        return self._advance_game()

    def _post_bet(self, p, amt):
        if amt > 0:
            p.stack -= amt
            p.bet += amt
            p.total_bet += amt
            self.pot += amt
        if p.stack == 0: p.all_in = True

    def _deal_community(self, count):
        for _ in range(count):
            if self.deck_idx < len(self.deck):
                self.board.append(self.deck[self.deck_idx])
                self.deck_idx += 1

    def _advance_game(self):
        active_players = [p for p in self.players if p.active and not p.all_in]
        
        # Check if hand ended (everyone folded)
        active_count = sum(1 for p in self.players if p.active)
        if active_count <= 1:
            return True # Hand Over

        # Check for round end
        # Round ends if:
        # 1. Bets are equal (not_matched is empty)
        # 2. All active players have acted at least once in this street
        
        high_bet = max(p.bet for p in self.players if p.active)
        not_matched = [p for p in self.players if p.active and p.bet < high_bet and not p.all_in]
        
        all_acted = all(p.seat_id in self.players_acted for p in active_players)
        
        if len(not_matched) == 0 and all_acted:
             # NEXT STREET
            self.stage += 1
            if self.stage > 3: return True # Hand Over
            
            if self.stage == 1: self._deal_community(3)
            else: self._deal_community(1)
            
            # Profile Stage
            actives = [p.seat_id for p in self.players if p.active]
            self.profiler.record_stage_transition(self.stage, actives)
            
            for pl in self.players: pl.bet = 0
            self.min_raise = self.bb_amt
            self.current_player = (self.button + 1) % self.n_players
            self._find_next_active()
            self.players_acted = set() # Reset for new street
            self._update_legal_moves()
            return False

        # Next player
        self.current_player = (self.current_player + 1) % self.n_players
        self._find_next_active()
        self._update_legal_moves()
        return False

    def _find_next_active(self):
        # Move current_player forward until we find an active, not-all-in player
        # Or if everyone is all-in/inactive, we might be stuck.
        # Safety break.
        start = self.current_player
        steps = 0
        while not self.players[self.current_player].active or self.players[self.current_player].all_in:
            self.current_player = (self.current_player + 1) % self.n_players
            steps += 1
            if steps > self.n_players: break # Everyone all in or inactive

    def _update_legal_moves(self):
        # 0: Fold
        # 1: Call/Check
        # 2: Min-Raise
        # 3: 33% Pot
        # 4: 75% Pot
        # 5: 150% Pot
        # 6: All-In
        
        # Start with simplistic legal moves (Fold, Call)
        self.legal_moves = [0, 1] 
        
        p = self.players[self.current_player]
        current_high = max(pl.bet for pl in self.players)
        to_call = current_high - p.bet
        
        # If we can't even call (stack <= to_call), we can only fold or call(all-in)
        # We cannot Raise.
        if p.stack <= to_call:
            return # Legal moves are just [0, 1]
            
        # If we have chips to raise:
        pot_base = self.pot + to_call
        
        # Check Bucket 2 (Min-Raise)
        if p.stack >= to_call + self.min_raise:
            self.legal_moves.append(2)
            
        # Check Buckets 3, 4, 5 (Geometric)
        fractions = [0.33, 0.75, 1.5]
        indices = [3, 4, 5]
        
        for frac, idx in zip(fractions, indices):
            raise_sz = frac * pot_base
            # Must be at least min_raise AND distinct from All-In (cost < stack)
            if raise_sz >= self.min_raise and (to_call + raise_sz) < p.stack:
                self.legal_moves.append(idx)
                
        # Check Bucket 6 (All-In)
        # Always legal if stack > to_call
        self.legal_moves.append(6)

    def seat_role(self, seat_id):
        # order = ["BTN","SB","BB","UTG","HJ","CO"][:self.n_players] 
        # But this depends on N.
        # Relative to button:
        # 0 = BTN
        # 1 = SB
        # 2 = BB
        # ...
        # Standard 6-max: BTN, SB, BB, UTG, HJ, CO.
        # 0, 1, 2, 3, 4, 5 (Relative)
        rel_pos = (seat_id - self.button) % self.n_players
        if rel_pos == 0: return "BTN"
        if rel_pos == 1: return "SB"
        if rel_pos == 2: return "BB"
        if self.n_players == 6:
            if rel_pos == 3: return "UTG"
            if rel_pos == 4: return "HJ"
            if rel_pos == 5: return "CO"
        return f"Pos_{rel_pos}"

    def resolve_hand(self):
        """
        Calculates payouts for every player.
        Handles Split Pots, Side Pots, and fold wins (no evaluation needed).
        Returns: List of floats (payout per player)
        """
        payouts = [0.0] * self.n_players

        # --- Fast Path: Single active player (everyone else folded) ---
        # No hand evaluation needed. They win the entire pot.
        active_players = [p for p in self.players if p.active]
        if len(active_players) == 1:
            winner = active_players[0]
            payouts[winner.seat_id] = self.pot
            return payouts

        # --- Showdown Path: 2+ players reached showdown ---
        # Profile Showdown
        actives = [p.seat_id for p in self.players if p.active]
        self.profiler.record_showdown(actives)

        # 1. Evaluate hands for active players
        scores = []
        for p in self.players:
            if p.active and p.hand:
                full_hand = p.hand + self.board
                rank = self.evaluator.evaluate(full_hand)
                if rank == 0:
                    raise ValueError(f"Evaluator lookup failed for hand {full_hand}. Board={self.board}. Stage={self.stage}.")
                scores.append((p.seat_id, rank))
            else:
                scores.append((p.seat_id, -1))  # Folded
        
        # 2. Iterative Pot Construction
        # We need to compute side pots.
        # Algorithm:
        # Collect all bets from this hand (total_bet).
        # Sort distinct bet amounts > 0.
        # Iterate levels.
        
        bets = [(p.seat_id, p.total_bet) for p in self.players]
        payouts = [0.0] * self.n_players
        
        # While there is money in bets
        while True:
            active_bets = [b[1] for b in bets if b[1] > 0]
            if not active_bets: break
            
            min_bet = min(active_bets)
            pot_chunk = 0
            contributors = []
            
            for i in range(len(bets)):
                seat, amount = bets[i]
                if amount > 0:
                    contribution = min(amount, min_bet)
                    pot_chunk += contribution
                    bets[i] = (seat, amount - contribution)
                    contributors.append(seat)
            
            # Who wins this chunk?
            # Must be a contributor and have best hand
            eligible_scores = [s for s in scores if s[0] in contributors and s[1] != -1]
            
            if not eligible_scores:
                # Everyone folded? (Shouldn't happen in showdown logic usually, but if everyone folded remaining money goes to last?)
                # If resolving hand, we assume showdown or single winner?
                # If only 1 active player left, they win everything.
                # Check for single active player case earlier? 
                pass
            else:
                best_rank = max(s[1] for s in eligible_scores)  # Higher rank = BETTER hand in this evaluator
                winners = [s[0] for s in eligible_scores if s[1] == best_rank]
                
                share = pot_chunk / len(winners)
                for w in winners:
                    payouts[w] += share
                    
        return payouts

    # ------------------------------------------------------------------
    # Undo stack — Live Advisor safety net (Phase 3)
    # ------------------------------------------------------------------

    def push_undo(self) -> None:
        """Save a deep copy of current state. Called before every advisor action."""
        self.undo_stack.append(self.get_state())

    def undo(self) -> bool:
        """
        Revert game to exactly the state before the last action.
        Returns True if successful, False if nothing to undo.
        """
        if not self.undo_stack:
            return False
        self.set_state(self.undo_stack.pop())
        return True

    def observe_action(self, seat_id: int, action_idx: int,
                       amount: float = 0) -> bool:
        """
        Live Advisor entry point. Forces an action from a specific seat,
        bypassing the internal current_player check, and saves state first
        so it can be instantly reverted with undo().

        Args:
            seat_id:    The seat taking the action.
            action_idx: 0=Fold, 1=Call/Check, 2-6=Raise buckets.
            amount:     Chip amount (only used for custom raise).

        Returns:
            done (bool): True if the hand is over after this action.
        """
        self.push_undo()                  # save state before mutation
        self.current_player = seat_id     # force the engine to accept this seat
        return self.step(action_idx, amount)