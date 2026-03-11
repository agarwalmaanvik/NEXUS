"""
NEXUS poker_bot_api.py — 6-Pillar Inference Engine (v2)

Pillars:
  1. Deep CFR (NEXUS_GTO_Net strategy_net) — GTO baseline
  2. RAG Retriever — solved subgame priors
  3. DDQN Agent — session-adaptive exploit layer
  4. Bluff Injector — equity-based stochastic bluffs
  5. Exploit Tilt — tilt toward best action vs readable opponents
  6. Bet Sizer — arithmetic sizing from action bucket + pot
"""

import os
import torch
import numpy as np

from engine_core       import GameState
from vectorizer        import PokerVectorizer
from networks          import NEXUS_GTO_Net, INPUT_DIM
from fast_evaluator    import FastEvaluator
from range_encoder     import RangeEncoder, hand_to_class
from tell_detector     import TellDetector
from rag_retriever     import SubgameRetriever
from preflop_tables    import PreflopOracle
from solver            import ExternalSamplingMCCFR
from bet_sizer         import bucket_to_amount
from ddqn_agent        import DDQNAgent
from equity_calc       import EquityCalc
from hand_classifier   import HandClassifier

N_ACTIONS     = 7
CHECKPOINT_DIR = "checkpoints"


class SOTAPokerBot:
    """
    NEXUS 6-Pillar Inference Engine.

    Decision flow (per call of get_action):
      1. Range Encoder: Bayesian range update; composite Threat Matrix (multi-way)
      2. Preflop Oracle: O(1) GTO lookup for preflop decisions
      3. Deep CFR: NEXUS_GTO_Net forward pass → advantage + value
      4. RAG: blend solved prior via learned α
      5. DDQN: session-adaptive exploit Q-values, gated by exploit_signal
      6. Guardrails: preflop premium check, post-flop shove guard
      7. Bluff Injector: stochastic equity-based bluffs
      8. Exploit Tilt: tilt toward best action vs readable opponents
      9. Bet Sizer: arithmetic sizing from action bucket + pot
      10. AnalysisReport: calibrated equity, hand name, EV estimates returned
    """

    def __init__(self, model_path: str = f"{CHECKPOINT_DIR}/nexus_latest.pt",
                 device: str = "cpu", temperature: float = 1.0,
                 bluff_freq: float = 0.05):
        self.device      = device
        self.temperature = temperature
        self.bluff_freq  = bluff_freq

        # Pillar 1: GTO Network (strategy_net for inference)
        print(f"Loading NEXUS brain from {model_path}...")
        self.net = NEXUS_GTO_Net(input_dim=INPUT_DIM).to(device)
        self.net.eval()
        try:
            data = torch.load(model_path, map_location=device)
            key  = 'strategy_net' if 'strategy_net' in data else 'model_state_dict'
            self.net.load_state_dict(data[key])
            print(f"Brain loaded. α={self.net.get_alpha():.3f}")
        except FileNotFoundError:
            print("⚠️  No trained model found. Playing from untrained network.")
        except Exception as e:
            print(f"Load error: {e}. Playing from untrained network.")

        # Pillar 2: Range Encoder + Tell Detector
        self._range_encoder = RangeEncoder(n_players=6)
        self._tell          = TellDetector()

        # Pillar 3: DDQN session-adaptive exploit agent
        self._ddqn = DDQNAgent(input_dim=INPUT_DIM, device=device)

        # Pillar 4: RAG + Evaluator + Oracle
        self._evaluator  = FastEvaluator()   # Cached — never rebuilt per action
        self._equity_calc = EquityCalc()
        self._classifier  = HandClassifier()
        self._rag    = SubgameRetriever(f"{CHECKPOINT_DIR}/rag.pkl")
        self._rag.load()
        self._oracle = PreflopOracle(f"{CHECKPOINT_DIR}/preflop_gto.db")

        # Inline search solver
        self._solver = ExternalSamplingMCCFR(
            net=self.net, device=device,
            depth_limit=6, n_traversals=100, rag=None)

        print(f"RAG index: {len(self._rag)} situations")

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------

    def new_hand(self, hero_seat: int) -> None:
        """Reset per-hand state. Call at the top of each new hand."""
        self._range_encoder.reset()

    # ------------------------------------------------------------------
    # Multi-way composite range (Phase 2)
    # ------------------------------------------------------------------

    def _composite_range(self, game_state_obj: GameState,
                         hero_seat_id: int) -> np.ndarray:
        """
        Merges all active opponents into a single Bayesian Threat Matrix.

        Why: CFR has no multi-player convergence guarantee. Treating every
        opponent as a separate range (5 simultaneous beliefs) creates cyclic
        instability. Merging them into one weighted composite gives the blueprint
        network a single clean signal — 'total threat from the table.'

        Weighting: Earlier position = tighter range = more credible signal.
        Relative position 0 = Button (widest), n_players-1 = UTG (tightest).
        """
        composite = np.zeros(169, dtype=np.float32)
        active_opponents = [
            p for p in game_state_obj.players
            if p.seat_id != hero_seat_id and p.active
        ]

        if not active_opponents:
            return np.ones(169, dtype=np.float32) / 169.0

        for opp in active_opponents:
            rel_pos    = (opp.seat_id - game_state_obj.button) % game_state_obj.n_players
            pos_weight = 1.0 + (game_state_obj.n_players - rel_pos) * 0.1
            opp_range  = self._range_encoder.to_numpy(opp.seat_id)
            composite += pos_weight * opp_range

        total = composite.sum()
        return composite / total if total > 1e-8 else np.ones(169, dtype=np.float32) / 169.0

    def observe_opponent_action(self, opp_seat: int, action: int,
                                 amount: float, board: list,
                                 pot: float, stage: int,
                                 hand_strength_est: float = 0.5) -> None:
        """Feed opponent actions into range/tell models before your turn."""
        self._range_encoder.update(opp_seat, action, amount, board, pot, stage)
        self._tell.record_action(opp_seat, action, amount, pot, stage,
                                  hand_strength_est)

    def record_outcome(self, state_vec: np.ndarray, action: int,
                        reward_bb: float, next_state_vec: np.ndarray,
                        done: bool = True) -> None:
        """Feed hand outcome to DDQN for session-level learning."""
        self._ddqn.record(state_vec, action, reward_bb, next_state_vec, done)
        self._ddqn.update()

    # ------------------------------------------------------------------
    # Core decision function
    # ------------------------------------------------------------------

    def get_action(self, game_state_obj: GameState, hero_seat_id: int,
                   use_search: bool = True):
        """
        Returns (action_idx, amount, market_data).

        action_idx 0-6 are standard buckets (amount ignored by engine).
        action_idx 7 is custom raise — amount = target total bet.
        """
        stage        = int(game_state_obj.stage)
        hero         = game_state_obj.players[hero_seat_id]
        legal_moves  = list(game_state_obj.legal_moves)
        current_high = max(p.bet for p in game_state_obj.players)
        to_call      = current_high - hero.bet
        board        = list(game_state_obj.board)
        bb_amt       = float(game_state_obj.bb_amt)
        stack_bb     = float(hero.stack) / max(bb_amt, 1.0)
        min_raise    = float(game_state_obj.min_raise)

        # Compute raise_count from history (no GameState attribute for this)
        raise_count = sum(1 for a in game_state_obj.history
                          if a.action_type >= 2 and a.stage == stage)

        # ── PILLAR 2: Build 355-dim state vector ──────────────────────────────
        game_vec = PokerVectorizer.state_to_tensor(game_state_obj, hero_seat_id)

        # Remove card blockers from ALL opponents (not just the first)
        for p in game_state_obj.players:
            if p.seat_id != hero_seat_id:
                self._range_encoder.remove_blocker(
                    p.seat_id, [int(c) for c in hero.hand])

        # Multi-way Threat Matrix: position-weighted union of all active ranges
        range_vec = self._composite_range(game_state_obj, hero_seat_id)  # [169]

        # Tell features: averaged across all active opponents
        # Exploit signal: max (respect the most exploitable player at the table)
        opp_seats = [
            p.seat_id for p in game_state_obj.players
            if p.seat_id != hero_seat_id and p.active
        ]
        if opp_seats:
            tell_vec = np.mean(
                [self._tell.get_features(s) for s in opp_seats], axis=0
            ).astype(np.float32)                                           # [8]
            exploit  = max(self._tell.get_exploit_signal(s) for s in opp_seats)
        else:
            tell_vec = np.zeros(8, dtype=np.float32)
            exploit  = 0.0

        # opp_primary used for equity calc in AnalysisReport
        opp_primary = opp_seats[0] if opp_seats else hero_seat_id

        state_vec    = np.concatenate([game_vec, range_vec, tell_vec]).astype(np.float32)
        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)

        # ── PILLAR 4A: Preflop Oracle ─────────────────────────────────────────
        oracle_strategy = None
        if stage == 0 and raise_count <= 1:
            oracle_strategy = self._oracle.lookup(
                hand=hero.hand,
                position=hero_seat_id,
                stack_bb=stack_bb,
                facing_raise=(raise_count == 1),
                raise_count=raise_count,
            )

        # ── PILLAR 4B: Post-flop RAG prior ────────────────────────────────────
        rag_prior = None
        if stage >= 1:
            try:
                emb = self._rag.embed_state(game_state_obj, hero_seat_id,
                                             range_belief=range_vec)
                rag_arr   = self._rag.retrieve(emb, k=3)
                rag_prior = torch.from_numpy(rag_arr).float().unsqueeze(0)
            except Exception:
                rag_prior = None

        # ── PILLAR 1: GTO Network ─────────────────────────────────────────────
        legal_mask_np = np.zeros(7, dtype=np.float32)
        for m in legal_moves:
            legal_mask_np[m] = 1.0
        legal_mask_t = torch.from_numpy(legal_mask_np).unsqueeze(0)

        with torch.no_grad():
            adv, val, _ = self.net(state_tensor)
        raw_regrets = adv.cpu().numpy()[0]
        win_prob    = float(torch.sigmoid(val.cpu().squeeze()).item())

        # ── Pillar 3: DDQN exploit Q-values ──────────────────────────────────
        ddqn_strat = self._ddqn.strategy(state_vec, legal_mask_np)

        # ── Strategy blend: Oracle / Search / Network + DDQN ─────────────────
        if oracle_strategy is not None:
            cfr_strategy = oracle_strategy.copy()
        elif use_search and stage >= 1:
            cfr_strategy = self._solver.solve(game_state_obj, hero_seat_id,
                                               range_encoder=self._range_encoder)
        else:
            clipped = np.maximum(raw_regrets, 0.0)
            alpha   = self.net.get_alpha()
            if rag_prior is not None:
                rag_np   = rag_prior.squeeze(0).numpy()
                clipped  = alpha * clipped + (1.0 - alpha) * rag_np
            clipped *= legal_mask_np
            total = clipped.sum()
            cfr_strategy = clipped / total if total > 1e-8 else legal_mask_np / max(legal_mask_np.sum(), 1e-8)

        # ── Pillar 5 blend: CFR (GTO floor) + DDQN (exploit ceiling) ─────────
        # exploit ∈ [0,1]: 0 = trust CFR fully, 1 = trust DDQN fully.
        # DDQN only meaningfully contributes after ~30 hands of session data.
        ddqn_weight = exploit if self._ddqn.hands_observed() >= 20 else 0.0
        strategy    = (1.0 - ddqn_weight) * cfr_strategy + ddqn_weight * ddqn_strat

        # ── Guardrails ────────────────────────────────────────────────────────

        # Pre-flop guard: prohibit shoves with non-premium hands
        if stage == 0 and 6 in legal_moves and not _is_preflop_premium(hero.hand):
            strategy[6] = 0.0

        # Post-flop all-in guard: suppress shove unless hand is genuinely strong
        if stage >= 1 and 6 in legal_moves and stack_bb >= 15:
            try:
                if len(board) >= 3:
                    rank = self._evaluator.evaluate(
                        [int(c) for c in hero.hand] + [int(c) for c in board])
                    if (rank / 7462.0) < 0.60:   # weaker than top-40% → no shove
                        strategy[6] = 0.0
            except Exception:
                pass

        # ── REALITY CHECK (The Monte Carlo Supervisor) ──────────────────
        equity_vs_rng = win_prob
        hand_name, hand_pct = ('Preflop', 0.0)
        pot_size = float(game_state_obj.pot)

        # 1. Calculate true equity vs the Bayesian Threat Matrix
        if stage >= 1 and len(hero.hand) == 2:
            try:
                rank = self._evaluator.evaluate(
                    [int(c) for c in hero.hand] + [int(c) for c in board])
                hand_name, hand_pct = self._classifier.classify(rank)
                range_belief = self._range_encoder.to_numpy(opp_primary)
                win_r, tie_r, _ = self._equity_calc.equity_vs_range(
                    hero_hand=list(hero.hand), board=board, 
                    range_belief=range_belief, n_samples=200)
                equity_vs_rng = win_r + 0.5 * tie_r
            except Exception:
                pass

        is_bluff = False
        
        # 2. Cash Game Apex Logic
        if stage >= 1:
            req_eq = float(to_call) / float(pot_size + to_call) if (pot_size + to_call) > 0 else 0.0
            
            # GATE A: The Nash Exploit (Humans don't bluff enough)
            if to_call > 0:
                # If we are mathematically dead, snap fold. (No hero-calling with 8-high)
                if equity_vs_rng < (req_eq - 0.05) and equity_vs_rng < 0.25:
                    strategy = np.zeros(7)
                    strategy[0] = 1.0  
                # If the human bombs the pot (>50% pot size), require at least 45% equity to continue
                elif to_call > (pot_size * 0.5) and equity_vs_rng < 0.45:
                    strategy = np.zeros(7)
                    strategy[0] = 1.0  

            # GATE B: The River Value Ban
            # Prevents 3-betting the river with Ace-high
            elif stage == 3 and equity_vs_rng < 0.50:
                strategy[3:] = 0.0 
                if strategy.sum() == 0: strategy[0] = 1.0 

            # GATE C: Constructed Semi-Bluffs (The Cash Game Crusher)
            # We bluff ONLY if our hand is currently weak (<30%), BUT we have outs/draws (>20% equity).
            elif self.bluff_freq > 0 and hand_pct < 0.30 and equity_vs_rng > 0.20 and to_call <= (pot_size * 0.5):
                # Because we restricted bluffs to actual draws, we can double the frequency to apply heavy pressure
                if np.random.random() < (self.bluff_freq * 2.0):
                    is_bluff = True
                    best_raise = 4 if 4 in legal_moves else (3 if 3 in legal_moves else 2) # 75% pot prefered
                    strategy = np.zeros(7)
                    strategy[best_raise] = 1.0

        # ── Exploit Bounding ──────────────────────────────────────────────────
        # Let the DDQN exploit human patterns, but cap it at 35% so the bot 
        # never deviates far enough from GTO to become exploitable itself.
        if exploit > 0.65 and not is_bluff and legal_mask_np.sum() > 0:
            masked_strat = strategy * legal_mask_np
            if masked_strat.sum() > 0:
                best = int(np.argmax(masked_strat))
                exploit_s = np.zeros(7); exploit_s[best] = 1.0
                blend  = min((exploit - 0.65) * 1.5, 0.35) 
                strategy = (1.0 - blend) * strategy + blend * exploit_s

        # Temperature scaling
        if self.temperature != 1.0 and not is_bluff:
            strategy = np.clip(strategy, 1e-9, None)
            log_s    = np.log(strategy) / self.temperature
            log_s   -= log_s.max()
            strategy = np.exp(log_s)

        # Final mask + normalise
        strategy *= legal_mask_np
        total = strategy.sum()
        if total > 1e-8:
            strategy /= total
        else:
            strategy = legal_mask_np / max(legal_mask_np.sum(), 1e-8)

        # ── Pillar 6: Bet Sizer (replaces Kelly) ──────────────────────────────
        action_idx = int(np.random.choice(7, p=strategy))
        action_idx = max(legal_moves, key=lambda a: strategy[a]) \
            if action_idx not in legal_moves else action_idx

        # ── Check-instead-of-fold guardrail ───────────────────────────────────
        # Folding when a free check is available is always strictly wrong.
        # If the bot picked FOLD (0) but to_call == 0 (no bet to face),
        # silently upgrade to CHECK (1) instead.
        if action_idx == 0 and to_call <= 0 and 1 in legal_moves:
            action_idx = 1
        amount = 0

        if action_idx >= 2:
            pot_size  = float(game_state_obj.pot)
            eff_stack = float(hero.stack)
            amount    = bucket_to_amount(
                bucket=action_idx,
                pot=pot_size,
                to_call=float(to_call),
                stack=eff_stack,
                min_raise=float(min_raise)
            )
            # Use action_idx=7 (custom amount) for non-all-in so engine gets
            # the exact chip count rather than a fixed fraction.
            if action_idx != 6:
                action_idx = 7

        # ── AnalysisReport ────────────────────────────────────────────────────
        # (calculated previously in REALITY CHECK block)

        # EV estimates: regret-weighted EV per action (relative, in BB)
        ev_by_action = {}
        for a in legal_moves:
            ev_by_action[a] = round(float(raw_regrets[a] * bb_amt), 2)

        analysis = {
            'win_prob':       round(win_prob, 4),
            'equity_vs_rng':  round(equity_vs_rng, 4),
            'hand_pct':       round(hand_pct, 4),
            'hand_name':      hand_name,
            'range_entropy':  float(self._range_encoder.entropy(opp_primary)),
            'exploit_signal': round(exploit, 4),
            'ddqn_hands':     self._ddqn.hands_observed(),
            'ev_by_action':   ev_by_action,
        }

        return action_idx, amount, analysis


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _is_preflop_premium(hand) -> bool:
    """Top ~10% preflop hands: AA, KK, QQ, JJ, AK."""
    if len(hand) < 2:
        return False
    # Engine encoding: card = rank*4 + suit  → rank = card // 4
    r1 = int(hand[0]) // 4
    r2 = int(hand[1]) // 4
    ranks = tuple(sorted([r1, r2], reverse=True))
    if ranks[0] == ranks[1] and ranks[0] >= 9:  # JJ+ (rank 9=J, 10=Q, 11=K, 12=A)
        return True
    if ranks == (12, 11):  # AK
        return True
    return False
