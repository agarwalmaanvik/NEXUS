import numpy as np
import torch
from engine_core import GameState
from vectorizer import PokerVectorizer

class ParallelPokerEnv:
    def __init__(self, n_envs=128, n_players=6):
        self.n_envs = n_envs
        self.n_players = n_players
        # Create N independent games (Training Mode = No Persistence)
        self.envs = [GameState(n_players=n_players, training_mode=True) for _ in range(n_envs)]
        self.vectorizer = PokerVectorizer
        
    def reset(self):
        """Resets all games and returns initial states."""
        states = []
        for env in self.envs:
            env.reset()
            
            # --- PERSONA INJECTION (Phase 9) ---
            # Randomly assign a persona to Player 1 (The Opponent)
            # 0=Maniac, 1=Nit, 2=Station, 3=Reg
            p_type = np.random.randint(0, 4)
            
            if p_type == 0: # Maniac (Loose-Aggressive)
                vpip = np.random.uniform(0.6, 0.9)
                pfr = np.random.uniform(0.4, 0.8)
                afq = np.random.uniform(0.6, 0.9)
                wtsd = np.random.uniform(0.3, 0.5)
            elif p_type == 1: # Nit (Tight-Passive/Agg)
                vpip = np.random.uniform(0.1, 0.2)
                pfr = np.random.uniform(0.05, 0.15)
                afq = np.random.uniform(0.1, 0.3)
                wtsd = np.random.uniform(0.2, 0.4)
            elif p_type == 2: # Station (Loose-Passive)
                vpip = np.random.uniform(0.4, 0.7)
                pfr = np.random.uniform(0.05, 0.2)
                afq = np.random.uniform(0.1, 0.3)
                wtsd = np.random.uniform(0.5, 0.8)
            else: # Reg (Balanced)
                vpip = np.random.uniform(0.2, 0.3)
                pfr = np.random.uniform(0.15, 0.25)
                afq = np.random.uniform(0.3, 0.5)
                wtsd = np.random.uniform(0.4, 0.6)
            
            # Set Stats for Player 1
            env.profiler.set_persona(1, vpip, pfr, afq, wtsd)
            
            # Vectorize state for the current player of that environment
            vec = self.vectorizer.state_to_tensor(env, env.current_player)
            states.append(vec)
        return np.array(states) # Shape: [n_envs, 171]

    def step(self, actions, amounts):
        """
        Applies batch actions to all environments.
        actions: list/array of ints [n_envs]
        amounts: list/array of ints [n_envs]
        """
        next_states = []
        dones = []
        payouts = []
        
        for i, env in enumerate(self.envs):
            # 1. Apply Action
            is_done = env.step(actions[i], amounts[i])
            
            # 2. Handle Outcome
            if is_done:
                pay = env.resolve_hand() # [P0, P1, ...]
                payouts.append(pay)
                # Auto-reset and re-inject a fresh random persona for the opponent
                env.reset()
                # Re-randomize opponent persona each hand for training diversity
                p_type = np.random.randint(0, 4)
                if p_type == 0:
                    vpip, pfr, afq, wtsd = (np.random.uniform(0.6,0.9), np.random.uniform(0.4,0.8),
                                            np.random.uniform(0.6,0.9), np.random.uniform(0.3,0.5))
                elif p_type == 1:
                    vpip, pfr, afq, wtsd = (np.random.uniform(0.1,0.2), np.random.uniform(0.05,0.15),
                                            np.random.uniform(0.1,0.3), np.random.uniform(0.2,0.4))
                elif p_type == 2:
                    vpip, pfr, afq, wtsd = (np.random.uniform(0.4,0.7), np.random.uniform(0.05,0.2),
                                            np.random.uniform(0.1,0.3), np.random.uniform(0.5,0.8))
                else:
                    vpip, pfr, afq, wtsd = (np.random.uniform(0.2,0.3), np.random.uniform(0.15,0.25),
                                            np.random.uniform(0.3,0.5), np.random.uniform(0.4,0.6))
                env.profiler.set_persona(1, vpip, pfr, afq, wtsd)
                dones.append(True)
            else:
                payouts.append(None)
                dones.append(False)
            
            # 3. Get Next State
            vec = self.vectorizer.state_to_tensor(env, env.current_player)
            next_states.append(vec)
            
        return np.array(next_states), dones, payouts

    def get_legal_moves_mask(self):
        """Returns boolean mask [n_envs, 7] of valid moves."""
        mask = np.zeros((self.n_envs, 7), dtype=bool)
        for i, env in enumerate(self.envs):
            for m in env.legal_moves:
                mask[i, m] = True
        return mask
        
    def get_current_players(self):
        """Returns list of who is acting in each env."""
        return [env.current_player for env in self.envs]

    def get_env_info(self):
        """Returns [(pot, stage)] per env for trajectory credit assignment."""
        return [(env.pot, env.stage) for env in self.envs]

    def get_current_states(self):
        """Returns [n_envs, 171] raw state vectors using current player perspective."""
        states = []
        for env in self.envs:
            vec = self.vectorizer.state_to_tensor(env, env.current_player)
            states.append(vec)
        return np.array(states, dtype=np.float32)
