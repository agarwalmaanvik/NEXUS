import numpy as np
import json
import os

class OpponentProfiler:
    def __init__(self, n_players=2, training_mode=False, filename="opponent_memory.json"):
        self.n = n_players
        self.training_mode = training_mode
        self.filename = filename
        
        # Stats Storage
        self.hands_played = np.zeros(n_players)
        self.vpip_count = np.zeros(n_players)
        self.pfr_count = np.zeros(n_players)
        self.agg_moves = np.zeros(n_players)
        self.passive_moves = np.zeros(n_players)
        self.wtsd_count = np.zeros(n_players)
        self.saw_flop_count = np.zeros(n_players)
        
        # Hand-Specific Flags
        self.did_vpip = np.zeros(n_players, dtype=bool) 
        self.did_pfr = np.zeros(n_players, dtype=bool)
        self.saw_flop_current = np.zeros(n_players, dtype=bool)
        
        # Load Memory if not in training mode
        if not self.training_mode:
            self.load_memory()

    def load_memory(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    # We only load stats for Player 1 (The Human) in 1v1
                    # Player 0 is Bot (we don't track bot history here usually, or maybe we do?)
                    # Let's assume P1 is the human for now.
                    p1_data = data.get("player_1", {})
                    if p1_data:
                        self.hands_played[1] = p1_data.get("hands", 0)
                        self.vpip_count[1] = p1_data.get("vpip", 0)
                        self.pfr_count[1] = p1_data.get("pfr", 0)
                        self.agg_moves[1] = p1_data.get("agg", 0)
                        self.passive_moves[1] = p1_data.get("pas", 0)
                        self.wtsd_count[1] = p1_data.get("wtsd", 0)
                        self.saw_flop_count[1] = p1_data.get("saw_flop", 0)
                    print(f"🧠 Profiler Loaded: {int(self.hands_played[1])} hands on record.")
            except Exception as e:
                print(f"⚠️ Memory Corrupt: {e}")

    def save_memory(self):
        if self.training_mode: return
        
        data = {
            "player_1": {
                "hands": float(self.hands_played[1]),
                "vpip": float(self.vpip_count[1]),
                "pfr": float(self.pfr_count[1]),
                "agg": float(self.agg_moves[1]),
                "pas": float(self.passive_moves[1]),
                "wtsd": float(self.wtsd_count[1]),
                "saw_flop": float(self.saw_flop_count[1])
            }
        }
        try:
            with open(self.filename, 'w') as f:
                json.dump(data, f)
        except: pass

    def start_hand(self):
        """Called at the start of a new hand."""
        self.hands_played += 1
        self.did_vpip[:] = False
        self.did_pfr[:] = False
        self.saw_flop_current[:] = False
        
        # Auto-Save every 5 hands
        if not self.training_mode and self.hands_played[1] % 5 == 0:
            self.save_memory()

    def record_action(self, player_id, action_type, stage):
        """
        action_type: 0=Fold, 1=Check/Call, 2+=Raise
        stage: 0=PreFlop, 1=Flop, 2=Turn, 3=River
        """
        # 1. AFq Tracking
        if action_type >= 2: # Raise
            self.agg_moves[player_id] += 1
        elif action_type == 1: # Call/Check
            self.passive_moves[player_id] += 1
        elif action_type == 0: # Fold
            self.passive_moves[player_id] += 1
            
        # 2. Pre-Flop Stats (VPIP / PFR)
        if stage == 0:
            if action_type >= 2: # Raise
                if not self.did_pfr[player_id]:
                    self.pfr_count[player_id] += 1
                    self.did_pfr[player_id] = True
                if not self.did_vpip[player_id]:
                    self.vpip_count[player_id] += 1
                    self.did_vpip[player_id] = True
                    
            elif action_type == 1: # Call
                if not self.did_vpip[player_id]:
                    self.vpip_count[player_id] += 1
                    self.did_vpip[player_id] = True

    def record_stage_transition(self, new_stage, active_players):
        """Called when game moves to next street."""
        if new_stage == 1: # FLOP
            for pid in active_players:
                if not self.saw_flop_current[pid]:
                    self.saw_flop_count[pid] += 1
                    self.saw_flop_current[pid] = True
                    
    def record_showdown(self, active_players):
        """Called at end of hand if it went to showdown."""
        for pid in active_players:
            self.wtsd_count[pid] += 1
            
        if not self.training_mode:
            self.save_memory()

    def get_stats(self, player_id):
        """Returns [VPIP, PFR, AFq, WTSD] normalized 0-1."""
        hp = max(1, self.hands_played[player_id])
        saw_flop = max(1, self.saw_flop_count[player_id])
        
        vpip = self.vpip_count[player_id] / hp
        pfr = self.pfr_count[player_id] / hp
        
        tot_moves = self.agg_moves[player_id] + self.passive_moves[player_id]
        if tot_moves > 0:
            afq = self.agg_moves[player_id] / tot_moves
        else:
            afq = 0.0
            
        wtsd = self.wtsd_count[player_id] / saw_flop
        
        return np.array([vpip, pfr, afq, wtsd], dtype=np.float32)

    def get_archetype(self, player_id):
        """Returns string description of player style."""
        stats = self.get_stats(player_id) # [VPIP, PFR, AFq, WTSD]
        vpip = stats[0]
        afq = stats[2]
        
        if vpip > 0.5:
            if afq > 0.4: return "MANIAC"
            return "STATION"
        if vpip < 0.2:
            return "NIT"
        return "REGULAR"

    # For Training: Inject Random Personas
    def set_persona(self, player_id, vpip, pfr, afq, wtsd):
        # We fudge the counts to match the desired ratios
        # Set "hands_played" to 100 to give it weight
        self.hands_played[player_id] = 100
        self.vpip_count[player_id] = int(vpip * 100)
        self.pfr_count[player_id] = int(pfr * 100)
        
        # AFq is ratio, so:
        # Agg = AFq * 100, Pas = (1-AFq) * 100
        self.agg_moves[player_id] = int(afq * 100)
        self.passive_moves[player_id] = int((1.0 - afq) * 100)
        
        # WTSD = Count / SawFlop
        self.saw_flop_count[player_id] = 100
        self.wtsd_count[player_id] = int(wtsd * 100)
