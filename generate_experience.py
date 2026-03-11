import pandas as pd
import numpy as np
import random
from engine_core import GameState
from vectorizer import PokerVectorizer

# --- CONFIG ---
NUM_GAMES = 5000 # Start small to verify, then scale to 1M
OUTPUT_FILE = "deep_poker_data.csv"

def generate_billion_samples():
    print(f"🏭 Starting The Factory... Target: {NUM_GAMES} Games.")
    
    # Initialize Engine
    game = GameState(n_players=6)
    vectorizer = PokerVectorizer()
    
    data_buffer = []
    
    for g in range(NUM_GAMES):
        game.reset()
        done = False
        
        while not done:
            # 1. Get State for Current Player
            state_vec = vectorizer.state_to_tensor(game, game.current_player)
            
            # 2. Pick Action (Random for now - Exploring the Tree)
            # We bias slightly towards Check/Call to make hands last longer
            valid = game.legal_moves
            if 1 in valid and random.random() < 0.6: action = 1 # Call/Check bias
            else: action = random.choice(valid)
            
            # Amount Logic (Randomize Raise Sizes)
            amt = 0
            if action == 2:
                # Raise between Min and All-In
                min_r = game.min_raise
                max_r = game.players[game.current_player].stack
                if max_r > min_r:
                    amt = random.randint(int(min_r), int(max_r))
                else:
                    amt = max_r
            
            # 3. Step
            # Store transition? For now, we just store State -> Action
            # In Phase 2, we store State -> Reward
            
            # Simple Row: [State_0 ... State_135, Action_Taken]
            row = np.append(state_vec, action)
            data_buffer.append(row)
            
            done = game.step(action, amt)
            
        if g % 100 == 0:
            print(f"   Game {g}/{NUM_GAMES} | Buffer: {len(data_buffer)} samples")

    # Save
    print("💾 Flashing Memory to Disk...")
    cols = [f'f{i}' for i in range(136)] + ['target_action']
    df = pd.DataFrame(data_buffer, columns=cols)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_billion_samples()