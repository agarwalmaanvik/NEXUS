
import random
import numpy as np
import time
from engine_core import GameState, Action
from vectorizer import PokerVectorizer

def random_policy(game, legal_moves):
    return random.choice(legal_moves)

def rollout(game):
    while True:
        # Check end conditions
        if game.stage > 3: break
        active_count = sum(1 for p in game.players if p.active)
        if active_count <= 1: break
        
        legal = game.legal_moves
        if not legal: break
        
        action = random_policy(game, legal)
        amt = 0
        if action == 2: amt = game.min_raise
        
        game.step(action, amt)
    
    return game.resolve_hand()

def generate_sample():
    # 1. Create Random Situation
    game = GameState(n_players=6)
    game.reset()
    
    # 0 to 50 random moves
    steps = random.randint(0, 50)
    
    for _ in range(steps):
        if game.stage > 3: break
        active = sum(1 for p in game.players if p.active)
        if active <= 1: break
        
        legal = game.legal_moves
        if not legal: break
        
        a = random_policy(game, legal)
        amt = 0
        if a == 2: amt = game.min_raise
            
        game.step(a, amt)
        
    # Check if terminal
    active = sum(1 for p in game.players if p.active)
    if active <= 1 or game.stage > 3:
        return None
        
    # Snapshot Input
    try:
        input_tensor = PokerVectorizer.state_to_tensor(game, game.current_player)
    except Exception as e:
        # Vectorizer might fail on weird states?
        return None
    
    # Snapshot State for Rollout
    state_dict = game.get_state()
    clone = GameState(n_players=6)
    clone.set_state(state_dict)
    
    # Rollout
    payouts = rollout(clone)
    
    total_pot = sum(payouts)
    if total_pot == 0: return None
    
    hero_payout = payouts[game.current_player]
    label = hero_payout / total_pot
    
    return input_tensor, label

if __name__ == "__main__":
    print("Generating samples...")
    t0 = time.time()
    count = 0
    for i in range(20):
        res = generate_sample()
        if res:
            x, y = res
            print(f"Sample {count}: Label={y:.2f} TensorShape={x.shape}")
            count += 1
            if count >= 5: break
            
    print(f"Done in {time.time()-t0:.2f}s")
