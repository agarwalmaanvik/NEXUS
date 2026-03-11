import torch
import torch.nn as nn
from cfr_agent import CFRAgent
from memory import ReplayBuffer
from engine_core import GameState
from vectorizer import PokerVectorizer

class PokerTrainer:
    def __init__(self, device="cpu"):
        self.device = device
        self.agent = CFRAgent(device)
        self.memory = ReplayBuffer(capacity=100000)
        self.game = GameState(n_players=6)
        self.vectorizer = PokerVectorizer
        
        self.batch_size = 64
        self.min_buffer_size = 100
        
    def run_iteration(self):
        """
        1. Self-Play: Generate data
        2. Train: Update networks
        """
        # 1. Self Play (Traverse)
        # How many traversals?
        # A few hands per iteration.
        for _ in range(10):
            self.agent.traverse(self.game, self.vectorizer, self.memory)
            
    def train_step(self):
        """
        Updates the network from ReplayBuffer.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states, advs, vals = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        advs = advs.to(self.device)
        vals = vals.to(self.device)
        
        # Forward Pass
        pred_adv, pred_val = self.agent.net(states)
        
        # Losses
        # A. Advantage Loss (MSE)
        # Target is [0, 0, reward] (one-hot masked)
        # But we only want to penalize specific action slots?
        # Actually our target vector IS specific.
        # But for non-taken actions, target is 0. And we don't want to train them to be 0 if we didn't explore them?
        # Since we use outcome sampling, forcing others to 0 is roughly OK ("advantage of unknown actions is assumed baseline 0" - simplified).
        # Better: Mask loss?
        # For this milestone, pure MSE is standard for "Advantage Regression".
        
        loss_adv = nn.MSELoss()(pred_adv, advs)
        
        # B. Value Loss (MSE)
        loss_val = nn.MSELoss()(pred_val, vals)
        
        total_loss = loss_adv + loss_val
        
        # Update
        self.agent.optimizer.zero_grad()
        total_loss.backward()
        self.agent.optimizer.step()
        
        return total_loss.item()
        
    def save(self, path="checkpoint.pt"):
        torch.save(self.agent.net.state_dict(), path)
        
    def load(self, path="checkpoint.pt"):
        self.agent.net.load_state_dict(torch.load(path))

if __name__ == "__main__":
    print("Starting Training Loop...")
    trainer = PokerTrainer()
    
    for i in range(100):
        trainer.run_iteration()
        loss = trainer.train_step()
        if i % 10 == 0:
            print(f"Iter {i}: Loss={loss:.4f} Memory={len(trainer.memory)}")
            
    print("Training Complete.")
