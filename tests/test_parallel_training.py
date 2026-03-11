
import unittest
import torch
import sys
import os
import shutil
# Need to import from parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_master import train_master, NUM_ENVS, STEPS_PER_ITER
from parallel_env import ParallelPokerEnv
from cfr_agent import CFRAgent
from memory import ReplayBuffer

class TestParallelTraining(unittest.TestCase):
    def test_parallel_env_init(self):
        penv = ParallelPokerEnv(n_envs=4, n_players=6)
        states = penv.reset()
        self.assertEqual(states.shape, (4, 163))
        
    def test_parallel_step(self):
        penv = ParallelPokerEnv(n_envs=4, n_players=6)
        penv.reset()
        actions = [1, 1, 1, 1] # Call
        amounts = [0, 0, 0, 0]
        next_states, dones, payouts = penv.step(actions, amounts)
        self.assertEqual(next_states.shape, (4, 163))
        self.assertEqual(len(dones), 4)
        
    def test_training_loop_mock(self):
        # Run a mini-version of train_master logic
        # Initialize
        agent = CFRAgent(device="cpu") # Force CPU for test
        penv = ParallelPokerEnv(n_envs=2)
        memory = ReplayBuffer(capacity=100)
        
        current_states = penv.reset()
        trajectories = [[] for _ in range(2)]
        
        # Run 1 step
        state_tensor = torch.tensor(current_states, dtype=torch.float32)
        legal_mask = penv.get_legal_moves_mask()
        actions, amounts, _ = agent.get_batch_strategy(state_tensor, legal_mask)
        
        # Check output shapes
        self.assertEqual(len(actions), 2)
        
        # Step
        next_states, dones, payouts = penv.step(actions, amounts)
        
        # Verify valid transition
        self.assertEqual(next_states.shape, (2, 163))

if __name__ == "__main__":
    unittest.main()
