import unittest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfr_agent import CFRAgent
from engine_core import GameState
from vectorizer import PokerVectorizer
from memory import ReplayBuffer

class TestCFRAgent(unittest.TestCase):
    def setUp(self):
        self.agent = CFRAgent()
        self.game = GameState(n_players=2)
        self.vectorizer = PokerVectorizer
        self.memory = ReplayBuffer(capacity=100)

    def test_get_strategy(self):
        # Fake input tensor
        state = torch.zeros(163, dtype=torch.float32)
        strategy = self.agent.get_strategy(state)
        
        self.assertEqual(len(strategy), 3)
        # Should sum to 1.0 (approx due to float)
        self.assertAlmostEqual(sum(strategy), 1.0, places=5)
        
    def test_get_batch_strategy(self):
        # Test batch of 4
        batch_size = 4
        states = torch.zeros((batch_size, 163), dtype=torch.float32)
        legal = np.ones((batch_size, 3), dtype=bool)
        
        actions, amounts, strategies = self.agent.get_batch_strategy(states, legal)
        
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(amounts), batch_size)
        self.assertEqual(strategies.shape, (batch_size, 3))
        
        # Check specific mask
        legal[0] = [True, False, False] # Only fold allowed
        actions, _, strats = self.agent.get_batch_strategy(states, legal)
        
        # Strategy for item 0 should be [1, 0, 0]
        self.assertAlmostEqual(strats[0][0], 1.0)
        self.assertAlmostEqual(strats[0][1], 0.0)
        self.assertAlmostEqual(strats[0][2], 0.0)
        self.assertEqual(actions[0], 0)

if __name__ == '__main__':
    unittest.main()
