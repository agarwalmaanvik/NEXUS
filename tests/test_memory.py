
import unittest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def test_add_sample(self):
        buffer = ReplayBuffer(capacity=10)
        state = np.zeros(163, dtype=np.float32)
        adv = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        val = 1.0
        
        buffer.add(state, adv, val)
        self.assertEqual(len(buffer), 1)

    def test_sample_batch(self):
        buffer = ReplayBuffer(capacity=10)
        for i in range(5):
            state = np.zeros(163, dtype=np.float32)
            state[0] = i # Differentiate
            adv = np.array([0, 0, 0], dtype=np.float32)
            val = float(i)
            buffer.add(state, adv, val)
            
        states, advs, vals = buffer.sample(3)
        
        self.assertEqual(states.shape, (3, 163))
        self.assertEqual(advs.shape, (3, 3))
        self.assertEqual(vals.shape, (3, 1))
        self.assertTrue(torch.is_tensor(states))

    def test_capacity(self):
        buffer = ReplayBuffer(capacity=2)
        buffer.add(np.zeros(163), np.zeros(3), 0)
        buffer.add(np.zeros(163), np.zeros(3), 1)
        buffer.add(np.zeros(163), np.zeros(3), 2)
        
        # Should be 2 (oldest dropped)
        self.assertEqual(len(buffer), 2)

if __name__ == '__main__':
    unittest.main()
