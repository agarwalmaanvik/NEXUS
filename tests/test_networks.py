import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import DeepCFR_Network

class TestDeepCFRNetwork(unittest.TestCase):
    def test_forward_pass(self):
        # Batch of 4, 163 dimensions
        x = torch.randn(4, 163)
        net = DeepCFR_Network(input_dim=163)
        
        adv, val = net(x)
        
        # Advantage shape: [4, 3] (Fold, Call, Raise)
        self.assertEqual(adv.shape, (4, 3))
        
        # Value shape: [4, 1]
        self.assertEqual(val.shape, (4, 1))
        
        # Value can be anything (Chip Count)
        self.assertTrue(torch.is_tensor(val))

if __name__ == '__main__':
    unittest.main()
