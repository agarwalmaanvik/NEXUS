import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer import PokerTrainer
from cfr_agent import CFRAgent
from memory import ReplayBuffer

class TestTrainer(unittest.TestCase):
    def test_train_step(self):
        # Initialize Trainer
        trainer = PokerTrainer()
        
        # Manually fill memory with some junk
        # Needs at least batch_size samples (default usually 32 or 64)
        # We can mock config batch size 
        trainer.batch_size = 4
        
        for _ in range(10):
            state = test_state = torch.randn(163).numpy()
            adv = test_adv = torch.randn(3).numpy()
            val = test_val = 1.0
            trainer.memory.add(state, adv, val)
            
        # Run one step
        initial_loss = trainer.train_step()
        
        # Verify loss is returned and network parameters changed (mocking check or just execution)
        self.assertIsInstance(initial_loss, float)
        
    def test_loop_integration(self):
        # Test full loop: traverse -> train
        trainer = PokerTrainer()
        trainer.batch_size = 2
        
        # Traverse
        trainer.run_iteration()
        # Should populate memory
        self.assertGreater(len(trainer.memory), 0)
        
        # Train
        loss = trainer.train_step()
        self.assertIsInstance(loss, float)

if __name__ == '__main__':
    unittest.main()
