
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine_core import GameState

class TestResetFix(unittest.TestCase):
    def test_stack_reload(self):
        game = GameState(n_players=2)
        game.reset()
        
        # Verify initial stack
        self.assertEqual(game.players[0].stack, 1990.0) # SB posts 10. Stack 2000->1990.
        # Wait, reset() posts blinds.
        # If 2 players:
        # P0=SB, P1=BB or vice versa.
        # Check starting_stack is 2000.
        self.assertEqual(game.players[0].starting_stack, 2000.0)
        
        # Simulate loss
        game.players[0].stack = 0
        game.players[1].stack = 4000
        
        # Reset again
        game.reset()
        
        # Verify stacks are back to ~2000 (minus blinds)
        # One of them will be SB (10), one BB (20).
        # Stack should be 1990 or 1980.
        
        s0 = game.players[0].stack
        s1 = game.players[1].stack
        
        self.assertTrue(s0 >= 1980.0)
        self.assertTrue(s1 >= 1980.0)
        self.assertTrue(game.players[0].active)

if __name__ == '__main__':
    unittest.main()
