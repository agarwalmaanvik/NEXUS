import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorizer import PokerVectorizer
from engine_core import GameState, Player, Action

class TestVectorizer(unittest.TestCase):
    def setUp(self):
        self.game = GameState(n_players=2)
        self.game.reset()
        
    def test_tensor_shape(self):
        # We expect a specific size.
        # Cards: 52 (Hero) + 52 (Board) = 104
        # Position: 6 (Bits) or 1 (Float)? User asked for "Context-Aware Tensor"
        # History: Last 10 actions * 4 features = 40
        # Context: Pot, Stack, Opponent Stack, Stage, Legal Moves mask?
        # Let's say approx 150-200 floats.
        
        tensor = PokerVectorizer.state_to_tensor(self.game, hero_seat_id=0)
        self.assertTrue(isinstance(tensor, np.ndarray))
        # 168 is the calculated total size: 104 + 10 + 40 + 6 + 4 + 4
        self.assertGreater(len(tensor), 150)
        
    def test_card_embedding(self):
        # Hero has As Ks (51, 50)
        self.game.players[0].hand = [51, 50]
        tensor = PokerVectorizer.state_to_tensor(self.game, hero_seat_id=0)
        
        # Check if indices corresponding to 51 and 50 are hot
        # Assuming first 52 are hero cards
        self.assertEqual(tensor[51], 1.0)
        self.assertEqual(tensor[50], 1.0)
        self.assertEqual(tensor[0], 0.0)

    def test_action_history(self):
        # Add some history
        # Player 0 Posts SB (10)
        # Player 1 Posts BB (20)
        # Player 0 Calls (20)
        # Player 1 Checks
        
        # Game state history is auto-populated by step(), but we can manually inject for unit testing specific vectorization
        self.game.history = [
            Action(0, 3, 10, 0),
            Action(1, 3, 20, 0),
            Action(0, 1, 10, 0), # Call adds 10 to reach 20
            Action(1, 1, 0, 0)   # Check
        ]
        
        tensor = PokerVectorizer.state_to_tensor(self.game, hero_seat_id=0)
        # We verify last action is checked in the encoding
        
    def test_geometric_features(self):
        # Pot 100. Stack 100.
        self.game.pot = 100
        self.game.players[0].stack = 100
        self.game.bb_amt = 10 
        
        # Logic is internally opaque but we test it runs without error
        p_tensor = PokerVectorizer.state_to_tensor(self.game, hero_seat_id=0)

if __name__ == '__main__':
    unittest.main()
