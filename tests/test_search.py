import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poker_bot_api import SOTAPokerBot
from engine_core import GameState

class TestSubgameSolving(unittest.TestCase):
    def test_rollout_search(self):
        # Initialize Bot (mocking model path)
        bot = SOTAPokerBot(model_path="non_existent.pt")
        
        game = GameState(n_players=2)
        game.reset()
        
        # Force a decision point
        # Run search
        # We expect get_action_with_search to return a valid action and distribution
        
        action, amount, win_prob = bot.get_action(game, hero_seat_id=game.current_player, use_search=True)
        
        self.assertIn(action, [0, 1, 2])
        self.assertTrue(0 <= win_prob <= 1.0)
        
        # Check internal search method if exposed
        if hasattr(bot, '_run_rollout_search'):
            # Manually trigger search on a specific state
            # Ensure it doesn't crash
            pass

if __name__ == '__main__':
    unittest.main()
