import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poker_bot_api import SOTAPokerBot
from engine_core import GameState

class TestPokerBotAPI(unittest.TestCase):
    def test_api_inference(self):
        # Initialize Bot (mocking model path to avoid loading actual file if not exists, or handle graceful fail)
        # The API handles FileNotFoundError by initializing random model.
        bot = SOTAPokerBot(model_path="non_existent.pt")
        
        game = GameState(n_players=6)
        game.reset()
        
        try:
            # Get Action
            action, amount, market_data = bot.get_action(game, hero_seat_id=game.current_player)
            
            print(f"DEBUG: Action={action}, MarketData={market_data}")
            
            self.assertIn(action, [0, 1, 2])
            self.assertIsInstance(amount, int)
            
            # Check Market Data
            if market_data is not None:
                self.assertIsInstance(market_data, dict)
                self.assertIn("fair_value_chips", market_data)
                self.assertIn("volatility", market_data)
                self.assertIn("bid", market_data)
                self.assertIn("ask", market_data)
            else:
                pass
                
        except Exception as e:
            with open("crash_log.txt", "w") as f:
                import traceback
                f.write(traceback.format_exc())
            self.fail(f"Crashed: {e}")

if __name__ == '__main__':
    unittest.main()
