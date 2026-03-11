import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine_core import GameState, Player, Action

class TestPokerEngine(unittest.TestCase):
    def setUp(self):
        self.game = GameState(n_players=2)
        self.game.reset()

    def test_initialization(self):
        self.assertEqual(len(self.game.players), 2)
        self.assertEqual(self.game.pot, 30) # SB(10) + BB(20)
        self.assertEqual(len(self.game.history), 2) # Blinds posted

    def test_game_flow(self):
        # SB is button + 1 = Player 1
        # BB is button + 2 = Player 0 (mod 2)
        # Current player should be SB (Head's up rules are tricky, but generally dealer is SB, non-dealer is BB. 
        # But this engine uses standard ring game logic: Button=0, SB=1, BB=0 for 2 players. So current player should be SB=1 acting first.)
        # Button=0. SB=(0+1)%2=1. BB=(0+2)%2=0.
        # Current = (BB+1)%2 = 1. So Player 1 acts first.
        
        # Player 1 calls 10 (total 20)
        self.game.step(1) 
        self.assertEqual(self.game.players[1].bet, 20)
        self.assertEqual(self.game.pot, 40)
        
        # Player 0 checks (matches 20)
        self.game.step(1)
        
        # Should be Flop now
        self.assertEqual(self.game.stage, 1)
        self.assertEqual(len(self.game.board), 3)

    def test_state_restore(self):
        # Play some moves
        self.game.step(1) # Call
        self.game.step(1) # Check -> Flop
        
        # Capture state
        state = self.game.get_state()
        
        # Create new game and load
        game2 = GameState(n_players=2)
        game2.set_state(state)
        
        # Verify
        self.assertEqual(self.game.pot, game2.pot)
        self.assertEqual(self.game.board, game2.board)
        self.assertEqual(self.game.current_player, game2.current_player)
        self.assertEqual(self.game.players[0].stack, game2.players[0].stack)
        self.assertEqual(len(self.game.history), len(game2.history))
        
        # Continue game 2
        game2.step(1) # Check on flop
        self.assertEqual(len(game2.history), len(self.game.history) + 1)

    def test_arbitrary_state(self):
        # Test creating a specific scenario: River, Pot 1000, P1 all-in
        game = GameState(n_players=2)
        game.reset()
        
        state = game.get_state()
        state['stage'] = 3 # River
        state['pot'] = 1000.0
        state['board'] = [0, 1, 2, 3, 4] # 2c, 3c, 4c, 5c, 6c
        state['players'][0]['stack'] = 0
        state['players'][0]['all_in'] = True
        state['players'][0]['active'] = True
        state['players'][1]['bet'] = 500
        state['current_player'] = 1 
        
        game.set_state(state)
        
        self.assertEqual(game.stage, 3)
        self.assertEqual(game.pot, 1000.0)
        self.assertTrue(game.players[0].all_in)

if __name__ == '__main__':
    unittest.main()
