
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_evaluator import FastEvaluator

class TestFastEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = FastEvaluator()

    def test_royal_flush(self):
        # Royal Flush: 10, J, Q, K, A of Spades (Spades=3, Hearts=2, Diamonds=1, Clubs=0)
        # 0-12=Clubs, 13-25=Diamonds, 26-38=Hearts, 39-51=Spades
        # 10s=48, Js=49, Qs=50, Ks=51, As=39? No, A is 12, 25, 38, 51.
        # Wait, standard mapping: 0=2c ... 12=Ac. 13=2d ... 25=Ad.
        # Spades (3): 39=2s ... 51=As.
        # Royal: 10s(47), Js(48), Qs(49), Ks(50), As(51).
        
        # Let's verify mapping in code or define it here.
        # 0=2, 8=10, 9=J, 10=Q, 11=K, 12=A.
        # Suit offsets: 0, 13, 26, 39.
        
        hand = [47, 48, 49, 50, 51] 
        rank = self.evaluator.evaluate(hand)
        self.assertEqual(rank, 7462, "Royal Flush should be max rank")

    def test_straight_flush(self):
        # 9s, 10s, Js, Qs, Ks -> 46, 47, 48, 49, 50
        hand = [46, 47, 48, 49, 50]
        rank = self.evaluator.evaluate(hand)
        self.assertTrue(rank < 7462)
        self.assertTrue(rank > 7000) # Arbitrary threshold for Straight Flush

    def test_quads(self):
        # 4 Aces + 2c
        # As=51, Ah=38, Ad=25, Ac=12. 2c=0.
        hand = [51, 38, 25, 12, 0]
        rank = self.evaluator.evaluate(hand)
        
        # 4 Kings + 2c
        # Ks=50, Kh=37, Kd=24, Kc=11.
        hand2 = [50, 37, 24, 11, 0]
        rank2 = self.evaluator.evaluate(hand2)
        
        self.assertTrue(rank > rank2, "Quad Aces should beat Quad Kings")

    def test_full_house(self):
        # A A A K K
        hand = [51, 38, 25, 50, 37]
        rank = self.evaluator.evaluate(hand)
        
        # A A A Q Q
        hand2 = [51, 38, 25, 49, 36]
        rank2 = self.evaluator.evaluate(hand2)
        
        self.assertTrue(rank > rank2)

    def test_flush_beats_straight(self):
        # Flush: 2s 4s 6s 8s 10s
        hand_flush = [39, 41, 43, 45, 47]
        
        # Straight: 2c 3d 4h 5s 6c
        # 2=0, 3=14, 4=28, 5=42, 6=4
        hand_straight = [0, 14, 28, 42, 4]
        
        r_flush = self.evaluator.evaluate(hand_flush)
        r_straight = self.evaluator.evaluate(hand_straight)
        
        self.assertTrue(r_flush > r_straight)

    def test_ties(self):
        # Split pot case
        h1 = [0, 1, 2, 3, 5] # 2,3,4,5,7 (High 7)
        h2 = [13, 14, 15, 16, 18] # 2,3,4,5,7 (High 7, different suits)
        
        r1 = self.evaluator.evaluate(h1)
        r2 = self.evaluator.evaluate(h2)
        self.assertEqual(r1, r2)

if __name__ == '__main__':
    unittest.main()
