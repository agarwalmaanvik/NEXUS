import itertools
from collections import defaultdict

class FastEvaluator:
    """
    High-Performance Integer-Based Evaluator.
    Uses Prime Products to evaluate hands in O(1) time.
    """
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    
    # We use the standard equivalence 1 = 7-5 Lowball? No.
    # We want 1 = Worst hand (7-5-4-3-2 unsuited), 7462 = Royal Flush.
    # The original Cactus Kev table maps 1=Royal Flush. We will invert it.
    MAX_RANK = 7462

    # Global lookup tables
    _lookup_generated = False
    _flush_lookup = {}
    _unsuited_lookup = {}

    def __init__(self):
        if not FastEvaluator._lookup_generated:
            self._precompute_tables()
            FastEvaluator._lookup_generated = True

    def _precompute_tables(self):
        print("⚡ Precomputing FastEvaluator Tables...")
        # Helpers
        
        # Rank Key Function: Returns tuple for correct sorting
        def rank_key(hand, is_flush=False, is_straight=False):
            # Sort descending
            r = sorted(list(hand), reverse=True)
            
            # Special Case: Wheel (A-2-3-4-5)
            # Normal: 12, 3, 2, 1, 0
            if r == [12, 3, 2, 1, 0]:
                r = [3, 2, 1, 0, -1] # Treat A as low
                
            if is_flush and is_straight: return (8, r)
            if is_flush: return (5, r)
            if is_straight: return (4, r)
            
            # Counts
            counts = defaultdict(int)
            for x in hand: counts[x] += 1
            freqs = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
            # Pattern: (Count, Rank)
            
            pattern = tuple(f[1] for f in freqs)
            kickers = tuple(f[0] for f in freqs)
            # Kickers are already sorted by rank descending in the tuple due to key
            
            if pattern == (4, 1): return (7, kickers)
            if pattern == (3, 2): return (6, kickers)
            if pattern == (3, 1, 1): return (3, kickers)
            if pattern == (2, 2, 1): return (2, kickers)
            if pattern == (2, 1, 1, 1): return (1, kickers)
            return (0, kickers)

        # 1. Generate ALL 5-card combinations of ranks (1287 unique sets)
        # These cover Flushes, Straights, and High Cards.
        # But wait, Pairs/Trips are NOT combinations of 5 distinct ranks.
        # They involve repeated ranks.
        
        # We need a list of ALL valid hand types.
        # Total poker hands = 2,598,960.
        # Total EQUIVALENCE CLASSES = 7462.
        
        # We generate equivalence classes by category.
        
        all_hands = []
        
        # A. Unique Ranks (No Pairs)
        # Can be High Card, Straight, Flush, Straight Flush.
        for ranks in itertools.combinations(range(13), 5):
            is_str = self._is_straight(ranks)
            
            # Option 1: Flush / SF
            score_f = rank_key(ranks, is_flush=True, is_straight=is_str)
            all_hands.append({
                'type': 'flush',
                'ranks': ranks,
                'score': score_f,
                'bitmask': sum(1 << r for r in ranks)
            })
            
            # Option 2: High Card / Straight (Unsuited)
            if not is_str:
                score_nf = rank_key(ranks, is_flush=False, is_straight=False)
                all_hands.append({
                    'type': 'non_flush',
                    'ranks': ranks,
                    'score': score_nf,
                    'prime': self._get_prime_product(ranks)
                })
            else:
                # Straight (Unsuited)
                score_nf = rank_key(ranks, is_flush=False, is_straight=True)
                all_hands.append({
                    'type': 'non_flush',
                    'ranks': ranks,
                    'score': score_nf,
                    'prime': self._get_prime_product(ranks)
                })

        # B. Pairs/Trips/Quads/FH (Always non-flush)
        
        # Quads (4+1)
        for r4 in range(13):
            for r1 in range(13):
                if r4==r1: continue
                hand_tuple = tuple([r4]*4 + [r1])
                score = rank_key(hand_tuple)
                all_hands.append({'type':'non_flush', 'score':score, 'prime':self._get_prime_product(hand_tuple)})

        # FH (3+2)
        for r3 in range(13):
            for r2 in range(13):
                if r3==r2: continue
                hand_tuple = tuple([r3]*3 + [r2]*2)
                score = rank_key(hand_tuple)
                all_hands.append({'type':'non_flush', 'score':score, 'prime':self._get_prime_product(hand_tuple)})

        # Trips (3+1+1)
        for r3 in range(13):
            for kickers in itertools.combinations([r for r in range(13) if r!=r3], 2):
                hand_tuple = tuple([r3]*3 + list(kickers))
                score = rank_key(hand_tuple)
                all_hands.append({'type':'non_flush', 'score':score, 'prime':self._get_prime_product(hand_tuple)})
                
        # Two Pair (2+2+1)
        for pairs in itertools.combinations(range(13), 2):
            for kicker in range(13):
                if kicker in pairs: continue
                hand_tuple = tuple([pairs[1]]*2 + [pairs[0]]*2 + [kicker]) # Pairs need order? No combination is sorted.
                score = rank_key(hand_tuple)
                all_hands.append({'type':'non_flush', 'score':score, 'prime':self._get_prime_product(hand_tuple)})

        # Pair (2+1+1+1)
        for r2 in range(13):
            for kickers in itertools.combinations([r for r in range(13) if r!=r2], 3):
                hand_tuple = tuple([r2]*2 + list(kickers))
                score = rank_key(hand_tuple)
                all_hands.append({'type':'non_flush', 'score':score, 'prime':self._get_prime_product(hand_tuple)})

        # Sort ALL hands by score
        # Note: 'score' tuple (Type, Ranks) handles sorting naturally.
        # But for non-flushes, we have multiple entries?
        # A specific 'prime product' maps to exactly one rank.
        
        all_len = len(all_hands)
        print(f"Generated {all_len} hand classes (approx 7462 expected + redundancy). Sorting...")
        
        all_hands.sort(key=lambda x: x['score'])
        
        # Assign Ranks
        current_rank = 1
        last_score = None
        
        # (3, 2, 4, 5, 7) has same score as (2, 3, 5, 7, 4).
        # But our generation ensures uniqueness of prime product for unsuited.
        # For flushes, bitmask is unique.
        
        # Standard approach: Simple iteration.
        # list should be strictly unique equivalence classes.
        
        for i, h in enumerate(all_hands):
            # Check for strict increase
            if last_score is not None and h['score'] > last_score:
                current_rank += 1
            elif last_score is not None and h['score'] < last_score:
                # Should not happen due to sort
                pass
                
            last_score = h['score']
            
            if h['type'] == 'flush':
                FastEvaluator._flush_lookup[h['bitmask']] = current_rank
            else:
                FastEvaluator._unsuited_lookup[h['prime']] = current_rank

        print(f"✅ Tables Ready. Max Rank: {current_rank}")

    def _is_straight(self, ranks):
        # ranks is list of 5 ints
        s = sorted(list(set(ranks)))
        if len(s) != 5: return False
        if s[-1] - s[0] == 4: return True
        if s == [0, 1, 2, 3, 12]: return True
        return False

    def _get_prime_product(self, ranks):
        p = 1
        for r in ranks:
            p *= FastEvaluator.PRIMES[r]
        return p

    def evaluate(self, cards):
        """
        Input: List of Integers (0-51). Handles numpy integer types (int8, int64, etc.)
        Returns: Int (1 to 7462). Returns 0 only if cards list is empty or malformed.
        """
        # Cast ALL cards to plain Python int immediately.
        # np.int8 bitshifts overflow at 8 bits (e.g. 1 << np.int8(12) = -64, not 4096).
        # This was silently producing wrong bitmasks and prime products on every lookup.
        cards = [int(c) for c in cards]

        # For 7 cards, iterate best 5.
        if len(cards) > 5:
            best = 0
            for hand in itertools.combinations(cards, 5):
                r = self.evaluate(list(hand))
                if r > best: best = r
            return best

        # 5-Card Eval
        # Engine encodes cards as rank*4 + suit  (rank 0-12, suit 0-3)
        s0 = cards[0] % 4
        is_flush = all(c % 4 == s0 for c in cards[1:])

        current_ranks = [c // 4 for c in cards]

        if is_flush:
            bitmask = sum(1 << r for r in current_ranks)
            return FastEvaluator._flush_lookup.get(bitmask, 0)
        else:
            prime_product = 1
            for r in current_ranks:
                prime_product *= FastEvaluator.PRIMES[r]
            return FastEvaluator._unsuited_lookup.get(prime_product, 0)