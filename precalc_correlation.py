import random
import time
from treys import Card, Evaluator, Deck

# Target: 50,000 boards per hand for accuracy without taking hours.
NUM_TRIALS = 50000

# Treys ranking threshold for Two Pair or better:
# 1 (Royal Flush) to 3325 (lowest Two Pair)
MONSTER_THRESHOLD = 3325

def get_169_hands():
    """Returns a list of 169 starting hand tuples (treys.Card format, human string)."""
    hands = []
    ranks = 'AKQJT98765432'
    # To get 169 hands, we define representative actual cards.
    # Pairs:
    for r in ranks:
        hands.append(([Card.new(r+'s'), Card.new(r+'h')], r*2))
    
    # Suited and Offsuit
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            r1, r2 = ranks[i], ranks[j]
            # Suited: Both spades
            hands.append(([Card.new(r1+'s'), Card.new(r2+'s')], r1+r2+'s'))
            # Offsuit: Spades + Hearts
            hands.append(([Card.new(r1+'s'), Card.new(r2+'h')], r1+r2+'o'))
    return hands

def main():
    evaluator = Evaluator()
    hands = get_169_hands()
    correlation_dict = {}
    
    print(f"Starting simulation of {len(hands)} starting hands...")
    print(f"Running {NUM_TRIALS} runouts per hand.\n")
    
    start_time = time.time()
    
    for idx, (hole_cards, name) in enumerate(hands):
        hits = 0
        # Initialize an optimized loop instead of instantiating Deck repeatedly
        # The remaining 50 cards
        remaining_deck = [c for c in Deck.GetFullDeck() if c not in hole_cards]
        
        for _ in range(NUM_TRIALS):
            # random.sample is fast enough for 5 cards
            board = random.sample(remaining_deck, 5)
            # Evaluate
            rank = evaluator.evaluate(board, hole_cards)
            if rank <= MONSTER_THRESHOLD:
                hits += 1
                
        score = hits / NUM_TRIALS
        correlation_dict[name] = round(score, 4)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/169 hands. Time elapsed: {time.time() - start_time:.1f}s")
            
    print("\n--- SIMULATION COMPLETE ---")
    print(f"Total time: {time.time() - start_time:.1f}s\n")
    
    # Let's print the dictionary definition so we can copy it directly into our script
    dict_str = "PREFLOP_CORRELATION = {\n"
    items = sorted(correlation_dict.items(), key=lambda x: x[1], reverse=True)
    chunks = []
    for k, v in items:
        chunks.append(f"'{k}': {v}")
    
    for i in range(0, len(chunks), 8):
        dict_str += "    " + ", ".join(chunks[i:i+8]) + ",\n"
    dict_str += "}\n"
    
    print(dict_str)
    
    # Also write it to a file
    with open("preflop_correlation_dict.py", "w") as f:
        f.write("# Generated structurally by 50,000 runouts per hand\n")
        f.write(dict_str)

if __name__ == "__main__":
    main()
