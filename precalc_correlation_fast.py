import random
import time
from treys import Card, Evaluator, Deck

# Target: 5000 boards per hand for speed right now. Law of large numbers makes this within +/- 1%
NUM_TRIALS = 5000
MONSTER_THRESHOLD = 3325

def get_169_hands():
    hands = []
    ranks = 'AKQJT98765432'
    for r in ranks:
        hands.append(([Card.new(r+'s'), Card.new(r+'h')], r*2))
    
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            r1, r2 = ranks[i], ranks[j]
            hands.append(([Card.new(r1+'s'), Card.new(r2+'s')], r1+r2+'s'))
            hands.append(([Card.new(r1+'s'), Card.new(r2+'h')], r1+r2+'o'))
    return hands

def main():
    evaluator = Evaluator()
    hands = get_169_hands()
    correlation_dict = {}
    
    start_time = time.time()
    
    for idx, (hole_cards, name) in enumerate(hands):
        hits = 0
        remaining_deck = [c for c in Deck.GetFullDeck() if c not in hole_cards]
        
        for _ in range(NUM_TRIALS):
            board = random.sample(remaining_deck, 5)
            rank = evaluator.evaluate(board, hole_cards)
            if rank <= MONSTER_THRESHOLD:
                hits += 1
                
        score = hits / NUM_TRIALS
        correlation_dict[name] = round(score, 4)
        print(f"[{idx+1}/169] {name}: {score:.4f}", flush=True)
            
    # Also write it to a file
    dict_str = "PREFLOP_CORRELATION = {\\n"
    items = sorted(correlation_dict.items(), key=lambda x: x[1], reverse=True)
    chunks = []
    for k, v in items:
        chunks.append(f"'{k}': {v}")
    
    for i in range(0, len(chunks), 8):
        dict_str += "    " + ", ".join(chunks[i:i+8]) + ",\\n"
    dict_str += "}\\n"

    with open("preflop_correlation_dict.py", "w") as f:
        f.write("# Generated securely by 5,000 runouts per hand\\n")
        f.write(dict_str)
    
    print("DONE writing preflop_correlation_dict.py")

if __name__ == "__main__":
    main()
