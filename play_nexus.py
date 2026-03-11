"""
play_nexus.py  —  Interactive heads-up poker vs the trained NEXUS bot
Run:  python play_nexus.py

Controls (when it's your turn):
  f          → Fold
  c          → Call / Check
  r <amount> → Raise to total of <amount> chips  (e.g. "r 200")
  a          → All-in
  q          → Quit session
"""

import os
import sys

# ── Colour helpers ───────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def col(text, colour): return f"{colour}{text}{RESET}"

# ── Card rendering ───────────────────────────────────────────────────────────
RANKS = "23456789TJQKA"
SUITS = "cdhs"
SUIT_SYM = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
SUIT_COL = {"c": GREEN, "d": RED, "h": RED, "s": CYAN}

def card_str(card_int: int) -> str:
    rank = RANKS[card_int // 4]
    suit = SUITS[card_int % 4]
    return col(f"{rank}{SUIT_SYM[suit]}", SUIT_COL[suit] + BOLD)

def hand_str(cards):
    return "  ".join(card_str(c) for c in cards) if cards else col("??  ??", DIM)

def board_str(cards):
    if not cards: return col("(no board cards yet)", DIM)
    return "  ".join(card_str(c) for c in cards)

STAGES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}
ACTION_NAMES = {
    0:"FOLD", 1:"CHECK/CALL", 2:"RAISE min", 3:"RAISE 33%pot",
    4:"RAISE 75%pot", 5:"RAISE 1.5xpot", 6:"ALL-IN", 7:"RAISE"
}

def sep(char="─", w=62): print(col(char * w, DIM))

# ── Table display ─────────────────────────────────────────────────────────────
def print_table(gs, hero_seat, hand_num, session_profit):
    bot_seat = 1 - hero_seat
    hero = gs.players[hero_seat]
    bot  = gs.players[bot_seat]
    current_high = max(p.bet for p in gs.players)
    to_call = max(0, current_high - hero.bet)
    stage_name = STAGES.get(int(gs.stage), "?")

    os.system("cls" if os.name == "nt" else "clear")
    print(col("  ╔══════════════════════════════════════════════════════════════╗", CYAN))
    print(col("  ║  N E X U S   Heads-Up Poker                                 ║", CYAN + BOLD))
    print(col("  ╚══════════════════════════════════════════════════════════════╝", CYAN))
    pnl_col = GREEN if session_profit >= 0 else RED
    print(f"  Hand #{hand_num}  │  {col(stage_name, CYAN+BOLD)}  │  "
          f"Pot: {col(f'${gs.pot:.0f}', YELLOW+BOLD)}  │  "
          f"P&L: {col(f'{session_profit:+.0f}', pnl_col+BOLD)}")
    sep()
    bot_status = col("[folded]", RED) if not bot.active else ""
    print(f"  {col('NEXUS BOT', RED+BOLD):<30}  Stack: {col(f'${bot.stack:.0f}', BOLD)}   Bet: ${bot.bet:.0f}  {bot_status}")
    print(f"  Cards: {col('?? ??', DIM)}  (hidden)")
    sep("·")
    print(f"  Board:  {board_str(gs.board)}")
    sep("·")
    print(f"  {col('YOU', GREEN+BOLD):<30}  Stack: {col(f'${hero.stack:.0f}', BOLD)}   Bet: ${hero.bet:.0f}  to_call: ${to_call:.0f}")
    print(f"  Hand:  {hand_str(hero.hand)}")
    sep()

# ── Human input ───────────────────────────────────────────────────────────────
def get_human_action(gs, hero_seat):
    hero = gs.players[hero_seat]
    legal = list(gs.legal_moves)
    current_high = max(p.bet for p in gs.players)
    to_call = max(0, current_high - hero.bet)

    opts = []
    if 0 in legal: opts.append(col("f=fold", RED))
    if 1 in legal: opts.append(col("c=" + ("check" if to_call == 0 else f"call ${to_call:.0f}"), CYAN))
    if any(m >= 2 for m in legal):
        opts.append(col(f"r <chips>=raise (min raise: ${gs.min_raise:.0f})", YELLOW))
        opts.append(col("a=all-in", YELLOW + BOLD))
    opts.append(col("q=quit", DIM))

    print(f"\n  Options: {' | '.join(opts)}")

    while True:
        try:
            raw = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None, None

        if raw == "q":
            return None, None
        if raw == "f" and 0 in legal:
            return 0, 0
        if raw == "c" and 1 in legal:
            return 1, 0
        if raw == "a":
            return 7, int(hero.stack + hero.bet)   # all-in raise-to
        if raw.startswith("r"):
            parts = raw.split()
            if len(parts) < 2:
                print(col("  Usage: r <total chips>  e.g.  r 200", RED))
                continue
            try:
                amt = int(float(parts[1]))
                min_total = int(current_high + gs.min_raise)
                if amt < min_total:
                    print(col(f"  Must be at least ${min_total} (min raise)", RED))
                    continue
                if amt >= int(hero.stack + hero.bet):
                    return 7, int(hero.stack + hero.bet)   # cap at all-in
                return 7, amt
            except ValueError:
                print(col("  Invalid amount.", RED))
                continue
        print(col("  Unknown command.", RED))

# ── Session ───────────────────────────────────────────────────────────────────
def play_session():
    from engine_core  import GameState
    from poker_bot_api import SOTAPokerBot

    os.system("cls" if os.name == "nt" else "clear")
    print(col("""
  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
""", CYAN + BOLD))
    print(col("  Heads-Up vs NEXUS — Loading AI brain (~20s)...\n", DIM))

    nexus = SOTAPokerBot(device="cpu")
    gs     = GameState(n_players=2, small_blind=10, big_blind=20, training_mode=False)
    hero_seat = 0
    bot_seat  = 1

    hand_num  = 0
    session_pnl = 0.0

    print(col("\n  ✅ Ready!  You are Seat 0.  NEXUS is Seat 1.  Blinds: 10/20.\n", GREEN))
    input(col("  [Press Enter to start]", DIM))

    while True:
        hand_num += 1

        # Capture stacks BEFORE reset (blinds are posted inside reset)
        hero_chips_pre = gs.players[hero_seat].stack
        bot_chips_pre  = gs.players[bot_seat].stack

        gs.reset(reset_stacks=False)
        nexus.new_hand(hero_seat)

        hand_done = False

        while not hand_done:
            print_table(gs, hero_seat, hand_num, session_pnl)

            cp = gs.current_player

            # Skip if only one active player left (fold win)
            active = [p for p in gs.players if p.active]
            if len(active) <= 1:
                break

            # Skip players who are all-in (no action needed)
            hero = gs.players[hero_seat]
            bot  = gs.players[bot_seat]

            if cp == hero_seat:
                # ── Human turn ────────────────────────────────────────
                action_idx, amount = get_human_action(gs, hero_seat)
                if action_idx is None:
                    _summary(hand_num - 1, session_pnl)
                    return
                done = gs.step(action_idx, amount or 0)

            else:
                # ── NEXUS turn ────────────────────────────────────────
                # Feed last human action into tell/range model
                recent = [a for a in gs.history if a.player_id == hero_seat]
                if recent:
                    last = recent[-1]
                    nexus.observe_opponent_action(
                        opp_seat=hero_seat,
                        action=last.action_type,
                        amount=float(last.amount),
                        board=list(gs.board),
                        pot=float(gs.pot),
                        stage=int(gs.stage)
                    )

                print_table(gs, hero_seat, hand_num, session_pnl)
                print(col("\n  🤖 NEXUS is thinking...", DIM))

                try:
                    action_idx, amount, market = nexus.get_action(gs, bot_seat, use_search=True)
                except Exception as e:
                    print(col(f"  ⚠️  NEXUS error: {e}  → defaulting CHECK/CALL", RED))
                    action_idx, amount, market = 1, 0, {}

                # Render what NEXUS did
                aname = ACTION_NAMES.get(action_idx, f"?{action_idx}")
                if action_idx == 0:
                    act_display = col(f"  🤖 NEXUS → {aname}", RED + BOLD)
                elif action_idx == 1:
                    act_display = col(f"  🤖 NEXUS → {aname}", CYAN + BOLD)
                else:
                    act_display = col(f"  🤖 NEXUS → {aname}  (to ${amount})", YELLOW + BOLD)
                print(act_display)

                if market:
                    wp   = market.get('win_prob', 0.5)
                    eq   = market.get('equity_vs_rng', 0.5)
                    hn   = market.get('hand_name', 'Unknown')
                    expl = market.get('exploit_signal', 0.0)
                    print(col(f"  [Market] Equity: {eq:.1%} | GTO win%: {wp:.1%} | Hand: {hn} | Exploit: {expl:.2f}", DIM))

                done = gs.step(action_idx, amount or 0)
                input(col("  [Enter to continue]", DIM))

            if done:
                hand_done = True

        # ── Hand resolution ───────────────────────────────────────────
        payouts = gs.resolve_hand()

        # Add winnings back to stacks
        gs.players[hero_seat].stack += payouts[hero_seat]
        gs.players[bot_seat].stack  += payouts[bot_seat]

        hero_net = gs.players[hero_seat].stack - hero_chips_pre
        session_pnl += hero_net
        # NEXUS record_outcome expects (state, action, reward, next_state, done)
        dummy_vec = np.zeros(355, dtype=np.float32)
        nexus.record_outcome(dummy_vec, 1, hero_net / gs.bb_amt, dummy_vec, True)

        # ── Showdown display ──────────────────────────────────────────
        print_table(gs, hero_seat, hand_num, session_pnl)
        sep("═")

        hero_obj = gs.players[hero_seat]
        bot_obj  = gs.players[bot_seat]

        print(col("  ── SHOWDOWN / RESULT ──", YELLOW + BOLD))
        print(f"  Your hand   : {hand_str(hero_obj.hand)}")
        if bot_obj.hand and bot_obj.active:
            print(f"  NEXUS hand  : {hand_str(bot_obj.hand)}  {col('(revealed)', DIM)}")
        print(f"  Board       : {board_str(gs.board)}")
        sep()

        if payouts[hero_seat] > payouts[bot_seat]:
            print(col(f"  🏆 YOU WIN  +${payouts[hero_seat]:.0f} !!", GREEN + BOLD))
        elif payouts[bot_seat] > payouts[hero_seat]:
            print(col(f"  💀 NEXUS wins  +${payouts[bot_seat]:.0f}", RED + BOLD))
        else:
            print(col(f"  🤝 CHOP POT  (${payouts[hero_seat]:.0f} each)", YELLOW))

        print(f"  Net this hand: {col(f'{hero_net:+.0f} chips  ({hero_net/gs.bb_amt:+.1f} BB)', GREEN+BOLD if hero_net >= 0 else RED+BOLD)}")
        sep("═")

        again = input(col("\n  Play another hand? (y/n): ", CYAN)).strip().lower()
        if again != "y":
            _summary(hand_num, session_pnl)
            return

        # Auto-rebuy if broke
        for p in gs.players:
            if p.stack < gs.bb_amt * 10:
                print(col(f"  💳 Seat {p.seat_id} rebuys to $2000", DIM))
                p.stack = 2000.0


def _summary(hands, pnl):
    sep("═")
    print(col(f"\n  Session over — {hands} hand(s) played", BOLD))
    pnl_col = GREEN if pnl >= 0 else RED
    bb_pnl  = pnl / 20.0
    print(col(f"  Net P&L   : {pnl:+.0f} chips  ({bb_pnl:+.1f} BB)", pnl_col + BOLD))
    if hands > 0:
        print(col(f"  BB / hand : {bb_pnl/hands:+.2f}", pnl_col))
    sep("═")
    print()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    play_session()
