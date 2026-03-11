"""
nexus_gui.py  —  Original Poker Pro layout with NEXUS AI engine
Run:  python nexus_gui.py

Controls  (your turn):
  FOLD / CHECK·CALL / RAISE buttons at the bottom
  Type a number in the raise box and press Enter
  NEXT HAND after each hand ends
"""

import os, sys, threading
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np

# ── NEXUS engine ──────────────────────────────────────────────────────────────
from engine_core   import GameState
from poker_bot_api import SOTAPokerBot

# ── Config (identical to original) ───────────────────────────────────────────
SCREEN_WIDTH  = 1000
SCREEN_HEIGHT = 700
BG_COLOR      = (34, 139, 34)
WHITE         = (255, 255, 255)
BLACK         = (0,   0,   0)
RED           = (220,  20,  60)
GOLD          = (255, 215,   0)
GRAY          = (200, 200, 200)
BUTTON_COLOR  = (50,  50,  50)
BUTTON_HOVER  = (100, 100, 100)

HERO_SEAT = 0
BOT_SEAT  = 1

# Card encoding helpers (NEXUS uses rank*4 + suit)
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_SUIT_SYMS = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}


def _card_display(card_int):
    """Return (rank_str, colour, suit_symbol) from engine card int."""
    rank_str = _RANKS[card_int // 4]
    suit_str = _SUITS[card_int % 4]
    col = RED if suit_str in 'dh' else BLACK
    return rank_str, col, _SUIT_SYMS[suit_str]


def draw_card(screen, card_int, x, y, font_big, font_small, hidden=False):
    """Draw one card at (x,y). Matches original draw_card signature exactly."""
    rect = pygame.Rect(x, y, 80, 120)
    if hidden:
        pygame.draw.rect(screen, (0, 50, 150), rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, rect, 2, border_radius=5)
        return
    pygame.draw.rect(screen, WHITE, rect, border_radius=5)
    pygame.draw.rect(screen, BLACK, rect, 2, border_radius=5)
    r, c, s = _card_display(card_int)
    screen.blit(font_small.render(r, True, c), (x + 5,  y + 5))
    screen.blit(font_small.render(s, True, c), (x + 5,  y + 20))
    screen.blit(font_big.render(s,   True, c), (x + 20, y + 35))


class Button:
    """Identical to original Button class."""
    def __init__(self, text, x, y, w, h, code):
        self.rect  = pygame.Rect(x, y, w, h)
        self.text  = text
        self.code  = code
        self.active = True
        self.hover  = False

    def draw(self, screen, font):
        bg    = (30, 30, 30)  if not self.active else \
                (BUTTON_HOVER if self.hover else BUTTON_COLOR)
        txt_c = GRAY if not self.active else WHITE
        pygame.draw.rect(screen, bg,    self.rect, border_radius=8)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=8)
        t = font.render(self.text, True, txt_c)
        screen.blit(t, (self.rect.centerx - t.get_width()  // 2,
                        self.rect.centery - t.get_height() // 2))


class PokerApp:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NEXUS Poker — AI Brain Active")
        self.clock   = pygame.time.Clock()
        self.f_ui    = pygame.font.SysFont("Arial", 24, bold=True)
        self.f_big   = pygame.font.SysFont("Segoe UI Symbol", 48)
        self.f_small = pygame.font.SysFont("Segoe UI Symbol", 20)

        # ── FIX 1: Lock protecting card snapshot lists read by draw() ─────────
        self._card_lock = threading.Lock()

        # ── FIX 3: Quit flag so bot thread exits cleanly ──────────────────────
        self._quit = False

        # ── Loading screen ────────────────────────────────────────────────────
        self.log_msg = ["⏳  Loading NEXUS — please wait (~20 s)…"]
        self._draw_loading()

        # ── Load NEXUS engine ─────────────────────────────────────────────────
        self.nexus = SOTAPokerBot(device="cpu")
        self.gs    = GameState(n_players=2, small_blind=10, big_blind=20,
                               training_mode=False)

        # Session info
        self.session_pnl  = 0.0
        self.hero_pre     = 2000.0   # stack before current hand starts
        self.market       = {}
        self._bot_busy    = False    # True while NEXUS thread is running

        # ── Buttons (same positions as original) ──────────────────────────────
        y = SCREEN_HEIGHT - 80
        self.btns = [
            Button("FOLD",        50,  y, 100, 50, 'f'),
            Button("CHECK/CALL", 170,  y, 180, 50, 'c'),
            Button("RAISE",      370,  y, 100, 50, 'r'),
        ]
        self.next_btn  = Button("NEXT HAND", 800, 320, 150, 60, 'n')
        self.input_mode = False
        self.input_txt  = ""

        self.log_msg = [
            "NEXUS loaded ✅",
            f"RAG: {len(self.nexus._rag):,} situations",
            "Click NEXT HAND to begin.",
        ]
        self.turn = "WAITING"
        self._set_btns_active(False)
        self.next_btn.active = True

        # Current hand cards (list of engine card ints) — protected by _card_lock
        self.p_hand: list[int] = []
        self.b_hand: list[int] = []
        self.board:  list[int] = []

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _draw_loading(self):
        self.screen.fill((20, 60, 20))
        for i, m in enumerate(self.log_msg):
            s = self.f_ui.render(m, True, GOLD)
            self.screen.blit(s, (SCREEN_WIDTH // 2 - s.get_width() // 2,
                                  SCREEN_HEIGHT // 2 - 20 + i * 35))
        pygame.display.flip()
        pygame.event.pump()

    def log(self, t):
        self.log_msg.append(t)
        if len(self.log_msg) > 8:
            self.log_msg.pop(0)
            
        try:
            with open("game_log.txt", "a", encoding="utf-8") as f:
                f.write(t + "\n")
        except:
            pass

    def _set_btns_active(self, value: bool):
        for b in self.btns:
            b.active = value

    def _sync_cards_from_gs(self):
        """Pull current card lists from GameState into local attrs (thread-safe)."""
        # Take a snapshot of gs lists BEFORE acquiring lock so we hold
        # the lock for as little time as possible.
        p_snap = list(self.gs.players[HERO_SEAT].hand)
        b_snap = list(self.gs.players[BOT_SEAT].hand)
        bd_snap = list(self.gs.board)           # ← copy while engine is idle
        with self._card_lock:
            self.p_hand = p_snap
            self.b_hand = b_snap
            self.board  = bd_snap

    # ── Hand lifecycle ────────────────────────────────────────────────────────
    def start_hand(self):
        self.next_btn.active = False
        self._set_btns_active(False)
        self.market = {}

        # Stacks before blinds are posted
        self.hero_pre = self.gs.players[HERO_SEAT].stack
        self.bot_pre  = self.gs.players[BOT_SEAT].stack

        self.gs.reset(reset_stacks=False)
        self.nexus.new_hand(HERO_SEAT)
        self._sync_cards_from_gs()
        self.log("─── New Hand ───")
        self._advance()

    def _advance(self):
        """Check game state and set turn correctly."""
        self._sync_cards_from_gs()

        active = [p for p in self.gs.players if p.active]
        if len(active) <= 1:
            self._end_hand()
            return

        # ── All-in runout check ───────────────────────────────────────────────
        # If every still-active player is all-in, nobody can make decisions.
        # Auto-deal the remaining community cards and go straight to showdown.
        # Also covers the case where _find_next_active() landed on an all-in
        # player because it couldn't find anyone free to act.
        all_allin = all(p.all_in for p in active)
        cp_is_allin = self.gs.players[self.gs.current_player].all_in
        if all_allin or cp_is_allin:
            self._run_allin_runout()
            return

        cp = self.gs.current_player
        if cp == HERO_SEAT:
            self.turn = "PLAYER"
            self._refresh_buttons()
        else:
            self.turn = "BOT"
            self._set_btns_active(False)
            self._bot_busy = True
            threading.Thread(target=self._run_bot, daemon=True).start()

    def _run_allin_runout(self):
        """All active players are all-in: deal remaining board cards and end hand."""
        gs = self.gs
        # Deal each missing street directly — no step() call, no extra actions recorded.
        # Expected board length: stage0→0, stage1→3, stage2→4, stage3→5
        while gs.stage < 3:
            gs.stage += 1
            target_len = 3 if gs.stage == 1 else (gs.stage + 2)  # stage2→4, stage3→5
            needed = target_len - len(gs.board)
            if needed > 0:
                gs._deal_community(needed)
            self._sync_cards_from_gs()
            pygame.time.wait(500)   # brief pause between each street reveal
        self._end_hand()

    def _refresh_buttons(self):
        hero = self.gs.players[HERO_SEAT]
        legal = list(self.gs.legal_moves)
        current_high = max(p.bet for p in self.gs.players)
        to_call = max(0, current_high - hero.bet)

        for b in self.btns:
            if b.code == 'f': b.active = (0 in legal)
            if b.code == 'c':
                b.active = (1 in legal)
                b.text = "CHECK" if to_call == 0 else f"CALL {to_call:.0f}"
            if b.code == 'r': b.active = any(m >= 2 for m in legal)

    # ── Bot turn (background thread) ──────────────────────────────────────────
    def _run_bot(self):
        try:
            if self._quit:          # FIX 3: honour quit flag before doing work
                return

            recent = [a for a in self.gs.history if a.player_id == HERO_SEAT]
            if recent:
                last = recent[-1]
                self.nexus.observe_opponent_action(
                    opp_seat=HERO_SEAT,
                    action=last.action_type,
                    amount=float(last.amount),
                    board=list(self.gs.board),
                    pot=float(self.gs.pot),
                    stage=int(self.gs.stage),
                )

            # FIX 3: use_search=False avoids potentially unbounded search time
            act_idx, amount, market = self.nexus.get_action(
                self.gs, BOT_SEAT, use_search=False)
            self.market = market or {}
        except Exception as e:
            self.log(f"⚠ NEXUS err: {e}")
            act_idx, amount = 1, 0

        if self._quit:              # FIX 3: don't touch state if quitting
            return

        names = {0:"FOLD", 1:"CHECK/CALL",
                 2:"RAISE 0.5x", 3:"RAISE 1x", 4:"RAISE 2x",
                 5:"RAISE 3x", 6:"ALL-IN", 7:"RAISE"}
        aname = names.get(act_idx, f"?{act_idx}")
        amt_str = f" → ${amount}" if act_idx >= 2 else ""
        self.log(f"BOT: {aname}{amt_str}")

        done = self.gs.step(act_idx, amount or 0)

        # ── FIX 1: sync cards after engine step, BEFORE clearing _bot_busy ──
        self._sync_cards_from_gs()
        self._bot_busy = False

        pygame.time.wait(400)   # brief pause so player can read the action

        if done:
            self._end_hand()
        else:
            self._advance()

    # ── Player action ─────────────────────────────────────────────────────────
    def do_player(self, code, raise_amount=0):
        if self.turn != "PLAYER":
            return
        self._set_btns_active(False)
        self.turn = "ACTING"

        hero  = self.gs.players[HERO_SEAT]
        legal = list(self.gs.legal_moves)

        done = None   # FIX 2: track whether we actually executed an action

        if code == 'f' and 0 in legal:
            self.log("You: FOLD")
            done = self.gs.step(0, 0)
        elif code == 'c' and 1 in legal:
            current_high = max(p.bet for p in self.gs.players)
            to_call = max(0, current_high - hero.bet)
            self.log("You: CHECK" if to_call == 0 else f"You: CALL {to_call:.0f}")
            done = self.gs.step(1, 0)
        elif code == 'r':
            # FIX 2: validate raise amount; fall back to check/call if invalid
            try:
                amt = int(raise_amount)
            except (ValueError, TypeError):
                amt = 0
            if amt > 0:
                amt = min(amt, int(hero.stack + hero.bet))
                self.log(f"You: RAISE to {amt}")
                done = self.gs.step(7, amt)
            else:
                # Invalid raise → treat as check/call if legal
                if 1 in legal:
                    current_high = max(p.bet for p in self.gs.players)
                    to_call = max(0, current_high - hero.bet)
                    self.log("You: CHECK" if to_call == 0 else f"You: CALL {to_call:.0f}")
                    done = self.gs.step(1, 0)

        # FIX 2: if no valid action was executed, restore PLAYER turn
        if done is None:
            self.turn = "PLAYER"
            self._refresh_buttons()
            return

        if done:
            self._end_hand()
        else:
            self._advance()

    # ── Hand end ──────────────────────────────────────────────────────────────
    def _end_hand(self):
        self.turn = "WAITING"
        self._set_btns_active(False)
        self._sync_cards_from_gs()
        
        try:
            with open("game_log.txt", "a", encoding="utf-8") as f:
                def fc(cards): return " ".join([f"{_card_display(c)[0]}{_card_display(c)[2]}" for c in cards])
                f.write(f"\n--- SHOWDOWN ---\n")
                f.write(f"Board: {fc(self.board)}\n")
                f.write(f"Hero:  {fc(self.p_hand)}\n")
                f.write(f"Bot:   {fc(self.b_hand)}\n")
                m = getattr(self, 'market', {})
                if m: f.write(f"Market: {m}\n")
                f.write("\n")
        except Exception as e:
            pass

        payouts = self.gs.resolve_hand()

        for i, p in enumerate(self.gs.players):
            p.stack += payouts[i]

        hero_net = self.gs.players[HERO_SEAT].stack - self.hero_pre
        self.session_pnl += hero_net
        # NEXUS record_outcome expects (state, action, reward, next_state, done)
        # For the interactive GUI, we pass placeholders to avoid crashing legacy DDQN pillar.
        dummy_vec = np.zeros(355, dtype=np.float32)
        self.nexus.record_outcome(dummy_vec, 1, hero_net / self.gs.bb_amt, dummy_vec, True)

        hp, bp = payouts[HERO_SEAT], payouts[BOT_SEAT]
        if hp > bp:   self.log(f"You win! +${hp:.0f}  (net {hero_net:+.0f})")
        elif bp > hp: self.log(f"BOT wins! +${bp:.0f}  (net {hero_net:+.0f})")
        else:         self.log(f"SPLIT POT  (net {hero_net:+.0f})")

        # Auto-rebuy
        for p in self.gs.players:
            if p.stack < 200:
                p.stack = 2000.0
                self.log(f"Seat {p.seat_id} rebuys to $2000")

        self.next_btn.active = True

    # ── Draw (same layout as original) ────────────────────────────────────────
    def draw(self):
        gs  = self.gs
        hero = gs.players[HERO_SEAT]
        bot  = gs.players[BOT_SEAT]

        self.screen.fill(BG_COLOR)

        # Board felt + pot
        pygame.draw.rect(self.screen, (20, 80, 20),
                         (200, 250, 600, 200), border_radius=20)
        self.screen.blit(
            self.f_ui.render(f"POT: {gs.pot:.0f}", True, GOLD), (450, 260))

        # FIX 1: take a snapshot under the lock so the draw loop sees a
        # consistent set of cards even if the bot thread is updating them.
        with self._card_lock:
            board_snap  = list(self.board)
            p_hand_snap = list(self.p_hand)
            b_hand_snap = list(self.b_hand)

        # Community cards (capped at 5 for safety)
        for i, c in enumerate(board_snap[:5]):
            draw_card(self.screen, c, 260 + i*95, 300,
                      self.f_big, self.f_small)

        # Bot info + cards
        self.screen.blit(
            self.f_ui.render(f"BOT: {bot.stack:.0f}  |  P&L: {self.session_pnl:+.0f}",
                             True, WHITE), (370, 50))
        show_b = self.turn in ("WAITING",)
        for i, c in enumerate(b_hand_snap[:2]):
            draw_card(self.screen, c, 400 + i*90, 80,
                      self.f_big, self.f_small, hidden=not show_b)

        # Market data (small, top-right)
        if self.market:
            wp  = self.market.get('win_prob', 0.5)
            eq  = self.market.get('equity_vs_rng', 0.5)
            hn  = self.market.get('hand_name', 'Unknown')
            expl = self.market.get('exploit_signal', 0.0)
            
            wc = (80, 220, 80) if eq > 0.5 else RED
            lines = [
                f"Equity: {eq:.1%}",
                f"GTO WP: {wp:.1%}",
                f"Hand: {hn}",
                f"Exploit: {expl:.2f}"
            ]
            for i, l in enumerate(lines):
                ms = self.f_small.render(l, True, wc if "Equity" in l else GOLD)
                self.screen.blit(ms, (780, 200 + i*22))

        # Hero info + cards
        self.screen.blit(
            self.f_ui.render(f"YOU: {hero.stack:.0f}", True, WHITE), (450, 630))
        for i, c in enumerate(p_hand_snap[:2]):
            draw_card(self.screen, c, 400 + i*90, 500,
                      self.f_big, self.f_small)

        # Buttons
        if self.turn == "PLAYER":
            for b in self.btns:
                b.draw(self.screen, self.f_ui)
        elif self.turn == "WAITING":
            self.next_btn.active = True
            self.next_btn.draw(self.screen, self.f_ui)

        # Bot-thinking indicator
        if self.turn == "BOT" or self._bot_busy:
            t = self.f_ui.render("🤖  NEXUS thinking…", True, GOLD)
            self.screen.blit(t, (370, 460))

        # Raise input overlay
        if self.input_mode:
            box = pygame.Rect(350, 300, 300, 100)
            pygame.draw.rect(self.screen, (40, 40, 40), box)
            pygame.draw.rect(self.screen, WHITE,       box, 2)
            self.screen.blit(
                self.f_ui.render(f"Raise: {self.input_txt}", True, GOLD),
                (400, 340))

        # Log (same position as original)
        for i, m in enumerate(self.log_msg):
            self.screen.blit(
                self.f_small.render(m, True, (200, 255, 200)),
                (20, 20 + i*20))

        pygame.display.flip()

    # ── Main loop (mirrors original structure) ────────────────────────────────
    def run(self):
        while True:
            mp = pygame.mouse.get_pos()

            for e in pygame.event.get():
                # FIX 3: QUIT is always handled first — never blocked by bot thread
                if e.type == pygame.QUIT:
                    self._quit = True
                    pygame.quit()
                    sys.exit()

                # Raise input mode
                if self.input_mode:
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_RETURN:
                            try:
                                amt = int(self.input_txt)
                            except ValueError:
                                amt = 0
                            self.input_mode = False
                            self.input_txt  = ""
                            # FIX 2: always call do_player; it handles amt=0 gracefully
                            self.do_player('r', amt)
                        elif e.key == pygame.K_ESCAPE:
                            self.input_mode = False
                            self.input_txt  = ""
                            # FIX 2: restore buttons so player can still act
                            if self.turn in ("PLAYER", "ACTING"):
                                self.turn = "PLAYER"
                                self._refresh_buttons()
                        elif e.key == pygame.K_BACKSPACE:
                            self.input_txt = self.input_txt[:-1]
                        elif e.unicode.isdigit():
                            self.input_txt += e.unicode
                    continue

                if e.type == pygame.MOUSEBUTTONDOWN:
                    if self.turn == "PLAYER":
                        for b in self.btns:
                            if b.rect.collidepoint(mp) and b.active:
                                if b.code == 'r':
                                    # Pre-fill with min-raise
                                    hero = self.gs.players[HERO_SEAT]
                                    cur_high = max(p.bet for p in self.gs.players)
                                    self.input_txt = str(int(cur_high + self.gs.min_raise))
                                    self.input_mode = True
                                else:
                                    self.do_player(b.code)
                    if self.turn == "WAITING" and self.next_btn.rect.collidepoint(mp):
                        self.start_hand()

                # Hover effects
                for b in self.btns + [self.next_btn]:
                    b.hover = b.rect.collidepoint(mp) and b.active

            self.draw()
            self.clock.tick(30)


if __name__ == "__main__":
    PokerApp().run()
