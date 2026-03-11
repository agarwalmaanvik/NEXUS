"""
advisor_gui.py — NEXUS Live Advisor (v2 Stealth HUD)

Objective: 
  A minimal, numpad-driven interface for live play with friends.
  Designed for speed and "invisibility" (low-profile layout).

Hotkeys:
  Numpad 0-6: Fold, Call/Check, Raise1, Raise2, Raise3, Raise4, Shove
  Backspace:   Undo last action (reverts GameState)
  Enter:       Confirm action/Advance stage
  Esc:         Exit
"""

import sys
import os
import pygame
import numpy as np
from engine_core import GameState, Action
from poker_bot_api import SOTAPokerBot

# --- Design Tokens (Stealth Mode) ---
SCREEN_SIZE = (400, 500)
BG_COLOR    = (15, 15, 15)       # Nearly black
TEXT_COLOR  = (180, 180, 180)    # Soft grey (low contrast)
ACCENT_BLUE = (60, 120, 240)
ACCENT_GOLD = (240, 180, 60)
CRITICAL_RED = (200, 50, 50)

class AdvisorGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("NEXUS Advisor")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Outfit", 18)
        self.font_mid   = pygame.font.SysFont("Outfit", 24)
        self.font_big   = pygame.font.SysFont("Outfit", 32, bold=True)

        # 1. State Init
        self.n_players = 6
        self.gs = GameState(n_players=self.n_players)
        self.gs.reset()
        
        # 2. Bot Init (Advisor uses current checkpoint)
        self.bot = SOTAPokerBot(device="cpu") 
        self.hero_seat = 0
        
        self.last_analysis = {}
        self.input_buffer = ""
        self.running = True

    def run(self):
        while self.running:
            self._handle_events()
            self._update_bot()
            self._draw()
            self.clock.tick(30)
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                # Numpad Mapping
                if pygame.K_KP0 <= event.key <= pygame.K_KP6:
                    action_idx = event.key - pygame.K_KP0
                    self.gs.observe_action(self.gs.current_player, action_idx)
                
                elif event.key == pygame.K_BACKSPACE:
                    self.gs.undo()
                
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def _update_bot(self):
        # Only query the bot if it's hero's turn
        if self.gs.current_player == self.hero_seat and not self.gs.stage == 4:
            # We don't take the action automatically (user must confirm)
            # but we show the recommendation.
            _, _, analysis = self.bot.get_action(self.gs, self.hero_seat, use_search=True)
            self.last_analysis = analysis

    def _draw(self):
        self.screen.fill(BG_COLOR)
        y = 20
        
        # 1. HUD Header
        title = self.font_big.render("NEXUS HUD", True, ACCENT_BLUE)
        self.screen.blit(title, (20, y))
        y += 50

        # 2. Game State Info
        stage_names = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
        stage_text = f"Stage: {stage_names[self.gs.stage]}"
        pot_text   = f"Pot: {self.gs.pot:.0f}"
        
        r_stage = self.font_mid.render(stage_text, True, TEXT_COLOR)
        r_pot   = self.font_mid.render(pot_text, True, ACCENT_GOLD)
        self.screen.blit(r_stage, (20, y))
        self.screen.blit(r_pot, (200, y))
        y += 40

        # 3. Decision HUD (The Advisor)
        if self.last_analysis:
            # Equity & Strength
            eq = self.last_analysis.get('equity_vs_rng', 0.0)
            hs = self.last_analysis.get('hand_name', 'Unknown')
            pct = self.last_analysis.get('hand_pct', 0.0)
            
            self._draw_stat("Equity", f"{eq:.1%}", 20, y)
            self._draw_stat("Strength", f"{hs} ({pct:.0%})", 20, y + 30)
            y += 80

            # EV Bars
            y += 10
            evs = self.last_analysis.get('ev_by_action', {})
            actions = ["Fold", "Call", "R-33", "R-50", "R-75", "R-100", "SHOVE"]
            for i, label in enumerate(actions):
                ev = evs.get(i, 0.0)
                color = (100, 200, 100) if ev >= 0 else (200, 100, 100)
                
                # Draw Bar
                bar_w = min(abs(ev) * 2, 150)
                pygame.draw.rect(self.screen, color, (100, y, bar_w, 15))
                
                # Draw Label
                txt = self.font_small.render(f"{label}:", True, TEXT_COLOR)
                self.screen.blit(txt, (20, y - 5))
                
                # Draw EV Value
                val = self.font_small.render(f"{ev:+.1f}", True, color)
                self.screen.blit(val, (260, y - 5))
                y += 25

        # 4. Footer (Controls)
        footer_y = SCREEN_SIZE[1] - 40
        msg = "[Numpad 0-6]: Log  [BS]: Undo"
        r_msg = self.font_small.render(msg, True, (80, 80, 80))
        self.screen.blit(r_msg, (20, footer_y))

        pygame.display.flip()

    def _draw_stat(self, label, value, x, y):
        l_txt = self.font_mid.render(f"{label}:", True, TEXT_COLOR)
        v_txt = self.font_mid.render(value, True, ACCENT_GOLD if "Equity" in label else ACCENT_BLUE)
        self.screen.blit(l_txt, (x, y))
        self.screen.blit(v_txt, (x + 100, y))

if __name__ == "__main__":
    AdvisorGUI().run()
