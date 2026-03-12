import os
import json
import asyncio
import logging
import threading
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# NEXUS engine imports
from engine_core import GameState, Action
from poker_bot_api import SOTAPokerBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus-poker-server")

app = FastAPI(title="NEXUS Poker Server")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the bot once at startup
# This ensures weights are loaded once and shared across sessions
logger.info("Loading NEXUS AI Engine (this may take ~20s)...")
nexus_bot = SOTAPokerBot(device="cpu")
logger.info("NEXUS Engine Loaded.")

HERO_SEAT = 0
BOT_SEAT = 1

class GameSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.gs = GameState(n_players=2, small_blind=10, big_blind=20, training_mode=False)
        self.nexus = nexus_bot
        self.is_active = False
        self.bot_thinking = False
        self.market = {}
        self.hero_pre = 2000.0

    def start_new_hand(self):
        self.market = {}
        self.hero_pre = self.gs.players[HERO_SEAT].stack
        self.gs.reset(reset_stacks=False)
        self.nexus.new_hand(HERO_SEAT)
        self.is_active = True
        return self.get_client_state()

    def get_client_state(self, reveal_bot_cards: bool = False):
        """
        Returns a JSON-serializable dict of the current state for the client.
        Note: We filter out the bot's cards unless reveal_bot_cards is True (showdown).
        """
        hero = self.gs.players[HERO_SEAT]
        bot = self.gs.players[BOT_SEAT]
        
        # Check if hand is actually over to reveal cards
        active_count = len([p for p in self.gs.players if p.active])
        hand_over = active_count <= 1 or (self.gs.stage == 3 and self.gs.current_player == -1)
        
        should_reveal = reveal_bot_cards or hand_over

        state = {
            "pot": float(self.gs.pot),
            "board": [int(c) for c in self.gs.board],
            "stage": int(self.gs.stage),
            "current_player": int(self.gs.current_player),
            "legal_moves": list(self.gs.legal_moves),
            "min_raise": float(self.gs.min_raise),
            "hero": {
                "stack": float(hero.stack),
                "bet": float(hero.bet),
                "hand": [int(c) for c in hero.hand],
                "active": bool(hero.active),
                "all_in": bool(hero.all_in)
            },
            "bot": {
                "stack": float(bot.stack),
                "bet": float(bot.bet),
                "hand": [int(c) for c in bot.hand] if should_reveal else [], 
                "active": bool(bot.active),
                "all_in": bool(bot.all_in)
            },
            "history": [
                {"player_id": int(a.player_id), "action": int(a.action_type), "amount": float(a.amount), "stage": int(a.stage)}
                for a in self.gs.history
            ],
            "is_hand_over": hand_over,
            "bot_thinking": self.bot_thinking,
            "market": self.market if should_reveal else {} # Only show market data to owner (or at end)
        }
        return state

    async def run_bot_turn(self, websocket: WebSocket):
        self.bot_thinking = True
        await websocket.send_json({"type": "STATE_UPDATE", "state": self.get_client_state()})

        # Run bot inference in a separate thread to keep the event loop responsive
        def bot_inference():
            try:
                # Observe hero's last action
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
                
                # Get bot action
                act_idx, amount, market = self.nexus.get_action(self.gs, BOT_SEAT, use_search=False)
                return act_idx, amount, market
            except Exception as e:
                logger.error(f"Bot inference error: {e}")
                return 1, 0, {}

        loop = asyncio.get_event_loop()
        act_idx, amount, market = await loop.run_in_executor(None, bot_inference)
        
        self.market = market or {}
        self.bot_thinking = False
        
        done = self.gs.step(act_idx, amount or 0)
        
        if done:
            await self.end_hand(websocket)
        else:
            await self.advance_game(websocket)

    async def advance_game(self, websocket: WebSocket):
        active_players = [p for p in self.gs.players if p.active]
        if len(active_players) <= 1:
            await self.end_hand(websocket)
            return

        # Handle all-in runouts
        all_allin = all(p.all_in for p in active_players)
        cp_is_allin = self.gs.players[self.gs.current_player].all_in
        if all_allin or cp_is_allin:
            while self.gs.stage < 3:
                self.gs.stage += 1
                target_len = 3 if self.gs.stage == 1 else (self.gs.stage + 2)
                needed = target_len - len(self.gs.board)
                if needed > 0:
                    self.gs._deal_community(needed)
                await websocket.send_json({"type": "STATE_UPDATE", "state": self.get_client_state()})
                await asyncio.sleep(1) # Slow down for visualization
            await self.end_hand(websocket)
            return

        # If it's the bot's turn, trigger bot logic
        if self.gs.current_player == BOT_SEAT:
            await self.run_bot_turn(websocket)
        else:
            await websocket.send_json({"type": "STATE_UPDATE", "state": self.get_client_state()})

    async def end_hand(self, websocket: WebSocket):
        payouts = self.gs.resolve_hand()
        for i, p in enumerate(self.gs.players):
            p.stack += payouts[i]
        
        # Record outcome for DDQN (placeholder logic)
        hero_net = self.gs.players[HERO_SEAT].stack - self.hero_pre
        dummy_vec = np.zeros(355, dtype=np.float32)
        self.nexus.record_outcome(dummy_vec, 1, hero_net / self.gs.bb_amt, dummy_vec, True)

        # Final state update with revealed cards
        await websocket.send_json({"type": "HAND_OVER", "state": self.get_client_state(reveal_bot_cards=True), "payouts": payouts})

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, GameSession] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        if session_id not in self.sessions:
            self.sessions[session_id] = GameSession(session_id)
        
        # Send initial state
        await websocket.send_json({"type": "INIT", "state": self.sessions[session_id].get_client_state()})

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def handle_message(self, session_id: str, data: dict):
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        websocket = self.active_connections[session_id]
        
        msg_type = data.get("type")
        
        if msg_type == "START_HAND":
            state = session.start_new_hand()
            await session.advance_game(websocket)
            
        elif msg_type == "PLAYER_ACTION":
            action_idx = data.get("action")
            amount = data.get("amount", 0)
            
            if session.gs.current_player == HERO_SEAT:
                done = session.gs.step(action_idx, amount)
                if done:
                    await session.end_hand(websocket)
                else:
                    await session.advance_game(websocket)

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_message(session_id, data)
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"Session {session_id} disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
