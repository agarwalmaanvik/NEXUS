import React, { useState, useEffect, useMemo, useRef } from 'react';
import PokerTable from './components/PokerTable';

const SESSION_ID = Math.random().toString(36).substring(7);
const WS_URL = `ws://localhost:8000/ws/${SESSION_ID}`;

function App() {
    const [gameState, setGameState] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const socketRef = useRef(null);

    useEffect(() => {
        const connect = () => {
            const socket = new WebSocket(WS_URL);

            socket.onopen = () => {
                setIsConnected(true);
                console.log('Connected to NEXUS Server');
            };

            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Received:', data);
                if (data.type === 'INIT' || data.type === 'STATE_UPDATE' || data.type === 'HAND_OVER') {
                    setGameState(data.state);
                }
                setLastMessage(data);
            };

            socket.onclose = () => {
                setIsConnected(false);
                console.log('Disconnected from NEXUS Server');
                // Auto-reconnect after 3 seconds
                setTimeout(connect, 3000);
            };

            socketRef.current = socket;
        };

        connect();

        return () => {
            if (socketRef.current) socketRef.current.close();
        };
    }, []);

    const sendAction = (type, payload = {}) => {
        if (socketRef.current && isConnected) {
            socketRef.current.send(JSON.stringify({ type, ...payload }));
        }
    };

    const handleStartHand = () => sendAction('START_HAND');
    const handlePlayerAction = (action, amount = 0) => sendAction('PLAYER_ACTION', { action, amount });

    return (
        <div className="min-h-screen flex flex-col items-center justify-center p-4">
            <header className="absolute top-8 left-8 flex items-center gap-4">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <h1 className="text-2xl font-bold tracking-tight text-white/90">NEXUS <span className="text-blue-500">AI ARENA</span></h1>
            </header>

            {gameState ? (
                <PokerTable
                    state={gameState}
                    onAction={handlePlayerAction}
                    onStart={handleStartHand}
                />
            ) : (
                <div className="glass p-8 rounded-2xl text-center max-w-md">
                    <h2 className="text-xl font-semibold mb-2">Connecting to Engine...</h2>
                    <p className="text-text-muted">Ensure the NEXUS server is running at localhost:8000</p>
                </div>
            )}

            <footer className="absolute bottom-4 text-xs text-text-muted">
                NEXUS 6-Pillar Inference Engine v2.0
            </footer>
        </div>
    );
}

export default App;
