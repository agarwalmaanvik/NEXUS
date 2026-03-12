import React, { useState, useEffect, useMemo, useRef } from 'react';
import PokerTable from './components/PokerTable';

const SESSION_ID = Math.random().toString(36).substring(7);
// Use localhost if we are running locally, otherwise use the live DigitalOcean IP
const WS_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? "ws://localhost:8000"
    : "ws://137.184.41.164:8000";
const WS_URL = `${WS_BASE_URL}/ws/${SESSION_ID}`;
console.log("Connecting to NEXUS at:", WS_URL);

function App() {
    const [gameState, setGameState] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [logs, setLogs] = useState([]);
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

                    // Logic to extract new history items and add to logs
                    if (data.state?.history) {
                        const history = data.state.history;
                        const newEntries = history.map((h, i) => {
                            const pName = h.player_id === 1 ? 'Bot' : 'You';
                            let msg = "";
                            if (h.action === 0) msg = `${pName} folded`;
                            else if (h.action === 1) msg = h.amount === 0 ? `${pName} checked` : `${pName} called $${h.amount.toFixed(0)}`;
                            else if (h.action === 2) msg = `${pName} raised to $${h.amount.toFixed(0)}`;
                            else if (h.action === 3) msg = `${pName} posted blind $${h.amount.toFixed(0)}`;
                            return { id: `h-${i}-${data.state.stage}`, msg, pid: h.player_id };
                        });

                        // We uniquely add messages to avoid duplicates
                        setLogs(prev => {
                            const existing = new Set(prev.map(p => p.msg + p.id));
                            const uniqueNew = newEntries.filter(n => !existing.has(n.msg + n.id));
                            return [...prev, ...uniqueNew].slice(-50); // Keep last 50
                        });
                    }
                }
            };

            socket.onclose = () => {
                setIsConnected(false);
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
    const handleAction = (type, payload = {}) => {
        if (type === 'RESET_GAME') {
            sendAction('RESET_GAME');
            setLogs(prev => [...prev, { id: Date.now(), msg: "--- Game Reset ($2k) ---", pid: -1 }]);
        } else {
            sendAction('PLAYER_ACTION', payload);
        }
    };

    return (
        <div style={{ backgroundColor: '#0f172a', width: '100vw', height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
            {gameState ? (
                <PokerTable
                    state={gameState}
                    logs={logs}
                    onAction={(action, amount) => handleAction('PLAYER_ACTION', { action, amount })}
                    onReset={() => handleAction('RESET_GAME')}
                    onStart={handleStartHand}
                />
            ) : (
                <div className="flex flex-col items-center justify-center h-full">
                    <div className="glass" style={{ padding: '40px', borderRadius: '24px', textAlign: 'center', maxWidth: '400px' }}>
                        <div className="thinking-text" style={{ marginBottom: '16px' }}>Establishing Uplink</div>
                        <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px', color: 'white' }}>Connecting to NEXUS</h2>
                        <p style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '24px' }}>Initializing 6-Pillar Inference Engine...</p>
                        <div className="flex flex-col items-center gap-3">
                            <div className="flex flex-row items-center gap-2 px-3 py-1 rounded-full text-sm font-bold uppercase tracking-widest" style={{ backgroundColor: isConnected ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)', color: isConnected ? '#22c55e' : '#ef4444' }}>
                                <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`} />
                                {isConnected ? 'Signal Stable' : 'Searching for Host'}
                            </div>
                            {!isConnected && (
                                <div className="font-mono opacity-50" style={{ marginTop: '16px', fontSize: '9px', color: '#94a3b8' }}>
                                    Target: {WS_URL}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;
