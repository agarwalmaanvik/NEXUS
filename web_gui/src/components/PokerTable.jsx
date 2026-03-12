import React, { useState, useEffect, useRef } from 'react';
import Card from './Card';

const PokerTable = ({ state, logs, onAction, onStart, onReset }) => {
    const [sliderValue, setSliderValue] = useState(0);
    const scrollEndRef = useRef(null);

    if (!state) return (
        <div className="flex flex-col items-center justify-center h-full">
            <div className="thinking-text">Syncing with NEXUS Engine...</div>
        </div>
    );

    const { pot, board, hero, bot, current_player, legal_moves, min_raise, is_hand_over, bot_thinking, history, stage } = state;
    const isHeroTurn = current_player === 0 && !is_hand_over && !bot_thinking;

    // Game is in "Lobby" if no cards are dealt yet
    const isLobby = (hero?.hand?.length || 0) === 0 && (board?.length || 0) === 0 && (pot || 0) === 0;
    const showNextBtn = is_hand_over || isLobby;

    useEffect(() => {
        if (isHeroTurn) {
            setSliderValue(min_raise || 20);
        }
    }, [isHeroTurn, min_raise]);

    useEffect(() => {
        if (scrollEndRef.current) {
            scrollEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    const handleAction = (actionIdx, amount = 0) => {
        if (!isHeroTurn) return; // UI Safety
        const val = parseInt(amount) || 0;
        onAction(actionIdx, val);
    };

    const getActionMessage = (pid) => {
        if (!history || history.length === 0) return null;
        const relevant = history.filter(h => h.player_id === pid && h.action !== 3);
        const last = relevant[relevant.length - 1];
        if (!last || last.stage !== stage) return null;

        const pName = pid === 1 ? 'Bot' : 'You';
        if (last.action === 1 && last.amount === 0) return `${pName} Checked`;
        if (last.action === 1) return `${pName} Called $${last.amount.toFixed(0)}`;
        if (last.action === 2) return `${pName} Raised $${last.amount.toFixed(0)}`;
        if (last.action === 0) return `${pName} Folded`;
        return null;
    };

    const botMessage = getActionMessage(1);
    const heroMessage = getActionMessage(0);

    return (
        <div className="felt-bg">
            {/* Header / Stats Overlay */}
            <div className="absolute flex flex-col gap-1" style={{ top: '16px', left: '24px', zIndex: 10, pointerEvents: 'none' }}>
                <div className="text-white font-mono text-sm uppercase font-bold tracking-widest" style={{ opacity: 0.8 }}>NEXUS</div>
            </div>

            {/* Sidebar Log Panel */}
            <div className="absolute glass flex flex-col" style={{ top: '16px', right: '16px', width: '220px', height: '200px', zIndex: 20, borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', overflow: 'hidden' }}>
                <div className="text-[10px] font-bold uppercase tracking-widest p-2 border-b border-white/10 text-gold bg-black/20">Game Feed</div>
                <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2" style={{ fontSize: '11px' }}>
                    {logs.length === 0 ? (
                        <div className="text-white/20 italic">Waiting for activity...</div>
                    ) : logs.map((l, i) => (
                        <div key={l.id || i} style={{ color: l.pid === 1 ? '#fbbf24' : l.pid === -1 ? '#60a5fa' : '#fff', opacity: 0.8 }}>
                            <span style={{ opacity: 0.4, marginRight: '6px' }}>•</span>
                            {l.msg}
                        </div>
                    ))}
                    <div ref={scrollEndRef} />
                </div>
            </div>

            {/* Internal 1000x700 Layout */}
            <div className="relative w-full h-full" style={{ padding: '40px' }}>

                {/* Bot Section (Top) */}
                <div className="absolute flex flex-col items-center gap-2" style={{ top: '40px', left: '50%', transform: 'translateX(-50%)' }}>
                    <div className="text-white font-bold text-lg uppercase">NEXUS Bot: ${bot?.stack?.toFixed(0) || 0}</div>
                    <div className="flex flex-row gap-2">
                        {bot?.hand?.length > 0 ? (
                            bot.hand.map((c, i) => <Card key={i} cardInt={c} />)
                        ) : (
                            <>
                                <Card hidden />
                                <Card hidden />
                            </>
                        )}
                    </div>
                    {bot_thinking ? (
                        <div className="text-gold font-bold animate-pulse text-xs tracking-widest mt-1">NEXUS THINKING...</div>
                    ) : botMessage && (
                        <div className="text-white/60 font-bold text-[11px] uppercase tracking-wider mt-1 bg-black/40 px-3 py-1 rounded-full">{botMessage}</div>
                    )}
                </div>

                {/* Pot Center Area */}
                <div className="absolute flex flex-col items-center gap-4" style={{ top: '190px', left: '50%', transform: 'translateX(-50%)' }}>
                    {isLobby ? (
                        <div className="text-white font-black text-4xl uppercase opacity-20" style={{ marginTop: '80px', letterSpacing: '0.2em', fontStyle: 'italic' }}>Waiting for Deal</div>
                    ) : (
                        <>
                            <div className="glass" style={{ padding: '8px 32px', borderRadius: '8px', position: 'relative', zIndex: 10 }}>
                                <span className="text-gold font-black text-2xl font-mono">POT: ${pot?.toFixed(0) || 0}</span>
                            </div>
                            <div className="flex flex-row gap-3" style={{ height: '120px' }}>
                                {board?.map((c, i) => <Card key={i} cardInt={c} />)}
                                {[...Array(Math.max(0, 5 - (board?.length || 0)))].map((_, i) => (
                                    <div key={i} className="card" style={{ backgroundColor: 'rgba(0,0,0,0.1)', border: '2px dashed rgba(255,255,255,0.1)', boxShadow: 'none' }} />
                                ))}
                            </div>
                        </>
                    )}
                </div>

                {/* Next Hand / Reset Buttons */}
                {showNextBtn && (
                    <div className="absolute" style={{ top: '45%', right: '40px', transform: 'translateY(-50%)', display: 'flex', flexDirection: 'column', gap: '16px', zIndex: 100 }}>
                        {!isHeroTurn && !bot_thinking && (
                            <button
                                className="bg-gold text-black font-black hover:scale-105 transition-transform"
                                style={{ padding: '20px 40px', borderRadius: '12px', border: '3px solid rgba(255,255,255,0.3)', cursor: 'pointer', fontSize: '18px', textTransform: 'uppercase', boxShadow: '0 0 30px rgba(251, 191, 36, 0.3)' }}
                                onClick={onStart}
                            >
                                {isLobby ? 'Start Hand' : 'Next Hand'}
                            </button>
                        )}
                        {(bot?.stack < 20 || hero?.stack < 20) && (
                            <button
                                className="bg-red-600 text-white font-black hover:scale-105 transition-transform"
                                style={{ padding: '12px 24px', borderRadius: '8px', border: '2px solid rgba(255,255,255,0.2)', cursor: 'pointer', fontSize: '12px', textTransform: 'uppercase', backgroundColor: '#dc2626' }}
                                onClick={onReset}
                            >
                                Restart Session ($2k)
                            </button>
                        )}
                    </div>
                )}

                {/* Hero Position (Bottom) */}
                <div className="absolute flex flex-col items-center gap-3" style={{ bottom: '90px', left: '50%', transform: 'translateX(-50%)' }}>
                    <div className="flex flex-row gap-2" style={{ minHeight: '120px' }}>
                        {hero?.hand?.length > 0 ? (
                            hero.hand.map((c, i) => <Card key={i} cardInt={c} />)
                        ) : (
                            <div className="text-white font-mono italic text-xs uppercase opacity-20" style={{ alignSelf: 'center' }}>Cards will appear here</div>
                        )}
                    </div>
                    <div className="text-white font-bold text-lg uppercase">You: ${hero?.stack?.toFixed(0) || 0}</div>
                    {heroMessage && !is_hand_over && (
                        <div className="text-white/60 font-bold text-[11px] uppercase tracking-wider bg-black/40 px-3 py-1 rounded-full">{heroMessage}</div>
                    )}
                </div>

                {/* Action Buttons Row (Bottom Fixed) */}
                <div className="absolute flex flex-row items-end justify-between gap-4" style={{ bottom: '24px', left: '50%', transform: 'translateX(-50%)', width: '100%', padding: '0 48px' }}>

                    {/* Action Group */}
                    <div className="flex flex-row gap-2" style={{ height: '50px' }}>
                        {!is_hand_over && !isLobby && (
                            <>
                                <button
                                    className="btn btn-danger" style={{ width: '90px', height: '100%' }}
                                    onClick={() => handleAction(0)}
                                    disabled={!isHeroTurn || !legal_moves?.includes(0)}
                                >
                                    Fold
                                </button>
                                <button
                                    className="btn btn-action" style={{ width: '160px', height: '100%' }}
                                    onClick={() => handleAction(1)}
                                    disabled={!isHeroTurn || !legal_moves?.includes(1)}
                                >
                                    {hero?.bet < Math.max(hero?.bet || 0, bot?.bet || 0) ? `Call $${Math.max(0, (bot?.bet || 0) - (hero?.bet || 0))}` : 'Check'}
                                </button>
                                <button
                                    className="btn btn-primary" style={{ width: '90px', height: '100%' }}
                                    onClick={() => handleAction(7, sliderValue)}
                                    disabled={!isHeroTurn || !legal_moves?.includes(2)}
                                >
                                    Raise
                                </button>
                            </>
                        )}
                    </div>

                    {/* Wager Control (New Input Box) */}
                    {isHeroTurn && legal_moves?.includes(2) && !isLobby && (
                        <div className="glass flex flex-row items-center gap-4" style={{ padding: '8px 16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)' }}>
                            <div className="flex flex-col gap-1">
                                <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.5)', fontWeight: 'bold' }}>Wager Amount</span>
                                <input
                                    type="number"
                                    min={min_raise || 20}
                                    max={hero?.stack || 2000}
                                    value={sliderValue}
                                    onChange={(e) => setSliderValue(parseInt(e.target.value) || 0)}
                                    disabled={!isHeroTurn}
                                    style={{ background: 'transparent', border: 'none', color: '#fbbf24', fontSize: '20px', fontWeight: 'bold', outline: 'none', width: '100px', opacity: isHeroTurn ? 1 : 0.5 }}
                                />
                            </div>
                            <div className="flex flex-col gap-2" style={{ width: '140px' }}>
                                <input
                                    type="range"
                                    min={min_raise || 20}
                                    max={hero?.stack || 2000}
                                    step="10"
                                    value={sliderValue}
                                    onChange={(e) => setSliderValue(parseInt(e.target.value))}
                                    disabled={!isHeroTurn}
                                    style={{ width: '100%', accentColor: '#fbbf24', height: '4px', opacity: isHeroTurn ? 1 : 0.5 }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Background Texture Overlay */}
            <div className="absolute inset-0 pointer-events-none opacity-5 mix-blend-overlay" style={{ backgroundImage: 'url("https://www.transparenttextures.com/patterns/felt.png")' }}></div>
        </div>
    );
};

export default PokerTable;
