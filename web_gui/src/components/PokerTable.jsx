import React, { useState } from 'react';
import Card from './Card';

const PokerTable = ({ state, onAction, onStart }) => {
    const [raiseAmount, setRaiseAmount] = useState('');

    const { pot, board, hero, bot, current_player, legal_moves, min_raise, is_hand_over, bot_thinking } = state;
    const isHeroTurn = current_player === 0 && !is_hand_over;

    const handleRaise = () => {
        const amt = parseInt(raiseAmount);
        if (!isNaN(amt) && amt >= min_raise) {
            onAction(2, amt); // 2 = RAISE in engine_core (though server might use action_idx mapping)
            // Actually, in engine_core: 0=FOLD, 1=CALL/CHECK, 2=RAISE
            // But in nexus_server.py: we handle action_idx directly from client
            setRaiseAmount('');
        }
    };

    return (
        <div className="w-full max-w-5xl aspect-[1.4/1] flex flex-col items-center justify-between py-12">
            {/* Bot Area */}
            <div className="flex flex-col items-center gap-4">
                <div className={`glass px-6 py-2 rounded-full border-2 ${current_player === 1 ? 'border-gold shadow-[0_0_15px_rgba(245,158,11,0.3)]' : 'border-transparent'}`}>
                    <span className="font-bold text-lg">🤖 NEXUS BOT</span>
                    <span className="ml-4 text-gold font-mono">${bot.stack.toFixed(0)}</span>
                </div>
                <div className="flex gap-2">
                    {bot.hand.length > 0 ? (
                        bot.hand.map((c, i) => <Card key={i} cardInt={c} />)
                    ) : (
                        <>
                            <Card hidden />
                            <Card hidden />
                        </>
                    )}
                </div>
            </div>

            {/* Center Table */}
            <div className="felt-bg w-[85%] h-[45%] flex flex-col items-center justify-center gap-6">
                <div className="bg-black/30 px-8 py-2 rounded-full">
                    <span className="text-gold font-black text-2xl tracking-widest">POT: ${pot.toFixed(0)}</span>
                </div>

                <div className="flex gap-3 h-28">
                    {board.map((c, i) => <Card key={i} cardInt={c} />)}
                    {[...Array(5 - board.length)].map((_, i) => (
                        <div key={i} className="w-[70px] h-[100px] border-2 border-white/5 rounded-6" />
                    ))}
                </div>

                {bot_thinking && (
                    <div className="absolute top-4 italic animate-pulse text-gold">
                        NEXUS is calculating ranges...
                    </div>
                )}
            </div>

            {/* Hero Area */}
            <div className="flex flex-col items-center gap-6 w-full">
                <div className="flex gap-4">
                    {hero.hand.map((c, i) => <Card key={i} cardInt={c} />)}
                </div>

                <div className={`glass px-8 py-3 rounded-full border-2 ${isHeroTurn ? 'border-primary shadow-[0_0_15px_rgba(59,130,246,0.3)]' : 'border-transparent'} flex items-center gap-6`}>
                    <div className="flex flex-col">
                        <span className="text-text-muted text-xs uppercase font-bold">Your Stack</span>
                        <span className="text-xl font-bold font-mono">${hero.stack.toFixed(0)}</span>
                    </div>
                    <div className="w-[1px] h-8 bg-white/10" />
                    <div className="flex flex-col">
                        <span className="text-text-muted text-xs uppercase font-bold">Current Bet</span>
                        <span className="text-xl font-bold text-blue-400">${hero.bet.toFixed(0)}</span>
                    </div>
                </div>

                {/* Controls */}
                <div className="flex gap-3 items-center">
                    {is_hand_over ? (
                        <button onClick={onStart} className="btn btn-primary px-12 py-4 text-xl shadow-lg ring-4 ring-blue-500/20">
                            Deal Next Hand
                        </button>
                    ) : (
                        <>
                            <button
                                disabled={!isHeroTurn || !legal_moves.includes(0)}
                                onClick={() => onAction(0)}
                                className="btn btn-danger"
                            >
                                Fold
                            </button>
                            <button
                                disabled={!isHeroTurn || !legal_moves.includes(1)}
                                onClick={() => onAction(1)}
                                className="btn glass hover:bg-white/10"
                            >
                                {hero.bet < Math.max(hero.bet, bot.bet) ? `Call $${Math.max(0, bot.bet - hero.bet)}` : 'Check'}
                            </button>

                            <div className="flex items-center glass p-1 rounded-lg">
                                <input
                                    type="number"
                                    disabled={!isHeroTurn || !legal_moves.includes(2)}
                                    value={raiseAmount}
                                    onChange={(e) => setRaiseAmount(e.target.value)}
                                    placeholder={`Min ${min_raise}`}
                                    className="bg-transparent border-none outline-none px-3 w-24 font-mono text-white"
                                />
                                <button
                                    disabled={!isHeroTurn || !legal_moves.includes(2)}
                                    onClick={handleRaise}
                                    className="btn btn-primary py-2 px-4"
                                >
                                    Raise
                                </button>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PokerTable;
