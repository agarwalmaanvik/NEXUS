import React from 'react';

const _RANKS = "23456789TJQKA";
const _SUITS = "cdhs";
const _SUIT_SYMS = { 'c': '♣', 'd': '♦', 'h': '♥', 's': '♠' };

const Card = ({ cardInt, hidden }) => {
    if (hidden) {
        return (
            <div className="card hidden border-2 border-white/20">
                <div className="w-full h-full opacity-20 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]" />
            </div>
        );
    }

    const rankIdx = Math.floor(cardInt / 4);
    const suitIdx = cardInt % 4;
    const rank = _RANKS[rankIdx];
    const suitChar = _SUITS[suitIdx];
    const suitSym = _SUIT_SYMS[suitChar];
    const isRed = suitChar === 'd' || suitChar === 'h';

    return (
        <div className="card animate-in fade-in zoom-in duration-300">
            <div className={`text-xl font-bold ${isRed ? 'text-red-600' : 'text-black'}`}>
                {rank}
            </div>
            <div className={`text-sm ${isRed ? 'text-red-500' : 'text-black'}`}>
                {suitSym}
            </div>
            <div className={`card-suit-big ${isRed ? 'text-red-500' : 'text-black'}`}>
                {suitSym}
            </div>
        </div>
    );
};

export default Card;
