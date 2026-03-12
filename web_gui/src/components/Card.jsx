import React from 'react';

const _RANKS = "23456789TJQKA";
const _SUITS = "cdhs";
const _SUIT_SYMS = { 'c': '♣', 'd': '♦', 'h': '♥', 's': '♠' };

const Card = ({ cardInt, hidden }) => {
    if (hidden) {
        return (
            <div className="card hidden">
                <div className="card-back-pattern" />
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
        <div className="card">
            <div className={`text-lg font-black ${isRed ? 'text-red' : 'text-black'}`} style={{ lineHeight: 1 }}>
                {rank}
            </div>
            <div className={`text-sm font-bold ${isRed ? 'text-red' : 'text-black'}`} style={{ lineHeight: 1 }}>
                {suitSym}
            </div>
            <div className={`card-suit-big ${isRed ? 'text-red' : 'text-black'}`} style={{ opacity: 0.4 }}>
                {suitSym}
            </div>
        </div>
    );
};

export default Card;
