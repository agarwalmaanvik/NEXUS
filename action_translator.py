"""
action_translator.py — Maps arbitrary real-world bet sizes to abstraction buckets.

In the Live Advisor, opponents bet non-standard amounts (e.g. $42 into a $100 pot).
The MCCFR solver only knows 7 action buckets. This module translates any bet to a
weighted blend of the two nearest buckets, preserving the full EV calculation.

Example:
    $42 into $100 pot → fraction 0.42
    Nearest: Bucket3 (33%) and Bucket4 (75%)
    Weights: {3: 0.75, 4: 0.25}

    Solver evaluates: EV = 0.75*EV(Bucket3) + 0.25*EV(Bucket4)
    This is more accurate than hard-rounding to a single bucket.

Reference: Libratus action translation (Brown & Sandholm, 2017).
"""

# Bucket → pot fraction mapping (must match bet_sizer.py + engine_core.py)
_BUCKETS: dict[int, float] = {
    3: 0.33,   # 33% pot
    4: 0.75,   # 75% pot
    5: 1.50,   # 150% pot
    6: 9999.0, # All-in (large sentinel)
}

# Sentinels for special actions
_CALL_BUCKETS  = {1}
_FOLD_BUCKETS  = {0}


def translate_raise(bet_amount: float, pot: float) -> dict[int, float]:
    """
    Map a raise amount to a weighted blend of two abstraction buckets.

    Args:
        bet_amount: The raise amount in chips (raise increment, not total).
        pot:        Pot size before this raise in chips.

    Returns:
        Dict mapping bucket_id → weight, where weights sum to 1.0.
        E.g. {3: 0.75, 4: 0.25}
    """
    if pot <= 0:
        return {4: 1.0}  # default to 75% pot if pot info missing

    frac = bet_amount / pot

    # Sort buckets by distance to the observed fraction (exclude all-in sentinel)
    workable = {b: f for b, f in _BUCKETS.items() if b != 6}
    sorted_buckets = sorted(workable.items(), key=lambda kv: abs(kv[1] - frac))

    b1, f1 = sorted_buckets[0]

    # If extremely close to one bucket, no need to blend
    if abs(f1 - frac) < 0.05:
        return {b1: 1.0}

    if len(sorted_buckets) < 2:
        return {b1: 1.0}

    b2, f2 = sorted_buckets[1]
    d1 = abs(f1 - frac)
    d2 = abs(f2 - frac)
    total_d = d1 + d2

    if total_d < 1e-9:
        return {b1: 1.0}

    w1 = 1.0 - d1 / total_d   # closer bucket gets higher weight
    w2 = 1.0 - w1

    return {b1: round(w1, 4), b2: round(w2, 4)}


def translate_to_single_bucket(bet_amount: float, pot: float) -> int:
    """
    Hard-round a bet to the nearest single bucket. Used when the solver
    needs a discrete action (e.g. for tree traversal at a specific node).
    """
    blend = translate_raise(bet_amount, pot)
    return max(blend, key=blend.get)


def describe_translation(bet_amount: float, pot: float) -> str:
    """Human-readable description for debugging/display."""
    blend = translate_raise(bet_amount, pot)
    frac  = bet_amount / max(pot, 1.0)
    parts = [f"Bucket{b}({w*100:.0f}%)" for b, w in sorted(blend.items())]
    return f"${bet_amount:.0f}/{pot:.0f}pot ({frac:.0%}) → {' + '.join(parts)}"
