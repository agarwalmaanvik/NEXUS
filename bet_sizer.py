"""
bet_sizer.py — Game-tree-derived bet sizing for NEXUS v2.

Replaces KellySizer in the decision path. Maps action bucket index
to exact chip amount using pot-fraction arithmetic. No bankroll math,
no neural net — just clean poker sizing.

Action buckets (matches legal_moves encoding in engine_core.py):
  0 = Fold
  1 = Check / Call
  2 = Min-raise
  3 = 33% pot raise
  4 = 75% pot raise
  5 = 150% pot raise
  6 = All-in
"""

# Fractional pot-sizes for each raise bucket
_BUCKET_FRACS = {
    3: 0.33,
    4: 0.75,
    5: 1.50,
}


def bucket_to_amount(bucket: int, pot: float, to_call: float,
                     stack: float, min_raise: float) -> int:
    """
    Convert an action bucket index to a chip amount.

    Args:
        bucket:    Action bucket index (2–6).
        pot:       Current pot in chips (before our action).
        to_call:   Chips we must call to stay in hand.
        stack:     Our remaining stack (effective stack cap).
        min_raise: Minimum legal raise total.

    Returns:
        Total chips to put in (call + raise), clamped to [min_raise, stack].
        Returns 0 for fold/check/call buckets (caller handles those).
    """
    if bucket <= 1:
        return 0
    if bucket == 6:
        return int(stack)

    frac = _BUCKET_FRACS.get(bucket, 0.75)
    raise_increment = frac * max(pot, 1.0)
    total = to_call + raise_increment

    total = max(total, float(min_raise))
    total = min(total, float(stack))
    return int(round(total))


def amount_to_display(amount: int, bb_amt: float) -> str:
    """Returns a human-readable string like '3.5 BB' or '$140'."""
    if bb_amt > 0:
        bbs = amount / bb_amt
        return f"{bbs:.1f} BB"
    return f"{amount} chips"
