import numpy as np

class KellySizer:
    """
    Fractional Kelly criterion for poker bet sizing.

    WHY NOT PURE KELLY: Our win_prob comes from a neural net estimate.
    A 5% overestimate causes pure Kelly to massively over-bet and blow
    the stack (gambler's ruin). Quant desks use Half-Kelly as standard.

    Formula: f_actual = multiplier × (p(b+1) - 1) / b
      p = win probability (from value head, 0-1)
      b = ratio of villain's call to our bet (pot odds denominator)
      multiplier = 0.5 (Half-Kelly), variance-adaptive 0.25–0.75
    """

    def __init__(self, kelly_multiplier: float = 0.5, min_multiplier: float = 0.25,
                 max_multiplier: float = 0.75):
        self.base_multiplier = kelly_multiplier
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self._session_outcomes: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_bet(self, win_prob: float, pot: float, effective_stack: float,
                    to_call: float = 0.0) -> float:
        """
        Returns the optimal bet size in chips.

        Args:
            win_prob:        Hero's estimated win probability [0, 1].
            pot:             Current pot size in chips (before our bet).
            effective_stack: Smallest stack at the table (chips we can win).
            to_call:         Cost to call any existing bet (0 if we're first to act).

        Returns:
            Bet amount in chips (clamped to [0, effective_stack]).
        """
        win_prob = float(np.clip(win_prob, 0.01, 0.99))
        pot = max(float(pot), 1.0)
        effective_stack = max(float(effective_stack), 1.0)

        # Pot odds: if we bet X into pot P, villain calls X to win (P + X).
        # We iterate over a candidate bet range and pick the Kelly-optimal one.
        # For a simpler closed form: b = pot / bet (pot-odds multiplier for villain).
        # Kelly fraction f gives us: bet = f × effective_stack.
        # We solve: f = multiplier × (p(b+1) - 1) / b with b = pot / bet.
        # Substituting and solving gives: bet = multiplier × (2p - 1) × pot
        # (first-order approximation valid for moderate bet sizes).
        multiplier = self._adaptive_multiplier()
        q = 1.0 - win_prob

        # Simplified Kelly for poker: bet proportional to edge × pot
        edge = win_prob - q  # = 2p - 1
        if edge <= 0:
            # Negative edge: only call/check, never bet for value.
            return 0.0

        optimal_bet = multiplier * edge * pot

        # Clamp to sensible poker limits: at least 1 chip, at most effective stack.
        optimal_bet = float(np.clip(optimal_bet, 1.0, effective_stack))
        return round(optimal_bet)

    def compute_raise(self, win_prob: float, pot: float, effective_stack: float,
                      to_call: float, min_raise: float) -> float:
        """
        Returns total raise amount (call + raise increment).
        Used when facing a bet/raise ourselves.
        """
        base_bet = self.compute_bet(win_prob, pot + to_call, effective_stack, to_call)
        total = to_call + base_bet
        total = max(total, float(min_raise))
        return float(np.clip(total, min_raise, effective_stack))

    def record_outcome(self, profit_in_bb: float) -> None:
        """Record a hand outcome (in BBs) for variance tracking."""
        self._session_outcomes.append(float(profit_in_bb))
        # Keep last 200 hands
        if len(self._session_outcomes) > 200:
            self._session_outcomes = self._session_outcomes[-200:]

    def get_current_multiplier(self) -> float:
        return self._adaptive_multiplier()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _adaptive_multiplier(self) -> float:
        """
        Variance guard: reduce Kelly fraction during high-variance stretches.
        Mirrors a prop-desk risk limit — scale back exposure automatically.
        """
        if len(self._session_outcomes) < 20:
            return self.base_multiplier

        recent = self._session_outcomes[-50:]
        variance = float(np.var(recent))

        # Normalise variance: a variance of 25 BB² is already "high" for a session
        # (std ≈ 5 BB per hand is significant downswing territory)
        norm_var = min(variance / 25.0, 1.0)

        # Linear interpolation from base_multiplier down to min_multiplier
        multiplier = self.base_multiplier - norm_var * (self.base_multiplier - self.min_multiplier)
        return float(np.clip(multiplier, self.min_multiplier, self.max_multiplier))
