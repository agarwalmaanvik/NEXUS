import numpy as np
from collections import defaultdict, deque

class TellDetector:
    """
    Session-level psychological profiling of opponents.

    Tracks betting patterns, sizing tells, and aggression shifts to produce:
      1. get_exploit_signal(opp_id): float in [0,1] — how much to deviate
         from GTO and exploit this specific opponent this session.
         0 = they're balanced (play GTO), 1 = they're a fish (pure exploit).
      2. get_features(opp_id): 8-dim float32 vector for the NEXUS_GTO_Net.
    """

    MAX_HISTORY = 100  # hands to track per opponent

    def __init__(self):
        # Per-opponent action logs
        self._actions      = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        self._bet_sizes    = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        self._showdowns    = defaultdict(list)   # [(hand_rank, expected_rank)]
        self._hands_seen   = defaultdict(int)
        self._total_hands  = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_action(self, opp_id: int, action: int, amount: float,
                      pot: float, stage: int,
                      hand_strength_estimate: float = 0.5) -> None:
        """
        Log an opponent action for pattern analysis.

        Args:
            opp_id:                   Seat ID of the opponent.
            action:                   Action index (0=fold,1=call,2+=raise).
            amount:                   Bet/raise amount in chips.
            pot:                      Pot size before this action.
            stage:                    Street (0=pre, 1=flop, 2=turn, 3=river).
            hand_strength_estimate:   Our estimate of their hand strength [0,1].
        """
        self._actions[opp_id].append((action, stage))
        if pot > 0 and action >= 2:
            bet_fraction = amount / pot
            self._bet_sizes[opp_id].append((bet_fraction, hand_strength_estimate, stage))
        self._hands_seen[opp_id] += 0  # incremented at hand end
        self._total_hands += 1

    def record_hand_end(self, opp_id: int) -> None:
        """Call once per hand to update hand counters."""
        self._hands_seen[opp_id] += 1

    def record_showdown(self, opp_id: int, actual_rank: float,
                        expected_rank: float) -> None:
        """
        Log a showdown: compare actual hand strength to what betting implied.
        actual_rank / expected_rank both in [0,1] (1=best hand).
        """
        self._showdowns[opp_id].append((actual_rank, expected_rank))
        if len(self._showdowns[opp_id]) > 50:
            self._showdowns[opp_id] = self._showdowns[opp_id][-50:]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_exploit_signal(self, opp_id: int) -> float:
        """
        Returns a float in [0, 1].
          ~0 → opponent is balanced/GTO → play GTO yourself
          ~1 → opponent is exploitable → deviate to exploit

        Low sample size → 0.5 (uncertain, stay GTO).
        """
        hands = self._hands_seen.get(opp_id, 0)
        if hands < 10:
            return 0.5  # Insufficient data

        features = self.get_features(opp_id)
        # Features: [vpip, pfr, afq, wtsd, size_tell, showdown_gap, recent_agg, sample_conf]
        vpip, pfr, afq, wtsd, size_tell, showdown_gap, recent_agg, conf = features

        # Exploit signals:
        # High VPIP + low PFR → Loose-passive station → value bet mercilessly
        looseness     = vpip - 0.25           # positive if looser than average
        passiveness   = max(0.0, 0.3 - pfr)   # positive if tighter than normal PFR

        # Showdown gap: if they show hands weaker than their betting implied → they bluff too much
        bluff_tendency = showdown_gap  # positive → they over-bluffed at showdown

        # Bet sizing tell: if large bets correlate with weak hands (polarized bluffer)
        sizing_exploit = size_tell

        # Aggregate
        raw = 0.4 * (looseness + passiveness) + 0.35 * bluff_tendency + 0.25 * sizing_exploit
        raw = float(np.clip(raw, -1.0, 1.0))

        # Map [-1, 1] → [0, 1]
        signal = (raw + 1.0) / 2.0
        # Dampen with sample confidence
        signal = 0.5 + conf * (signal - 0.5)
        return float(np.clip(signal, 0.0, 1.0))

    def get_features(self, opp_id: int) -> np.ndarray:
        """
        Returns 8-dim float32 feature vector for the NEXUS_GTO_Net input.
        Always returns valid numbers (defaults if no data).

        Features:
          [0] vpip          — Voluntarily Put In Preflop [0,1]
          [1] pfr           — Pre-Flop Raise frequency [0,1]
          [2] afq           — Aggression Frequency post-flop [0,1]
          [3] wtsd          — Went To Showdown given saw flop [0,1]
          [4] size_tell     — Bet-size consistency score [0,1] (1=readable)
          [5] showdown_gap  — Bluff overfrequency signal [-1,1] normalized [0,1]
          [6] recent_agg    — Aggression in last 20 actions [0,1]
          [7] sample_conf   — Confidence in estimates (hands seen / 50, capped 1) [0,1]
        """
        actions = list(self._actions.get(opp_id, []))
        bet_sizes = list(self._bet_sizes.get(opp_id, []))
        showdowns = self._showdowns.get(opp_id, [])
        hands = max(self._hands_seen.get(opp_id, 0), 1)

        # VPIP: preflop call or raise / hands
        preflop_actions = [(a, s) for a, s in actions if s == 0]
        vpip = len([a for a, s in preflop_actions if a >= 1]) / max(len(preflop_actions), 1)

        # PFR: preflop raise / hands
        pfr = len([a for a, s in preflop_actions if a >= 2]) / max(len(preflop_actions), 1)

        # AFq: post-flop raises / (raises + calls + checks)
        postflop = [(a, s) for a, s in actions if s >= 1]
        afq = len([a for a, s in postflop if a >= 2]) / max(len(postflop), 1)

        # WTSD: approximate from showdown count vs hands
        wtsd = min(len(showdowns) / max(hands, 1), 1.0) * 2  # rough
        wtsd = float(np.clip(wtsd, 0.0, 1.0))

        # Bet size tell: do large bets predict strong hands?
        # Low variance in size→strength correlation = readable tell
        if len(bet_sizes) >= 5:
            sizes = np.array([s for s, h, st in bet_sizes])
            strengths = np.array([h for s, h, st in bet_sizes])
            # Correlation: if high, their bets are readable
            corr = float(np.corrcoef(sizes, strengths)[0, 1])
            if np.isnan(corr): corr = 0.0
            size_tell = abs(corr)  # high correlation = tells us something
        else:
            size_tell = 0.5

        # Showdown gap: actual vs expected strength
        if len(showdowns) >= 3:
            gaps = [actual - expected for actual, expected in showdowns]
            mean_gap = float(np.mean(gaps))
            # Positive gap → stronger than expected (nit), negative → weaker (bluffer)
            showdown_gap = float(np.clip((mean_gap + 1) / 2, 0.0, 1.0))
        else:
            showdown_gap = 0.5

        # Recent aggression: last 20 actions
        recent = actions[-20:] if len(actions) >= 20 else actions
        recent_agg = len([a for a, s in recent if a >= 2]) / max(len(recent), 1)

        # Sample confidence
        sample_conf = min(hands / 50.0, 1.0)

        return np.array([vpip, pfr, afq, wtsd, size_tell, showdown_gap,
                         recent_agg, sample_conf], dtype=np.float32)
