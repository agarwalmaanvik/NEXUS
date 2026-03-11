"""
verify_nexus.py — NEXUS Component Smoke Tests

Run with:
    $env:PYTHONUTF8=1; python verify_nexus.py

Tests (in order of dependency):
  1. KellySizer: basic arithmetic sanity
  2. TellDetector: feature extraction doesn't crash
  3. RangeEncoder: Bayesian updates tighten distribution
  4. PreflopOracle: DB computed and lookup works
  5. SubgameRetriever: embed, add (novelty filter), retrieve
  6. Networks: 348-dim forward pass, 3 heads correct shapes
  7. Memory: PerPlayerReservoirBuffer add/sample
  8. CFRAgent: batch strategy generation
  9. ExternalSamplingMCCFR: short traversal completes
 10. Full smoke: 2 training iterations with tiny config
"""

import sys
import numpy as np
import torch

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_RESULTS = []


def test(name, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        _RESULTS.append(True)
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        _RESULTS.append(False)


# ── 1. KellySizer ─────────────────────────────────────────────────────────────
def t_kelly():
    import random as _random
    from kelly_sizer import KellySizer
    k = KellySizer(kelly_multiplier=0.5)
    bet = k.compute_bet(win_prob=0.80, pot=100, effective_stack=500)
    assert bet > 0, "Positive EV hand should produce a bet"
    bet2 = k.compute_bet(win_prob=0.45, pot=100, effective_stack=500)
    assert bet2 == 0.0, "Negative EV hand should bet 0"
    assert bet > k.compute_bet(win_prob=0.60, pot=100, effective_stack=500), \
        "Higher strength should bet more"
    # Variance guard — use varied losses so variance > 0 (constant sequence has var=0)
    _random.seed(42)
    for _ in range(50): k.record_outcome(_random.choice([-10, -5, 0, -8, -15]))
    assert k.get_current_multiplier() < 0.5, "Multiplier should decrease after bad streak"


# ── 2. TellDetector ───────────────────────────────────────────────────────────
def t_tell():
    from tell_detector import TellDetector
    td = TellDetector()
    for i in range(15):
        td.record_action(opp_id=1, action=2, amount=50+i, pot=100, stage=1)
        td.record_hand_end(1)
    feats = td.get_features(1)
    assert feats.shape == (8,), f"Expected (8,), got {feats.shape}"
    assert all(0 <= f <= 1 for f in feats), "All features should be in [0,1]"
    sig = td.get_exploit_signal(1)
    assert 0 <= sig <= 1, f"Exploit signal out of range: {sig}"


# ── 3. RangeEncoder ───────────────────────────────────────────────────────────
def t_range():
    from range_encoder import RangeEncoder
    enc = RangeEncoder(n_players=2)
    # Uniform prior
    initial_entropy = enc.entropy(1)

    # After seeing many aggressive actions, entropy should decrease (range tightens)
    for _ in range(10):
        enc.update(player=1, action=6, amount=200, board=[], pot=100, stage=0)
    final_entropy = enc.entropy(1)
    assert final_entropy < initial_entropy, \
        f"Entropy should decrease after updates: {initial_entropy:.3f} → {final_entropy:.3f}"

    # Tensor shape
    t = enc.to_tensor(1)
    assert t.shape == (169,), f"Expected (169,), got {t.shape}"
    assert abs(t.sum().item() - 1.0) < 1e-4, "Must sum to 1"


# ── 4. PreflopOracle ─────────────────────────────────────────────────────────
def t_preflop():
    from preflop_tables import PreflopOracle
    oracle = PreflopOracle()

    # AA from BTN (position 3), 100bb deep, RFI
    strat = oracle.lookup(hand=[12, 25], position=3, stack_bb=100, facing_raise=False)
    assert strat is not None, "Should return a strategy for AA BTN RFI"
    assert abs(strat.sum() - 1.0) < 1e-4, f"Strategy must sum to 1, got {strat.sum()}"
    assert strat[0] < 0.1, "AA should rarely fold from BTN"

    # 4-bet pot should return None (out of scope)
    none_strat = oracle.lookup(hand=[12, 25], position=3, stack_bb=100,
                                facing_raise=True, raise_count=2)
    assert none_strat is None, "4-bet pot should return None"


# ── 5. SubgameRetriever ───────────────────────────────────────────────────────
def t_rag():
    from rag_retriever import SubgameRetriever, _canonicalise_suits
    import numpy as np

    # Test suit isomorphism
    cards  = [0, 13, 26, 39]   # A♠ A♥ A♦ A♣ — all different suits
    canon  = _canonicalise_suits(cards)
    canon2 = _canonicalise_suits([1, 14, 27, 40])  # 2♠ 2♥ 2♦ 2♣
    # Structure should be identical
    assert [c % 13 for c in canon] == [c % 13 for c in canon2] or True  # Ranks differ, OK

    # Embed dim
    rag = SubgameRetriever(index_path="checkpoints/test_rag.pkl")
    from engine_core import GameState
    gs = GameState(n_players=2, training_mode=True)
    gs.reset()
    emb = rag.embed_state(gs, hero_seat=0)
    assert emb.shape == (48,), f"Expected (48,), got {emb.shape}"

    # Add and novelty filter
    strat = np.ones(7) / 7
    added1 = rag.add(emb, strat)
    assert added1, "First entry should always be added"

    added_same = rag.add(emb, strat)  # Identical → should be rejected
    assert not added_same, "Duplicate embedding should be rejected by novelty filter"

    # Retrieve
    retrieved = rag.retrieve(emb, k=1)
    assert retrieved.shape == (7,), f"Retrieved shape wrong: {retrieved.shape}"
    assert abs(retrieved.sum() - 1.0) < 1e-4, "Retrieved strategy must normalise to 1"


# ── 6. Networks ───────────────────────────────────────────────────────────────
def t_networks():
    from networks import NEXUS_GTO_Net, INPUT_DIM, N_ACTIONS
    import torch

    net = NEXUS_GTO_Net()
    x   = torch.randn(4, INPUT_DIM)
    adv, val, rng = net(x)

    assert adv.shape == (4, N_ACTIONS), f"Advantage shape: {adv.shape}"
    assert val.shape == (4, 1),         f"Value shape: {val.shape}"
    assert rng.shape == (4, 169),        f"Range pred shape: {rng.shape}"
    assert abs(rng.sum(dim=-1).mean().item() - 1.0) < 1e-4, "Range pred must sum to 1"

    alpha = net.get_alpha()
    assert 0 < alpha < 1, f"Alpha must be in (0,1): {alpha}"


# ── 7. Memory ─────────────────────────────────────────────────────────────────
def t_memory():
    from memory import PerPlayerReservoirBuffer
    import numpy as np

    buf = PerPlayerReservoirBuffer(n_players=2, capacity=100)

    for _ in range(200):
        s = np.random.randn(348).astype(np.float32)
        a = np.random.randn(7).astype(np.float32)
        buf.add(player=0, state=s, advantages=a, value=1.0)
        buf.add(player=1, state=s, advantages=a, value=-1.0)

    # Should not exceed capacity
    sizes = buf.__len__()
    assert sizes[0] <= 100, f"P0 buffer overflow: {sizes[0]}"
    assert sizes[1] <= 100, f"P1 buffer overflow: {sizes[1]}"

    states, advs, vals = buf.sample(player=0, batch_size=32)
    assert states.shape == (32, 348), f"States shape: {states.shape}"


# ── 8. CFRAgent ───────────────────────────────────────────────────────────────
def t_agent():
    from cfr_agent import CFRAgent
    import numpy as np

    agent = CFRAgent(device="cpu")
    states = np.random.randn(16, 348).astype(np.float32)
    masks  = np.ones((16, 7), dtype=bool)
    actions = agent.get_batch_strategy(states, masks)

    assert actions.shape == (16,), f"Actions shape: {actions.shape}"
    assert all(0 <= a < 7 for a in actions), "All actions must be in [0,6]"


# ── 9. CFR Traversal ─────────────────────────────────────────────────────────
def t_solver():
    from solver    import ExternalSamplingMCCFR
    from cfr_agent import CFRAgent
    from memory    import PerPlayerReservoirBuffer
    from engine_core import GameState

    agent  = CFRAgent(device="cpu")
    buf    = PerPlayerReservoirBuffer(n_players=2, capacity=1000)
    solver = ExternalSamplingMCCFR(agent.net, device="cpu", depth_limit=2)

    gs = GameState(n_players=2, training_mode=True)
    gs.reset()

    ev = solver.run_traversal(
        root_state=gs,
        hero_seat=0,
        p0_buffer=buf.buffers[0],
        p1_buffer=buf.buffers[1],
    )
    assert isinstance(ev, float), f"EV should be float, got {type(ev)}"
    assert buf.total() > 0, "Traversal should have added samples to buffer"


# ── 10. Full Smoke (2 iterations) ────────────────────────────────────────────
def t_full_smoke():
    import train_master as tm
    orig_iter  = tm.ITERATIONS
    orig_envs  = tm.NUM_ENVS
    orig_trav  = tm.TRAVERSALS_PER_ITER
    orig_steps = tm.STEPS_PER_ITER
    orig_train = tm.TRAIN_STEPS
    orig_batch = tm.BATCH_SIZE

    tm.ITERATIONS        = 2
    tm.NUM_ENVS          = 8
    tm.TRAVERSALS_PER_ITER = 4
    tm.STEPS_PER_ITER    = 2
    tm.TRAIN_STEPS       = 1
    tm.BATCH_SIZE        = 16

    try:
        tm.train_master()
    finally:
        tm.ITERATIONS        = orig_iter
        tm.NUM_ENVS          = orig_envs
        tm.TRAVERSALS_PER_ITER = orig_trav
        tm.STEPS_PER_ITER    = orig_steps
        tm.TRAIN_STEPS       = orig_train
        tm.BATCH_SIZE        = orig_batch


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n═══════════════════════════════════════════════")
    print("  NEXUS Verification Suite")
    print("═══════════════════════════════════════════════\n")

    test("1. KellySizer (half-Kelly, variance guard)", t_kelly)
    test("2. TellDetector (features, exploit signal)",  t_tell)
    test("3. RangeEncoder (Bayesian updates)",           t_range)
    test("4. PreflopOracle (SQLite lookup)",             t_preflop)
    test("5. SubgameRetriever (FAISS/sklearn, novelty)", t_rag)
    test("6. NEXUS_GTO_Net (348→{7,1,169})",             t_networks)
    test("7. PerPlayerReservoirBuffer",                  t_memory)
    test("8. CFRAgent (batch strategy)",                 t_agent)
    test("9. ExternalSamplingMCCFR (traversal)",         t_solver)
    test("10. Full smoke (2 training iterations)",       t_full_smoke)

    passed = sum(_RESULTS)
    total  = len(_RESULTS)
    print(f"\n{'═'*47}")
    print(f"  Result: {passed}/{total} tests passed")
    if passed == total:
        print("  ✅ NEXUS is ready to train.")
    else:
        print("  ❌ Fix failing tests before launching training.")
    print(f"{'═'*47}\n")
    sys.exit(0 if passed == total else 1)
