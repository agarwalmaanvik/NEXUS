"""
NEXUS train_master.py — Full CFR Training Loop (v2 correct interface)

Training pipeline:
  Phase 1: External Sampling MCCFR traversals → per-player advantage buffers
            + FAISS RAG index population
  Phase 2: Parallel game experience (supplementary diversity signal)
  Phase 3: Dual-network gradient updates (advantage_net + strategy_net)
  Phase 4: FSP ghost pool management
  Phase 5: Checkpoint & logging
"""

import os
import gc
import glob
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from cfr_agent     import CFRAgent
from parallel_env  import ParallelPokerEnv
from memory        import PerPlayerReservoirBuffer
from solver        import ExternalSamplingMCCFR
from range_encoder import RangeEncoder
from rag_retriever import SubgameRetriever
from preflop_tables import PreflopOracle
from networks      import INPUT_DIM
from engine_core   import GameState

# ── Hyperparameters ───────────────────────────────────────────────────────────────────────────────
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
ITERATIONS         = 50_000       # Deep convergence
NUM_ENVS           = 128          # Parallel games
STEPS_PER_ITER     = 64          # Richer gradient signal per iteration
TRAVERSALS_PER_ITER = 32         # CFR traversals per iteration
BUFFER_CAPACITY    = 500_000     # 500K transitions — ~1.45 GB RAM per player; safe on most machines
BATCH_SIZE         = 512          # More stable gradients
TRAIN_STEPS        = 48           # Gradient steps per iteration
LR                 = 1e-4
STARTING_STACK     = 2000
BB_AMT             = 20
CHECKPOINT_DIR     = "checkpoints"
HISTORY_DIR        = "checkpoints/history"
MODEL_NAME        = "nexus"       # Matches nexus_latest.pt naming
SAVE_EVERY         = 500
LOG_EVERY          = 10

STAGE_WEIGHTS      = {0: 0.5, 1: 0.70, 2: 0.85, 3: 1.0}


def build_full_vec(state_vec: np.ndarray, range_enc: RangeEncoder,
                   opp_seat: int) -> np.ndarray:
    """Concatenates 178-dim state + 169-dim range + 8-dim zeros = 355."""
    range_vec = range_enc.to_numpy(opp_seat)
    tell_vec  = np.zeros(8, dtype=np.float32)
    return np.concatenate([state_vec, range_vec, tell_vec]).astype(np.float32)


def train_master():
    print(f"NEXUS training on {DEVICE}. INPUT_DIM={INPUT_DIM}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Initialise components ─────────────────────────────────────────────────
    agent  = CFRAgent(device=DEVICE, lr=LR, input_dim=INPUT_DIM)
    memory = PerPlayerReservoirBuffer(n_players=2, capacity=BUFFER_CAPACITY)
    rag    = SubgameRetriever(os.path.join(CHECKPOINT_DIR, "rag.pkl"))
    oracle = PreflopOracle(os.path.join(CHECKPOINT_DIR, "preflop_gto.db"))

    # Load existing checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_latest.pt")
    if agent.load(ckpt_path):
        print(f"Resumed from {ckpt_path}")
    else:
        print("Fresh start.")

    optimizer  = agent.optimizer
    # Cosine Annealing: LR cools from 1e-4 → 1e-5 over ITERATIONS steps.
    # Starts fast (explores) then cools to micro-crawl (converges on Nash).
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ITERATIONS, eta_min=1e-5
    )

    rag.load()
    print(f"RAG: {len(rag)} situations")

    history_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ghost_*.pt")))
    print(f"FSP pool: {len(history_files)} snapshots")

    # Parallel envs for supplementary experience
    penv = ParallelPokerEnv(n_envs=NUM_ENVS, n_players=2)
    penv.reset()
    range_encoders = [RangeEncoder(n_players=2) for _ in range(NUM_ENVS)]

    # CFR solver (populates buffers + RAG during traversal)
    solver = ExternalSamplingMCCFR(
        net=agent.net, device=DEVICE,
        depth_limit=4, n_traversals=1,  # 1 traversal per call for speed
        rag=rag)

    print(f"Training {ITERATIONS} iterations...")

    for iteration in range(1, ITERATIONS + 1):
        t_start = time.time()
        agent.net.eval()

        # ── PHASE 1: CFR TRAVERSAL ─────────────────────────────────────────
        traversal_evs = []
        for _ in range(TRAVERSALS_PER_ITER):
            gs = GameState(n_players=2, training_mode=True)
            gs.reset()
            re = RangeEncoder(n_players=2)
            hero = random.randint(0, 1)
            try:
                ev = solver.run_traversal(
                    root_state=gs,
                    hero_seat=hero,
                    p0_buffer=memory.buffers[0],
                    p1_buffer=memory.buffers[1],
                    range_encoder=re,
                )
                traversal_evs.append(ev)
            except Exception:
                pass  # Rare degenerate game states; skip

        # ── PHASE 2: PARALLEL EXPERIENCE ──────────────────────────────────
        raw_states    = penv.get_current_states()       # [N, 171]
        active_players = penv.get_current_players()    # [N] ints
        legal_masks   = penv.get_legal_moves_mask()    # [N, 7] bool
        env_info      = penv.get_env_info()            # [(pot, stage)]

        current_states = np.zeros((NUM_ENVS, INPUT_DIM), dtype=np.float32)
        for i in range(NUM_ENVS):
            opp = 1 - active_players[i]
            current_states[i] = build_full_vec(raw_states[i], range_encoders[i], opp)

        actions      = agent.get_batch_strategy(current_states, legal_masks)
        dummy_amounts = np.zeros(NUM_ENVS, dtype=int)
        _, dones, payouts = penv.step(actions, dummy_amounts)

        for i in range(NUM_ENVS):
            if dones[i] and payouts[i] is not None:
                p = active_players[i]
                reward = payouts[i][p] - STARTING_STACK
                norm_r = float(np.clip(reward / STARTING_STACK, -2.0, 2.0))

                stage_w  = STAGE_WEIGHTS.get(env_info[i][1], 1.0)
                pot_frac = min(env_info[i][0] / (STARTING_STACK * 2), 1.0)
                commit_w = 0.5 + 0.5 * pot_frac
                w_reward = norm_r * stage_w * commit_w

                target_adv = np.zeros(7, dtype=np.float32)
                target_adv[actions[i]] = w_reward
                memory.buffers[p].add(current_states[i], target_adv, w_reward)
                range_encoders[i].reset()

        # ── PHASE 3: NETWORK UPDATES ───────────────────────────────────────
        min_size = memory.min_size()
        adv_loss_total = val_loss_total = range_loss_total = 0.0

        if min_size >= BATCH_SIZE:
            agent.net.train()
            agent.strategy_net.train()

            update_player = iteration % 2  # Alternate which player's buffer

            for step in range(TRAIN_STEPS):
                # ── Advantage network update ──────────────────────────────
                states, advs, vals = memory.sample(update_player, BATCH_SIZE)
                states, advs, vals = states.to(DEVICE), advs.to(DEVICE), vals.to(DEVICE)

                pred_adv, pred_val, pred_range = agent.net(states)

                # Masked MSE on advantages (only non-zero targets)
                action_mask = (advs != 0).float()
                adv_diff    = (pred_adv - advs) * action_mask
                loss_adv    = (adv_diff ** 2).mean()

                loss_val    = F.mse_loss(pred_val, vals)

                # Range head: auxiliary KL vs uniform (self-supervised)
                uniform     = torch.ones_like(pred_range) / pred_range.shape[-1]
                loss_range  = F.kl_div(pred_range.log() + 1e-10,
                                        uniform, reduction='batchmean')

                loss = loss_adv + 0.5 * loss_val + 0.1 * loss_range
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 1.0)
                agent.optimizer.step()

                adv_loss_total   += loss_adv.item()
                val_loss_total   += loss_val.item()
                range_loss_total += loss_range.item()

                # ── Strategy network update (average strategy convergence) ─
                s2, a2, v2 = memory.sample(update_player, BATCH_SIZE)
                s2, a2, v2 = s2.to(DEVICE), a2.to(DEVICE), v2.to(DEVICE)

                pred_strat, pred_sval, _ = agent.strategy_net(s2)

                # Target = regret-matched output of current advantage_net
                with torch.no_grad():
                    adv_ref, _, _ = agent.net(s2)
                    ref_clipped = torch.clamp(adv_ref, min=0.0)
                    ref_sum = ref_clipped.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    target_strat = ref_clipped / ref_sum

                loss_s = F.mse_loss(pred_strat, target_strat) + \
                         0.3 * F.mse_loss(pred_sval, v2)
                agent.strategy_optimizer.zero_grad()
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(agent.strategy_net.parameters(), 1.0)
                agent.strategy_optimizer.step()

        # ── PHASE 4: FSP GHOST MANAGEMENT ─────────────────────────────────
        save_now = (iteration <= 100 and iteration % 10 == 0) \
                   or (iteration % 100 == 0)
        if save_now:
            snap = agent.save_snapshot()
            history_files.append(snap)
            history_files = history_files[-50:]  # Cap pool at 50

        if iteration % 25 == 0 and history_files:
            agent.load_historical(random.choice(history_files))

        # Step the cosine LR scheduler once per iteration
        scheduler.step()

        # ── PHASE 5: LOGGING & PERSISTENCE ────────────────────────────────
        if iteration % LOG_EVERY == 0:
            elapsed  = time.time() - t_start
            buf      = memory.__len__()
            mean_ev  = float(np.mean(traversal_evs)) if traversal_evs else 0.0
            ts       = TRAIN_STEPS if min_size >= BATCH_SIZE else 0
            lr       = scheduler.get_last_lr()[0]

            print(f"Iter {iteration:05d} | "
                  f"AdvL:{adv_loss_total/max(ts,1):.4f} "
                  f"ValL:{val_loss_total/max(ts,1):.4f} "
                  f"RngL:{range_loss_total/max(ts,1):.4f} | "
                  f"EV:{mean_ev:.1f} | "
                  f"Buf[0]:{buf.get(0,0):>6} Buf[1]:{buf.get(1,0):>6} | "
                  f"RAG:{len(rag):>5} | "
                  f"α:{agent.net.get_alpha():.3f} | "
                  f"LR:{lr:.2e} | "
                  f"{elapsed:.1f}s")

        if iteration % SAVE_EVERY == 0:
            agent.save(ckpt_path)
            rag.save()
            gc.collect()  # Reclaim fragmented memory after heavy save ops
            print(f"Saved at iteration {iteration}")

    agent.save(ckpt_path)
    rag.save()
    print("Training complete.")


if __name__ == "__main__":
    train_master()
