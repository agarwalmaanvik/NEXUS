**[Read the Full Technical Whitepaper Here](./NEXUS_Whitepaper.pdf)**

# NEXUS Poker Bot

**NEXUS** is a state-of-the-art, 6-pillar hybrid Texas Hold'em engine designed to achieve near-Game Theory Optimal (GTO) play while actively exploiting human opponents, all while running efficiently on consumer hardware.

This repository contains the core engine, the deep neural network architecture, and the exploitation layers that power the bot. The trained model weights (`.pt`), large solved indices (`RAG`), and GTO databases (`.db`) are kept private and are not included in this repository.

## The Architecture
NEXUS solves the computationally infeasible problem of real-time GTO play through a Tri-Layer Hybrid Architecture. The bot achieves this by compressing massive CFR training data into a deep neural network, gating exploiting behavior with hard mathematical constraints, and overriding hallucinations with deterministic Monte Carlo simulation.

### Pillar 1: Deep CFR Blueprint (The GTO Baseline)
A 355-dimensional state vector (representing cards, board, pot geometry, and opponent ranges) is fed through a 512-width, 6-block Pre-activation ResNet. This "subconscious" layer outputs regret-matched action probabilities and Absolute Expected Value (EV) for every game state, approximating GTO play learned from 50,000 iterations of Deep Counterfactual Regret Minimization (CFR).

### Pillar 2: L2-RAG Subgame Memory
To handle computationally expensive post-flop and river decisions without cluster-level compute, NEXUS uses Retrieval-Augmented Generation (RAG). Solved subgames are stored in a FAISS flat L2 index, compressed 24x using suit isomorphism terminology. If the neural network encounters a board state structurally close to a pre-solved state, it retrieves the precise GTO solution with near-zero latency.

### Pillar 3: The Bayesian Threat Matrix
To navigate non-zero-sum multi-player dynamics without cyclic instability, NEXUS tracks a continuous Bayesian probability distribution over 169 canonical hand categories for every opponent. Actions are dynamically weighted based on pot geometry and position, feeding a single, composite Threat Matrix into the Deep CFR network.

### Pillar 4: The DDQN Exploit Agent
Pure GTO minimizes loss but leaves money on the table against predictable human players. A secondary Double Deep Q-Network (DDQN) actively tracks opponent deviations from equilibrium to find profitable exploits. To prevent reverse-exploitation, the DDQN's influence on the final action distribution is strictly hard-capped at a 35% variance limit.

### Pillars 5 & 6: The Monte Carlo Supervisor
Unconstrained neural networks hallucinate irrational plays when range estimates contain errors (e.g., hero-calling an overbet with 8-high). The MC Supervisor acts as a deterministic circuit breaker, running ~1,000 O(1) Monte Carlo equity calculations per decision using a prime-product `FastEvaluator`. 

The Supervisor enforces three strict rules:
1. **Gate A (Nash Pot Odds):** Absolute snap-folds if the required break-even math isn't met.
2. **Gate B (The River Value Ban):** Blocks aggressive 3-betting on the river without genuine showdown value.
3. **Gate C (Constructed Semi-Bluffs):** Forces all stochastic bluffs to have true mathematical "outs" or equity improvement potential (e.g., flush draws).

## Directory Structure
*   `poker_bot_api.py`: The entry point and inference engine driving the 6-pillar decision flow
*   `engine_core.py`: Core poker state mechanics and action processing
*   `networks.py`: The `NEXUS_GTO_Net` neural network architecture
*   `solver.py`: The External Sampling MCCFR logic implementation
*   `ddqn_agent.py`: The session-adaptive exploit layer
*   `range_encoder.py`: The Bayesian Threat Matrix probability tracker
*   `fast_evaluator.py`: High-performance O(1) prime-product hand evaluator
*   `advisor_gui.py` / `nexus_gui.py`: Interactive user interfaces for live play
*   `train_master.py`: The orchestrator for the Deep CFR training loop

## Usage
The source code provided here allows researchers and developers to inspect the architecture, compile the engine, study the CFR pipelines, and run the testing suite found in `tests/`.

