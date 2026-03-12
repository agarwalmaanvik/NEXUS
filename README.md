# NEXUS: Deep CFR & Bounded Exploit Poker Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-1A82E2.svg)](https://github.com/facebookresearch/faiss)

> A state-of-the-art, consumer-hardware-deployable Texas Hold'em engine. NEXUS navigates non-zero-sum, imperfect-information environments by synthesizing Deep Counterfactual Regret Minimization (CFR), Bayesian inference, and variance-capped Reinforcement Learning.

**[Read the Full Technical Whitepaper Here](./NEXUS_Whitepaper.pdf)**

---

## The Architecture Problem
Pure Game Theory Optimal (GTO) solvers (e.g., *Pluribus*) require massive compute clusters for real-time subgame solving. Conversely, pure Reinforcement Learning (RL) bots collapse into cyclic instability or hallucinate irrational "balancing" plays when range estimates contain errors. 

**NEXUS** solves this through a **Tri-Layer Hybrid Architecture**:
1. **The Subconscious:** A Deep CFR neural network providing an unexploitable structural baseline.
2. **The Predator:** A session-adaptive DDQN targeting opponent inefficiencies.
3. **The Supervisor:** A deterministic Monte Carlo risk-management layer that overrides neural network hallucinations.

---

## The 6-Pillar Inference Framework

### I. Deep CFR Blueprint (The GTO Baseline)
A 355-dimensional state vector (representing board texture, pot geometry, and opponent ranges) is fed through a 512-width, 6-block Pre-activation ResNet. Learned over **50,000 iterations**, this network outputs regret-matched action probabilities and Absolute Expected Value (EV). A self-supervised Range Prediction Head forces the hidden layers to learn hand-distribution semantics.

### II. L2-RAG Subgame Memory
To bypass cluster-level compute requirements for post-flop decisions, NEXUS utilizes Retrieval-Augmented Generation (RAG). Solved subgames are stored in a FAISS flat L2 index and compressed **24x using suit isomorphism**. If the network encounters a structurally familiar board state, it retrieves the precise GTO solution with near-zero latency.

### III. The Bayesian Threat Matrix
Standard CFR destabilizes in multi-way pots. NEXUS tracks a continuous Bayesian probability distribution over 169 canonical hand categories for every opponent. Likelihoods are dynamically shifted based on bet sizing geometry, feeding a single, composite Threat Matrix into the neural network.

### IV. The DDQN Exploit Agent
Pure GTO minimizes loss but leaves EV on the table against flawed human players. A Double Deep Q-Network (DDQN) actively tracks opponent deviations from equilibrium. To prevent catastrophic reverse-exploitation, the DDQN's influence on the final action distribution is strictly **hard-capped at a 35% variance limit**.

### V & VI. The Monte Carlo Supervisor (Circuit Breakers)
Unconstrained networks hallucinate (e.g., hero-calling an overbet with 8-high). The MC Supervisor acts as a deterministic risk manager, running ~1,000 $O(1)$ equity calculations per decision via a prime-product `FastEvaluator`. It enforces three strict rules:
* **Gate A (Nash Pot Odds Supremacy):** Absolute snap-folds if required break-even math isn't met.
* **Gate B (River Value Ban):** Blocks aggressive river 3-betting without genuine showdown value.
* **Gate C (Constructed Semi-Bluffs):** Extinguishes random high-card spew, forcing all stochastic bluffs to have true mathematical "outs" (e.g., flush/straight draws).

---

## Repository Map
This repository contains the core schematic, training loop, and inference APIs.

**Core Inference & API**
* `poker_bot_api.py` — The entry point synthesizing the 6-pillar decision flow.
* `engine_core.py` — Core poker state mechanics and action processing.

**Mathematics & Evaluation**
* `range_encoder.py` — The Bayesian Threat Matrix probability tracker.
* `fast_evaluator.py` — High-performance $O(1)$ prime-product hand evaluator.
* `equity_calc.py` — Monte Carlo calculator for equity vs Bayesian ranges.

**Neural Networks & Learning**
* `networks.py` — The `NEXUS_GTO_Net` neural network architecture.
* `solver.py` — External Sampling MCCFR logic implementation.
* `ddqn_agent.py` — The session-adaptive RL exploit layer.
* `train_master.py` — Master orchestrator for the Deep CFR training loop.

**Subgame Retrieval**
* `rag_retriever.py` — L2-distance FAISS subgame strategy retrieval.

---

## IP & Usage Disclaimer
This repository serves as a structural showcase of the NEXUS architecture for technical review. To protect proprietary alpha generated during development, **the following compiled assets are excluded from this public repository:**
* The 50,000-iteration trained model weights (`.pt`).
* The solved RAG indices and GTO databases (`.db`).
* The 2.5-million runout preflop correlation matrix (`preflop_correlation_dict.py`).

Without these assets, the bot will not execute live hands. Researchers are encouraged to review the whitepaper for empirical validation logs, convergence metrics, and training methodologies.

---
**Author:** Maanvik Agarwal
