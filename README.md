# SmartSignals: Deep Reinforcement Learning for Traffic Optimization

A practical exploration of how modern RL techniques can dramatically improve intersection efficiency.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Phases](#project-phases)
   1. [Phase 1 – Single-Intersection Baseline](#phase-1)
   2. [Phase 2 – Stochastic Validation (Upcoming)](#phase-2)
   3. [Phase 3 – Multi-Agent Expansion (Upcoming)](#phase-3)
3. [Phase 1 Results](#phase-1-results)
4. [Getting Started](#getting-started)
5. [Project Structure](#project-structure)
6. [Research & Methodology](#research--methodology)
7. [Roadmap](#roadmap)

---

## Introduction

Traditional traffic lights run on fixed schedules. They ignore real-time conditions, creating unnecessary stops, congestion, and emissions.  
This project asks a simple question:

> _Can a reinforcement-learning agent learn to run a single intersection better than a hand-crafted schedule?_

The answer (so far) is **yes—by a wide margin.**

---

## Methods

This project was planned to be iterative from the beginning. I started work in May 2025, since I hadn't interacted with reinforcement learning in any capacity, I did a lot of reading on different algorithms such as PPO, (the one initially used in this project) DQN, TRPO etc. I didn't try to grasp the theory of it too much rather their implementations and use cases to see which algorithm would suit the project best. I eventually landed on PPO due to its stability, good sample-efficiency on discrete action spaces, and strong off-the-shelf support in Stable Baselines3.

---

## Project Phases

### Phase 1 – Single-Intersection Baseline <a name="phase-1"></a>

- Goal: Train a PPO agent that decisively outperforms traditional fixed-time controllers on a single four-way intersection.
- Outcome: **97 % reduction** in average vehicle wait time and **64×** better reward signal.

### Phase 2 – Stochastic Validation (Upcoming) <a name="phase-2"></a>

- Generate 100 + traffic scenarios with varying demand patterns.
- Validate that improvements hold for worst-case (maximum) wait times.

### Phase 3 – Multi-Agent Expansion (Upcoming) <a name="phase-3"></a>

- Scale to a corridor or grid of intersections.
- Investigate cooperative versus independent control strategies.

---

## Phase 1 Results <a name="phase-1-results"></a>

| Controller              | Avg Wait Time | System Penalty (lower better) | Phase Changes |
| ----------------------- | ------------: | ----------------------------: | ------------: |
| **RL Agent (PPO)**      |    **0.07 s** |                       **-31** |            23 |
| Fixed-Time (30 s cycle) |        2.81 s |                         ‑1992 |             3 |
| Adaptive Fixed-Time     |        2.31 s |                         ‑1571 |             3 |

> **Key takeaway:** the agent almost eliminates waiting while remaining highly responsive.

<p align="center">
  <img alt="Performance Comparison" src="results/visualizations/phase_1/improvement_showcase.png" width="70%">
</p>

Additional visualisations (bar, radar, episode plots) are available in `results/visualizations` for deeper analysis.

---

## Getting Started <a name="getting-started"></a>

```bash
# clone repository
$ git clone https://github.com/yourusername/Smart_Signals.git
$ cd Smart_Signals

# install dependencies (Python 3.8+)
$ pip install -r requirements.txt

# run the Phase 1 comparison
$ python src/evaluation/compare_controllers.py

# generate the visualisations
$ python src/evaluation/create_visualizations.py
```

Training scripts (`src/training/`) allow you to retrain or experiment with alternative algorithms.

---

## Project Structure <a name="project-structure"></a>

```
Smart_Signals/
├── src/
│   ├── environments/   # SUMO-based Gymnasium environment
│   ├── agents/         # RL and baseline controllers
│   ├── training/       # Training scripts
│   └── evaluation/     # Comparison & visualisation tools
├── sumo_scenarios/     # Network / route definitions
├── results/            # Logged metrics & charts
└── models/             # Saved RL models (git-ignored)
```

---

## Research & Methodology <a name="research--methodology"></a>

- **State space:** 20-dimensional vector (vehicle counts, wait times, light phase).
- **Action space:** Discrete (4) – choose one of four traffic-light phases.
- **Reward:** negative sum of vehicle waiting times each step.
- **Algorithm:** PPO from Stable Baselines3.
- **Simulator:** SUMO via TraCI, 5 s simulation step, 120-step episodes.

The agent was trained for ~100k timesteps and evaluated over multiple independent episodes.

---

## Roadmap <a name="roadmap"></a>

1. **Phase 2 – Stochastic Validation**  
   • traffic generator  
   • 100 + scenario benchmarking  
   • confidence intervals on max-wait reduction
2. **Phase 3 – Multi-Agent Control**  
   • multi-intersection network  
   • coordination strategies  
   • scalability analysis
3. **Algorithm Benchmarks** (DQN, A2C, SAC, etc.)

_Last updated 2025-07-01_
