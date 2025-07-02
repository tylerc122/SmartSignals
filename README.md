# SmartSignals: Deep Reinforcement Learning for Traffic Optimization

A practical exploration of how modern RL techniques can dramatically improve intersection efficiency.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Phases](#project-phases)
   1. [Phase 1 – Multi-Scale Performance Validation](#phase-1)
   2. [Phase 2 – Stochastic Validation (Upcoming)](#phase-2)
   3. [Phase 3 – Multi-Agent Expansion (Upcoming)](#phase-3)
3. [Getting Started](#getting-started)
4. [Project Structure](#project-structure)
5. [Research & Methodology](#research--methodology)
6. [Roadmap](#roadmap)

---

## Introduction

Traditional traffic lights run on fixed schedules. They ignore real-time conditions, creating unnecessary stops, congestion, and emissions. This project asks a simple question:

> _Can a reinforcement-learning agent learn to run a single intersection better than a hand-crafted, static, schedule?_

The answer is **yes—by a dramatic margin.** The trained PPO agent achieves **98.6% reduction in vehicle wait times** compared to traditional fixed-time signals, with performance improving over longer time horizons. Comprehensive testing across 22.7 hours of simulated traffic validates real-world deployment potential.

---

## Methods

This project was planned to be iterative from the beginning. I started work in May 2025. Since I hadn't interacted with reinforcement learning in any capacity, I did a lot of reading on different algorithms such as PPO, (the one used in this project) DQN, TRPO etc. I didn't try to grasp the theory of it too much rather their implementations and use cases to see which algorithm would suit the project best. I eventually landed on PPO due to its stability, good sample-efficiency on discrete action spaces, and strong off-the-shelf support in Stable Baselines3.

---

## Project Phases

### Phase 1 – Multi-Scale Performance Validation <a name="phase-1"></a>

**Goal:** Train a PPO agent that decisively outperforms traditional traffic control strategies across realistic time horizons.  
**Outcome:** **98.6% reduction** in vehicle wait times with performance improving over longer time periods.

Phase 1 consisted of training one PPO agent (training details found here <- need to link training config later in readme), and creating two baseline controllers. The first baseline controller was on a fixed time schedule (30s green each direction creating a 2 minute schedule) and the other a slightly smarter fixed-time controller with different timing patterns allowing for asymmetric timing (e.g., longer green for busy directions, this is the one that most closely mimics real traffic lights).

After training and building each respective controller, a comparison script was written in order to have verifiable data that our agent was actually making good decisions within a given traffic scenario. The comparison script started out as a simple 10-minute testing period comparing all three controllers. Through an iterative process, it evolved into a more robust multi-scale validation system supporting short-term (10 minutes), medium-term (2 hours), and long-term (8 hours) testing durations. This evolution yielded much stronger evidence as the agent's performance advantages became more pronounced with longer test durations compared to the baseline controllers.

The current testing methodology ensures fairness by running all controllers under identical conditions across these three time scales: 120 steps/5 episodes, 1440 steps/3 episodes, and 5760 steps/2 episodes respectively. Each scale uses the same SUMO configuration and traffic demand patterns, with statistical measures calculated for comprehensive performance validation.

For each episode, the script collects key metrics:

- **Average waiting time** (primary performance indicator)
- **System penalty** (cumulative negative reward reflecting overall congestion)
- **Phase changes** (responsiveness measure)
- **Throughput** (vehicles successfully processed)

The comparison script automatically handles environment resets, action execution, and data logging, ensuring no human bias in the evaluation process. Results are saved to JSON files with timestamps for reproducibility.

## Results across time horizons:

_Average waiting time per vehicle (seconds):_

| Time Scale      | Duration   | RL Agent   | Adaptive Fixed | Fixed-Time | **RL Improvement**                  |
| --------------- | ---------- | ---------- | -------------- | ---------- | ----------------------------------- |
| **Short-term**  | 10 minutes | **0.074s** | 2.313s         | 2.812s     | **97.4% reduction in waiting time** |
| **Medium-term** | 2 hours    | **0.018s** | 0.972s         | 1.122s     | **98.4% reduction in waiting time** |
| **Long-term**   | 8 hours    | **0.004s** | 0.243s         | 0.280s     | **98.6% reduction in waiting time** |

**Key Insights:**

- **Performance improves with longer time horizons** – the agent optimizes better for sustained traffic patterns
- **98.6% wait time reduction** achieved over 8-hour simulations – exceeding initial 50% improvement targets
- **139× better system performance** (reward ratio) demonstrates massive efficiency gains
- **22.7 total hours simulated** across all tests for robust testing

<p align="center">
  <img alt="Multi-Scale Performance Comparison" src="results/phase_1_multi_scale_validation/visualizations/performance_comparison.png" width="80%">
</p>

<p align="center">
  <img alt="Performance Scaling Over Time" src="results/phase_1_multi_scale_validation/visualizations/performance_scaling.png" width="80%">
</p>

The multi-scale analysis proves that the RL agent not only provides immediate improvements but maintains and enhances performance over extended operational periods.

### Phase 2 – Stochastic Validation (Upcoming) <a name="phase-2"></a>

- Generate 100 + traffic scenarios with varying demand patterns.
- Validate that improvements hold for worst-case (maximum) wait times.

### Phase 3 – Multi-Agent Expansion (Upcoming) <a name="phase-3"></a>

- Scale to a corridor or grid of intersections.
- Investigate cooperative versus independent control strategies.

---

## Getting Started <a name="getting-started"></a>

```bash
# clone repository
$ git clone https://github.com/tylerc122/SmartSignals.git
$ cd SmartSignals

# install dependencies (Python 3.8+)
$ pip install -r requirements.txt

# run complete multi-scale analysis
$ python run_multi_scale_analysis.py

# or run individual components
$ python src/evaluation/multi_scale_comparison.py
$ python src/evaluation/create_clean_charts.py
```

Training scripts (`src/training/`) allow you to retrain or experiment with alternative algorithms.

---

## Project Structure <a name="project-structure"></a>

```
SmartSignals/
├── src/
│   ├── environments/   # SUMO-based Gymnasium environment
│   ├── agents/         # RL and baseline controllers
│   ├── training/       # Training scripts
│   ├── evaluation/     # Comparison & visualization tools
│   └── utils/          # Helper functions
├── sumo_scenarios/     # Network / route definitions
├── results/
│   ├── phase_1_multi_scale_validation/  # Multi-scale test results
│   └── visualizations/                  # Generated charts
├── configs/            # Training configuration files
├── models/             # Saved RL models (git-ignored)
├── data/               # Raw data storage
├── logs/               # Training logs (git-ignored)
├── tests/              # Unit tests
├── run_multi_scale_analysis.py  # Main analysis runner
├── test_environment.py          # Environment testing
└── start_gui.sh                 # SUMO GUI launcher
```

---

## Research & Methodology <a name="research--methodology"></a>

### Environment Design

- **State space:** 20-dimensional vector (vehicle counts, wait times, light phase).
- **Action space:** Discrete (4) – choose one of four traffic-light phases.
- **Reward:** negative sum of vehicle waiting times each step.
- **Simulator:** SUMO via TraCI, 5-second simulation steps, 300-second episodes (60 steps).

### Training Configuration

The model was trained for 100,000 timesteps (approximately 139 simulated hours of traffic). Training converged after ~80,000 timesteps, with performance stabilizing as the agent mastered optimal traffic flow patterns.

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

_Last updated 2025-07-02_
