# Rea-Maac: Multi-Agent DDPG with Attention for MIMO Reactor

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

**Rea-Maac** is a **research-grade Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** implementation with **attention mechanisms**, designed for controlling a **MIMO (Multiple Input Multiple Output) reactor** in industrial process automation.

Developed as part of a **Master's thesis**, this project focuses on multi-agent coordination, learning stability, and control efficiency in complex industrial environments. While initially intended for multiple industrial scenarios, the final implementation focuses on the **MIMO reactor** due to time constraints.

**Key Contributions:**

* MADDPG applied to industrial MIMO control
* Attention mechanisms improve agent coordination and learning stability
* Supports multiple actor architectures including **RNNs and MLPs** for flexible experimentation
* Modular framework for MARL research

---

## Intuition & Problem Statement

Controlling MIMO reactors is challenging due to **highly coupled nonlinear dynamics**. Key problems include:

* Actions by one agent influence multiple outputs
* Partial and noisy observations for each agent
* Non-stationarity caused by multiple agents learning simultaneously
* Credit assignment and coordination across agents

**Solution Approach:**

* **MADDPG:** Each agent controls a part of the reactor with continuous actions
* **Attention mechanisms in critics:** Dynamically weigh other agents’ observations and actions to improve coordinated learning
* **Replay buffers:** Stabilize training by mixing experiences
* **Iterative self-play:** Gradually improve coordinated policies

**Benefits:**

* Efficient **cooperative control** for complex industrial systems
* Robust to partial observability and noisy measurements
* Flexible actor architectures (RNN, MLP) enable experimentation with temporal dependencies
* Extensible framework for other multi-agent industrial applications

---

## Technical Details

### Actor Architectures

* **RNNs:** Capture temporal dependencies and historical state information
* **MLPs:** Simpler feedforward models for stateless control
* Configurable per agent via `agent_type` parameter in training

### Critic Networks

* Centralized with attention over all agents
* Evaluate joint actions to improve gradient signal and coordination

### Training Workflow

1. Initialize actors and critics for all agents
2. Create multiple environments (normal, derivative, NMPC reward types)
3. Collect experiences from agents interacting with the environment
4. Store experiences in replay buffers
5. Update critic using attention-weighted evaluation of other agents’ actions
6. Update actor networks via policy gradients
7. Save models periodically and log metrics to Tensorboard
8. Repeat for configured number of episodes

### Logging & Visualization

* Track mean rewards over episodes
* Visualize attention weights per agent and head to interpret agent coordination
* Supports Tensorboard integration for detailed analysis

---

## Repository Structure

```
.
├── agents/                 # Actor network implementations (RNN, MLP)
├── critics/                # Centralized critic networks with attention
├── environemet/            # MIMO reactor environment simulations
├── learning_algorithms/    # MADDPG & other RL algorithms
├── models/                 # Saved model architectures and checkpoints
├── utilities/              # Logging, Tensorboard utilities, plotting functions
├── train.py                # Training script for experiments
├── tensorboard/            # Tensorboard logs
└── README.md               # This file
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/alihadi2103/Rea-Maac.git
cd Rea-Maac
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

Train MADDPG agents on the MIMO reactor with attention and configurable actor architectures:

```bash
python train.py --alg ATT_maddpg --save-path ./ --episodes 2000
```

Key parameters configurable in `train.py`:

* `agent_type`: Choose between `rnn` or `mlp` for actor network
* `attention`: Enable/disable attention in critic
* `train_episodes_num`: Total training episodes
* `save_model_freq`: Frequency of saving model checkpoints
* `batch_size`: Training batch size

Visualize performance and attention weights in Tensorboard:

```bash
tensorboard --logdir ./tensorboard/
```

---

## Benefits of the Approach

* **Coordinated Multi-Agent Control:** Agents learn to collaborate efficiently
* **Stable Learning:** Replay buffers and attention reduce variance and improve convergence
* **Temporal Dependencies:** RNN actors capture sequential patterns in reactor dynamics
* **Flexible Architecture:** Switch between RNN and MLP actors depending on environment complexity
* **Research-Ready:** Modular design supports experimentation and extension

---

## Future Improvements

* Extend to additional industrial environments
* Explore multi-objective reward optimization
* Distributed multi-agent training on multiple GPUs
* Real-time deployment simulation with safety constraints
* Hybrid model-based and model-free reinforcement learning integration

---

## License

[MIT License](LICENSE)

---

## Author

Ali Hadi — [GitHub](https://github.com/alihadi2103)
