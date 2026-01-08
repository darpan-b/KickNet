# Component 1: Comparative study of few Deep RL algorithms on Atari Pong

## 📌 Project Overview
This component benchmarks three distinct Deep Reinforcement Learning (DRL) algorithms—**DQN (Value-Based)**, **A2C (Actor-Critic)**, and **PPO (Policy Gradient)**—on the high-dimensional discrete control task `PongNoFrameskip-v4`.

The objective is to strictly evaluate **sample efficiency**, **stability**, and **convergence behavior** under a constrained budget of 1 Million timesteps. Additionally, an ablation study comparing **Convolutional Neural Networks (CNN)** vs. **Multi-Layer Perceptrons (MLP)** is included to empirically validate the necessity of spatial feature extraction.

## 📂 Folder Structure

```text
study/
├── train_cnnpolicy.py       # Main training script (CNN Policy - 200K & 1M steps)
├── train_mlppolicy.py       # Ablation study script (MLP Policy - 200k steps)
├── plot_results.py          # Data visualization script (Parses TensorBoard logs)
├── gameplay.py              # Inference script to watch trained agents play
├── requirements.txt         # Dependencies (Stable-Baselines3, Ale-py, etc.)
├── README.md                # Project documentation
└── kicknet_atari_logs/      # (Generated post training) Stores checkpoints and event log
```

## 🚀 Installation & Setup
- Prerequisites: Python 3.8+ recommended
- Install Dependencies: ```pip install gymnasium[atari] stable-baselines3 shimmy ale-py pandas seaborn matplotlib tensorboard```

## 📊 Experimental Findings

| Experiment | Algorithm | Result (1M Steps) | Key Finding |
|-----------|-----------|-------------------|-------------|
| v2 (CNN) | DQN | Won (+3.0) | Superior sample efficiency. Solved the task. |
| v2 (CNN) | PPO | Lost (-15.0) | Stable trajectory but slower convergence. |
| v2 (CNN) | A2C | Lost (-15.0) | High volatility and lower stability. |
| v3 (MLP) | All | Failed (-21.0) | Proves spatial features (CNN) are mandatory. |

## 🛠️ Hyperparameters

**Environment:** PongNoFrameskip-v4 
- Stacked 4 frames 
- Grayscale 84×84 

**Total Timesteps:** 1,000,000 

### DQN
- Learning Rate: 1e-4 
- Replay Buffer Size: 50k 
- Exploration: 20% → 1% 

### PPO
- Learning Rate: 2.5e-4
- Batch Size: 256
- Clip Range: 0.2
- Entropy Coefficient: 0.02

### A2C
- Learning Rate: 7e-4
- Entropy Coefficient: 0.02

## 📝 License
This project was developed for an academic assessment on Deep Reinforcement Learning. Code is based on ```stable-baselines3``` documentation.
