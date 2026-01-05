🎮 Multi-Agent 3v3 Soccer (Competitive Reinforcement Learning)
📘 Overview

This project implements a 3 vs 3 competitive multi-agent reinforcement learning (MARL) soccer simulation.
Each of the six players is controlled by its own learned policy and competes in a dynamic soccer environment using self-play PPO.

The project demonstrates:

Emergent teamwork

Strategy formation

Competitive & cooperative behavior

A live UI match viewer where agents play continuously until manually closed

🎯 Objective

Build an autonomous 3v3 soccer AI where:

6 independent agents learn to play soccer competitively

Agents train via self-play and learn offensive + defensive strategies

The trained model can be evaluated in a visual soccer field UI

The UI runs continuously until the user closes the window/terminal

🧩 Concept
Each agent:

Observes:

Its position, velocity

Ball position

Teammates & opponents’ positions

Game score and time

Acts:

Move Up / Down / Left / Right

Dash / Sprint

Kick / Pass

Idle

Receives rewards based on:

Goals scored

Successful passes

Defensive stops

Ball possession

Fouls or collisions

Training uses:

Centralized critic, decentralized actors (CTDE)

Self-play PPO

Optional league training later

🏟️ 3v3 Soccer Environment Setup
Environment Features

3 Agents vs 3 Agents

Continuous 2D Soccer Field

Physics-based ball movement

Collision detection

Reward shaping for passes, goals, possession

Built using PettingZoo ParallelEnv API

Observations (per agent)

[x, y, vx, vy] of the agent

[x, y] of ball

[x, y] of teammates

[x, y] of opponents

Actions
Action	Meaning
0	Move Up
1	Move Down
2	Move Left
3	Move Right
4	Dash
5	Kick
6	Pass
7	Idle
⚙️ Algorithms Implemented
Algorithm	Description
Self-Play PPO	Agents train by playing against copies of themselves
Centralized Critic	One shared critic for stability
Decentralized Actors	Independent action policies
Curriculum Learning	Start with simple ball-chasing → full 3v3
🧠 Training Architecture
Training Flow
┌────────────────────────────────────────────────────────┐
│ Initialize 3v3 soccer environment                      │
├────────────────────────────────────────────────────────┤
│ For each episode:                                       │
│   • All 6 agents observe state                          │
│   • Agents take actions via PPO policy                  │
│   • Environment updates physics and ball movement       │
│   • Rewards assigned (goals, passes, possession, etc.)  │
│   • Store transitions in replay buffer                  │
│   • PPO update occurs after rollout length              │
└────────────────────────────────────────────────────────┘

📁 Directory Structure
├── envs/
│   └── soccer_env_3v3.py       # 3v3 soccer simulation
│
├── agents/
│   ├── ppo_agent.py            # PPO decentralized actors
│   ├── centralized_critic.py   # Shared critic network
│   └── selfplay_manager.py     # Self-play policy handling
│
├── training/
│   └── train_selfplay.py       # Main training loop
│
├── evaluation/
│   └── evaluate_match.py       # Runs UI 3v3 match viewer
│
├── ui/
│   └── soccer_viewer.py        # Live UI using pygame
│
├── results/                    # logs, graphs, training curves
├── models/                     # PPO saved weights
└── main.py                     # CLI runner

🎮 Live UI Viewer (3v3 Soccer)

After training, you can visualize the match where:

All 6 agents appear on the field

Ball moves based on physics

Scoreboard updates in real-time

Agents move, pass, defend

The UI stays open until you close the window / kill the terminal

The UI is built using pygame.

🚀 How to Run the Project
1️⃣ Install Dependencies
conda create -n marl_soccer python=3.10 -y
conda activate marl_soccer

pip install torch gymnasium pettingzoo stable-baselines3 pygame tensorboard matplotlib

2️⃣ Train the 3v3 Soccer Agents
python training/train_selfplay.py \
    --episodes 5000 \
    --rollout-length 256 \
    --log-dir results/tensorboard \
    --checkpoint-dir models \
    --save-interval 100

3️⃣ Evaluate the Trained Model (Runs the UI)
python evaluation/evaluate_match.py --model models/soccer_ppo_final.pth


➡️ This will open a soccer field UI showing all 6 agents playing.
➡️ The match continues until you manually close the pygame window or press CTRL+C.

📈 Evaluation Metrics
Metric	Meaning
Win Rate	Percent of matches won vs. previous policies
Goals Scored	Number of goals per episode
Pass Accuracy	% of completed passes
Possession Time	Ball control percentage
Reward Stability	Convergence of PPO training
📚 References

PettingZoo MARL Framework

PPO (Schulman et al., 2017)

AlphaStar (DeepMind, 2019)

Multi-Agent RL (Lowe et al., MADDPG, 2017)

👨‍💻 Contributors

Ronak Sarkar – Project Lead (RL + MARL + Simulation)

Group RR – Supporting Research and Development

🪙 License

MIT License © 2025 Ronak Sarkar

You may use, modify, and distribute this work with attribution.
