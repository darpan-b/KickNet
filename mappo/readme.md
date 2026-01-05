# 3v3 Multi-Agent Soccer with MAPPO

A complete implementation of a 3v3 multi-agent soccer game using **MAPPO (Multi-Agent Proximal Policy Optimization)** with Centralized Training, Decentralized Execution (CTDE). Each agent learns to play soccer through advanced reward shaping and coordinated team play.

## 🌟 Key Features

### MAPPO Implementation
- **Centralized Training, Decentralized Execution (CTDE)**: Agents use local observations during execution but share a centralized critic during training
- **Parameter Sharing**: Optional parameter sharing within teams for improved sample efficiency
- **PPO Clipping**: Stable policy updates with ε=0.2 clipping
- **GAE (Generalized Advantage Estimation)**: λ=0.95 for better variance-bias tradeoff
- **Entropy Bonus**: Encourages exploration during training

### Advanced Reward Shaping
- **Potential-Based Shaping**: Ball progression rewards using φ(s) = -distance(ball, goal)
- **Passing Bonus**: +0.5 reward for successful team passes
- **Pressure Penalty**: -0.02 for opponents threatening your goal
- **Effort Penalty**: -0.01 per step to encourage quick scoring
- **Sparse Goal Rewards**: ±10 for scoring/conceding goals

### Environment Features
- Realistic physics with ball friction and collision
- 600×400 pixel field with goals
- Continuous action space for smooth movement
- 28-dimensional local observations per agent
- 28-dimensional global state for centralized critic

## 📦 Project Structure

```
soccer_mappo/
├── soccer_env.py          # Enhanced environment with reward shaping
├── actor_critic.py        # MAPPO networks (Actor + Centralized Critic)
├── replay_buffer.py       # Rollout buffer for on-policy learning
├── train.py              # MAPPO training script
├── evaluate.py           # Evaluation with visualization
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 State and Action Spaces

### Local Observation (28D per agent)
Used by **decentralized Actor** during execution:
- Self position (x, y) - normalized to [-1, 1]
- Self velocity (vx, vy) - normalized
- Ball position (x, y) - normalized
- Ball velocity (vx, vy) - normalized
- Teammate positions & velocities (2 agents × 4 values)
- Opponent positions & velocities (3 agents × 4 values)

### Global State (28D)
Used by **centralized Critic** during training:
- All 6 agent positions and velocities (6 × 4 = 24D)
- Ball position and velocity (4D)

### Action Space (3D per agent)
- Move X: [-1, 1] - horizontal movement
- Move Y: [-1, 1] - vertical movement
- Kick: [0, 1] - kick power

## 🎯 Reward Structure

The reward function implements potential-based shaping:

$$R'(s, a, s') = R(s, a) + \gamma \Phi(s') - \Phi(s)$$

Where $\Phi(s) = -\text{distance}(\text{ball}, \text{opponent\_goal})$

### Reward Components

| Component | Value | Description |
|-----------|-------|-------------|
| **Goal Scored** | +10.0 | When your team scores |
| **Goal Conceded** | -10.0 | When opponent scores |
| **Ball Progression** | 0.1 × Δφ | Moving ball toward opponent goal |
| **Passing Bonus** | +0.5 | Successful pass to teammate |
| **Pressure Penalty** | -0.02 | Opponent near your goal with ball |
| **Ball Proximity** | -dist/1000 | Encourages moving toward ball |
| **Effort Penalty** | -0.01 | Per step (encourages efficiency) |

## 🏋️ Training

Run the MAPPO training script:

```bash
python train.py
```

### Training Configuration

Default hyperparameters in `train.py`:

```python
config = {
    'num_episodes': 3000,        # Total episodes
    'rollout_steps': 1000,       # Steps per rollout
    'batch_size': 128,           # Mini-batch size
    'ppo_epochs': 4,             # PPO update epochs
    'lr_actor': 3e-4,           # Actor learning rate
    'lr_critic': 3e-4,          # Critic learning rate
    'gamma': 0.99,               # Discount factor
    'gae_lambda': 0.95,         # GAE lambda
    'clip_epsilon': 0.2,        # PPO clip range
    'entropy_coef': 0.01,       # Entropy bonus
    'value_coef': 0.5,          # Value loss weight
    'max_grad_norm': 0.5,       # Gradient clipping
    'parameter_sharing': True,   # Share params within teams
}
```

### Training Output

```
Episode 100/3000
  Total Steps: 100000
  Avg Score - Team1: 0.45, Team2: 0.38
  Episodes this interval: 8
  Team1 - Actor Loss: 0.0234, Critic Loss: 0.1456, Entropy: 1.234
  Team2 - Actor Loss: 0.0198, Critic Loss: 0.1389, Entropy: 1.198
```

Models are automatically saved every 500 episodes to `models/soccer_mappo/`.

## 🎮 Evaluation

Evaluate trained agents with real-time visualization:

```bash
python evaluate.py --model_dir models/soccer_mappo/final --num_episodes 10
```

### Evaluation Options

```bash
# Basic evaluation
python evaluate.py

# Custom model directory
python evaluate.py --model_dir models/soccer_mappo/checkpoint_ep1500

# More episodes
python evaluate.py --num_episodes 50

# Without rendering (faster)
python evaluate.py --no_render

# Without parameter sharing (if trained that way)
python evaluate.py --no_parameter_sharing
```

### Visualization Controls

- **ESC**: Exit visualization
- **Visual Elements**:
  - Blue circles: Team 1 players (numbered 1-3)
  - Red circles: Team 2 players (numbered 1-3)
  - White circle with black outline: Ball
  - Score and step counter at top

### Evaluation Output

```
Episode 1: Team1 2 - 1 Team2
  Avg Reward - Team1: 3.45, Team2: 2.87

Episode 2: Team1 1 - 1 Team2
  Avg Reward - Team1: 2.98, Team2: 3.12

==================================================
Evaluation Summary
==================================================
Total Episodes: 10
Average Score - Team1: 1.50, Team2: 1.20
Team1 Wins: 6, Team2 Wins: 3, Draws: 1
==================================================
```

## 🧠 MAPPO Architecture Details

### Actor Network (Decentralized)
- **Input**: 28D local observation
- **Hidden layers**: 3 × 256 units (ReLU)
- **Output**: Mean and log_std for action distribution
- **Distribution**: Normal with tanh squashing
- **Usage**: Each agent has its own actor (or shares with team)

### Centralized Critic Network
- **Input**: 28D global state (all agents + ball)
- **Hidden layers**: 3 × 256 units (ReLU)
- **Output**: Single value estimate
- **Usage**: Shared across all agents during training

### Training Algorithm (MAPPO)

For each rollout:
1. **Collection Phase** (Decentralized Execution):
   - Each agent selects actions using local observations
   - Store (state, global_state, action, reward, log_prob, value, done)

2. **Update Phase** (Centralized Training):
   - Compute GAE advantages using centralized critic
   - Perform PPO updates with clipping
   - Update both actor (local obs) and critic (global state)

3. **Key Difference from IAC**:
   - Critic sees global state → better coordination
   - Actor still uses local obs → scales to execution
   - Parameter sharing → sample efficiency

## 📈 Expected Training Progress

### Early Training (Episodes 0-500)
- Agents learn basic movement
- Random kicking and positioning
- Scores typically 0-1 per game
- High exploration (high entropy)

### Mid Training (Episodes 500-1500)
- Agents start coordinating
- Basic passing emerges
- Defensive positioning improves
- Scores increase to 1-2 per game

### Late Training (Episodes 1500-3000)
- Complex team strategies
- Consistent passing and coordination
- Strategic positioning
- Scores stabilize at 2-3 per game

## 🔧 Hyperparameter Tuning Guide

### For Faster Learning
- Increase `lr_actor` and `lr_critic` to 5e-4
- Increase `rollout_steps` to 2000
- Risk: Training instability

### For More Stable Training
- Decrease learning rates to 1e-4
- Increase `ppo_epochs` to 8
- Decrease `clip_epsilon` to 0.1
- Risk: Slower convergence

### For Better Exploration
- Increase `entropy_coef` to 0.05
- Use higher initial temperature
- Risk: Slower convergence to optimal policy

### For Better Coordination
- Enable `parameter_sharing`
- Increase `rollout_steps`
- Tune reward shaping weights

## 🎓 Key MAPPO Concepts Explained

### 1. Centralized Training, Decentralized Execution (CTDE)
**Training**: Critic sees global state → better value estimates
**Execution**: Actor uses only local observations → scalable

### 2. Generalized Advantage Estimation (GAE)
Balances bias and variance in advantage computation:

$$A^{\text{GAE}}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### 3. PPO Clipping
Prevents large policy updates:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$

### 4. Parameter Sharing
- One set of weights for all agents in a team
- Reduces parameters from 6 agents to 2 teams
- Improves sample efficiency
- Natural for homogeneous agents

## 🆚 MAPPO vs IAC Comparison

| Feature | Independent AC | MAPPO |
|---------|---------------|-------|
| **Critic Input** | Local observation | Global state |
| **Coordination** | Implicit only | Explicit via centralized critic |
| **Sample Efficiency** | Lower | Higher (with param sharing) |
| **Scalability** | Better | Good (decentralized execution) |
| **Training Stability** | Lower | Higher (PPO clipping) |
| **Convergence Speed** | Slower | Faster |

## 🔬 Advanced Modifications

### 1. Add Communication Channels
```python
# In actor network, add communication layer
self.comm_encoder = nn.Linear(state_dim, comm_dim)
self.comm_decoder = nn.Linear(comm_dim * num_teammates, hidden_dim)
```

### 2. Hierarchical Policies
```python
# High-level: Select strategy (attack/defend)
# Low-level: Execute movement given strategy
```

### 3. Opponent Modeling
```python
# Add opponent prediction module
self.opponent_model = nn.LSTM(obs_dim, hidden_dim)
```

### 4. Curriculum Learning
```python
# Start with fewer opponents, gradually increase
if episode < 1000:
    num_opponents = 1
elif episode < 2000:
    num_opponents = 2
else:
    num_opponents = 3
```

## 🐛 Troubleshooting

### Training is unstable
- Reduce learning rates
- Increase gradient clipping (`max_grad_norm`)
- Reduce `clip_epsilon`
- Check for NaN values in losses

### Agents not coordinating
- Enable `parameter_sharing`
- Increase `rollout_steps`
- Verify centralized critic is receiving global state
- Tune passing bonus higher

### Low scores/poor performance
- Train longer (3000+ episodes)
- Adjust reward shaping weights
- Increase entropy coefficient for more exploration
- Check if agents are getting stuck in local minima

### Slow training
- Use GPU if available
- Increase `batch_size`
- Reduce `ppo_epochs`
- Enable parameter sharing

## 📚 References

1. **MAPPO**: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2021)
2. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
3. **GAE**: Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
4. **Reward Shaping**: Ng et al. "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping" (1999)

## 📝 Citation

```bibtex
@software{soccer_mappo_2025,
  title={3v3 Multi-Agent Soccer with MAPPO},
  author={Your Name},
  year={2025},
  description={MAPPO implementation for multi-agent soccer with centralized training and decentralized execution}
}
```

## 📄 License

MIT License - Free for research and educational purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Communication between agents
- More complex reward structures
- Different team sizes
- Tournament mode with multiple teams
- Self-play training