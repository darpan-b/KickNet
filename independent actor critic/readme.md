# 3v3 Multi-Agent Soccer with Independent Actor-Critic

A complete implementation of a 3v3 multi-agent soccer game using Independent Actor-Critic (IAC) reinforcement learning. Each agent learns independently to play soccer in a team environment.

## Features

- **Multi-Agent Environment**: 3v3 soccer game with realistic physics
- **Independent Actor-Critic**: Each of the 6 agents has its own actor-critic network
- **Continuous Action Space**: Smooth movement and kicking actions
- **Real-time Visualization**: Pygame-based rendering with player movements and ball physics
- **Experience Replay**: Separate replay buffers for each agent
- **Team Coordination**: Agents learn to coordinate through shared rewards

## Project Structure

```
soccer_iac/
├── soccer_env.py          # Soccer environment implementation
├── actor_critic.py        # Actor-Critic network architectures
├── replay_buffer.py       # Experience replay buffers
├── train.py              # Training script
├── evaluate.py           # Evaluation with visualization
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Details

### State Space (28 dimensions per agent)
- Agent position (x, y) - normalized
- Agent velocity (vx, vy) - normalized
- Ball position (x, y) - normalized
- Ball velocity (vx, vy) - normalized
- Teammate positions and velocities (2 agents × 4 values)
- Opponent positions and velocities (3 agents × 4 values)

### Action Space (3 dimensions per agent)
- Move X: [-1, 1] - horizontal movement
- Move Y: [-1, 1] - vertical movement
- Kick: [0, 1] - kick power

### Rewards
- Distance to ball: Negative reward for being far from ball
- Ball progress: Reward for moving ball toward opponent's goal
- Goal scored: +10 for scoring team, -10 for conceding team
- Small step penalties to encourage efficiency

## Training

Run the training script:

```bash
python train.py
```

### Training Configuration

The default configuration in `train.py`:
- Episodes: 5000
- Batch size: 128
- Replay buffer size: 100,000
- Learning rate (Actor): 3e-4
- Learning rate (Critic): 3e-4
- Discount factor (γ): 0.99
- Learning starts: 1000 steps

Models are saved every 500 episodes to `models/soccer_iac/`.

### Monitoring Training

The training script logs:
- Average rewards per team
- Average scores per team
- Episode count and total steps
- Replay buffer sizes

Example output:
```
Episode 100/5000
  Total Steps: 45231
  Avg Reward - Team1: 2.456, Team2: 2.234
  Avg Score - Team1: 0.34, Team2: 0.28
  Buffer Size: 45231
```

## Evaluation

Evaluate trained agents with visualization:

```bash
python evaluate.py --model_dir models/soccer_iac/final --num_episodes 10
```

### Evaluation Options

- `--model_dir`: Directory containing trained models (default: `models/soccer_iac/final`)
- `--num_episodes`: Number of episodes to evaluate (default: 10)
- `--no_render`: Disable visualization for faster evaluation

### Visualization Controls

- **ESC**: Exit the visualization
- The visualization shows:
  - Blue players: Team 1
  - Red players: Team 2
  - White ball with black outline
  - Current score and step count

## Game Rules

1. **Field**: 600×400 pixel rectangular field with goals on each end
2. **Teams**: 2 teams of 3 players each
3. **Objective**: Score goals by getting the ball into the opponent's goal
4. **Ball Physics**: 
   - Ball has inertia and friction
   - Players can kick when within range (20 pixels)
   - Kick power affects ball velocity
5. **Episode Length**: Maximum 1000 steps per episode

## Architecture Details

### Actor Network
- Input: 28-dimensional state
- Hidden layers: 3 × 256 units (ReLU activation)
- Output: Mean and log standard deviation for action distribution
- Actions sampled from Normal distribution with tanh squashing

### Critic Network
- Input: 28-dimensional state
- Hidden layers: 3 × 256 units (ReLU activation)
- Output: Single value estimate

### Training Algorithm

For each agent independently:
1. Sample actions from policy
2. Execute actions in environment
3. Store transitions in agent-specific replay buffer
4. Sample mini-batch from replay buffer
5. Update Critic: minimize TD error
6. Update Actor: maximize advantage-weighted log probability

## Tips for Better Performance

1. **Training Duration**: Train for at least 3000-5000 episodes for good coordination
2. **Hyperparameter Tuning**:
   - Increase learning rate for faster learning (risk: instability)
   - Decrease learning rate for stability (risk: slower convergence)
   - Adjust reward weights to encourage specific behaviors
3. **Curriculum Learning**: Start with simpler scenarios (fewer agents) and gradually increase complexity
4. **Reward Shaping**: Modify reward function in `soccer_env.py` to encourage desired behaviors

## Customization

### Modify Environment
Edit `soccer_env.py`:
- Field dimensions
- Number of agents per team
- Physics parameters (speed limits, friction)
- Reward structure

### Modify Network Architecture
Edit `actor_critic.py`:
- Hidden layer sizes
- Number of layers
- Activation functions

### Modify Training
Edit `train.py`:
- Number of episodes
- Batch size
- Learning rates
- Save intervals

## Troubleshooting

### Agents not learning
- Increase training episodes
- Adjust learning rates
- Check reward scaling
- Ensure replay buffer has enough samples before training

### Unstable training
- Reduce learning rates
- Add gradient clipping (already implemented)
- Increase batch size
- Check for NaN values in losses

### Poor coordination
- Train longer (coordination emerges over time)
- Adjust team reward structure
- Consider adding communication channels between agents

## Future Improvements

1. **Communication**: Add explicit communication between teammates
2. **Centralized Training**: Use QMIX or MADDPG for better coordination
3. **Hierarchical RL**: Add high-level strategy selection
4. **Opponent Modeling**: Learn to predict opponent actions
5. **Transfer Learning**: Pre-train on simpler tasks
6. **Self-Play**: Continuously improve by playing against past versions

## Citation

If you use this code, please cite:

```
@software{soccer_iac_2025,
  title={3v3 Multi-Agent Soccer with Independent Actor-Critic},
  author={BSB},
  year={2025}
}
```

## License

MIT License - Feel free to use for research and educational purposes.
