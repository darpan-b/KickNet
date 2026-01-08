## Component 2: KickNet (3v3 Multi-Agent Soccer)

**KickNet** is a custom 2D soccer environment designed to foster coordination and competition among agents.

---

## Environment Features
- **Dynamics**: Incorporates ball velocity, friction (0.96 decay), and elastic collisions.
- **Action space**: Continuous 2D or 3D vectors representing movement and kicking.
- **Observation space**: Ranges from 5-dimensional relative vectors to 28-dimensional global state vectors.

## MARL Implementation Progression
- **Shared policy**: A basic policy gradient approach where agents share a single policy network to reduce complexity. 
- **Independent Actor-Critic (IAC)**: Each agent learns autonomously with its own policy and value networks, utilizing replay buffers to improve sample efficiency.
- **MAPPO (CTDE)**: Multi-Agent PPO using *Centralized Training with Decentralized Execution (CTDE)*. The *Critic* utilizes a global state, while *Actors* operate on local observations.

## MAPPO Advantage Calculation
MAPPO utilizes **Generalized Advantage Estimation (GAE)** with:

$$\gamma = 0.95\ and\ \lambda = 0.99$$

The temporal-difference residual is defined as:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The GAE advantage estimate is then:

$$
A_t^{\mathrm{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}
$$
