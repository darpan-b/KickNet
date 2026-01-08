import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SoccerEnv(gym.Env):
    """3v3 Multi-Agent Soccer Environment with MAPPO support"""
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Field dimensions
        self.field_width = 600
        self.field_height = 400
        self.goal_width = 100
        self.goal_depth = 20
        
        # Game parameters
        self.num_agents_per_team = 3
        self.max_steps = 1000
        self.ball_speed_limit = 20.0
        self.player_speed_limit = 9.0
        self.kick_range = 20.0
        self.kick_power = 10.0
        
        # Reward shaping parameters
        self.ball_progress_scale = 0.1
        self.passing_bonus = 0.5
        self.pressure_penalty = -0.02
        self.effort_penalty = -0.01
        
        # Observation and action spaces (per agent)
        # Local obs: [self_x, self_y, self_vx, self_vy, ball_x, ball_y, ball_vx, ball_vy, 
        #             teammates (2x4), opponents (3x4)]
        obs_dim = 8 + 2*4 + 3*4  # 28 dimensions
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Global state dimension for centralized critic
        # All players (6x4) + ball (4) = 28 dimensions
        self.global_state_dim = 6*4 + 4
        
        # Actions: [move_x, move_y, kick]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]), 
            high=np.array([1, 1, 1]), 
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.last_kicker = None  # Track last agent who kicked the ball
        self.last_ball_to_goal_dist = None  # For potential-based shaping
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize ball at center
        self.ball_pos = np.array([self.field_width/2, self.field_height/2], dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        
        # Initialize team 1 (blue) positions - left side
        self.team1_pos = np.array([
            [150, 100],   # Defender
            [150, 200],   # Midfielder
            [150, 300],   # Forward
        ], dtype=np.float32)
        self.team1_vel = np.zeros((3, 2), dtype=np.float32)
        
        # Initialize team 2 (red) positions - right side
        self.team2_pos = np.array([
            [450, 100],   # Forward
            [450, 200],   # Midfielder
            [450, 300],   # Defender
        ], dtype=np.float32)
        self.team2_vel = np.zeros((3, 2), dtype=np.float32)
        
        self.steps = 0
        self.score = [0, 0]  # [team1, team2]
        self.done = False
        self.last_kicker = None
        
        # Initialize potential function values
        self.last_ball_to_goal_dist = {
            'team1': np.linalg.norm(self.ball_pos - np.array([self.field_width, self.field_height/2])),
            'team2': np.linalg.norm(self.ball_pos - np.array([0, self.field_height/2]))
        }
        
        return self._get_obs(), self._get_global_state(), {}
    
    def _get_potential(self, team):
        """Potential function: negative distance to opponent's goal"""
        if team == 'team1':
            goal_pos = np.array([self.field_width, self.field_height/2])
        else:
            goal_pos = np.array([0, self.field_height/2])
        
        return -np.linalg.norm(self.ball_pos - goal_pos)
    
    def _get_obs(self):
        """Get local observations for all agents"""
        obs = {}
        
        # Normalize positions and velocities
        def normalize_pos(pos):
            return np.array([
                2 * pos[0] / self.field_width - 1,
                2 * pos[1] / self.field_height - 1
            ])
        
        def normalize_vel(vel):
            return vel / self.player_speed_limit
        
        ball_pos_norm = normalize_pos(self.ball_pos)
        ball_vel_norm = self.ball_vel / self.ball_speed_limit
        
        # Team 1 observations
        for i in range(3):
            agent_pos = normalize_pos(self.team1_pos[i])
            agent_vel = normalize_vel(self.team1_vel[i])
            
            # Teammates
            teammates = []
            for j in range(3):
                if i != j:
                    teammates.extend(normalize_pos(self.team1_pos[j]))
                    teammates.extend(normalize_vel(self.team1_vel[j]))
            
            # Opponents
            opponents = []
            for j in range(3):
                opponents.extend(normalize_pos(self.team2_pos[j]))
                opponents.extend(normalize_vel(self.team2_vel[j]))
            
            obs[f'team1_agent{i}'] = np.concatenate([
                agent_pos, agent_vel, ball_pos_norm, ball_vel_norm,
                teammates, opponents
            ]).astype(np.float32)
        
        # Team 2 observations
        for i in range(3):
            agent_pos = normalize_pos(self.team2_pos[i])
            agent_vel = normalize_vel(self.team2_vel[i])
            
            # Teammates
            teammates = []
            for j in range(3):
                if i != j:
                    teammates.extend(normalize_pos(self.team2_pos[j]))
                    teammates.extend(normalize_vel(self.team2_vel[j]))
            
            # Opponents
            opponents = []
            for j in range(3):
                opponents.extend(normalize_pos(self.team1_pos[j]))
                opponents.extend(normalize_vel(self.team1_vel[j]))
            
            obs[f'team2_agent{i}'] = np.concatenate([
                agent_pos, agent_vel, ball_pos_norm, ball_vel_norm,
                teammates, opponents
            ]).astype(np.float32)
        
        return obs
    
    def _get_global_state(self):
        """Get global state for centralized critic"""
        def normalize_pos(pos):
            return np.array([
                2 * pos[0] / self.field_width - 1,
                2 * pos[1] / self.field_height - 1
            ])
        
        def normalize_vel(vel):
            return vel / self.player_speed_limit
        
        # Concatenate all agent states and ball state
        global_state = []
        
        # Team 1 agents
        for i in range(3):
            global_state.extend(normalize_pos(self.team1_pos[i]))
            global_state.extend(normalize_vel(self.team1_vel[i]))
        
        # Team 2 agents
        for i in range(3):
            global_state.extend(normalize_pos(self.team2_pos[i]))
            global_state.extend(normalize_vel(self.team2_vel[i]))
        
        # Ball
        global_state.extend(normalize_pos(self.ball_pos))
        global_state.extend(self.ball_vel / self.ball_speed_limit)
        
        return np.array(global_state, dtype=np.float32)
    
    def step(self, actions):
        """Execute actions for all agents"""
        self.steps += 1
        
        # Parse actions
        team1_actions = [actions[f'team1_agent{i}'] for i in range(3)]
        team2_actions = [actions[f'team2_agent{i}'] for i in range(3)]
        
        # Update player velocities and positions
        for i in range(3):
            # Team 1
            move = team1_actions[i][:2] * self.player_speed_limit
            self.team1_vel[i] = move
            self.team1_pos[i] += self.team1_vel[i]
            self.team1_pos[i] = np.clip(self.team1_pos[i], 
                                        [10, 10], 
                                        [self.field_width-10, self.field_height-10])
            
            # Team 2
            move = team2_actions[i][:2] * self.player_speed_limit
            self.team2_vel[i] = move
            self.team2_pos[i] += self.team2_vel[i]
            self.team2_pos[i] = np.clip(self.team2_pos[i], 
                                        [10, 10], 
                                        [self.field_width-10, self.field_height-10])
        
        # Track who kicked the ball for passing bonus
        current_kicker = None
        
        # Handle kicking
        for i in range(3):
            # Team 1 kicks
            if team1_actions[i][2] > 0.5:
                dist = np.linalg.norm(self.team1_pos[i] - self.ball_pos)
                if dist < self.kick_range:
                    direction = self.ball_pos - self.team1_pos[i]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        self.ball_vel = direction * self.kick_power * team1_actions[i][2]
                        current_kicker = ('team1', i)
            
            # Team 2 kicks
            if team2_actions[i][2] > 0.5:
                dist = np.linalg.norm(self.team2_pos[i] - self.ball_pos)
                if dist < self.kick_range:
                    direction = self.ball_pos - self.team2_pos[i]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        self.ball_vel = direction * self.kick_power * team2_actions[i][2]
                        current_kicker = ('team2', i)
        
        # Update ball position
        self.ball_pos += self.ball_vel
        self.ball_vel *= 0.95  # Friction
        
        # Ball collision with walls
        if self.ball_pos[1] < 0 or self.ball_pos[1] > self.field_height:
            self.ball_vel[1] *= -0.8
            self.ball_pos[1] = np.clip(self.ball_pos[1], 0, self.field_height)
        
        # Calculate rewards with shaping
        rewards = self._calculate_shaped_rewards(current_kicker)
        
        # Check for goals
        goal_scored = False
        if self.ball_pos[0] < 0:  # Team 2 scores
            goal_y = self.field_height / 2
            if abs(self.ball_pos[1] - goal_y) < self.goal_width / 2:
                self.score[1] += 1
                goal_scored = True
                rewards = self._goal_rewards(team=2)
        
        if self.ball_pos[0] > self.field_width:  # Team 1 scores
            goal_y = self.field_height / 2
            if abs(self.ball_pos[1] - goal_y) < self.goal_width / 2:
                self.score[0] += 1
                goal_scored = True
                rewards = self._goal_rewards(team=1)
        
        # Reset ball after goal
        if goal_scored:
            self.ball_pos = np.array([self.field_width/2, self.field_height/2])
            self.ball_vel = np.zeros(2)
            self.last_kicker = None
            self.last_ball_to_goal_dist = {
                'team1': np.linalg.norm(self.ball_pos - np.array([self.field_width, self.field_height/2])),
                'team2': np.linalg.norm(self.ball_pos - np.array([0, self.field_height/2]))
            }
        else:
            # Update last kicker
            if current_kicker is not None:
                self.last_kicker = current_kicker
        
        # Clip ball to field
        self.ball_pos[0] = np.clip(self.ball_pos[0], 0, self.field_width)
        
        # Check termination
        terminated = self.steps >= self.max_steps
        truncated = False
        
        obs = self._get_obs()
        global_state = self._get_global_state()
        infos = {agent: {"score": self.score} for agent in obs.keys()}
        
        dones = {agent: terminated for agent in obs.keys()}
        dones['__all__'] = terminated
        
        return obs, global_state, rewards, dones, infos
    
    def _calculate_shaped_rewards(self, current_kicker):
        """Calculate rewards with potential-based shaping"""
        rewards = {}
        
        # Current potentials
        current_potential_team1 = self._get_potential('team1')
        current_potential_team2 = self._get_potential('team2')
        
        # Ball progression rewards (potential-based)
        ball_progress_team1 = self.ball_progress_scale * (
            current_potential_team1 - self.last_ball_to_goal_dist['team1']
        )
        ball_progress_team2 = self.ball_progress_scale * (
            current_potential_team2 - self.last_ball_to_goal_dist['team2']
        )
        
        # Update last potential
        self.last_ball_to_goal_dist['team1'] = current_potential_team1
        self.last_ball_to_goal_dist['team2'] = current_potential_team2
        
        # Passing bonus
        passing_bonus_team1 = 0
        passing_bonus_team2 = 0
        
        if current_kicker is not None and self.last_kicker is not None:
            if (current_kicker[0] == self.last_kicker[0] and 
                current_kicker[1] != self.last_kicker[1]):
                # Successful pass within same team
                if current_kicker[0] == 'team1':
                    passing_bonus_team1 = self.passing_bonus
                else:
                    passing_bonus_team2 = self.passing_bonus
        
        # Pressure penalty (opponent near own goal with ball)
        pressure_team1 = 0
        pressure_team2 = 0
        
        # Check if opponents are threatening
        team1_goal = np.array([0, self.field_height/2])
        team2_goal = np.array([self.field_width, self.field_height/2])
        
        ball_near_team1_goal = np.linalg.norm(self.ball_pos - team1_goal) < 150
        ball_near_team2_goal = np.linalg.norm(self.ball_pos - team2_goal) < 150
        
        # Check if opponent has possession (close to ball)
        team2_has_ball = any(np.linalg.norm(self.team2_pos[i] - self.ball_pos) < 30 for i in range(3))
        team1_has_ball = any(np.linalg.norm(self.team1_pos[i] - self.ball_pos) < 30 for i in range(3))
        
        if ball_near_team1_goal and team2_has_ball:
            pressure_team1 = self.pressure_penalty
        if ball_near_team2_goal and team1_has_ball:
            pressure_team2 = self.pressure_penalty
        
        # Team 1 rewards
        for i in range(3):
            dist_to_ball = np.linalg.norm(self.team1_pos[i] - self.ball_pos)
            
            reward = self.effort_penalty  # Small penalty for each step
            reward += -dist_to_ball / 1000.0  # Proximity to ball
            reward += ball_progress_team1  # Ball progression
            reward += passing_bonus_team1  # Passing bonus
            reward += pressure_team1  # Pressure penalty
            
            rewards[f'team1_agent{i}'] = reward
        
        # Team 2 rewards
        for i in range(3):
            dist_to_ball = np.linalg.norm(self.team2_pos[i] - self.ball_pos)
            
            reward = self.effort_penalty
            reward += -dist_to_ball / 1000.0
            reward += ball_progress_team2
            reward += passing_bonus_team2
            reward += pressure_team2
            
            rewards[f'team2_agent{i}'] = reward
        
        return rewards
    
    def _goal_rewards(self, team):
        """Rewards when a goal is scored"""
        rewards = {}
        
        if team == 1:
            for i in range(3):
                rewards[f'team1_agent{i}'] = 10.0
                rewards[f'team2_agent{i}'] = -10.0
        else:
            for i in range(3):
                rewards[f'team1_agent{i}'] = -10.0
                rewards[f'team2_agent{i}'] = 10.0
        
        return rewards
    
    def render(self):
        """Render is handled by the visualization script"""
        pass
    
    def get_state(self):
        """Get full state for rendering"""
        return {
            'ball_pos': self.ball_pos.copy(),
            'team1_pos': self.team1_pos.copy(),
            'team2_pos': self.team2_pos.copy(),
            'score': self.score.copy(),
            'steps': self.steps
        }