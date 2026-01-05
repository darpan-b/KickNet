import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer for each agent"""
    
    def __init__(self, capacity=15000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class MultiAgentReplayBuffer:
    """Replay buffers for all agents"""
    
    def __init__(self, num_agents=6, capacity=100000):
        self.buffers = {
            f'team1_agent{i}': ReplayBuffer(capacity) for i in range(3)
        }
        self.buffers.update({
            f'team2_agent{i}': ReplayBuffer(capacity) for i in range(3)
        })
    
    def push(self, agent_id, state, action, reward, next_state, done):
        """Add experience for specific agent"""
        self.buffers[agent_id].push(state, action, reward, next_state, done)
    
    def sample(self, agent_id, batch_size):
        """Sample batch for specific agent"""
        return self.buffers[agent_id].sample(batch_size)
    
    def ready(self, agent_id, batch_size):
        """Check if agent has enough samples"""
        return len(self.buffers[agent_id]) >= batch_size
    
    def size(self, agent_id):
        """Get buffer size for agent"""
        return len(self.buffers[agent_id])