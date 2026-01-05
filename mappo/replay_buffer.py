import numpy as np
from collections import deque

class RolloutBuffer:
    """Rollout buffer for on-policy MAPPO training"""
    
    def __init__(self):
        self.states = []
        self.global_states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def push(self, state, global_state, action, reward, log_prob, value, done):
        """Add experience to buffer"""
        self.states.append(state)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        """Get all experiences"""
        return (
            np.array(self.states),
            np.array(self.global_states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.log_probs),
            np.array(self.values),
            np.array(self.dones)
        )
    
    def clear(self):
        """Clear buffer"""
        self.states.clear()
        self.global_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class MultiAgentRolloutBuffer:
    """Rollout buffers for all agents in MAPPO"""
    
    def __init__(self, num_agents=6):
        self.buffers = {
            f'team1_agent{i}': RolloutBuffer() for i in range(3)
        }
        self.buffers.update({
            f'team2_agent{i}': RolloutBuffer() for i in range(3)
        })
    
    def push(self, agent_id, state, global_state, action, reward, log_prob, value, done):
        """Add experience for specific agent"""
        self.buffers[agent_id].push(state, global_state, action, reward, log_prob, value, done)
    
    def get(self, agent_id):
        """Get experiences for specific agent"""
        return self.buffers[agent_id].get()
    
    def clear(self, agent_id=None):
        """Clear buffer(s)"""
        if agent_id is None:
            # Clear all buffers
            for buffer in self.buffers.values():
                buffer.clear()
        else:
            self.buffers[agent_id].clear()
    
    def __len__(self):
        """Return length of first buffer (all should be same)"""
        return len(list(self.buffers.values())[0])