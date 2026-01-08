import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    """Actor network for continuous action space (Decentralized)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log_std for continuous actions
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state):
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        """Evaluate action log probability and entropy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        
        # Inverse tanh to get the pre-tanh action
        action_pre_tanh = torch.atanh(torch.clamp(action, -0.999, 0.999))
        
        log_prob = normal.log_prob(action_pre_tanh)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        entropy = normal.entropy().sum(1, keepdim=True)
        
        return log_prob, entropy
    
    def get_action(self, state, deterministic=False):
        """Get action for evaluation"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
        
        return action


class CentralizedCritic(nn.Module):
    """Centralized Critic network using global state"""
    
    def __init__(self, global_state_dim, hidden_dim=256):
        super(CentralizedCritic, self).__init__()
        
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, global_state):
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value(x)
        
        return value


class MAPPOAgent:
    """Multi-Agent PPO agent with centralized critic"""
    
    def __init__(self, state_dim, global_state_dim, action_dim, 
                 lr_actor=3e-4, lr_critic=3e-4, 
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                 device='cpu'):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks - Actor uses local obs, Critic uses global state
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = CentralizedCritic(global_state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # For tracking
        self.action_dim = action_dim
        
    def select_action(self, state, deterministic=False):
        """Select action given local state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_action(state, deterministic)
        
        # Scale actions appropriately
        action = action.cpu().numpy()[0]
        # Actions are [move_x, move_y, kick]
        # move_x, move_y are in [-1, 1]
        # kick should be in [0, 1]
        action[2] = (action[2] + 1) / 2  # Convert from [-1,1] to [0,1]
        
        return action
    
    def select_action_with_logprob(self, state):
        """Select action and return log probability for training"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        action, log_prob = self.actor.sample(state)
        
        # Scale kick action
        action_scaled = action.clone()
        action_scaled[:, 2] = (action_scaled[:, 2] + 1) / 2
        
        return action_scaled.detach().cpu().numpy()[0], log_prob.cpu().item()
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        
        # Reverse iteration for GAE computation
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def update_ppo(self, states, global_states, actions, old_log_probs, 
                    advantages, returns, epochs=4, batch_size=64):
        """PPO update with multiple epochs"""
        
        states = torch.FloatTensor(states).to(self.device)
        global_states = torch.FloatTensor(global_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        updates = 0
        
        for epoch in range(epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                if end > dataset_size:
                    continue
                
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_global_states = global_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                values = self.critic(batch_global_states)
                
                # PPO actor loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy bonus
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                # Total actor loss
                total_actor_loss_batch = actor_loss + entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic loss (MSE with returns)
                critic_loss = self.value_coef * F.mse_loss(values, batch_returns.unsqueeze(1))
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                updates += 1
        
        return {
            'actor_loss': total_actor_loss / max(updates, 1),
            'critic_loss': total_critic_loss / max(updates, 1),
            'entropy': total_entropy / max(updates, 1)
        }
    
    def get_value(self, global_state):
        """Get value estimate from centralized critic"""
        global_state = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(global_state)
        
        return value.cpu().item()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])