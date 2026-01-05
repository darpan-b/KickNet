import torch
import numpy as np
from soccer_env import SoccerEnv
from actor_critic import MAPPOAgent
from replay_buffer import MultiAgentRolloutBuffer
import os
from datetime import datetime
import json

class MAPPOTrainer:
    """Trainer for MAPPO multi-agent soccer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Environment
        self.env = SoccerEnv()
        
        # Get dimensions
        state_dim = self.env.observation_space.shape[0]  # Local observation
        global_state_dim = self.env.global_state_dim  # Global state for critic
        action_dim = self.env.action_space.shape[0]
        
        print(f"State dim (local): {state_dim}")
        print(f"Global state dim: {global_state_dim}")
        print(f"Action dim: {action_dim}")
        
        # Parameter sharing for homogeneous teams
        if config['parameter_sharing']:
            print("Using parameter sharing for homogeneous teams")
            # One agent per team
            self.shared_agents = {
                'team1': MAPPOAgent(
                    state_dim, global_state_dim, action_dim,
                    lr_actor=config['lr_actor'],
                    lr_critic=config['lr_critic'],
                    gamma=config['gamma'],
                    gae_lambda=config['gae_lambda'],
                    clip_epsilon=config['clip_epsilon'],
                    entropy_coef=config['entropy_coef'],
                    value_coef=config['value_coef'],
                    max_grad_norm=config['max_grad_norm'],
                    device=self.device
                ),
                'team2': MAPPOAgent(
                    state_dim, global_state_dim, action_dim,
                    lr_actor=config['lr_actor'],
                    lr_critic=config['lr_critic'],
                    gamma=config['gamma'],
                    gae_lambda=config['gae_lambda'],
                    clip_epsilon=config['clip_epsilon'],
                    entropy_coef=config['entropy_coef'],
                    value_coef=config['value_coef'],
                    max_grad_norm=config['max_grad_norm'],
                    device=self.device
                )
            }
            
            # Map agent IDs to shared agents
            self.agents = {}
            for i in range(3):
                self.agents[f'team1_agent{i}'] = self.shared_agents['team1']
                self.agents[f'team2_agent{i}'] = self.shared_agents['team2']
        else:
            # Independent agents
            print("Using independent agents (no parameter sharing)")
            self.agents = {}
            for i in range(3):
                self.agents[f'team1_agent{i}'] = MAPPOAgent(
                    state_dim, global_state_dim, action_dim,
                    lr_actor=config['lr_actor'],
                    lr_critic=config['lr_critic'],
                    gamma=config['gamma'],
                    gae_lambda=config['gae_lambda'],
                    clip_epsilon=config['clip_epsilon'],
                    entropy_coef=config['entropy_coef'],
                    value_coef=config['value_coef'],
                    max_grad_norm=config['max_grad_norm'],
                    device=self.device
                )
                self.agents[f'team2_agent{i}'] = MAPPOAgent(
                    state_dim, global_state_dim, action_dim,
                    lr_actor=config['lr_actor'],
                    lr_critic=config['lr_critic'],
                    gamma=config['gamma'],
                    gae_lambda=config['gae_lambda'],
                    clip_epsilon=config['clip_epsilon'],
                    entropy_coef=config['entropy_coef'],
                    value_coef=config['value_coef'],
                    max_grad_norm=config['max_grad_norm'],
                    device=self.device
                )
            self.shared_agents = None
        
        # Rollout buffers
        self.rollout_buffer = MultiAgentRolloutBuffer(num_agents=6)
        
        # Tracking
        self.episode_rewards = {agent: [] for agent in ['team1_agent0', 'team1_agent1', 'team1_agent2',
                                                          'team2_agent0', 'team2_agent1', 'team2_agent2']}
        self.episode_scores = []
        
        # Create save directory
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
    
    def collect_rollouts(self, num_steps):
        """Collect rollouts for training"""
        obs, global_state, _ = self.env.reset()
        
        for step in range(num_steps):
            # Select actions for all agents
            actions = {}
            log_probs = {}
            values = {}
            
            for agent_id, agent in self.agents.items():
                action, log_prob = agent.select_action_with_logprob(obs[agent_id])
                value = agent.get_value(global_state)
                
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
                values[agent_id] = value
            
            # Step environment
            next_obs, next_global_state, rewards, dones, infos = self.env.step(actions)
            done = dones['__all__']
            
            # Store transitions
            for agent_id in ['team1_agent0', 'team1_agent1', 'team1_agent2',
                            'team2_agent0', 'team2_agent1', 'team2_agent2']:
                self.rollout_buffer.push(
                    agent_id,
                    obs[agent_id],
                    global_state,
                    actions[agent_id],
                    rewards[agent_id],
                    log_probs[agent_id],
                    values[agent_id],
                    float(done)
                )
            
            obs = next_obs
            global_state = next_global_state
            
            if done:
                # Store episode statistics
                final_score = list(infos.values())[0]['score']
                self.episode_scores.append(final_score)
                
                obs, global_state, _ = self.env.reset()
    
    def train(self):
        """Main training loop"""
        config = self.config
        total_steps = 0
        
        for episode in range(config['num_episodes']):
            # Collect rollouts
            self.collect_rollouts(config['rollout_steps'])
            
            # Update agents
            update_stats = {}
            
            if config['parameter_sharing']:
                # Update shared agents
                for team_name, agent in self.shared_agents.items():
                    # Aggregate data from all agents in the team
                    all_states = []
                    all_global_states = []
                    all_actions = []
                    all_rewards = []
                    all_log_probs = []
                    all_values = []
                    all_dones = []
                    
                    agent_ids = [f'{team_name}_agent{i}' for i in range(3)]
                    
                    for agent_id in agent_ids:
                        states, global_states, actions, rewards, log_probs, values, dones = \
                            self.rollout_buffer.get(agent_id)
                        
                        all_states.append(states)
                        all_global_states.append(global_states)
                        all_actions.append(actions)
                        all_rewards.append(rewards)
                        all_log_probs.append(log_probs)
                        all_values.append(values)
                        all_dones.append(dones)
                    
                    # Concatenate all data
                    all_states = np.concatenate(all_states, axis=0)
                    all_global_states = np.concatenate(all_global_states, axis=0)
                    all_actions = np.concatenate(all_actions, axis=0)
                    all_rewards = np.concatenate(all_rewards, axis=0)
                    all_log_probs = np.concatenate(all_log_probs, axis=0)
                    all_values = np.concatenate(all_values, axis=0)
                    all_dones = np.concatenate(all_dones, axis=0)
                    
                    # Compute GAE
                    next_value = 0  # Assuming terminal state
                    advantages, returns = agent.compute_gae(
                        all_rewards, all_values, next_value, all_dones
                    )
                    
                    # PPO update
                    stats = agent.update_ppo(
                        all_states, all_global_states, all_actions,
                        all_log_probs, advantages, returns,
                        epochs=config['ppo_epochs'],
                        batch_size=config['batch_size']
                    )
                    
                    update_stats[team_name] = stats
            else:
                # Update independent agents
                for agent_id in ['team1_agent0', 'team1_agent1', 'team1_agent2',
                                'team2_agent0', 'team2_agent1', 'team2_agent2']:
                    agent = self.agents[agent_id]
                    
                    states, global_states, actions, rewards, log_probs, values, dones = \
                        self.rollout_buffer.get(agent_id)
                    
                    # Compute GAE
                    next_value = 0
                    advantages, returns = agent.compute_gae(
                        rewards, values, next_value, dones
                    )
                    
                    # PPO update
                    stats = agent.update_ppo(
                        states, global_states, actions,
                        log_probs, advantages, returns,
                        epochs=config['ppo_epochs'],
                        batch_size=config['batch_size']
                    )
                    
                    if agent_id not in update_stats:
                        update_stats[agent_id] = stats
            
            # Clear buffers
            self.rollout_buffer.clear()
            
            total_steps += config['rollout_steps']
            
            # Logging
            if (episode + 1) % config['log_interval'] == 0:
                recent_scores = self.episode_scores[-config['log_interval']:]
                if len(recent_scores) > 0:
                    avg_team1_score = np.mean([s[0] for s in recent_scores])
                    avg_team2_score = np.mean([s[1] for s in recent_scores])
                    
                    print(f"\nEpisode {episode + 1}/{config['num_episodes']}")
                    print(f"  Total Steps: {total_steps}")
                    print(f"  Avg Score - Team1: {avg_team1_score:.2f}, Team2: {avg_team2_score:.2f}")
                    print(f"  Episodes this interval: {len(recent_scores)}")
                    
                    if config['parameter_sharing']:
                        if 'team1' in update_stats:
                            print(f"  Team1 - Actor Loss: {update_stats['team1']['actor_loss']:.4f}, "
                                  f"Critic Loss: {update_stats['team1']['critic_loss']:.4f}, "
                                  f"Entropy: {update_stats['team1']['entropy']:.4f}")
                        if 'team2' in update_stats:
                            print(f"  Team2 - Actor Loss: {update_stats['team2']['actor_loss']:.4f}, "
                                  f"Critic Loss: {update_stats['team2']['critic_loss']:.4f}, "
                                  f"Entropy: {update_stats['team2']['entropy']:.4f}")
                    else:
                        sample_agent = 'team1_agent0'
                        if sample_agent in update_stats:
                            print(f"  Sample Agent ({sample_agent}) - "
                                  f"Actor Loss: {update_stats[sample_agent]['actor_loss']:.4f}, "
                                  f"Critic Loss: {update_stats[sample_agent]['critic_loss']:.4f}, "
                                  f"Entropy: {update_stats[sample_agent]['entropy']:.4f}")
            
            # Save models
            if (episode + 1) % config['save_interval'] == 0:
                self.save_models(f"checkpoint_ep{episode + 1}")
                print(f"Models saved at episode {episode + 1}")
        
        # Save final models
        self.save_models("final")
        self.save_training_stats()
        print("\nTraining completed!")
    
    def save_models(self, name):
        """Save all agent models"""
        save_path = os.path.join(self.save_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        if self.config['parameter_sharing']:
            # Save shared agents
            for team_name, agent in self.shared_agents.items():
                agent.save(os.path.join(save_path, f"{team_name}.pt"))
        else:
            # Save independent agents
            saved_agents = set()
            for agent_id, agent in self.agents.items():
                if id(agent) not in saved_agents:
                    agent.save(os.path.join(save_path, f"{agent_id}.pt"))
                    saved_agents.add(id(agent))
    
    def save_training_stats(self):
        """Save training statistics"""
        stats = {
            'episode_scores': self.episode_scores,
            'config': self.config
        }
        
        with open(os.path.join(self.save_dir, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    # Training configuration for MAPPO
    config = {
        'num_episodes': 10000,
        'rollout_steps': 3000,  # Steps per rollout
        'batch_size': 128,
        'ppo_epochs': 4,  # Multiple epochs per update
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,  # GAE parameter
        'clip_epsilon': 0.2,  # PPO clipping
        'entropy_coef': 0.01,  # Entropy bonus
        'value_coef': 0.5,  # Value loss coefficient
        'max_grad_norm': 0.5,  # Gradient clipping
        'parameter_sharing': True,  # Share parameters within teams
        'log_interval': 50,
        'save_interval': 50,
        'save_dir': 'models/soccer_mappo'
    }
    
    print("="*50)
    print("MAPPO Multi-Agent Soccer Training")
    print("="*50)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    # Create trainer and train
    trainer = MAPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()