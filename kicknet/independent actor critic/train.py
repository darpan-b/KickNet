import torch
import numpy as np
from soccer_env import SoccerEnv
from actor_critic import ActorCriticAgent
from replay_buffer import MultiAgentReplayBuffer
import os
from datetime import datetime
import json

class MultiAgentTrainer:
    """Trainer for multi-agent soccer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Environment
        self.env = SoccerEnv()
        
        # Get dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create agents
        self.agents = {}
        for i in range(3):
            self.agents[f'team1_agent{i}'] = ActorCriticAgent(
                state_dim, action_dim, 
                lr_actor=config['lr_actor'],
                lr_critic=config['lr_critic'],
                gamma=config['gamma'],
                device=self.device
            )
            self.agents[f'team2_agent{i}'] = ActorCriticAgent(
                state_dim, action_dim,
                lr_actor=config['lr_actor'],
                lr_critic=config['lr_critic'],
                gamma=config['gamma'],
                device=self.device
            )
        
        # Replay buffers
        self.replay_buffer = MultiAgentReplayBuffer(
            num_agents=6,
            capacity=config['buffer_size']
        )
        
        # Tracking
        self.episode_rewards = {agent: [] for agent in self.agents.keys()}
        self.episode_scores = []
        
        # Create save directory
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        config = self.config
        total_steps = 0
        
        for episode in range(config['num_episodes']):
            obs, _ = self.env.reset()
            episode_reward = {agent: 0 for agent in self.agents.keys()}
            done = False
            
            while not done:
                # Select actions for all agents
                actions = {}
                for agent_id, agent in self.agents.items():
                    actions[agent_id] = agent.select_action(obs[agent_id])
                
                # Step environment
                next_obs, rewards, dones, infos = self.env.step(actions)
                done = dones['__all__']
                
                # Store transitions
                for agent_id in self.agents.keys():
                    self.replay_buffer.push(
                        agent_id,
                        obs[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_obs[agent_id],
                        float(done)
                    )
                    episode_reward[agent_id] += rewards[agent_id]
                
                # Update agents
                if total_steps > config['learning_starts']:
                    for agent_id, agent in self.agents.items():
                        if self.replay_buffer.ready(agent_id, config['batch_size']):
                            batch = self.replay_buffer.sample(agent_id, config['batch_size'])
                            agent.update(*batch)
                
                obs = next_obs
                total_steps += 1
            
            # Track episode statistics
            for agent_id in self.agents.keys():
                self.episode_rewards[agent_id].append(episode_reward[agent_id])
            
            final_score = list(infos.values())[0]['score']
            self.episode_scores.append(final_score)
            
            # Logging
            if (episode + 1) % config['log_interval'] == 0:
                avg_reward_team1 = np.mean([
                    self.episode_rewards[f'team1_agent{i}'][-config['log_interval']:]
                    for i in range(3)
                ])
                avg_reward_team2 = np.mean([
                    self.episode_rewards[f'team2_agent{i}'][-config['log_interval']:]
                    for i in range(3)
                ])
                
                recent_scores = self.episode_scores[-config['log_interval']:]
                avg_team1_score = np.mean([s[0] for s in recent_scores])
                avg_team2_score = np.mean([s[1] for s in recent_scores])
                
                print(f"Episode {episode + 1}/{config['num_episodes']}")
                print(f"  Total Steps: {total_steps}")
                print(f"  Avg Reward - Team1: {avg_reward_team1:.3f}, Team2: {avg_reward_team2:.3f}")
                print(f"  Avg Score - Team1: {avg_team1_score:.2f}, Team2: {avg_team2_score:.2f}")
                print(f"  Buffer Size: {self.replay_buffer.size('team1_agent0')}")
                print()
            
            # Save models
            if (episode + 1) % config['save_interval'] == 0:
                self.save_models(f"checkpoint_ep{episode + 1}")
                print(f"Models saved at episode {episode + 1}")
        
        # Save final models
        self.save_models("final")
        self.save_training_stats()
        print("Training completed!")
    
    def save_models(self, name):
        """Save all agent models"""
        save_path = os.path.join(self.save_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent.save(os.path.join(save_path, f"{agent_id}.pt"))
    
    def save_training_stats(self):
        """Save training statistics"""
        stats = {
            'episode_rewards': {k: v for k, v in self.episode_rewards.items()},
            'episode_scores': self.episode_scores,
            'config': self.config
        }
        
        with open(os.path.join(self.save_dir, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    # Training configuration
    config = {
        'num_episodes': 5000,
        'batch_size': 128,
        'buffer_size': 1000,
        'learning_starts': 1000,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'log_interval': 100,
        'save_interval': 100,
        'save_dir': 'models/soccer_iac'
    }
    
    # Create trainer and train
    trainer = MultiAgentTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()