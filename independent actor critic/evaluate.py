import torch
import numpy as np
from soccer_env import SoccerEnv
from actor_critic import ActorCriticAgent
import pygame
import sys
import os

class SoccerVisualizer:
    """Pygame visualizer for soccer environment"""
    
    def __init__(self, env, fps=15):
        pygame.init()
        
        self.env = env
        self.fps = fps
        
        # Display settings
        self.scale = 1.2
        self.width = int(env.field_width * self.scale)
        self.height = int(env.field_height * self.scale)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3v3 Multi-Agent Soccer")
        
        self.clock = pygame.time.Clock()
        
        # Colors
        self.GREEN = (34, 139, 34)
        self.DARK_GREEN = (0, 100, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.BLACK = (0, 0, 0)
        
        # Font
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)
    
    def scale_pos(self, pos):
        """Scale position to display coordinates"""
        return (int(pos[0] * self.scale), int(pos[1] * self.scale))
    
    def draw_field(self):
        """Draw soccer field"""
        # Field background
        self.screen.fill(self.GREEN)
        
        # Field border
        pygame.draw.rect(self.screen, self.WHITE, 
                        (0, 0, self.width, self.height), 3)
        
        # Center line
        center_x = self.width // 2
        pygame.draw.line(self.screen, self.WHITE,
                        (center_x, 0), (center_x, self.height), 3)
        
        # Center circle
        center_y = self.height // 2
        pygame.draw.circle(self.screen, self.WHITE,
                          (center_x, center_y), 
                          int(50 * self.scale), 3)
        
        # Goals
        goal_width = int(self.env.goal_width * self.scale)
        goal_y = (self.height - goal_width) // 2
        
        # Left goal (Team 1 defends)
        pygame.draw.rect(self.screen, self.WHITE,
                        (0, goal_y, int(10 * self.scale), goal_width), 3)
        
        # Right goal (Team 2 defends)
        pygame.draw.rect(self.screen, self.WHITE,
                        (self.width - int(10 * self.scale), goal_y, 
                         int(10 * self.scale), goal_width), 3)
    
    def draw_players(self, state):
        """Draw players"""
        # Team 1 (Blue)
        for i, pos in enumerate(state['team1_pos']):
            screen_pos = self.scale_pos(pos)
            pygame.draw.circle(self.screen, self.BLUE, screen_pos, 12, 0)
            pygame.draw.circle(self.screen, self.WHITE, screen_pos, 12, 2)
            
            # Player number
            text = self.small_font.render(str(i+1), True, self.WHITE)
            text_rect = text.get_rect(center=screen_pos)
            self.screen.blit(text, text_rect)
        
        # Team 2 (Red)
        for i, pos in enumerate(state['team2_pos']):
            screen_pos = self.scale_pos(pos)
            pygame.draw.circle(self.screen, self.RED, screen_pos, 12, 0)
            pygame.draw.circle(self.screen, self.WHITE, screen_pos, 12, 2)
            
            # Player number
            text = self.small_font.render(str(i+1), True, self.WHITE)
            text_rect = text.get_rect(center=screen_pos)
            self.screen.blit(text, text_rect)
    
    def draw_ball(self, state):
        """Draw ball"""
        ball_pos = self.scale_pos(state['ball_pos'])
        pygame.draw.circle(self.screen, self.WHITE, ball_pos, 8, 0)
        pygame.draw.circle(self.screen, self.BLACK, ball_pos, 8, 2)
    
    def draw_score(self, state):
        """Draw score"""
        score_text = f"Step: {state['steps']}     {state['score'][0]} - {state['score'][1]}"
        text = self.font.render(score_text, True, self.WHITE)
        text_rect = text.get_rect(center=(self.width // 2, 30))
        
        # Background for text
        bg_rect = text_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, self.BLACK, bg_rect)
        pygame.draw.rect(self.screen, self.WHITE, bg_rect, 2)
        
        self.screen.blit(text, text_rect)
    
    def render(self, state):
        """Render the current state"""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # Draw everything
        self.draw_field()
        self.draw_players(state)
        self.draw_ball(state)
        self.draw_score(state)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        return True
    
    def close(self):
        """Close the visualizer"""
        pygame.quit()


def load_agents(model_dir, device):
    """Load trained agents"""
    agents = {}
    state_dim = 28  # From environment
    action_dim = 3
    
    for i in range(3):
        # Team 1
        agent_id = f'team1_agent{i}'
        agent = ActorCriticAgent(state_dim, action_dim, device=device)
        model_path = os.path.join(model_dir, f"{agent_id}.pt")
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded {agent_id}")
        agents[agent_id] = agent
        
        # Team 2
        agent_id = f'team2_agent{i}'
        agent = ActorCriticAgent(state_dim, action_dim, device=device)
        model_path = os.path.join(model_dir, f"{agent_id}.pt")
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded {agent_id}")
        agents[agent_id] = agent
    
    return agents


def evaluate(model_dir, num_episodes=10, render=True):
    """Evaluate trained agents"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = SoccerEnv()
    
    # Load agents
    agents = load_agents(model_dir, device)
    
    # Create visualizer if rendering
    visualizer = SoccerVisualizer(env) if render else None
    
    # Evaluation loop
    total_scores = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = {agent: 0 for agent in agents.keys()}
        
        while not done:
            # Select actions
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.select_action(obs[agent_id], deterministic=True)
            
            # Step
            next_obs, rewards, dones, infos = env.step(actions)
            done = dones['__all__']
            
            # Track rewards
            for agent_id in agents.keys():
                episode_reward[agent_id] += rewards[agent_id]
            
            # Render
            if render and visualizer:
                state = env.get_state()
                if not visualizer.render(state):
                    if visualizer:
                        visualizer.close()
                    return
            
            obs = next_obs
        
        # Get final score
        final_score = list(infos.values())[0]['score']
        total_scores.append(final_score)
        
        print(f"Episode {episode + 1}: Team1 {final_score[0]} - {final_score[1]} Team2")
        
        avg_reward_team1 = np.mean([episode_reward[f'team1_agent{i}'] for i in range(3)])
        avg_reward_team2 = np.mean([episode_reward[f'team2_agent{i}'] for i in range(3)])
        print(f"  Avg Reward - Team1: {avg_reward_team1:.2f}, Team2: {avg_reward_team2:.2f}\n")
    
    # Summary
    print("\n=== Evaluation Summary ===")
    print(f"Total Episodes: {num_episodes}")
    avg_score_team1 = np.mean([s[0] for s in total_scores])
    avg_score_team2 = np.mean([s[1] for s in total_scores])
    print(f"Average Score - Team1: {avg_score_team1:.2f}, Team2: {avg_score_team2:.2f}")
    
    if visualizer:
        visualizer.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained soccer agents')
    parser.add_argument('--model_dir', type=str, default='models/soccer_iac/final',
                       help='Directory containing trained models')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--no_render', action='store_true',
                       help='Disable rendering')
    
    args = parser.parse_args()
    
    evaluate(args.model_dir, args.num_episodes, render=not args.no_render)


if __name__ == "__main__":
    main()