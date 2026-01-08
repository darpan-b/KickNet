import os
import gymnasium as gym
import ale_py  # <--- CRITICAL: Registers Atari environments
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# --- Configuration ---
ENV_ID = "PongNoFrameskip-v4" 
LOG_DIR = "./kicknet_atari_logs/"
TOTAL_TIMESTEPS = 200_000
# ~ TOTAL_TIMESTEPS = 1_000_000

os.makedirs(LOG_DIR, exist_ok=True)

def create_vectorized_env(algo_name):
    # Create the environment with standard Atari wrappers
    # We pass n_envs=1 for DQN to avoid buffer shape mismatches in older SB3 versions
    n_envs = 4 if algo_name != "DQN" else 1
    
    env = make_atari_env(
        ENV_ID, 
        n_envs=n_envs, 
        seed=42, 
        monitor_dir=f"{LOG_DIR}/{algo_name}"
    )
    
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env

def benchmark_dqn():
    print(f"\n--- Starting DQN Benchmark on {ENV_ID} ---")
    env = create_vectorized_env("DQN")
    
    model = DQN(
        "MlpPolicy", 
        env, 
        buffer_size=50_000, 
        learning_rate=1e-4, 
        exploration_fraction=0.1,  # <--- FIXED: parameter name was 'fraction_explored'
        exploration_final_eps=0.01, # Good practice to set the final epsilon value
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="DQN_Pong")
    model.save(f"{LOG_DIR}/dqn_pong")
    print("DQN Complete.")
    env.close()

def benchmark_a2c():
    print(f"\n--- Starting A2C Benchmark on {ENV_ID} ---")
    env = create_vectorized_env("A2C")
    
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01, 
        vf_coef=0.5,
        tensorboard_log=LOG_DIR
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="A2C_Pong")
    model.save(f"{LOG_DIR}/a2c_pong")
    print("A2C Complete.")
    env.close()

def benchmark_ppo():
    print(f"\n--- Starting PPO Benchmark on {ENV_ID} ---")
    env = create_vectorized_env("PPO")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        clip_range=0.1,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_Pong")
    model.save(f"{LOG_DIR}/ppo_pong")
    print("PPO Complete.")
    env.close()
    

if __name__ == "__main__":
    # Ensure the gymnasium registry contains Atari games
    print(f"Registered environments: {len(gym.envs.registry)}")
    
    benchmark_dqn()
    benchmark_a2c()
    benchmark_ppo()
