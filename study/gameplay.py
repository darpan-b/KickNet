import warnings
warnings.filterwarnings('ignore')

import os
import time
import gymnasium as gym
import ale_py  # Critical for Atari registration
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# --- Configuration ---
ENV_ID = "PongNoFrameskip-v4"
LOG_DIR = "./kicknet_atari_logs"

def create_eval_env():
    """
    Creates an environment with 'human' render mode so a window pops up.
    """
    # FIX: Pass render_mode inside 'env_kwargs', NOT 'wrapper_kwargs'
    env = make_atari_env(
        ENV_ID, 
        n_envs=1, 
        seed=42, 
        env_kwargs={"render_mode": "human"}  # <--- This is the correct place
    )
    # Apply same wrappers as training (Frame Stacking + Transpose)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env

def watch_model(algo_name, model_class):
    # Construct file path
    # model_path = os.path.join(LOG_DIR, f"{algo_name.lower()}_pong.zip")
    model_path = os.path.join(LOG_DIR, f"{algo_name.lower()}_pong_v3.zip")
    
    # v1 --> 200K timesteps w/ CnnPolicy
    # v2 --> 1M timesteps w/ CnnPolicy
    # v3 --> 200K timesteps w/ MlpPolicy
    
    print(f"\n{'='*40}")
    print(f"PREPARING TO WATCH: {algo_name}")
    print(f"Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: File not found! Did you finish training {algo_name}?")
        print(f"   (Tip: check if the file is named slightly differently in {LOG_DIR})")
        return

    try:
        # Load Model
        print(f"Loading weights... ", end="")
        model = model_class.load(model_path)
        print("Done.")
        
        # Create Window
        print("Launching game window... (Check your taskbar if hidden)")
        env = create_eval_env()
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"--- Playing {algo_name} (Press Ctrl+C in terminal to stop) ---")
        
        while True:
            # Predict action (Deterministic = True means use best known strategy)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            # Slow down slightly so humans can follow the ball (0.02s ~ 50 FPS)
            time.sleep(0.02)
            
            if done:
                print(f"Game Over! Final Score: {int(total_reward)}")
                time.sleep(1.0) # Pause briefly on game over
                obs = env.reset()
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n❌ detailed error: {e}")
    finally:
        try:
            env.close()
        except:
            pass
            

if __name__ == "__main__":
    print("Choose an algorithm to watch:")
    print("1. DQN")
    print("2. A2C")
    print("3. PPO")
    
    choice = input("Enter number (1-3): ").strip()
    
    if choice == "1":
        watch_model("DQN", DQN)
    elif choice == "2":
        watch_model("A2C", A2C)
    elif choice == "3":
        watch_model("PPO", PPO)
    else:
        print("Invalid choice.")
