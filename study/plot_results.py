import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorflow.python.summary.summary_iterator import summary_iterator

# --- Configuration ---
LOG_DIR = "./kicknet_atari_logs/"
SMOOTHING = 0.9

def read_tb_data(log_dir):
    data = []
    print(f"Scanning {log_dir}...")

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                folder_name = os.path.basename(root)
                
                # 1. Identify Algorithm
                if "DQN" in folder_name: algo = "DQN"
                elif "A2C" in folder_name: algo = "A2C"
                elif "PPO" in folder_name: algo = "PPO"
                else: continue

                # 2. Auto-Detect Version Suffix
                # Assumes folder ends with "_1", "_2", etc.
                try:
                    # Splits "DQN_Pong_1" -> ["DQN", "Pong", "1"]
                    parts = folder_name.split("_")
                    
                    # Check if the last part is a number (the version)
                    if parts[-1].isdigit():
                        version_suffix = f"_{parts[-1]}" # e.g., "_1"
                    else:
                        continue # Skip folders that don't look like versions
                        
                    path = os.path.join(root, file)
                    
                    for e in summary_iterator(path):
                        for v in e.summary.value:
                            if v.tag == "rollout/ep_rew_mean":
                                data.append({
                                    "Algorithm": algo,
                                    "Version": version_suffix, # Stores "_1", "_2", etc.
                                    "Timesteps": e.step,
                                    "Reward": v.simple_value
                                })
                except Exception as e:
                    print(f"    Error reading {file}: {e}")

    return pd.DataFrame(data)

def format_human_readable(x, pos):
    if x >= 1_000_000: return f'{x*1e-6:.1f}M'
    if x >= 1_000: return f'{x*1e-3:.0f}k'
    return f'{x:.0f}'

def plot_specific_version(df, target_version, title, output_file):
    # Filter for the specific version requested
    df_filtered = df[df["Version"] == target_version].copy()
    
    if df_filtered.empty:
        print(f"⚠️  Skipping {target_version}: No data found.")
        return

    print(f"Generating plot for {target_version}...")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Apply Smoothing
    df_filtered["Reward"] = df_filtered.groupby("Algorithm")["Reward"].transform(
        lambda x: x.ewm(alpha=(1 - SMOOTHING)).mean()
    )

    sns.lineplot(
        data=df_filtered, 
        x="Timesteps", 
        y="Reward", 
        hue="Algorithm",
        style="Algorithm",
        linewidth=2.5,
        palette="deep"
    )

    plt.title(title, fontsize=15, weight='bold')
    plt.ylabel("Average Episode Reward", fontsize=12)
    plt.xlabel("Training Timesteps", fontsize=12)
    # plt.axhline(y=-21, color='r', linestyle='--', alpha=0.5, label="Random")
    # plt.axhline(y=0, color='g', linestyle='--', alpha=0.5, label="Winning")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_human_readable))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Saved {output_file}")
    plt.close() # Close plot to save memory

if __name__ == "__main__":
    df = read_tb_data(LOG_DIR)
    print(df)
    print(df['Version'].unique())
    if not df.empty:
        # --- DEFINE YOUR PLOTS HERE ---
        # You can add or remove any version you want easily.
        
        PLOTS_TO_GENERATE = [
            {
                "version": "_1", 
                "title": "Experiment v1: Short Run (CNN - 200k)", 
                "filename": "results_v1_short.png"
            },
            {
                "version": "_2", 
                "title": "Experiment v2: Pro Run (CNN - 1 Million Steps)", 
                "filename": "results_v2_long.png"
            },
            {
                "version": "_3", 
                "title": "Experiment v3: Architecture Failure (MLP - 200k)", 
                "filename": "results_v3_mlp.png"
            },
            # You can add "_4" here later!
        ]

        for p in PLOTS_TO_GENERATE:
            plot_specific_version(df, p["version"], p["title"], p["filename"])
            
    else:
        print("No data found.")    