"""
Train an A2C agent to play MsPacman using Stable Baselines3 + OCAtari
A2C: Advantage Actor-Critic - Policy-based method
"""
import gymnasium as gym
import ale_py
import numpy as np
import os
import time
import cv2
from gymnasium import spaces

from ocatari.core import OCAtari
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from preprocess import PreprocessFrame

# Register ALE environments
gym.register_envs(ale_py)

# Get number of CPU cores for parallel environments
N_ENVS = min(os.cpu_count(), 20)  # Use all available cores (max 20)

# Create directories for saving models and logs
os.makedirs("models/a2c", exist_ok=True)
os.makedirs("logs/a2c", exist_ok=True)


def make_env():
    """Create and wrap the MsPacman environment using OCAtari"""
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="rgb_array",
                  mode="vision")
    # Preprocess frames to grayscale 84x84 before monitoring/vectorization
    env = PreprocessFrame(env, width=84, height=84, force_image=True)
    env = Monitor(env)
    return env

def train_a2c():
    """Train the A2C agent"""
    print("=" * 60)
    print("Training A2C Agent for MsPacman with OCAtari")
    print(f"Using {N_ENVS} parallel environments")
    print("=" * 60)
    print("Creating environment...")

    # Create vectorized environment with parallel subprocesses for faster training
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    env = VecTransposeImage(env)  # Transpose (H,W,C) -> (C,H,W) for CNN
    env = VecFrameStack(env, n_stack=4)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    print("Initializing A2C agent...")

    # A2C hyperparameters optimized for Atari
    model = A2C(
        "CnnPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,                      # Steps per update per environment
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,                  # Entropy coefficient
        vf_coef=0.5,                    # Value function coefficient
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        normalize_advantage=False,
        verbose=1,
        tensorboard_log="./logs/a2c/",
        device="auto"
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/a2c/",
        name_prefix="mspacman_a2c"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/a2c/best/",
        log_path="./logs/a2c/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print("Starting training...")
    print("Monitor progress: tensorboard --logdir=./logs/a2c/")
    print("-" * 60)

    start_time = time.time()

    # Train the agent
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    training_time = time.time() - start_time

    # Save final model
    final_model_path = "models/a2c/mspacman_a2c_final.zip"
    model.save(final_model_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to models/a2c/best/")
    print("=" * 60)

    return model

if __name__ == "__main__":
    train_a2c()