"""
Train a DQN agent to play MsPacman using Stable Baselines3 + OCAtari with reward shaping
DQN: Deep Q-Network - Value-based method
"""
import gymnasium as gym
import ale_py
import numpy as np
import os
import time

from ocatari.core import OCAtari
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from preprocess import PreprocessFrame
from environment.reward_shaping_wrapper_3 import StabilityFixedRewardShaper as RewardShapingWrapper

# Đăng ký ALE environments
gym.register_envs(ale_py)

# Get number of CPU cores for parallel environments
N_ENVS = min(os.cpu_count(), 12)  # Use all available cores (max 20)

# Get project root directory (parent of shaping_reward/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tạo thư mục lưu model và log
os.makedirs(os.path.join(PROJECT_ROOT, "models/dqn"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "logs/dqn"), exist_ok=True)


# ======= Create env =======
def make_env():
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="rgb_array",
                  mode="both")
    env = RewardShapingWrapper(env, enable_logging=True, 
                              log_file=os.path.join(PROJECT_ROOT, "shaping_reward_dqn_log.txt"))
    env = PreprocessFrame(env, width=84, height=84, force_image=True)
    env = Monitor(env)
    return env


def train_dqn():
    """Train the DQN agent"""
    print("=" * 60)
    print("Training DQN Agent for MsPacman with OCAtari")
    print(f"Using {N_ENVS} parallel environments with reward shaping")
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

    print("Initializing DQN agent...")

    # DQN hyperparameters optimized for Atari
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=50000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log=os.path.join(PROJECT_ROOT, "logs/dqn/"),
        device="cuda"
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(PROJECT_ROOT, "models/dqn/"),
        name_prefix="mspacman_dqn"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(PROJECT_ROOT, "models/dqn/best/"),
        log_path=os.path.join(PROJECT_ROOT, "logs/dqn/eval/"),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print("Starting training...")
    print("Monitor progress: tensorboard --logdir=./logs/dqn/")
    print("-" * 60)

    start_time = time.time()

    # Train the agent
    model.learn(
        total_timesteps=40000000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    training_time = time.time() - start_time

    # Save final model
    final_model_path = os.path.join(PROJECT_ROOT, "models/dqn/mspacman_dqn_final.zip")
    model.save(final_model_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {os.path.join(PROJECT_ROOT, 'models/dqn/best/')}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train_dqn()