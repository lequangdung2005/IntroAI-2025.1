"""
Train a Rainbow DQN agent to play Pacman
Rainbow: Combines multiple DQN improvements (Double DQN, Dueling, Prioritized Experience Replay, etc.)
Note: Rainbow is not directly available in Stable Baselines3, so we use sb3-contrib
"""
import gymnasium as gym
import ale_py
try:
    from sb3_contrib import QRDQN  # Quantile Regression DQN (closest to Rainbow in SB3)
except ImportError:
    print("Warning: sb3-contrib not installed. Installing...")
    print("Run: pip install sb3-contrib")
    print("Using regular DQN as fallback")
    from stable_baselines3 import DQN as QRDQN
    
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import time

# Register ALE environments
gym.register_envs(ale_py)

# Create directories for saving models and logs
os.makedirs("models/rainbow", exist_ok=True)
os.makedirs("logs/rainbow", exist_ok=True)

def make_env():
    """Create and wrap the Pacman environment"""
    env = gym.make("ALE/Pacman-v5", 
                   render_mode="rgb_array",
                   frameskip=1)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

def train_rainbow():
    """Train the Rainbow (QR-DQN) agent"""
    print("=" * 60)
    print("Training Rainbow/QR-DQN Agent for Pacman")
    print("=" * 60)
    print("Creating environment...")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    print("Initializing Rainbow/QR-DQN agent...")
    
    # Rainbow (QR-DQN) hyperparameters optimized for Atari
    # QR-DQN uses quantile regression for better value estimation
    model = QRDQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200000,              # Reduced for memory
        learning_starts=50000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/rainbow/",
        device="auto"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/rainbow/",
        name_prefix="pacman_rainbow"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/rainbow/best/",
        log_path="./logs/rainbow/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    print("Starting training...")
    print("Monitor progress: tensorboard --logdir=./logs/rainbow/")
    print("-" * 60)
    
    start_time = time.time()
    
    # Train the agent
    model.learn(
        total_timesteps=900000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = "models/rainbow/pacman_rainbow_final.zip"
    model.save(final_model_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to models/rainbow/best/")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    train_rainbow()
