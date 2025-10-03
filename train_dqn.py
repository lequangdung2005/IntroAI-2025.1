"""
Train a DQN agent to play Pacman using Stable Baselines3
DQN: Deep Q-Network - Value-based method
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import time

# Register ALE environments
gym.register_envs(ale_py)

# Create directories for saving models and logs
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

def make_env():
    """Create and wrap the Pacman environment"""
    env = gym.make("ALE/Pacman-v5", 
                   render_mode="rgb_array",
                   frameskip=1)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

def train_dqn():
    """Train the DQN agent"""
    print("=" * 60)
    print("Training DQN Agent for Pacman")
    print("=" * 60)
    print("Creating environment...")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    print("Initializing DQN agent...")
    
    # DQN hyperparameters optimized for Atari
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,              # Reduced for memory
        learning_starts=5000,
        batch_size=16384,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/dqn/",
        device="cuda"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/dqn/",
        name_prefix="pacman_dqn"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/dqn/best/",
        log_path="./logs/dqn/eval/",
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
        total_timesteps=1000000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = "models/dqn/pacman_dqn_final.zip"
    model.save(final_model_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to models/dqn/best/")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    train_dqn()
