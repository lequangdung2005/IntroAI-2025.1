"""
Train a DQN agent to play Pacman using Stable Baselines3
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

# Register ALE environments
gym.register_envs(ale_py)

# Create directories for saving models and logs
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def make_env():
    """Create and wrap the Pacman environment"""
    # Create the environment with proper settings
    env = gym.make("ALE/Pacman-v5", 
                   render_mode="rgb_array",
                   frameskip=1)
    
    # Wrap with Atari preprocessing (handles grayscale, resizing, etc.)
    env = AtariWrapper(env)
    
    # Monitor wrapper for logging
    env = Monitor(env)
    
    return env

def train_agent():
    """Train the DQN agent"""
    print("Creating environment...")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Stack 4 frames for temporal information
    env = VecFrameStack(env, n_stack=4)
    
    print("Initializing DQN agent...")
    
    # Create DQN model with optimized hyperparameters for Atari
    model = DQN(
        "CnnPolicy",                    # CNN policy for image inputs
        env,
        learning_rate=1e-4,             # Learning rate
        buffer_size=100000,             # Replay buffer size
        learning_starts=10000,          # Start training after this many steps
        batch_size=64,                  # Batch size for training
        tau=1.0,                        # Target network update rate (hard update)
        gamma=0.99,                     # Discount factor
        train_freq=4,                   # Train every 4 steps
        gradient_steps=1,               # Number of gradient steps per update
        target_update_interval=1000,    # Update target network every 1000 steps
        exploration_fraction=0.1,       # Fraction of training for epsilon decay
        exploration_initial_eps=1.0,    # Initial epsilon
        exploration_final_eps=0.01,     # Final epsilon
        verbose=1,                      # Print training info
        tensorboard_log="./logs/",      # Log directory for TensorBoard
        device="auto"                   # Use GPU if available
    )
    
    # Create callbacks
    # Save model checkpoints every 50,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="pacman_dqn"
    )
    
    print("Starting training...")
    print("This will take a while. Monitor progress in TensorBoard:")
    print("  tensorboard --logdir=./logs/")
    print("-" * 60)
    
    # Train the agent
    total_timesteps = 1000000  # 1 million steps (adjust as needed)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        log_interval=10,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = "models/pacman_dqn_final.zip"
    model.save(final_model_path)
    print(f"\nTraining complete! Model saved to {final_model_path}")
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("Training DQN Agent for Pacman")
    print("=" * 60)
    
    # Train the agent
    model = train_agent()
    
    print("\nTo watch your trained agent play, run:")
    print("  python play_pacman.py")
