"""
Train a PPO agent to play Pacman using Stable Baselines3
PPO: Proximal Policy Optimization - Policy-based method
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import time

# Register ALE environments
gym.register_envs(ale_py)

# Create directories for saving models and logs
os.makedirs("models/ppo", exist_ok=True)
os.makedirs("logs/ppo", exist_ok=True)

def make_env():
    """Create and wrap the Pacman environment"""
    env = gym.make("ALE/Pacman-v5", 
                   render_mode="rgb_array",
                   frameskip=1)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

def train_ppo():
    """Train the PPO agent"""
    print("=" * 60)
    print("Training PPO Agent for Pacman")
    print("=" * 60)
    print("Creating environment...")
    
    # Create vectorized environment (PPO works better with multiple parallel envs)
    env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel environments
    env = VecFrameStack(env, n_stack=4)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    print("Initializing PPO agent...")
    
    # PPO hyperparameters optimized for Atari
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,                    # Steps per update per environment
        batch_size=256,                 # Minibatch size
        n_epochs=4,                     # Number of epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        clip_range_vf=None,
        ent_coef=0.01,                  # Entropy coefficient
        vf_coef=0.5,                    # Value function coefficient
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/ppo/",
        device="auto"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/ppo/",
        name_prefix="pacman_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/ppo/best/",
        log_path="./logs/ppo/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    print("Starting training...")
    print("Monitor progress: tensorboard --logdir=./logs/ppo/")
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
    final_model_path = "models/ppo/pacman_ppo_final.zip"
    model.save(final_model_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to models/ppo/best/")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    train_ppo()
