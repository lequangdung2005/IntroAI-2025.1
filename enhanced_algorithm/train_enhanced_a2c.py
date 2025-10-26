"""
Train A2C agent for Pacman with Enhanced Reward Shaping
Uses OCAtari for object detection and custom reward wrapper
"""
import gymnasium as gym
import ale_py
import os
import time

from ocatari.core import OCAtari
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
from enhanced_wrapper import EnhancedPacmanRewardWrapper

# Register ALE environments
gym.register_envs(ale_py)

# Create directories
os.makedirs("models/enhanced_a2c", exist_ok=True)
os.makedirs("logs/enhanced_a2c", exist_ok=True)


def make_env():
    """Create environment with enhanced reward shaping"""
    # Use standard gym environment with AtariWrapper
    env = gym.make("ALE/Pacman-v5")
    env = EnhancedPacmanRewardWrapper(env)  # Add enhanced reward wrapper first
    env = AtariWrapper(env)  # Standard Atari preprocessing
    env = Monitor(env)
    return env


def train_a2c():
    """Train A2C agent"""
    print("=" * 60)
    print("Training A2C for Pacman (Enhanced)")
    print("=" * 60)
    print("Creating environment...")
    
    # Create vectorized environment with parallel envs
    env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel environments
    env = VecFrameStack(env, n_stack=4)
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    print("Initializing A2C agent...")
    
    model = A2C(
        "CnnPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        normalize_advantage=False,
        verbose=1,
        tensorboard_log="./logs/enhanced_a2c/",
        device="cuda"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/enhanced_a2c/",
        name_prefix="pacman_a2c_enhanced"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/enhanced_a2c/best/",
        log_path="./logs/enhanced_a2c/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    print("Starting training...")
    print("Monitor: tensorboard --logdir=./logs/enhanced_a2c/")
    print("-" * 60)
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=2000000,  # 2M steps for better learning
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    # Save final model
    final_path = "models/enhanced_a2c/pacman_a2c_enhanced_final.zip"
    model.save(final_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model: {final_path}")
    print(f"Best model: models/enhanced_a2c/best/")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    train_a2c()
