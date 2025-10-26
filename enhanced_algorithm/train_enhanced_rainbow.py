"""
Train Rainbow (QR-DQN) agent for Pacman with Enhanced Reward Shaping
Uses OCAtari for object detection and custom reward wrapper
"""
import gymnasium as gym
import ale_py
import os
import time

from ocatari.core import OCAtari
try:
    from sb3_contrib import QRDQN
except ImportError:
    print("Warning: sb3-contrib not installed, using regular DQN")
    from stable_baselines3 import DQN as QRDQN

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from preprocess import PreprocessFrame
from enhanced_wrapper import EnhancedPacmanRewardWrapper

# Register ALE environments
gym.register_envs(ale_py)

# Create directories
os.makedirs("models/enhanced_rainbow", exist_ok=True)
os.makedirs("logs/enhanced_rainbow", exist_ok=True)


def make_env():
    """Create environment with enhanced reward shaping"""
    env = OCAtari("ALE/Pacman-v5",  # Changed to regular Pacman
                  render_mode="rgb_array",
                  mode="vision")
    env = EnhancedPacmanRewardWrapper(env)  # Add enhanced reward wrapper
    env = PreprocessFrame(env, width=84, height=84, grayscale=False, force_image=True)  # RGB for VecTransposeImage
    env = Monitor(env)
    return env


def train_rainbow():
    """Train Rainbow (QR-DQN) agent"""
    print("=" * 60)
    print("Training Rainbow/QR-DQN for Pacman (Enhanced)")
    print("=" * 60)
    print("Creating environment...")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    print("Initializing Rainbow/QR-DQN agent...")
    
    model = QRDQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
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
        tensorboard_log="./logs/enhanced_rainbow/",
        device="cuda"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/enhanced_rainbow/",
        name_prefix="pacman_rainbow_enhanced"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/enhanced_rainbow/best/",
        log_path="./logs/enhanced_rainbow/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    print("Starting training...")
    print("Monitor: tensorboard --logdir=./logs/enhanced_rainbow/")
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
    final_path = "models/enhanced_rainbow/pacman_rainbow_enhanced_final.zip"
    model.save(final_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model: {final_path}")
    print(f"Best model: models/enhanced_rainbow/best/")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    train_rainbow()
