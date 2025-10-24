"""
Train an A2C agent to play MsPacman using Stable Baselines3 + OCAtari
A2C: Advantage Actor-Critic - Policy-based method
"""
import gymnasium as gym
import ale_py
import numpy as np
import os
import time

from ocatari.core import OCAtari
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

# Register ALE environments
gym.register_envs(ale_py)

# Create directories for saving models and logs
os.makedirs("models/a2c", exist_ok=True)
os.makedirs("logs/a2c", exist_ok=True)


class OCAtariVisionWrapper(gym.Wrapper):
    """
    Wrapper to ensure OCAtari returns RGB frames instead of object representations.
    Forces observation space to be (210, 160, 3) uint8 for CNN compatibility.
    """
    def __init__(self, env):
        super().__init__(env)
        self.oc_env = self._get_inner_ocatari()
        # Override observation space to RGB frame
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(210, 160, 3),
            dtype=np.uint8
        )

    def _get_inner_ocatari(self):
        """Get the actual OCAtari environment (unwrap all wrappers)."""
        oc_env = self.env
        while hasattr(oc_env, "env"):
            oc_env = oc_env.env
        return oc_env

    def _get_rgb_frame(self):
        """Return RGB frame for A2C to process."""
        if hasattr(self.oc_env, "getScreenRGB"):
            frame = self.oc_env.getScreenRGB()
        else:
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        return frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Always return RGB frame
        return self._get_rgb_frame(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Always return RGB frame
        rgb_frame = self._get_rgb_frame()
        return rgb_frame, reward, terminated, truncated, info


def make_env():
    """Create and wrap the MsPacman environment using OCAtari"""
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="rgb_array",
                  mode="vision")
    env = OCAtariVisionWrapper(env)
    env = Monitor(env)
    return env

def train_a2c():
    """Train the A2C agent"""
    print("=" * 60)
    print("Training A2C Agent for MsPacman with OCAtari")
    print("=" * 60)
    print("Creating environment...")

    # Create vectorized environment (A2C works better with multiple parallel envs)
    env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel environments
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

