"""
Train a PPO agent to play MsPacman using Stable Baselines3 + OCAtari with reward shaping
PPO: Proximal Policy Optimization - Policy-based method with reward logging
"""
import gymnasium as gym
import ale_py
import numpy as np
import os
import time

from ocatari.core import OCAtari
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from ocatari.vision.utils import find_objects
from preprocess import PreprocessFrame

# Đăng ký ALE environments
gym.register_envs(ale_py)

# Tạo thư mục lưu model và log
os.makedirs("models/ppo", exist_ok=True)
os.makedirs("logs/ppo", exist_ok=True)


# ======= Reward shaping wrapper (fix tìm env có getScreenRGB) =======
class RewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping wrapper that searches the wrapper chain for an environment
    providing getScreenRGB (OCAtari) and uses it to return RGB frames.
    """
    def __init__(self, env):
        super().__init__(env)
        self.oc_env = self._get_inner_ocatari()
        self.step_count = 0

    def _get_inner_ocatari(self):
        """Truy vào tận OCAtari gốc (bỏ qua các wrapper)."""
        oc_env = self.env
        while hasattr(oc_env, "env"):
            oc_env = oc_env.env
        return oc_env

    def _get_rgb_frame(self):
        """Trả về frame RGB gốc để DQN học."""
        # OCAtari luôn có hàm này trong mode=vision
        if hasattr(self.oc_env, "getScreenRGB"):
            frame = self.oc_env.getScreenRGB()
        else:
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        return frame

    def is_powered_up(self):
        """Phát hiện ghost đang ăn được (theo màu xanh)."""
        frame = self._get_rgb_frame()
        # find_objects expects list of RGB tuples
        eatable_ghosts = find_objects(frame, [(66, 114, 194)], min_distance=1)
        return bool(eatable_ghosts)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        objects = getattr(self.env, "objects", [])
        # === Reward shaping ===
        player = next((o for o in objects if getattr(o, "category", None) == "Player"), None)
        if player is not None:
            px, py = getattr(player, "x", 0), getattr(player, "y", 0)
            bonus_powerpill_reward = 0.0
            bonus_eating_ghost_reward = 0.0
            penalty_nearing_ghost_reward = 0.0
            is_powered_up = self.is_powered_up()

            # thưởng nếu gần PowerPill
            powerpills = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            if powerpills and not is_powered_up:
                dists = [np.linalg.norm([px - o.x, py - o.y]) for o in powerpills]
                dist = min(dists)
                bonus_powerpill_reward += 10.0 / (dist + 1)

            # phạt nếu gần Ghost
            ghosts = [o for o in objects if getattr(o, "category", None) == "Ghost"]
            for g in ghosts:
                gx, gy = g.x, g.y
                dist = np.linalg.norm([px - gx, py - gy])
                if dist < 10:
                    if is_powered_up:
                        bonus_eating_ghost_reward += 10.0 / (dist + 1)
                    else:
                        penalty_nearing_ghost_reward -= 10.0 / (dist + 1)
            # Chỉ ghi log mỗi 100 bước
            self.step_count += 1
            if self.step_count % 50 == 0:
                with open("shaping_reward_log.txt", "a") as f:
                    f.write(f"is_powered_up: {is_powered_up}, base_reward: {reward:.3f}, bonus_powerpill_reward: {bonus_powerpill_reward:.3f}, bonus_eating_ghost_reward: {bonus_eating_ghost_reward:.3f}, penalty_nearing_ghost_reward: {penalty_nearing_ghost_reward:.3f}\n")
        return obs, reward, terminated, truncated, info


# ======= Tạo env (không dùng AtariWrapper) =======
def make_env():
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="rgb_array",
                  mode="vision")
    env = RewardShapingWrapper(env)
    env = PreprocessFrame(env, width=84, height=84, force_image=True)
    env = Monitor(env)
    return env


def train_ppo():
    """Train the PPO agent"""
    print("=" * 60)
    print("Training PPO Agent for MsPacman with OCAtari")
    print("=" * 60)
    print("Creating environment...")

    # Create vectorized environment (PPO works better with multiple parallel envs)
    env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel environments
    env = VecTransposeImage(env)  # Transpose (H,W,C) -> (C,H,W) for CNN
    env = VecFrameStack(env, n_stack=4)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env)
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
        device="cuda"
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/ppo/",
        name_prefix="mspacman_ppo"
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
    final_model_path = "models/ppo/mspacman_ppo_final.zip"
    model.save(final_model_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to models/ppo/best/")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train_ppo()