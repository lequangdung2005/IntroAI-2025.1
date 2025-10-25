"""
Train an on-policy agent (A2C) to play MsPacman using Stable Baselines3 + OCAtari
This file was converted from the previous DQN-based implementation to A2C to avoid
large replay buffer memory usage on constrained platforms (e.g. Kaggle).
"""
import gymnasium as gym
import ale_py
import numpy as np
import os
import time
import cv2

from ocatari.core import OCAtari
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from ocatari.vision.utils import find_objects
from gymnasium import spaces
from preprocess import PreprocessFrame

# Đăng ký ALE environments
gym.register_envs(ale_py)

# Tạo thư mục lưu model và log
os.makedirs("models/a2c", exist_ok=True)
os.makedirs("logs/a2c", exist_ok=True)


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
            penaty_nearing_ghost_reward = 0.0
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
                        penaty_nearing_ghost_reward -= 10.0 / (dist + 1)
            # Chỉ ghi log mỗi 100 bước
            self.step_count += 1
            if self.step_count % 50 == 0:
                with open("shaping_reward_log.txt", "a") as f:
                    f.write(f"step: {self.step_count}, is_powered_up: {is_powered_up}, base_reward: {reward:.3f}, bonus_powerpill_reward: {bonus_powerpill_reward:.3f}, bonus_eating_ghost_reward: {bonus_eating_ghost_reward:.3f}, penaty_nearing_ghost_reward: {penaty_nearing_ghost_reward:.3f}\n")
        return obs, reward, terminated, truncated, info


# ======= Tạo env (không dùng AtariWrapper) =======
def make_env():
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="rgb_array",
                  mode="vision")
    env = RewardShapingWrapper(env)
    env = Monitor(env)
    # Preprocess as outermost wrapper (force because we requested vision)
    env = PreprocessFrame(env, width=84, height=84, force_image=True)
    return env


def train_a2c():
    """Train the A2C agent"""
    print("=" * 60)
    print("Training A2C Agent for MsPacman with OCAtari")
    print("=" * 60)
    print("Creating environment...")

    # Env chính
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)   # chuyển (H,W,C) -> (C,H,W)
    env = VecFrameStack(env, n_stack=4)  # stack AFTER transpose

    eval_env = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    print("Initializing A2C agent...")

    # A2C hyperparameters optimized for Atari

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
        tensorboard_log="./logs/a2c/",
        device="auto"
    )

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./models/a2c/", name_prefix="mspacman_a2c")
    eval_callback = EvalCallback(eval_env, best_model_save_path="./models/a2c/best/", log_path="./logs/a2c/eval/", eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False)

    start_time = time.time()
    model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, eval_callback], log_interval=10)
    training_time = time.time() - start_time

    final_model_path = "models/a2c/mspacman_a2c_final.zip"
    model.save(final_model_path)

    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    return model


if __name__ == "__main__":
    train_a2c()