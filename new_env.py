"""
Train a DQN agent to play MsPacman using Stable Baselines3 + OCAtari
DQN: Deep Q-Network - Value-based method
"""
import gymnasium as gym
import ale_py
import numpy as np
import os
import time

from ocatari.core import OCAtari
from stable_baselines3 import DQN
# removed AtariWrapper import - not compatible with OCAtari
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from ocatari.vision.utils import find_objects
from gymnasium import spaces

# Đăng ký ALE environments
gym.register_envs(ale_py)

# Tạo thư mục lưu model và log
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)


# ======= Wrapper đảm bảo image là uint8 trong [0,255] =======
class FixImageSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Box( low=0, high=255, shape=obs_space.shape, dtype=np.uint8 )

    def observation(self, obs):
        obs = np.clip(obs, 0, 255)
        return obs.astype(np.uint8)


# ======= Reward shaping wrapper (không phụ thuộc AtariWrapper) =======
class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper vừa làm reward shaping từ OCAtari objects,
    vừa đảm bảo DQN chỉ nhận ảnh RGB (vision).
    """
    def __init__(self, env):
        super().__init__(env)
        self.oc_env = self._get_inner_ocatari()
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(210, 160, 3),  # ảnh gốc Atari RGB
            dtype=np.uint8
        )

    def _get_inner_ocatari(self):
        """Truy vào tận OCAtari gốc (bỏ qua các wrapper)."""
        oc_env = self.env
        while hasattr(oc_env, "env"):
            oc_env = oc_env.env
        return oc_env

    def _get_rgb_frame(self):
        """Trả về frame RGB gốc để DQN học."""
        # OCAtari luôn có hàm này trong mode=vision hoặc mode=both
        if hasattr(self.oc_env, "get_screen_rgb"):
            frame = self.oc_env.get_screen_rgb()
        elif hasattr(self.oc_env, "getScreenRGB"):
            frame = self.oc_env.getScreenRGB()
        else:
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        return frame

    def is_powered_up(self):
        """Phát hiện ghost đang ăn được (theo màu xanh)."""
        frame = self._get_rgb_frame()
        eatable_ghosts = find_objects(frame, [66, 114, 194], min_distance=1)
        return bool(eatable_ghosts)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # luôn trả về frame ảnh cho DQN
        return self._get_rgb_frame(), info

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
            shaping_reward = 0.0
            is_powered_up = self.is_powered_up()

            # thưởng nếu gần PowerPill
            powerpills = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            if powerpills and not is_powered_up:
                dists = [np.linalg.norm([px - o.x, py - o.y]) for o in powerpills]
                dist = min(dists)
                bonus_powerpill_reward += 10.0 / (dist + 1)
                shaping_reward += 10.0 / (dist + 1)

            # phạt nếu gần Ghost
            ghosts = [o for o in objects if getattr(o, "category", None) == "Ghost"]
            for g in ghosts:
                gx, gy = g.x, g.y
                dist = np.linalg.norm([px - gx, py - gy])
                if dist < 10:
                    if is_powered_up:
                        bonus_eating_ghost_reward += 10.0 / (dist + 1)
                        shaping_reward += 10.0 / (dist + 1)
                    else:
                        penaty_nearing_ghost_reward -= 10.0 / (dist + 1)
                        shaping_reward -= 10.0 / (dist + 1)
            with open("shaping_reward_log.txt", "a") as f:
                f.write(f"is_powered_up: {is_powered_up}, base_reward: {reward:.3f}, total_shaping_reward: {shaping_reward:.3f}, bonus_powerpill_reward: {bonus_powerpill_reward:.3f}, bonus_eating_ghost_reward: {bonus_eating_ghost_reward:.3f}, penaty_nearing_ghost_reward: {penaty_nearing_ghost_reward:.3f}\n")

        # === DQN chỉ nhận ảnh ===
        rgb_frame = self._get_rgb_frame()
        return rgb_frame, reward, terminated, truncated, info


# ======= Tạo env (không dùng AtariWrapper) =======
def make_env():
    # Use mode="vision" (or "both" if you want objects + vision) -- here we use vision
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="rgb_array",
                  mode="both")   # change to "both" if you need info objects in info
    # If you want reward shaping, wrap here:
    env = RewardShapingWrapper(env)
    # Ensure observation dtype is uint8 in [0,255]
    env = FixImageSpace(env)
    env = Monitor(env)
    # print("Observation space:", env.observation_space)
    # print("Observation dtype:", env.observation_space.dtype)
    return env


# ======= Train =======
def train_dqn():
    print("=" * 60)
    print("Training DQN Agent for MsPacman with OCAtari")
    print("=" * 60)

    # Env chính
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)   # chuyển (H,W,C) -> (C,H,W)
    env = VecFrameStack(env, n_stack=4)  # stack AFTER transpose

    # Env eval
    eval_env = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    print("Initializing DQN agent...")

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
        tensorboard_log="./logs/dqn/",
        device="cuda"  # or "cpu" if you have trouble with MPS
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/dqn/",
        name_prefix="mspacman_dqn"
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

    model.learn(
        total_timesteps=750_000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    training_time = time.time() - start_time

    # Save final model
    final_model_path = "models/dqn/mspacman_dqn_final.zip"
    model.save(final_model_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Time: {training_time/3600:.2f} hours")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to models/dqn/best/")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train_dqn()
