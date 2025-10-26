"""Shared PreprocessFrame wrapper for OCAtari projects.
Converts image observations (H,W,C) into grayscale 84x84x1 uint8 for memory savings.
If the environment returns non-image observations (object-mode), the wrapper will
attempt to obtain an RGB frame from an inner OCAtari using getScreenRGB() or
env.render(); otherwise it passes observations through unchanged.
"""
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces


class PreprocessFrame(gym.ObservationWrapper):
    """Convert RGB frames to grayscale 84x84x1 uint8, safe for object-mode envs.

    Behavior:
    - If the wrapped env's observation_space is an image Box (ndim==3), the wrapper
      will preprocess returned image observations.
    - Otherwise, it will try to find an inner env that exposes getScreenRGB() and use
      that to obtain frames for preprocessing.
    - If neither is available, observations are returned unchanged (no-op).

    This makes it safe to add to OCAtari envs whether they are in vision or objects mode.
    """

    def __init__(self, env, width: int = 84, height: int = 84, force_image: bool = False):
        super().__init__(env)
        self.width = int(width)
        self.height = int(height)
        self._is_image = False
        obs_space = getattr(env, "observation_space", None)

        # Find inner env providing getScreenRGB (OCAtari)
        self._screen_env = None
        e = env
        visited = set()
        while True:
            if id(e) in visited:
                break
            visited.add(id(e))
            if hasattr(e, "getScreenRGB"):
                self._screen_env = e
                break
            if not hasattr(e, "env"):
                break
            e = e.env

        # If forced image mode, or declared observation space is image-like, mark as image
        if force_image or (isinstance(obs_space, spaces.Box) and hasattr(obs_space, "shape") and len(obs_space.shape) == 3):
            self._is_image = True
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        elif self._screen_env is not None:
            # inner env can render frames -> treat as image mode
            self._is_image = True
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            # passthrough: keep original space (object-mode or other)
            self.observation_space = obs_space

    def _process_image(self, frame):
        if frame is None:
            return np.zeros((self.height, self.width, 1), dtype=np.uint8)
        img = np.asarray(frame)
        # handle common shapes and channel orders
        if img.ndim == 3:
            ch = img.shape[2]
            if ch == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif ch == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif ch == 1:
                gray = img[:, :, 0]
            else:
                gray = np.mean(img, axis=2).astype(np.uint8)
        elif img.ndim == 2:
            gray = img.astype(np.uint8)
        else:
            # unexpected shape
            return np.zeros((self.height, self.width, 1), dtype=np.uint8)

        # resize to target
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized.astype(np.uint8), -1)

    def observation(self, obs):
        # If not image-mode, try to obtain a frame from inner env if available
        if not self._is_image:
            return obs

        # If obs looks like an image, process directly
        try:
            arr = np.asarray(obs)
            if arr.ndim == 3 or arr.ndim == 2:
                return self._process_image(arr)
        except Exception:
            pass

        # If obs isn't an image array, try the inner screen env (OCAtari)
        if self._screen_env is not None and hasattr(self._screen_env, "getScreenRGB"):
            try:
                frame = self._screen_env.getScreenRGB()
                return self._process_image(frame)
            except Exception:
                pass

        # fallback: return zeros image to preserve downstream shapes
        return np.zeros((self.height, self.width, 1), dtype=np.uint8)
