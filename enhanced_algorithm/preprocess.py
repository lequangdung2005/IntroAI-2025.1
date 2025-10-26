"""
Preprocessing wrapper for OCAtari environments
Handles frame preprocessing for enhanced Pacman training
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2


class PreprocessFrame(gym.ObservationWrapper):
    """
    Preprocess OCAtari frames for RL training
    - Convert to grayscale
    - Resize to specified dimensions
    - Normalize pixel values
    - Handle both vision mode and standard rendering
    """
    
    def __init__(self, env, width=84, height=84, grayscale=False, force_image=False):
        """
        Initialize preprocessing wrapper
        
        Args:
            env: OCAtari environment
            width: Target frame width (default: 84)
            height: Target frame height (default: 84)
            grayscale: Convert to grayscale (default: False for RGB)
            force_image: Force image observation even if vision mode (default: False)
        """
        super(PreprocessFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.force_image = force_image
        
        # Set observation space - default to RGB for VecTransposeImage compatibility
        if self.grayscale:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8
            )
    
    def observation(self, obs):
        """
        Process observation frame
        
        Args:
            obs: Raw observation from environment (RGB frame from ALE)
            
        Returns:
            Preprocessed frame (grayscale or RGB)
        """
        # obs is already the RGB frame from gym.make with render_mode="rgb_array"
        frame = obs
        
        # Ensure frame is numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Resize frame
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if requested
        if self.grayscale:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        return frame


class ScaleFrame(gym.ObservationWrapper):
    """
    Scale frame values to [0, 1] range
    Useful for some RL algorithms that expect normalized inputs
    """
    
    def __init__(self, env):
        super(ScaleFrame, self).__init__(env)
        low = np.zeros(self.observation_space.shape, dtype=np.float32)
        high = np.ones(self.observation_space.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def observation(self, obs):
        """Scale observation to [0, 1]"""
        return obs.astype(np.float32) / 255.0


class RepeatAction(gym.Wrapper):
    """
    Repeat action for multiple steps
    Useful for speeding up training
    """
    
    def __init__(self, env, repeat=4):
        """
        Initialize action repeat wrapper
        
        Args:
            env: Environment to wrap
            repeat: Number of times to repeat each action (default: 4)
        """
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat
    
    def step(self, action):
        """
        Step with action repeat
        
        Args:
            action: Action to repeat
            
        Returns:
            Last observation, accumulated reward, done flag, info
        """
        total_reward = 0.0
        done = False
        info = {}
        
        for _ in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        
        return obs, total_reward, done, truncated, info


class ClipReward(gym.RewardWrapper):
    """
    Clip reward to {-1, 0, +1}
    Common preprocessing for Atari games
    """
    
    def __init__(self, env):
        super(ClipReward, self).__init__(env)
    
    def reward(self, reward):
        """Clip reward to {-1, 0, +1}"""
        return np.sign(reward)


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observation using running statistics
    """
    
    def __init__(self, env, epsilon=1e-8):
        super(NormalizeObservation, self).__init__(env)
        self.epsilon = epsilon
        self.running_mean = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.running_var = np.ones(self.observation_space.shape, dtype=np.float32)
        self.count = 0
    
    def observation(self, obs):
        """Normalize observation using running statistics"""
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += delta * delta2
        
        std = np.sqrt(self.running_var / self.count + self.epsilon)
        return (obs - self.running_mean) / std


def make_preprocessed_env(env_name="ALE/Pacman-v5", width=84, height=84, 
                          grayscale=False, scale=False, clip_reward=False):
    """
    Create preprocessed environment with standard preprocessing pipeline
    
    Args:
        env_name: Environment name (default: "ALE/Pacman-v5")
        width: Frame width (default: 84)
        height: Frame height (default: 84)
        grayscale: Convert to grayscale (default: False for RGB)
        scale: Scale to [0, 1] (default: False)
        clip_reward: Clip rewards to {-1, 0, +1} (default: False)
        
    Returns:
        Preprocessed environment
    """
    import gymnasium as gym
    
    # Create standard gym environment
    env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
    
    # Apply preprocessing
    env = PreprocessFrame(env, width=width, height=height, grayscale=grayscale, force_image=True)
    
    if scale:
        env = ScaleFrame(env)
    
    if clip_reward:
        env = ClipReward(env)
    
    return env


if __name__ == "__main__":
    """Test preprocessing pipeline"""
    print("Testing preprocessing pipeline...")
    
    # Create environment
    env = make_preprocessed_env(
        env_name="ALE/Pacman-v5",
        width=84,
        height=84,
        grayscale=False,  # RGB for VecTransposeImage
        scale=False
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Reset observation dtype: {obs.dtype}")
    print(f"Reset observation range: [{obs.min()}, {obs.max()}]")
    
    # Test step
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    print(f"Step observation shape: {obs.shape}")
    print(f"Step reward: {reward}")
    
    env.close()
    print("✅ Preprocessing test passed!")
