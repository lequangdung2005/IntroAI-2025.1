"""
Enhanced Reward Shaping Wrapper for Pacman
Implements multiple reward bonuses to boost learning performance
Works with standard Gymnasium environments without requiring OCAtari
"""
import gymnasium as gym
import numpy as np


class EnhancedPacmanRewardWrapper(gym.Wrapper):
    """
    Enhanced reward shaping wrapper for Pacman with multiple reward components:
    1. Score bonus (amplify game score)
    2. Survival bonus (staying alive)
    3. Movement efficiency (avoid standing still)
    4. Score progress bonus (encourage consistent scoring)
    5. Death penalty (discourage dying)
    6. Level completion bonus
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0
        self.prev_lives = 3
        self.steps_without_score = 0
        self.total_steps = 0
        self.max_score = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_score = 0
        self.prev_lives = 3
        self.steps_without_score = 0
        self.total_steps = 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current game state from info
        current_score = info.get('episode_frame_number', 0) if 'episode_frame_number' in info else 0
        
        # Try to extract lives from ALE
        try:
            if hasattr(self.env.unwrapped, 'ale'):
                current_lives = self.env.unwrapped.ale.lives()
            else:
                current_lives = self.prev_lives
        except:
            current_lives = self.prev_lives
        
        # Initialize shaped reward with base reward
        shaped_reward = reward
        
        # Track steps without scoring
        score_from_reward = reward
        if score_from_reward > 0:
            self.steps_without_score = 0
            self.max_score = max(self.max_score, score_from_reward)
        else:
            self.steps_without_score += 1
        
        # ===== 1. Score bonus (amplify positive rewards) =====
        if reward > 0:
            shaped_reward += reward * 0.1  # 10% bonus on score
        
        # ===== 2. Survival bonus =====
        if not terminated:
            shaped_reward += 0.1  # Small bonus for staying alive each step
        
        # ===== 3. Movement bonus (encourage exploration) =====
        if action != 0:  # 0 is typically NOOP
            shaped_reward += 0.5  # Bonus for taking action
        
        # ===== 4. Score progress bonus =====
        # Bonus for maintaining scoring streak
        if self.steps_without_score == 0:
            shaped_reward += 2.0  # Just scored!
        elif self.steps_without_score < 50:
            shaped_reward += 0.5  # Recently scored
        
        # ===== 5. Death penalty =====
        if current_lives < self.prev_lives:
            shaped_reward -= 100  # Large penalty for dying
        
        # ===== 6. Stagnation penalty =====
        # Penalize if not scoring for too long
        if self.steps_without_score > 100:
            shaped_reward -= 0.5
        
        # ===== 7. Level completion bonus =====
        if terminated and current_lives >= self.prev_lives:
            shaped_reward += 200  # Huge bonus for completing level
        
        # ===== 8. Consistent performance bonus =====
        if score_from_reward > self.max_score * 0.8:
            shaped_reward += 1.0  # Bonus for high-scoring actions
        
        # Update state
        self.prev_score = score_from_reward
        self.prev_lives = current_lives
        self.total_steps += 1
        
        return obs, shaped_reward, terminated, truncated, info

