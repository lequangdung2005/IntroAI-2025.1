"""
Enhanced Reward Shaping Wrapper for Pacman
Implements multiple reward bonuses to boost learning performance
"""
import gymnasium as gym
import numpy as np
from ocatari.vision.utils import find_objects


class EnhancedPacmanRewardWrapper(gym.Wrapper):
    """
    Enhanced reward shaping wrapper for Pacman with multiple reward components:
    1. Dot collection bonus (encourage eating pellets)
    2. Power pellet seeking (when ghosts are dangerous)
    3. Ghost hunting (when powered up)
    4. Ghost avoidance (when vulnerable)
    5. Survival bonus (staying alive)
    6. Movement efficiency (avoid standing still)
    7. Corner penalty (avoid getting trapped)
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.oc_env = self._get_inner_ocatari()
        self.prev_score = 0
        self.prev_lives = 3
        self.prev_player_pos = None
        self.steps_without_score = 0
        self.total_steps = 0
        
    def _get_inner_ocatari(self):
        """Navigate to the OCAtari environment through wrapper chain"""
        oc_env = self.env
        while hasattr(oc_env, "env"):
            oc_env = oc_env.env
        return oc_env
    
    def _get_rgb_frame(self):
        """Get RGB frame from OCAtari"""
        if hasattr(self.oc_env, "getScreenRGB"):
            return self.oc_env.getScreenRGB()
        return np.zeros((210, 160, 3), dtype=np.uint8)
    
    def is_powered_up(self):
        """Detect if Pacman is powered up (ghosts turn blue)"""
        frame = self._get_rgb_frame()
        # Blue ghosts color: (66, 114, 194)
        eatable_ghosts = find_objects(frame, [(66, 114, 194)], min_distance=1)
        return bool(eatable_ghosts)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_score = 0
        self.prev_lives = 3
        self.prev_player_pos = None
        self.steps_without_score = 0
        self.total_steps = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get game objects
        objects = getattr(self.env, "objects", [])
        player = next((o for o in objects if getattr(o, "category", None) == "Player"), None)
        
        if player is None:
            return obs, reward, terminated, truncated, info
        
        # Current state
        current_score = info.get('score', 0)
        current_lives = info.get('lives', 3)
        px, py = getattr(player, "x", 0), getattr(player, "y", 0)
        
        # Initialize shaped reward
        shaped_reward = reward
        
        # Track steps without scoring
        if current_score > self.prev_score:
            self.steps_without_score = 0
        else:
            self.steps_without_score += 1
        
        # ===== 1. Score increase bonus (scale original reward) =====
        score_diff = current_score - self.prev_score
        if score_diff > 0:
            shaped_reward += score_diff * 0.01  # Small bonus for scoring
        
        # ===== 2. Power pellet bonus =====
        is_powered = self.is_powered_up()
        if not is_powered:
            # Encourage seeking power pellets when vulnerable
            powerpills = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            if powerpills:
                distances = [np.linalg.norm([px - o.x, py - o.y]) for o in powerpills]
                min_dist = min(distances)
                shaped_reward += 5.0 / (min_dist + 1)  # Closer = higher bonus
        
        # ===== 3. Ghost interaction =====
        ghosts = [o for o in objects if getattr(o, "category", None) == "Ghost"]
        for ghost in ghosts:
            gx, gy = ghost.x, ghost.y
            dist = np.linalg.norm([px - gx, py - gy])
            
            if is_powered:
                # Encourage chasing ghosts when powered up
                if dist < 20:
                    shaped_reward += 15.0 / (dist + 1)
            else:
                # Penalize being near ghosts when vulnerable
                if dist < 15:
                    shaped_reward -= 20.0 / (dist + 1)
        
        # ===== 4. Dot collection encouragement =====
        dots = [o for o in objects if getattr(o, "category", None) in ["Dot", "Pellet"]]
        if dots and not is_powered:
            # Encourage going towards nearest dot
            distances = [np.linalg.norm([px - o.x, py - o.y]) for o in dots[:5]]  # Check nearest 5
            if distances:
                min_dist = min(distances)
                shaped_reward += 2.0 / (min_dist + 1)
        
        # ===== 5. Movement bonus (avoid standing still) =====
        if self.prev_player_pos is not None:
            prev_px, prev_py = self.prev_player_pos
            moved = abs(px - prev_px) + abs(py - prev_py)
            if moved > 0:
                shaped_reward += 0.1  # Small bonus for moving
            else:
                shaped_reward -= 0.5  # Penalty for standing still
        
        # ===== 6. Corner penalty (avoid getting trapped) =====
        # Simple heuristic: penalize being near edges with ghosts nearby
        is_near_edge = (px < 20 or px > 140 or py < 20 or py > 190)
        if is_near_edge and ghosts and not is_powered:
            nearest_ghost_dist = min([np.linalg.norm([px - g.x, py - g.y]) for g in ghosts])
            if nearest_ghost_dist < 30:
                shaped_reward -= 5.0
        
        # ===== 7. Survival bonus =====
        if not terminated:
            shaped_reward += 0.05  # Small bonus for staying alive
        
        # ===== 8. Death penalty =====
        if current_lives < self.prev_lives:
            shaped_reward -= 100  # Large penalty for dying
        
        # ===== 9. Efficiency penalty (encourage scoring) =====
        if self.steps_without_score > 100:
            shaped_reward -= 0.01 * (self.steps_without_score - 100)
        
        # ===== 10. Level completion bonus =====
        if terminated and current_lives >= self.prev_lives:
            shaped_reward += 200  # Huge bonus for completing level
        
        # Update previous state
        self.prev_score = current_score
        self.prev_lives = current_lives
        self.prev_player_pos = (px, py)
        self.total_steps += 1
        
        # Logging (every 100 steps)
        if self.total_steps % 100 == 0:
            with open("enhanced_reward_log.txt", "a") as f:
                f.write(f"Step {self.total_steps}: powered={is_powered}, "
                       f"base_reward={reward:.2f}, shaped_reward={shaped_reward:.2f}, "
                       f"score={current_score}, lives={current_lives}\n")
        
        return obs, shaped_reward, terminated, truncated, info
