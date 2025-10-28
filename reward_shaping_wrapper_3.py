"""
Stability-Fixed Reward Shaping Wrapper
Addresses learning instability and extreme reward variance
"""
import gymnasium as gym
import numpy as np
from ocatari.vision.utils import find_objects


class StabilityFixedRewardShaper(gym.Wrapper):
    """
    Stability-fixed reward shaper with:
    - Capped bonuses to prevent extremes
    - Normalized rewards for stable learning  
    - Gentler penalties to reduce interference
    - Progress bonus scaling for consistency
    """
    
    # === STABILITY-FIXED PARAMETERS ===
    BONUS_POWERPILL_COEF = 0.250
    POWERPILL_RADIUS = 30.0
    POWERPILL_BONUS_CAP = 5.0  # NEW: Cap powerpill bonus
    
    BONUS_EATING_GHOST_COEF = 1.0
    GHOST_CHASE_RADIUS = 20.0
    GHOST_BONUS_CAP = 20.0  # NEW: Cap ghost bonus
    
    PENALTY_NEARING_GHOST_COEF = 0.250
    GHOST_AVOID_RADIUS = 15.0
    GHOST_PENALTY_CAP = -5.0  # NEW: Cap penalty
    
    # === STABILITY FIXES ===
    PROGRESS_BONUS_SCALE = 0.5  # REDUCED: From 2.0 to 0.5
    MAX_STEPS_WITHOUT_SCORE = 100  # INCREASED: From 50 to 100
    STALLING_PENALTY_RATE = -0.005  # REDUCED: From -0.01 to -0.005
    
    MOVEMENT_BONUS = 0.01
    CONSECUTIVE_NOOP_PENALTY = -0.005  # REDUCED: From -0.01
    
    # NEW: Reward normalization
    ENABLE_REWARD_NORMALIZATION = True
    NORMALIZATION_SCALE = 10.0
    
    def __init__(self, env, enable_logging=False, log_file="stable_shaping_log.txt"):
        super().__init__(env)
        self.oc_env = self.env
        self.enable_logging = enable_logging
        self.log_file = log_file
        
        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        
        # Anti-stalling tracking
        self.steps_without_score = 0
        self.consecutive_noops = 0
        
        # Progress tracking for ghost chasing
        self.prev_ghost_distances = {}
        
    def reset(self, **kwargs):
        """Reset environment and tracking variables"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset anti-stalling counters
        self.steps_without_score = 0
        self.consecutive_noops = 0
        
        # Reset ghost distance tracking
        self.prev_ghost_distances = {}
        
        return obs, info
    
    def _get_rgb_frame(self):
        """Return RGB frame for ghost detection"""
        if hasattr(self.oc_env, "getScreenRGB"):
            frame = self.oc_env.getScreenRGB()
        else:
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        return frame
    
    def is_powered_up(self):
        """Detect if Pac-Man is powered up (blue ghosts visible)"""
        frame = self._get_rgb_frame()
        eatable_ghosts = find_objects(frame, [(66, 114, 194)], min_distance=1)
        return bool(eatable_ghosts)
    
    def _apply_caps_and_normalization(self, bonus_powerpill_raw, bonus_eating_ghost_raw, 
                                    penalty_nearing_ghost_raw):
        """Apply caps and normalization for stability"""
        
        # Apply caps to prevent extremes
        bonus_powerpill_raw = min(bonus_powerpill_raw, self.POWERPILL_BONUS_CAP)
        bonus_eating_ghost_raw = min(bonus_eating_ghost_raw, self.GHOST_BONUS_CAP)
        penalty_nearing_ghost_raw = max(penalty_nearing_ghost_raw, self.GHOST_PENALTY_CAP)
        
        return bonus_powerpill_raw, bonus_eating_ghost_raw, penalty_nearing_ghost_raw
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        objects = getattr(self.env, "objects", [])
        
        # Update score tracking
        if reward > 0:
            self.steps_without_score = 0
        else:
            self.steps_without_score += 1
        
        # === Reward shaping ===
        bonus_powerpill_raw = 0.0
        bonus_eating_ghost_raw = 0.0
        penalty_nearing_ghost_raw = 0.0
        stalling_penalty = 0.0
        movement_bonus = 0.0
        
        player = next((o for o in objects if getattr(o, "category", None) == "Player"), None)
        if player is not None:
            px, py = getattr(player, "x", 0), getattr(player, "y", 0)
            is_powered_up = self.is_powered_up()
            
            # 1. Bonus for approaching PowerPill (when not powered up)
            powerpills = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            if powerpills and not is_powered_up:
                dists = [np.linalg.norm([px - o.x, py - o.y]) for o in powerpills]
                dist = min(dists)
                bonus_powerpill_raw += 10.0 * np.exp(-dist/self.POWERPILL_RADIUS)
            
            # 2. Bonus for chasing ghosts OR penalty for being near ghosts
            ghosts = [o for o in objects if getattr(o, "category", None) == "Ghost"]
            
            if is_powered_up and ghosts:
                # Find closest ghost
                ghost_distances = [(g, np.linalg.norm([px - g.x, py - g.y])) for g in ghosts]
                closest_ghost, closest_dist = min(ghost_distances, key=lambda x: x[1])
                
                # Distance bonus for all nearby ghosts
                for g, dist in ghost_distances:
                    bonus_eating_ghost_raw += 10.0 * np.exp(-dist/self.GHOST_CHASE_RADIUS)
                
                # STABILITY FIX: Reduced progress bonus scale
                if 'closest_ghost_dist' in self.prev_ghost_distances:
                    prev_closest = self.prev_ghost_distances['closest_ghost_dist']
                    if closest_dist < prev_closest:
                        progress = prev_closest - closest_dist
                        bonus_eating_ghost_raw += self.PROGRESS_BONUS_SCALE * progress
                
                self.prev_ghost_distances['closest_ghost_dist'] = closest_dist
                
            elif not is_powered_up and ghosts:
                self.prev_ghost_distances.clear()
                
                # Penalty for being near ghosts
                for g in ghosts:
                    dist = np.linalg.norm([px - g.x, py - g.y])
                    if dist < self.GHOST_AVOID_RADIUS*2:
                        penalty_nearing_ghost_raw -= 10.0 * np.exp(-dist/self.GHOST_AVOID_RADIUS)
            
            # STABILITY FIX: Apply caps
            bonus_powerpill_raw, bonus_eating_ghost_raw, penalty_nearing_ghost_raw = \
                self._apply_caps_and_normalization(bonus_powerpill_raw, bonus_eating_ghost_raw, 
                                                 penalty_nearing_ghost_raw)
            
            # 3. GENTLER stalling penalty
            if self.steps_without_score > self.MAX_STEPS_WITHOUT_SCORE:
                excess = self.steps_without_score - self.MAX_STEPS_WITHOUT_SCORE
                stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.5)
                if is_powered_up:
                    stalling_penalty *= 0.5
            
            # 4. Movement incentive
            if action != 0:
                self.consecutive_noops = 0
                movement_bonus = self.MOVEMENT_BONUS
            else:
                self.consecutive_noops += 1
                if self.consecutive_noops > 6:
                    movement_bonus = self.CONSECUTIVE_NOOP_PENALTY * (self.consecutive_noops - 6)
        
        # Apply coefficients
        bonus_powerpill = self.BONUS_POWERPILL_COEF * bonus_powerpill_raw
        bonus_eating_ghost = self.BONUS_EATING_GHOST_COEF * bonus_eating_ghost_raw
        penalty_nearing_ghost = self.PENALTY_NEARING_GHOST_COEF * penalty_nearing_ghost_raw
        
        # STABILITY FIX: Reward normalization
        if self.ENABLE_REWARD_NORMALIZATION:
            bonus_sum = bonus_powerpill + bonus_eating_ghost + penalty_nearing_ghost + stalling_penalty + movement_bonus
            normalized_bonus = np.tanh(bonus_sum / self.NORMALIZATION_SCALE) * 2.0  # Range [-2, +2]
            shaped_reward = reward + normalized_bonus
        else:
            shaped_reward = reward + bonus_powerpill + bonus_eating_ghost + penalty_nearing_ghost + stalling_penalty + movement_bonus
        
        # Logging
        if self.enable_logging:
            self.step_count += 1
            if self.step_count % 25 == 0:
                is_powered = self.is_powered_up() if player is not None else False
                with open(self.log_file, "a") as f:
                    f.write(
                        f"is_powered_up: {is_powered}, base_reward: {reward:.3f}, "
                        f"bonus_powerpill: {bonus_powerpill_raw:.3f}, "
                        f"bonus_eating_ghost: {bonus_eating_ghost_raw:.3f}, "
                        f"penalty_nearing_ghost: {penalty_nearing_ghost_raw:.3f}, "
                        f"stalling_penalty: {stalling_penalty:.3f}, "
                        f"movement_bonus: {movement_bonus:.3f}, "
                        f"steps_without_score: {self.steps_without_score}, "
                        f"shaped_reward: {shaped_reward:.3f}\n"
                    )
        
        return obs, shaped_reward, terminated, truncated, info