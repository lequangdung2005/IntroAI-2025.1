"""
Advanced Reward Shaping Wrapper v4
Improved version based on analysis and feedback from wrapper_3
Addresses all identified issues and improves agent survival
"""
import gymnasium as gym
import numpy as np
from ocatari.vision.utils import find_objects


class AdvancedRewardShaper(gym.Wrapper):
    """
    Advanced reward shaper v4 with:
    - Fixed PowerPill camping parameters based on distance analysis
    - Improved position tracking using variance instead of simple distance
    - Enhanced ghost interaction mechanics for better survival
    - Corrected reward normalization logic
    - Better penalty balancing for longer survival
    """
    
    # === POWERPILL PARAMETERS (Updated based on distance analysis) ===
    BONUS_POWERPILL_COEF = 0.250
    POWERPILL_RADIUS = 30.0
    POWERPILL_BONUS_CAP = 6.0
    POWERPILL_CAMPING_RADIUS = 22.0  # UPDATED: From 10.0 to 22.0 based on analysis
    POWERPILL_CAMPING_THRESHOLD = 1.5  # UPDATED: From 1.0 to 1.5 for better progress detection
    
    # === GHOST PARAMETERS (Improved for better survival) ===
    BONUS_EATING_GHOST_COEF = 1.0
    GHOST_CHASE_RADIUS = 30.0  # UPDATED: From 20.0 to 30.0 for wider detection
    GHOST_BONUS_CAP = 20.0
    
    PENALTY_NEARING_GHOST_COEF = 0.350  # INCREASED: From 0.250 for stronger avoidance
    GHOST_AVOID_RADIUS = 27.5  # UPDATED: From 15.0 to 27.5 for earlier warning
    GHOST_PENALTY_CAP = -7.5  # INCREASED: From -5.0 for stronger deterrent
    
    # === SURVIVAL IMPROVEMENTS ===
    PROGRESS_BONUS_SCALE = 0.5  # Progress bonus scale for ghost chasing
    MAX_STEPS_WITHOUT_SCORE = 75  # INCREASED: From 70 to allow more survival time
    STALLING_PENALTY_RATE = -0.010  # REDUCED: From -0.015 to be less harsh
    
    # === MOVEMENT TRACKING (Improved with variance) ===
    POSITION_TRACKING_WINDOW = 10
    MIN_POSITION_VARIANCE = 25.0  # NEW: Minimum variance in position (not distance)
    STUCK_PENALTY = -0.1  # Penalty coefficient for stuck behavior (cap: -0.1*25=-2.5)
    
    # === LIFE MANAGEMENT ===
    LIFE_LOSS_PENALTY = -25.0  # REDUCED: From -50.0 to -25.0 (less harsh)
    ENABLE_LIFE_LOSS_TRACKING = True
    
    # === REWARD NORMALIZATION (Fixed logic) ===
    ENABLE_REWARD_NORMALIZATION = True
    SHAPING_NORMALIZATION_SCALE = 5.0  # NEW: Separate scale for shaping rewards
    
    def __init__(self, env, enable_logging=False, log_file="advanced_shaping_log.txt"):
        super().__init__(env)
        self.oc_env = self.env
        self.enable_logging = enable_logging
        self.log_file = log_file
        
        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        
        # Anti-stalling tracking
        self.steps_without_score = 0
        
        # Improved position tracking
        self.position_history = []  # List of (x, y) tuples
        self.prev_position = None
        
        # Progress tracking for ghost chasing
        self.prev_ghost_distances = {}
        
        # PowerPill camping detection
        self.powerpill_min_distances = {}
        
        # Life tracking
        self.prev_lives = None
        
    def reset(self, **kwargs):
        """Reset environment and tracking variables"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset all tracking
        self.steps_without_score = 0
        self.position_history = []
        self.prev_position = None
        self.prev_ghost_distances = {}
        self.powerpill_min_distances = {}
        
        # Initialize life tracking
        if self.ENABLE_LIFE_LOSS_TRACKING:
            self.prev_lives = info.get('lives', None)
            if self.prev_lives is None and hasattr(self.env.unwrapped, 'ale'):
                self.prev_lives = self.env.unwrapped.ale.lives()
        else:
            self.prev_lives = None
        
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
    

    
    def _calculate_position_variance(self):
        """Calculate variance in positions to detect if stuck in area"""
        if len(self.position_history) < 5:
            return float('inf')  # Not enough data
        
        positions = np.array(self.position_history)
        x_var = np.var(positions[:, 0])
        y_var = np.var(positions[:, 1])
        
        return x_var + y_var  # Total variance
    
    def _apply_caps_and_normalization(self, bonus_powerpill_raw, bonus_eating_ghost_raw, 
                                    penalty_nearing_ghost_raw):
        """Apply caps for stability"""
        bonus_powerpill_raw = min(bonus_powerpill_raw, self.POWERPILL_BONUS_CAP)
        bonus_eating_ghost_raw = min(bonus_eating_ghost_raw, self.GHOST_BONUS_CAP)
        penalty_nearing_ghost_raw = max(penalty_nearing_ghost_raw, self.GHOST_PENALTY_CAP)
        
        return bonus_powerpill_raw, bonus_eating_ghost_raw, penalty_nearing_ghost_raw
    
    def _scale_base_reward(self, reward):
        """
        Simplified base reward scaling
        If reward > 800: 80 + (reward - 800) * 0.025
        Else: reward / 10
        """
        if reward <= 0:
            return reward  # No scaling for zero/negative rewards
        
        if reward > 800:
            return 80 + (reward - 800) * 0.025
        else:
            return reward / 10
    
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
            
            # 1. PowerPill bonus (closest only, improved camping detection)
            powerpills = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            if powerpills and not is_powered_up:
                # Find closest PowerPill
                closest_pill = None
                closest_dist = float('inf')
                for pill in powerpills:
                    dist = np.linalg.norm([px - pill.x, py - pill.y])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_pill = pill
                
                if closest_pill is not None:
                    current_pill_positions = {(int(o.x), int(o.y)) for o in powerpills}
                    closest_pill_pos = (int(closest_pill.x), int(closest_pill.y))
                    
                    # Clean up tracking
                    self.powerpill_min_distances = {
                        pos: dist for pos, dist in self.powerpill_min_distances.items()
                        if pos in current_pill_positions
                    }
                    
                    # Apply improved anti-camping logic
                    if closest_dist <= self.POWERPILL_CAMPING_RADIUS:
                        if closest_pill_pos not in self.powerpill_min_distances:
                            # First entry to camping zone
                            self.powerpill_min_distances[closest_pill_pos] = closest_dist
                            bonus_powerpill_raw += 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                        else:
                            # Check for progress with improved threshold
                            min_dist = self.powerpill_min_distances[closest_pill_pos]
                            if closest_dist < min_dist - self.POWERPILL_CAMPING_THRESHOLD:
                                self.powerpill_min_distances[closest_pill_pos] = closest_dist
                                bonus_powerpill_raw += 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                            # else: No bonus - camping detected
                    else:
                        # Outside camping zone - normal bonus
                        if closest_pill_pos in self.powerpill_min_distances:
                            del self.powerpill_min_distances[closest_pill_pos]
                        bonus_powerpill_raw += 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
            
            # 2. Improved ghost interaction for better survival
            ghosts = [o for o in objects if getattr(o, "category", None) == "Ghost"]
            
            if is_powered_up and ghosts:
                # Ghost chasing (when powered up)
                ghost_distances = [(g, np.linalg.norm([px - g.x, py - g.y])) for g in ghosts]
                closest_ghost, closest_dist = min(ghost_distances, key=lambda x: x[1])
                
                # Distance bonus for nearby ghosts (wider radius)
                for g, dist in ghost_distances:
                    if dist <= self.GHOST_CHASE_RADIUS:
                        bonus_eating_ghost_raw += 10.0 * np.exp(-dist/self.GHOST_CHASE_RADIUS)
                
                # Reduced progress bonus to prevent over-aggressive chasing
                if 'closest_ghost_dist' in self.prev_ghost_distances:
                    prev_closest = self.prev_ghost_distances['closest_ghost_dist']
                    if closest_dist < prev_closest:
                        progress = prev_closest - closest_dist
                        bonus_eating_ghost_raw += self.PROGRESS_BONUS_SCALE * progress
                
                self.prev_ghost_distances['closest_ghost_dist'] = closest_dist
                
            elif not is_powered_up and ghosts:
                # Ghost avoidance (when not powered up) - IMPROVED
                self.prev_ghost_distances.clear()
                
                # Stronger penalty for being near ghosts (earlier warning)
                for g in ghosts:
                    dist = np.linalg.norm([px - g.x, py - g.y])
                    if dist <= self.GHOST_AVOID_RADIUS:
                        # Exponential penalty that gets stronger as distance decreases
                        penalty_nearing_ghost_raw -= 15.0 * np.exp(-dist/self.GHOST_AVOID_RADIUS)
            
            # Apply caps
            bonus_powerpill_raw, bonus_eating_ghost_raw, penalty_nearing_ghost_raw = \
                self._apply_caps_and_normalization(bonus_powerpill_raw, bonus_eating_ghost_raw, 
                                                 penalty_nearing_ghost_raw)
            
            # 3. Improved stalling penalty with powered-up urgency
            if is_powered_up:
                # When powered up, use stricter threshold (half the normal steps)
                threshold = self.MAX_STEPS_WITHOUT_SCORE * 0.5
                if self.steps_without_score > threshold:
                    excess = self.steps_without_score - threshold
                    stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.0)
            else:
                # Normal threshold when not powered up
                if self.steps_without_score > self.MAX_STEPS_WITHOUT_SCORE:
                    excess = self.steps_without_score - self.MAX_STEPS_WITHOUT_SCORE
                    stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.0)
            
            # 4. Improved movement tracking using position variance
            self.position_history.append((px, py))
            
            if len(self.position_history) > self.POSITION_TRACKING_WINDOW:
                self.position_history.pop(0)
            
            # Check if stuck using variance instead of distance
            if len(self.position_history) >= self.POSITION_TRACKING_WINDOW:
                position_variance = self._calculate_position_variance()
                
                if position_variance < self.MIN_POSITION_VARIANCE:
                    movement_bonus = self.STUCK_PENALTY * (self.MIN_POSITION_VARIANCE - position_variance)  # penalty
                    
                    # EXTREME PENALTY for being completely stuck (variance = 0)
                    if position_variance == 0.0:
                        # Same position repeatedly - apply extreme penalty
                        movement_bonus *= 5.0  # 5x penalty for complete stillness
                    
                    if is_powered_up:
                        movement_bonus *= 1.5  # Increased penalty when powered up
        
        # Apply coefficients
        bonus_powerpill = self.BONUS_POWERPILL_COEF * bonus_powerpill_raw
        bonus_eating_ghost = self.BONUS_EATING_GHOST_COEF * bonus_eating_ghost_raw
        penalty_nearing_ghost = self.PENALTY_NEARING_GHOST_COEF * penalty_nearing_ghost_raw
        
        # FIXED: Proper reward normalization logic with EMERGENCY ESCAPE mechanism
        if self.ENABLE_REWARD_NORMALIZATION:
            # Use non-linear base reward scaling: 10->1, 50->5, 200->20, etc.
            scaled_base_reward = self._scale_base_reward(reward)
            
            # Calculate bonus sum
            bonus_sum = bonus_powerpill + bonus_eating_ghost + penalty_nearing_ghost + stalling_penalty + movement_bonus
            
            # EMERGENCY ESCAPE: If stuck (movement_bonus < -10) AND ghost nearby (penalty_nearing_ghost < -1.0)
            is_stuck = movement_bonus < -10.0
            is_ghost_nearby = penalty_nearing_ghost < -1.0
            
            if is_stuck and is_ghost_nearby:
                # BYPASS NORMALIZATION for emergency situation - apply raw penalties
                emergency_penalty = penalty_nearing_ghost + movement_bonus  # Raw values without normalization
                shaped_reward = scaled_base_reward + emergency_penalty
            else:
                # Normal normalization
                normalized_bonus = np.tanh(bonus_sum / self.SHAPING_NORMALIZATION_SCALE) * 2.0
                shaped_reward = scaled_base_reward + normalized_bonus
        else:
            shaped_reward = reward + bonus_powerpill + bonus_eating_ghost + penalty_nearing_ghost + stalling_penalty + movement_bonus
        
        # Life loss detection (reduced penalty)
        if self.ENABLE_LIFE_LOSS_TRACKING:
            current_lives = info.get('lives', None)
            if current_lives is None and hasattr(self.env.unwrapped, 'ale'):
                current_lives = self.env.unwrapped.ale.lives()
            
            if self.prev_lives is not None and current_lives is not None:
                if current_lives < self.prev_lives:
                    shaped_reward += self.LIFE_LOSS_PENALTY
                    info['life_lost'] = True
                    if self.enable_logging:
                        with open(self.log_file, "a") as f:
                            f.write(f"[LIFE LOST] Lives: {self.prev_lives} -> {current_lives}, Penalty: {self.LIFE_LOSS_PENALTY}\n")
                else:
                    info['life_lost'] = False
            else:
                info['life_lost'] = False
            
            self.prev_lives = current_lives
        else:
            info['life_lost'] = False
        
        # Enhanced logging (matching reward_shaping_wrapper_3 format)
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