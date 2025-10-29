"""
Enhanced Stability Wrapper - Version 2.0 (Universal)
Dựa trên phân tích 189k+ entries từ 4 thuật toán
ÁP DỤNG CHO TẤT CẢ: DQN, PPO, A2C, Rainbow

THAY ĐỔI CHÍNH:
1. BASE_REWARD_SCALE = 0.1 - Scale base rewards xuống 10x
2. NORMALIZATION_SCALE = 20.0 - Tăng normalization
3. PENALTY_NEARING_GHOST_COEF = 0.15 - Giảm penalty
4. BONUS_POWERPILL_COEF = 0.30 - Tăng nhẹ để A2C cũng tìm powerpill tốt
5. POWERPILL_RADIUS = 27.0 - Giảm để bonus mạnh hơn khi gần
"""
import gymnasium as gym
import numpy as np
from ocatari.vision.utils import find_objects


class EnhancedStabilityRewardShaper(gym.Wrapper):
    """
    Universal enhanced stability wrapper cho TẤT CẢ thuật toán với:
    - Base reward scaling để cân bằng với shaped rewards
    - Normalization mạnh hơn cho learning ổn định
    - Penalty nhẹ hơn để không overpower bonus
    - Powerpill bonus được tối ưu cho cả A2C (powered up rate thấp)
    """
    
    # === ENHANCED PARAMETERS (Based on analysis) ===
    
    # CRITICAL FIX: Scale base rewards xuống
    BASE_REWARD_SCALE = 0.1  # Base reward × 0.1 (10→1, 200→20)
    
    # Bonus coefficients (BALANCED cho tất cả thuật toán)
    BONUS_POWERPILL_COEF = 0.30  # Tăng từ 0.25 → giúp A2C tìm powerpill tốt hơn
    POWERPILL_RADIUS = 27.0      # Giảm từ 30.0 → bonus mạnh hơn khi gần
    POWERPILL_BONUS_CAP = 5.0
    
    BONUS_EATING_GHOST_COEF = 1.0
    GHOST_CHASE_RADIUS = 20.0
    GHOST_BONUS_CAP = 20.0
    
    # REDUCED: Penalty coefficient (từ 0.25 → 0.15)
    # Vì phân tích cho thấy 73% penalties đạt cap, quá mạnh
    PENALTY_NEARING_GHOST_COEF = 0.15  # GIẢM từ 0.25
    GHOST_AVOID_RADIUS = 15.0
    GHOST_PENALTY_CAP = -5.0
    
    # Stability parameters
    PROGRESS_BONUS_SCALE = 0.5
    MAX_STEPS_WITHOUT_SCORE = 100
    STALLING_PENALTY_RATE = -0.005
    
    MOVEMENT_BONUS = 0.01
    CONSECUTIVE_NOOP_PENALTY = -0.005
    
    # INCREASED: Normalization scale (từ 10.0 → 20.0)
    ENABLE_REWARD_NORMALIZATION = True
    NORMALIZATION_SCALE = 20.0  # TĂNG từ 10.0
    
    def __init__(self, env, enable_logging=False, log_file="enhanced_shaping_log.txt"):
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
        
        # Progress tracking
        self.prev_ghost_distances = {}
        
        # Statistics (for debugging)
        self.stats = {
            'base_rewards_original': [],
            'base_rewards_scaled': [],
            'shaped_rewards': [],
        }
        
    def reset(self, **kwargs):
        """Reset environment and tracking variables"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset counters
        self.steps_without_score = 0
        self.consecutive_noops = 0
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
        """Detect if Pac-Man is powered up"""
        frame = self._get_rgb_frame()
        eatable_ghosts = find_objects(frame, [(66, 114, 194)], min_distance=1)
        return bool(eatable_ghosts)
    
    def _apply_caps(self, bonus_powerpill_raw, bonus_eating_ghost_raw, 
                   penalty_nearing_ghost_raw):
        """Apply caps to prevent extremes"""
        bonus_powerpill_raw = min(bonus_powerpill_raw, self.POWERPILL_BONUS_CAP)
        bonus_eating_ghost_raw = min(bonus_eating_ghost_raw, self.GHOST_BONUS_CAP)
        penalty_nearing_ghost_raw = max(penalty_nearing_ghost_raw, self.GHOST_PENALTY_CAP)
        
        return bonus_powerpill_raw, bonus_eating_ghost_raw, penalty_nearing_ghost_raw
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        objects = getattr(self.env, "objects", [])
        
        # Track score
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
            
            # 1. Bonus for approaching PowerPill (when not powered)
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
                
                # Progress bonus
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
            
            # Apply caps
            bonus_powerpill_raw, bonus_eating_ghost_raw, penalty_nearing_ghost_raw = \
                self._apply_caps(bonus_powerpill_raw, bonus_eating_ghost_raw, 
                               penalty_nearing_ghost_raw)
            
            # 3. Stalling penalty
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
        
        # ========================================
        # CRITICAL FIX: Scale base reward + normalize
        # ========================================
        if self.ENABLE_REWARD_NORMALIZATION:
            # Scale base reward xuống 10x
            scaled_base = reward * self.BASE_REWARD_SCALE
            
            # Sum all shaping components
            total_shaping = (bonus_powerpill + bonus_eating_ghost + 
                           penalty_nearing_ghost + stalling_penalty + movement_bonus)
            
            # Normalize shaping với tanh
            normalized_shaping = np.tanh(total_shaping / self.NORMALIZATION_SCALE) * 2.0
            
            # Combine scaled base + normalized shaping
            shaped_reward = scaled_base + normalized_shaping
            
            # Track statistics
            self.stats['base_rewards_original'].append(reward)
            self.stats['base_rewards_scaled'].append(scaled_base)
            self.stats['shaped_rewards'].append(shaped_reward)
        else:
            # Fallback: no normalization
            shaped_reward = reward + bonus_powerpill + bonus_eating_ghost + penalty_nearing_ghost + stalling_penalty + movement_bonus
        
        # Logging
        if self.enable_logging:
            self.step_count += 1
            if self.step_count % 25 == 0:
                is_powered = self.is_powered_up() if player is not None else False
                with open(self.log_file, "a") as f:
                    f.write(
                        f"is_powered_up: {is_powered}, "
                        f"base_reward: {reward:.3f}, "
                        f"scaled_base: {reward * self.BASE_REWARD_SCALE:.3f}, "
                        f"bonus_powerpill: {bonus_powerpill_raw:.3f}, "
                        f"bonus_eating_ghost: {bonus_eating_ghost_raw:.3f}, "
                        f"penalty_nearing_ghost: {penalty_nearing_ghost_raw:.3f}, "
                        f"stalling_penalty: {stalling_penalty:.3f}, "
                        f"movement_bonus: {movement_bonus:.3f}, "
                        f"steps_without_score: {self.steps_without_score}, "
                        f"shaped_reward: {shaped_reward:.3f}\n"
                    )
        
        return obs, shaped_reward, terminated, truncated, info
    
    def get_statistics(self):
        """Lấy statistics để debug"""
        if not self.stats['base_rewards_original']:
            return None
        
        return {
            'avg_base_original': np.mean(self.stats['base_rewards_original']),
            'avg_base_scaled': np.mean(self.stats['base_rewards_scaled']),
            'avg_shaped': np.mean(self.stats['shaped_rewards']),
            'std_shaped': np.std(self.stats['shaped_rewards']),
            'shaped_to_base_ratio': np.mean(self.stats['shaped_rewards']) / np.mean(self.stats['base_rewards_original']) if np.mean(self.stats['base_rewards_original']) > 0 else 0
        }



