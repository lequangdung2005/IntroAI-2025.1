"""
Manual Play Test with Reward Shaping Logging
Play Pac-Man with keyboard and log all reward shaping details
"""
import cv2
import numpy as np
from ocatari.core import OCAtari
from environment.reward_shaping_wrapper_4 import AdvancedRewardShaper
import sys
from datetime import datetime
import time


# CV2 key mapping to Atari actions
# Use ord() for character keys
# Atari Ms. Pacman actions: ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
KEY_TO_ACTION = {
    ord('w'): 1,  # UP
    ord('W'): 1,  # UP
    ord('s'): 4,  # DOWN (was 2, corrected!)
    ord('S'): 4,  # DOWN (was 2, corrected!)
    ord('a'): 3,  # LEFT
    ord('A'): 3,  # LEFT
    ord('d'): 2,  # RIGHT (was 4, corrected!)
    ord('D'): 2,  # RIGHT (was 4, corrected!)
    ord(' '): 0,  # NOOP (space)
    27: -1,       # ESC to quit
}

ACTION_NAMES = {
    0: "NOOP",
    1: "UP", 
    2: "RIGHT",  # Corrected!
    3: "LEFT",
    4: "DOWN"    # Corrected!
}


class DetailedRewardLogger(AdvancedRewardShaper):
    """Extended wrapper that logs detailed reward breakdown using wrapper_4"""
    
    def __init__(self, env, log_file="manual_play_log.txt"):
        super().__init__(env, enable_logging=False)  # Disable default logging
        
        self.detail_log_file = log_file
        self.step_num = 0
        
        # Clear log file
        with open(self.detail_log_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MANUAL PLAY SESSION - REWARD SHAPING LOG\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def step(self, action):
        # 🔥 DEBUG: Print what action we're taking
        print(f"\n🎮 TAKING ACTION: {ACTION_NAMES.get(action, 'UNKNOWN')}")
        
        # Store original step count for detailed logging
        original_step_count = self.step_count
        
        # Get shaped reward from parent class (wrapper_4) but also get raw data for logging
        obs, reward, terminated, truncated, info = self.env.step(action)
        objects = getattr(self.env, "objects", [])
        
        # Initialize shaping components for detailed logging (calculate same as wrapper_4)
        bonus_powerpill_raw = 0.0
        bonus_eating_ghost_raw = 0.0
        penalty_nearing_ghost_raw = 0.0
        stalling_penalty = 0.0
        movement_bonus = 0.0
        
        # Get player position
        player = next((o for o in objects if getattr(o, "category", None) == "Player"), None)
        player_pos = None
        is_powered = False
        
        if player is not None:
            px, py = getattr(player, "x", 0), getattr(player, "y", 0)
            player_pos = (px, py)
            is_powered = self.is_powered_up()
            
            # PowerPill detection with detailed logging
            powerpills = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            
            # Simple notification when PowerPill is eaten
            if reward >= 50:  # PowerPill gives 50 points
                print(f"🍎 PowerPill eaten! (+{reward} points)")
            
            # Basic status info
            print(f"📍 Player: ({px:.1f}, {py:.1f}) | Powered: {is_powered} | PowerPills: {len(powerpills)}")
            
            # Track score for stalling penalty calculation
            if reward > 0:
                self.steps_without_score = 0
            else:
                self.steps_without_score += 1
            
            # 1. PowerPill bonus calculation - FIXED: Only closest PowerPill
            if powerpills and not is_powered:
                # Find closest PowerPill
                closest_pill = None
                closest_dist = float('inf')
                for pill in powerpills:
                    dist = np.linalg.norm([px - pill.x, py - pill.y])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_pill = pill
                
                print(f"  🎯 Closest PowerPill at distance: {closest_dist:.1f}")
                
                if closest_pill is not None:
                    # Clean up tracking for non-existing PowerPills
                    current_pill_positions = {(int(o.x), int(o.y)) for o in powerpills}
                    closest_pill_pos = (int(closest_pill.x), int(closest_pill.y))
                    
                    self.powerpill_min_distances = {
                        pos: dist for pos, dist in self.powerpill_min_distances.items()
                        if pos in current_pill_positions
                    }
                    
                    # Apply anti-camping logic ONLY to closest PowerPill
                    bonus_powerpill_raw = 0.0
                    
                    if closest_dist <= self.POWERPILL_CAMPING_RADIUS:
                        if closest_pill_pos not in self.powerpill_min_distances:
                            # First time in camping zone
                            self.powerpill_min_distances[closest_pill_pos] = closest_dist
                            bonus_powerpill_raw = 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                            print(f"  🎯 First entry to camping zone: +{bonus_powerpill_raw:.3f}")
                        else:
                            # Check for progress in camping zone
                            min_dist = self.powerpill_min_distances[closest_pill_pos]
                            if closest_dist < min_dist - self.POWERPILL_CAMPING_THRESHOLD:
                                # Made progress - update and reward
                                self.powerpill_min_distances[closest_pill_pos] = closest_dist
                                bonus_powerpill_raw = 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                                print(f"  📈 Progress in camping zone: {min_dist:.1f}→{closest_dist:.1f} +{bonus_powerpill_raw:.3f}")
                            else:
                                # No progress - camping detected!
                                print(f"  🚫 CAMPING detected! Distance {closest_dist:.1f} vs min {min_dist:.1f} (need -{self.POWERPILL_CAMPING_THRESHOLD}+ progress)")
                                bonus_powerpill_raw = 0.0
                    else:
                        # Outside camping zone - normal distance bonus
                        if closest_pill_pos in self.powerpill_min_distances:
                            del self.powerpill_min_distances[closest_pill_pos]
                            print(f"  🔄 Exited camping zone, reset tracking")
                        # Normal bonus for closest PowerPill outside camping zone
                        bonus_powerpill_raw = 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                        print(f"  ✅ Normal bonus (outside camping): +{bonus_powerpill_raw:.3f}")
                    
                    if bonus_powerpill_raw > 0:
                        print(f"  🎯 Total PowerPill bonus: {bonus_powerpill_raw:.3f}")
                else:
                    bonus_powerpill_raw = 0.0
            
            print("-" * 40)
            
            # 2. Ghost interaction
            ghosts = [o for o in objects if getattr(o, "category", None) == "Ghost"]
            
            if is_powered and ghosts:
                ghost_distances = [(g, np.linalg.norm([px - g.x, py - g.y])) for g in ghosts]
                closest_ghost, closest_dist = min(ghost_distances, key=lambda x: x[1])
                
                for g, dist in ghost_distances:
                    bonus_eating_ghost_raw += 10.0 * np.exp(-dist/self.GHOST_CHASE_RADIUS)
                
                if 'closest_ghost_dist' in self.prev_ghost_distances:
                    prev_closest = self.prev_ghost_distances['closest_ghost_dist']
                    if closest_dist < prev_closest:
                        progress = prev_closest - closest_dist
                        bonus_eating_ghost_raw += self.PROGRESS_BONUS_SCALE * progress
                
                self.prev_ghost_distances['closest_ghost_dist'] = closest_dist
                
            elif not is_powered and ghosts:
                self.prev_ghost_distances.clear()
                ghost_distances = [np.linalg.norm([px - g.x, py - g.y]) for g in ghosts]
                closest_dist = min(ghost_distances)
                
                if closest_dist < self.GHOST_AVOID_RADIUS*2:
                    penalty_nearing_ghost_raw -= 10.0 * np.exp(-closest_dist/self.GHOST_AVOID_RADIUS)
            
            # Apply caps
            bonus_powerpill_raw, bonus_eating_ghost_raw, penalty_nearing_ghost_raw = \
                self._apply_caps_and_normalization(bonus_powerpill_raw, bonus_eating_ghost_raw, 
                                                 penalty_nearing_ghost_raw)
            
            # 3. Stalling penalty
            if self.steps_without_score > self.MAX_STEPS_WITHOUT_SCORE:
                excess = self.steps_without_score - self.MAX_STEPS_WITHOUT_SCORE
                stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.5)
                if is_powered:
                    stalling_penalty *= 0.5
            
            # 4. Position-based movement tracking with wrapper_4 enhanced penalties
            if hasattr(self, 'position_history') and len(self.position_history) >= self.POSITION_TRACKING_WINDOW:
                position_variance = self._calculate_position_variance()
                    
                if position_variance < self.MIN_POSITION_VARIANCE:
                    # Calculate movement penalty with wrapper_4 enhanced logic
                    base_movement_penalty = self.STUCK_PENALTY * (self.MIN_POSITION_VARIANCE - position_variance)
                    
                    # EXTREME PENALTY for being completely stuck (variance = 0)
                    if position_variance == 0.0:
                        base_movement_penalty *= 5.0  # 5x penalty for complete stillness
                        extreme_info = " 🆘 EXTREME STUCK!"
                    else:
                        extreme_info = ""
                    
                    final_movement_penalty = base_movement_penalty * 1.5 if is_powered else base_movement_penalty
                    powered_info = " (x1.5 powered-up!)" if is_powered else ""
                    print(f"🚫 STUCK DETECTED! Variance: {position_variance:.1f} → penalty: {final_movement_penalty:.3f}{powered_info}{extreme_info}")
                    movement_bonus = final_movement_penalty
                else:
                    print(f"✅ Moving freely. Variance: {position_variance:.1f}")
            else:
                position_variance = 0.0
            
            # 5. Enhanced stalling penalty with powered-up urgency (wrapper_4 style)
            if is_powered:
                # When powered up, use stricter threshold (half the normal steps)
                threshold = self.MAX_STEPS_WITHOUT_SCORE * 0.5
                if self.steps_without_score > threshold:
                    excess = self.steps_without_score - threshold
                    stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.0)
                    print(f"⏰ POWERED-UP STALLING PENALTY: {stalling_penalty:.3f} (threshold: {threshold:.0f})")
            else:
                # Normal threshold when not powered up
                if self.steps_without_score > self.MAX_STEPS_WITHOUT_SCORE:
                    excess = self.steps_without_score - self.MAX_STEPS_WITHOUT_SCORE
                    stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.0)
                    if stalling_penalty < -0.01:
                        print(f"⏰ STALLING PENALTY: {stalling_penalty:.3f}")
        
        # Now get the actual shaped reward from parent wrapper_4
        # We need to call super().step() to get the real shaped reward
        obs_final, shaped_reward, terminated_final, truncated_final, info_final = super().step(action)
        
        # Apply coefficients
        bonus_powerpill = self.BONUS_POWERPILL_COEF * bonus_powerpill_raw
        bonus_eating_ghost = self.BONUS_EATING_GHOST_COEF * bonus_eating_ghost_raw
        penalty_nearing_ghost = self.PENALTY_NEARING_GHOST_COEF * penalty_nearing_ghost_raw
        
        # Apply coefficients for logging
        bonus_powerpill = self.BONUS_POWERPILL_COEF * bonus_powerpill_raw
        bonus_eating_ghost = self.BONUS_EATING_GHOST_COEF * bonus_eating_ghost_raw
        penalty_nearing_ghost = self.PENALTY_NEARING_GHOST_COEF * penalty_nearing_ghost_raw
        
        # Check for emergency escape conditions
        is_stuck = movement_bonus < -10.0
        is_ghost_nearby = penalty_nearing_ghost < -1.0
        emergency_escape = is_stuck and is_ghost_nearby
        
        if emergency_escape:
            print(f"🆘 EMERGENCY NORMALIZATION! Wider range [-5,+5] instead of [-2,+2]")
        
        # Life loss detection for logging
        life_lost = info_final.get('life_lost', False)
        
        # LOG EVERYTHING
        self.step_num += 1
        with open(self.detail_log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"STEP {self.step_num} | Action: {ACTION_NAMES.get(action, 'UNKNOWN')}\n")
            f.write(f"{'-'*80}\n")
            
            # Basic info
            f.write(f"Player Position: {player_pos}\n")
            f.write(f"Powered Up: {is_powered}\n")
            f.write(f"Base Reward: {reward:.3f}\n")
            f.write(f"Lives: {self.prev_lives}\n")
            
            f.write(f"\n--- Reward Shaping Breakdown ---\n")
            
            # Write PowerPill info to log file (simplified)
            powerpills_detected = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            f.write(f"🔍 PowerPills detected: {len(powerpills_detected)}\n")
            if len(self.powerpill_min_distances) > 0:
                f.write(f"   - Tracking dict: {self.powerpill_min_distances}\n")
            
            f.write(f"1. PowerPill Bonus (raw): {bonus_powerpill_raw:.4f}\n")
            f.write(f"   - After coefficient (x{self.BONUS_POWERPILL_COEF}): {bonus_powerpill:.4f}\n")
            
            f.write(f"2. Ghost Chase Bonus (raw): {bonus_eating_ghost_raw:.4f}\n")
            f.write(f"   - After coefficient (x{self.BONUS_EATING_GHOST_COEF}): {bonus_eating_ghost:.4f}\n")
            
            f.write(f"3. Ghost Avoid Penalty (raw): {penalty_nearing_ghost_raw:.4f}\n")
            f.write(f"   - After coefficient (x{self.PENALTY_NEARING_GHOST_COEF}): {penalty_nearing_ghost:.4f}\n")
            
            f.write(f"4. Stalling Penalty: {stalling_penalty:.4f}\n")
            f.write(f"   - Steps without score: {self.steps_without_score}\n")
            
            f.write(f"5. Movement Bonus/Penalty: {movement_bonus:.4f}\n")
            
            # Enhanced penalty details for wrapper_4
            if hasattr(self, 'position_history') and len(self.position_history) >= self.POSITION_TRACKING_WINDOW:
                position_variance = self._calculate_position_variance() if hasattr(self, '_calculate_position_variance') else 0.0
                f.write(f"   - Position variance: {position_variance:.3f} (min required: {self.MIN_POSITION_VARIANCE})\n")
                f.write(f"   - STUCK_PENALTY coefficient: {self.STUCK_PENALTY} (max penalty: {self.STUCK_PENALTY * self.MIN_POSITION_VARIANCE:.2f})\n")
                if position_variance < self.MIN_POSITION_VARIANCE:
                    base_penalty = self.STUCK_PENALTY * (self.MIN_POSITION_VARIANCE - position_variance)
                    f.write(f"   - Base stuck penalty: {base_penalty:.4f}\n")
                    
                    # Show extreme penalty for zero variance
                    if position_variance == 0.0:
                        extreme_penalty = base_penalty * 5.0
                        f.write(f"   - EXTREME PENALTY (variance=0): x5.0 = {extreme_penalty:.4f}\n")
                        base_penalty = extreme_penalty
                    
                    if is_powered:
                        f.write(f"   - Powered-up multiplier: x1.5 = {base_penalty * 1.5:.4f}\n")
                    f.write(f"   - STUCK PENALTY APPLIED\n")
            else:
                f.write(f"   - Not enough position history ({len(getattr(self, 'position_history', []))}/{self.POSITION_TRACKING_WINDOW})\n")
            
            # Enhanced stalling penalty details (wrapper_4 style)
            if is_powered:
                threshold = self.MAX_STEPS_WITHOUT_SCORE * 0.5
                f.write(f"   - Powered-up threshold: {threshold:.0f} steps (vs normal {self.MAX_STEPS_WITHOUT_SCORE})\n")
                if self.steps_without_score > threshold:
                    excess = self.steps_without_score - threshold
                    f.write(f"   - Excess steps: {excess:.0f}, penalty: {stalling_penalty:.4f}\n")
            else:
                if self.steps_without_score > self.MAX_STEPS_WITHOUT_SCORE:
                    excess = self.steps_without_score - self.MAX_STEPS_WITHOUT_SCORE
                    f.write(f"   - Normal threshold: {self.MAX_STEPS_WITHOUT_SCORE} steps\n")
                    f.write(f"   - Excess steps: {excess:.0f}, penalty: {stalling_penalty:.4f}\n")
            
            if life_lost:
                f.write(f"\n!!! LIFE LOST - Penalty: {self.LIFE_LOSS_PENALTY} !!!\n")
            
            f.write(f"\n--- Final Reward (from wrapper_4) ---\n")
            bonus_sum = bonus_powerpill + bonus_eating_ghost + penalty_nearing_ghost + stalling_penalty + movement_bonus
            f.write(f"Total Bonus (before normalization): {bonus_sum:.4f}\n")
            
            # Check and log emergency escape
            if emergency_escape:
                emergency_normalized = np.tanh(bonus_sum / self.SHAPING_NORMALIZATION_SCALE) * 5.0
                f.write(f"🆘 EMERGENCY NORMALIZATION ACTIVATED!\n")
                f.write(f"   - Stuck penalty: {movement_bonus:.4f}\n")
                f.write(f"   - Ghost penalty: {penalty_nearing_ghost:.4f}\n")
                f.write(f"   - Emergency normalized (x5.0): {emergency_normalized:.4f} (instead of x2.0)\n")
            elif self.ENABLE_REWARD_NORMALIZATION:
                normalized = np.tanh(bonus_sum / self.SHAPING_NORMALIZATION_SCALE) * 2.0
                f.write(f"Normalized Bonus (tanh): {normalized:.4f}\n")
            
            scaled_base = self._scale_base_reward(reward) if hasattr(self, '_scale_base_reward') else reward
            f.write(f"Scaled Base Reward: {scaled_base:.4f} (original: {reward:.3f})\n")
            f.write(f"SHAPED REWARD (wrapper_4): {shaped_reward:.4f}\n")
            
            if terminated_final or truncated_final:
                f.write(f"\n{'='*80}\n")
                f.write("EPISODE ENDED\n")
                f.write(f"Terminated: {terminated_final}, Truncated: {truncated_final}\n")
                f.write(f"{'='*80}\n")
        
        return obs_final, shaped_reward, terminated_final, truncated_final, info_final



def main():
    print("=" * 80)
    print("MANUAL PLAY TEST - Pac-Man with Reward Shaping")
    print("=" * 80)
    print("\nControls:")
    print("  W - Move UP")
    print("  S - Move DOWN")
    print("  A - Move LEFT")
    print("  D - Move RIGHT")
    print("  SPACE - Do nothing (NOOP)")
    print("  ESC - Quit")
    print("\nAll reward details will be logged to a timestamped file")
    print("=" * 80)
    
    # Create environment
    env_name = "ALE/MsPacman-v5"
    print(f"\nInitializing environment: {env_name}...")
    env = OCAtari(env_name, mode="both", render_mode="rgb_array")
    
    # Wrap with detailed logger
    log_file = f"manual_play_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    env = DetailedRewardLogger(env, log_file=log_file)
    
    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset()
    
    # Environment ready for play
    
    # Setup CV2 window - use actual game size
    window_name = "Pac-Man Manual Play - Reward Shaping Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Atari screen is 210x160, scale up 3x for visibility: 630x480
    cv2.resizeWindow(window_name, 630, 480)
    
    # Game state
    done = False
    current_action = 0  # Start with NOOP
    total_shaped_reward = 0
    step_count = 0
    
    print(f"\n✅ Game started! Logging to: {log_file}")
    print("\n🎮 STEP-BY-STEP MANUAL PLAY MODE")
    print("Game will PAUSE after each frame and wait for your input")
    print("Press W/A/S/D/SPACE for next action, ESC to quit")
    print("=" * 80)
    
    # Render initial frame
    def render_current_frame():
        try:
            frame = env.render()
            if frame is None:
                if hasattr(env, 'getScreenRGB'):
                    frame = env.getScreenRGB()
                elif hasattr(env.env, 'render'):
                    frame = env.env.render()
        except Exception as e:
            print(f"Render error: {e}")
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        
        if frame is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize to 3x for better visibility while maintaining aspect ratio
            # Original: 160x210, Scaled: 480x630
            frame_scaled = cv2.resize(frame_bgr, (480, 630), interpolation=cv2.INTER_NEAREST)
            
            # Show clean frame without any overlay text
            cv2.imshow(window_name, frame_scaled)
        
        return frame is not None
    
    # Initial render
    render_current_frame()
    
    while not done:
        # 🔥 STEP-BY-STEP: Wait for user input
        print(f"\n[Step {step_count + 1}] Waiting for your action (W/A/S/D/SPACE/ESC)...", end=" ")
        
        while True:
            key = cv2.waitKey(0)  # Wait indefinitely for key press!
            
            if key in KEY_TO_ACTION:
                action_value = KEY_TO_ACTION[key]
                
                if action_value == -1:  # ESC
                    print("ESC - Quitting")
                    done = True
                    break
                else:
                    current_action = action_value
                    print(f"{ACTION_NAMES[current_action]}")
                    break
            else:
                # Invalid key, keep waiting
                print("Invalid key! Use W/A/S/D/SPACE/ESC only.", end=" ")
                continue
        
        if done:
            break
        
        # Execute the chosen action
        obs, shaped_reward, terminated, truncated, info = env.step(current_action)
        
        total_shaped_reward += shaped_reward
        step_count += 1
        
        # Render frame AFTER taking action
        frame_rendered = render_current_frame()
        
        if not frame_rendered:
            print("⚠️ Could not render frame")
        
        # Print detailed info to console
        lives = getattr(env, 'prev_lives', '?')
        powered = "YES" if hasattr(env, 'is_powered_up') and env.is_powered_up() else "NO"
        
        print(f"    → Reward: {shaped_reward:.3f} | Total: {total_shaped_reward:.2f}")
        print(f"      Lives: {lives} | Powered Up: {powered}")
        
        # Print reward breakdown if significant
        if abs(shaped_reward) > 0.01:
            print(f"      Details logged to: {log_file}")
        
        print("-" * 50)
        
        # Check if episode ended
        if terminated or truncated:
            print(f"\n{'='*80}")
            print("🎮 EPISODE ENDED")
            print(f"{'='*80}")
            print(f"Total Steps: {step_count}")
            print(f"Total Shaped Reward: {total_shaped_reward:.2f}")
            print(f"Log saved to: {log_file}")
            print(f"{'='*80}")
            
            # Keep the final frame visible
            if frame_rendered:
                print("Game window will remain open until you press a key.")
            
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            done = True
    
    # Cleanup
    cv2.destroyAllWindows()
    env.close()
    
    print(f"\n✅ Session completed. Check {log_file} for detailed reward logs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
