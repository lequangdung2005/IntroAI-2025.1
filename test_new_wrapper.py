"""
Manual Play Test with Reward Shaping Logging - wrapper_4 version
Play Pac-Man with keyboard and log all reward shaping details
"""
import cv2
import numpy as np
from ocatari.core import OCAtari
from environment.reward_shaping_wrapper_4 import AdvancedRewardShaper
import sys
from datetime import datetime


# CV2 key mapping to Atari actions
KEY_TO_ACTION = {
    ord('w'): 1,  # UP
    ord('W'): 1,
    ord('s'): 4,  # DOWN
    ord('S'): 4,
    ord('a'): 3,  # LEFT
    ord('A'): 3,
    ord('d'): 2,  # RIGHT
    ord('D'): 2,
    ord(' '): 0,  # NOOP (space)
    27: -1,       # ESC to quit
}

ACTION_NAMES = {
    0: "NOOP",
    1: "UP", 
    2: "RIGHT",
    3: "LEFT",
    4: "DOWN"
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
            f.write("MANUAL PLAY SESSION - REWARD SHAPING LOG (wrapper_4)\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def step(self, action):
        # 🔥 DEBUG: Print what action we're taking
        print(f"\n🎮 TAKING ACTION: {ACTION_NAMES.get(action, 'UNKNOWN')}")
        
        # Call parent step method ONCE to get the actual shaped reward
        obs, shaped_reward, terminated, truncated, info = super().step(action)
        objects = getattr(self.env, "objects", [])
        
        # Get base reward for logging
        base_reward = info.get('base_reward', 0) if 'base_reward' in info else 0
        
        # Get player info
        player = next((o for o in objects if getattr(o, "category", None) == "Player"), None)
        player_pos = None
        is_powered = False
        
        # Recalculate components for detailed logging (matching wrapper_4 logic)
        bonus_powerpill_raw = 0.0
        bonus_eating_ghost_raw = 0.0
        penalty_nearing_ghost_raw = 0.0
        stalling_penalty = 0.0
        movement_bonus = 0.0
        position_variance = 0.0
        escape_bonus = 0.0
        
        if player is not None:
            px, py = getattr(player, "x", 0), getattr(player, "y", 0)
            player_pos = (px, py)
            is_powered = self.is_powered_up()
            
            # PowerPill detection
            powerpills = [o for o in objects if getattr(o, "category", None) == "PowerPill"]
            
            # Basic status info
            print(f"📍 Player: ({px:.1f}, {py:.1f}) | Powered: {is_powered} | PowerPills: {len(powerpills)}")
            
            # 1. PowerPill bonus calculation (matching wrapper_4 logic)
            if powerpills and not is_powered:
                # Find closest PowerPill
                closest_pill = None
                closest_dist = float('inf')
                for pill in powerpills:
                    dist = np.linalg.norm([px - pill.x, py - pill.y])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_pill = pill
                
                if closest_pill is not None:
                    closest_pill_pos = (int(closest_pill.x), int(closest_pill.y))
                    
                    # Apply anti-camping logic (matching wrapper_4)
                    if closest_dist <= self.POWERPILL_CAMPING_RADIUS:
                        if closest_pill_pos not in self.powerpill_min_distances:
                            bonus_powerpill_raw = 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                            print(f"  🎯 PowerPill first entry: +{bonus_powerpill_raw:.3f}")
                        else:
                            min_dist = self.powerpill_min_distances[closest_pill_pos]
                            if closest_dist < min_dist - self.POWERPILL_CAMPING_THRESHOLD:
                                bonus_powerpill_raw = 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                                print(f"  📈 PowerPill progress: {min_dist:.1f}→{closest_dist:.1f} +{bonus_powerpill_raw:.3f}")
                            else:
                                print(f"  🚫 PowerPill camping detected!")
                    else:
                        bonus_powerpill_raw = 10.0 * np.exp(-closest_dist/self.POWERPILL_RADIUS)
                        print(f"  ✅ PowerPill bonus: +{bonus_powerpill_raw:.3f}")
            
            # 2. Ghost interaction (matching wrapper_4 logic)
            ghosts = [o for o in objects if getattr(o, "category", None) == "Ghost"]
            
            if is_powered and ghosts:
                # Ghost chasing
                for g in ghosts:
                    dist = np.linalg.norm([px - g.x, py - g.y])
                    if dist <= self.GHOST_CHASE_RADIUS:
                        bonus_eating_ghost_raw += 10.0 * np.exp(-dist/self.GHOST_CHASE_RADIUS)
                
                if bonus_eating_ghost_raw > 0:
                    print(f"  👻 Ghost chase bonus: +{bonus_eating_ghost_raw:.3f}")
                
            elif not is_powered and ghosts:
                # Ghost avoidance with gradient penalty (matching wrapper_4)
                for g in ghosts:
                    dist = np.linalg.norm([px - g.x, py - g.y])
                    if dist <= self.GHOST_AVOID_RADIUS:
                        if dist > self.GHOST_AVOID_RADIUS * 0.6:
                            penalty_nearing_ghost_raw -= 10.0 * np.exp(-dist/self.GHOST_AVOID_RADIUS)
                        else:
                            penalty_nearing_ghost_raw -= 15.0 * np.exp(-dist/self.GHOST_AVOID_RADIUS)
                
                if penalty_nearing_ghost_raw < -0.01:
                    print(f"  👻 Ghost avoid penalty: {penalty_nearing_ghost_raw:.3f}")
            
            # Apply caps (matching wrapper_4)
            bonus_powerpill_raw = min(bonus_powerpill_raw, self.POWERPILL_BONUS_CAP)
            bonus_eating_ghost_raw = min(bonus_eating_ghost_raw, self.GHOST_BONUS_CAP)
            penalty_nearing_ghost_raw = max(penalty_nearing_ghost_raw, self.GHOST_PENALTY_CAP)
            
            # 3. Stalling penalty (matching wrapper_4 logic)
            if is_powered:
                threshold = self.MAX_STEPS_WITHOUT_SCORE * 0.5
                if self.steps_without_score > threshold:
                    excess = self.steps_without_score - threshold
                    stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.0)
                    print(f"  ⏰ Powered stalling: {stalling_penalty:.3f}")
            else:
                if self.steps_without_score > self.MAX_STEPS_WITHOUT_SCORE:
                    excess = self.steps_without_score - self.MAX_STEPS_WITHOUT_SCORE
                    stalling_penalty = max(self.STALLING_PENALTY_RATE * (excess ** 1.1), -1.0)
                    if stalling_penalty < -0.01:
                        print(f"  ⏰ Stalling penalty: {stalling_penalty:.3f}")
            
            # 4. Movement tracking (matching wrapper_4 logic with escape bonus)
            if len(self.position_history) >= self.POSITION_TRACKING_WINDOW:
                position_variance = self._calculate_position_variance()
                
                # Check for escape bonus
                if len(self.position_history) >= 2:
                    prev_positions = self.position_history[:-1]
                    if len(prev_positions) >= self.POSITION_TRACKING_WINDOW - 1:
                        prev_variance = np.var([p[0] for p in prev_positions[-(self.POSITION_TRACKING_WINDOW-1):]]) + \
                                      np.var([p[1] for p in prev_positions[-(self.POSITION_TRACKING_WINDOW-1):]])
                        
                        if (prev_variance < self.MIN_POSITION_VARIANCE and 
                            position_variance >= self.MIN_POSITION_VARIANCE):
                            escape_bonus = 0.4
                            print(f"  🏃 ESCAPE BONUS: +{escape_bonus:.3f}")
                
                if position_variance < self.MIN_POSITION_VARIANCE:
                    # Stuck penalty
                    movement_bonus = self.STUCK_PENALTY * (self.MIN_POSITION_VARIANCE - position_variance)
                    
                    if position_variance == 0.0:
                        movement_bonus *= 1.5  # Extreme penalty
                        print(f"  🚫 EXTREME STUCK! Penalty: {movement_bonus:.3f}")
                    else:
                        print(f"  🚫 Stuck (var={position_variance:.1f}): {movement_bonus:.3f}")
                    
                    if is_powered:
                        movement_bonus *= 1.5
                else:
                    # Movement reward
                    base_movement_reward = min(position_variance / (self.MIN_POSITION_VARIANCE * 1.5), 2.5)
                    movement_bonus = base_movement_reward * 1.2
                    print(f"  ✅ Moving (var={position_variance:.1f}): +{movement_bonus:.3f}")
                
                # Add escape bonus
                movement_bonus += escape_bonus
        
        # Apply coefficients (matching wrapper_4)
        bonus_powerpill = self.BONUS_POWERPILL_COEF * bonus_powerpill_raw
        bonus_eating_ghost = self.BONUS_EATING_GHOST_COEF * bonus_eating_ghost_raw
        penalty_nearing_ghost = self.PENALTY_NEARING_GHOST_COEF * penalty_nearing_ghost_raw
        
        # Check for emergency conditions
        is_stuck = movement_bonus < -1.0
        is_ghost_nearby = penalty_nearing_ghost < -1.0
        emergency_escape = is_stuck and is_ghost_nearby
        
        if emergency_escape:
            print(f"  🆘 EMERGENCY NORMALIZATION! (x3.2 instead of x2.0)")
        
        # Life loss detection
        life_lost = info.get('life_lost', False)
        
        # LOG EVERYTHING TO FILE
        self.step_num += 1
        with open(self.detail_log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"STEP {self.step_num} | Action: {ACTION_NAMES.get(action, 'UNKNOWN')}\n")
            f.write(f"{'-'*80}\n")
            
            # Basic info
            f.write(f"Player Position: {player_pos}\n")
            f.write(f"Powered Up: {is_powered}\n")
            f.write(f"Base Reward: {base_reward:.3f}\n")
            f.write(f"Lives: {self.prev_lives}\n")
            
            f.write(f"\n--- Reward Shaping Breakdown ---\n")
            
            # PowerPill info
            powerpills_count = len([o for o in objects if getattr(o, "category", None) == "PowerPill"])
            f.write(f"🔍 PowerPills detected: {powerpills_count}\n")
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
            if is_powered:
                threshold = self.MAX_STEPS_WITHOUT_SCORE * 0.5
                f.write(f"   - Powered-up threshold: {threshold:.0f} steps (vs normal {self.MAX_STEPS_WITHOUT_SCORE})\n")
                if self.steps_without_score > threshold:
                    excess = self.steps_without_score - threshold
                    f.write(f"   - Excess steps: {excess:.0f}\n")
            else:
                if self.steps_without_score > self.MAX_STEPS_WITHOUT_SCORE:
                    excess = self.steps_without_score - self.MAX_STEPS_WITHOUT_SCORE
                    f.write(f"   - Normal threshold: {self.MAX_STEPS_WITHOUT_SCORE} steps\n")
                    f.write(f"   - Excess steps: {excess:.0f}\n")
            
            f.write(f"5. Movement Bonus/Penalty: {movement_bonus:.4f}\n")
            
            # Position variance details
            if len(self.position_history) >= self.POSITION_TRACKING_WINDOW:
                f.write(f"   - Position variance: {position_variance:.3f} (min required: {self.MIN_POSITION_VARIANCE})\n")
                f.write(f"   - STUCK_PENALTY coefficient: {self.STUCK_PENALTY}\n")
                
                if position_variance < self.MIN_POSITION_VARIANCE:
                    base_penalty = self.STUCK_PENALTY * (self.MIN_POSITION_VARIANCE - position_variance)
                    f.write(f"   - Base stuck penalty: {base_penalty:.4f}\n")
                    
                    if position_variance == 0.0:
                        f.write(f"   - EXTREME PENALTY (variance=0): x1.5 = {base_penalty * 1.5:.4f}\n")
                    
                    if is_powered:
                        f.write(f"   - Powered-up multiplier: x1.5\n")
                    
                    f.write(f"   - STUCK PENALTY APPLIED\n")
                else:
                    base_reward = min(position_variance / (self.MIN_POSITION_VARIANCE * 1.5), 2.5)
                    f.write(f"   - Base movement reward: {base_reward:.4f}\n")
                    f.write(f"   - Enhanced multiplier: x1.2 = {base_reward * 1.2:.4f}\n")
                    f.write(f"   - MOVEMENT REWARD APPLIED\n")
                
                if escape_bonus > 0:
                    f.write(f"   - 🏃 ESCAPE BONUS: +{escape_bonus:.4f}\n")
            else:
                f.write(f"   - Not enough position history ({len(self.position_history)}/{self.POSITION_TRACKING_WINDOW})\n")
            
            if life_lost:
                f.write(f"\n!!! LIFE LOST - Penalty: {self.LIFE_LOSS_PENALTY} !!!\n")
            
            f.write(f"\n--- Final Reward (from wrapper_4) ---\n")
            bonus_sum = bonus_powerpill + bonus_eating_ghost + penalty_nearing_ghost + stalling_penalty + movement_bonus
            f.write(f"Total Bonus (before normalization): {bonus_sum:.4f}\n")
            
            if emergency_escape:
                emergency_normalized = np.tanh(bonus_sum / self.SHAPING_NORMALIZATION_SCALE) * 3.2
                f.write(f"🆘 EMERGENCY CONDITIONS DETECTED:\n")
                f.write(f"   - Stuck: {is_stuck} (movement_bonus < -1.0)\n")
                f.write(f"   - Ghost nearby: {is_ghost_nearby} (penalty_nearing_ghost < -1.0)\n")
                f.write(f"   - Emergency normalized (x3.2): {emergency_normalized:.4f} (instead of x2.0)\n")
            else:
                normalized = np.tanh(bonus_sum / self.SHAPING_NORMALIZATION_SCALE) * 2.0
                f.write(f"Normalized Bonus (tanh): {normalized:.4f}\n")
            
            scaled_base = self._scale_base_reward(base_reward)
            f.write(f"Scaled Base Reward: {scaled_base:.4f} (original: {base_reward:.3f})\n")
            f.write(f"SHAPED REWARD (wrapper_4): {shaped_reward:.4f}\n")
            
            if terminated or truncated:
                f.write(f"\n{'='*80}\n")
                f.write("EPISODE ENDED\n")
                f.write(f"Terminated: {terminated}, Truncated: {truncated}\n")
                f.write(f"{'='*80}\n")
        
        return obs, shaped_reward, terminated, truncated, info


def main():
    print("=" * 80)
    print("MANUAL PLAY TEST - Pac-Man with Reward Shaping (wrapper_4)")
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
    env = OCAtari(env_name, mode="both", render_mode="rgb_array")  # Use "both" mode for accuracy
    
    # Wrap with detailed logger
    log_file = f"manual_play_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    env = DetailedRewardLogger(env, log_file=log_file)
    
    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset()
    
    # Setup CV2 window
    window_name = "Pac-Man Manual Play - Reward Shaping Test (wrapper_4)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 480, 630)  # 3x scale
    
    # Game state
    done = False
    current_action = 0
    total_shaped_reward = 0
    step_count = 0
    
    print(f"\n✅ Game started! Logging to: {log_file}")
    print("\n🎮 STEP-BY-STEP MANUAL PLAY MODE")
    print("Game will PAUSE after each frame and wait for your input")
    print("Press W/A/S/D/SPACE for next action, ESC to quit")
    print("=" * 80)
    
    # Render function
    def render_current_frame():
        try:
            frame = env.render()
            if frame is None:
                frame = np.zeros((210, 160, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Render error: {e}")
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_scaled = cv2.resize(frame_bgr, (480, 630), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, frame_scaled)
        
        return frame is not None
    
    # Initial render
    render_current_frame()
    
    while not done:
        print(f"\n[Step {step_count + 1}] Waiting for your action (W/A/S/D/SPACE/ESC)...", end=" ")
        
        while True:
            key = cv2.waitKey(0)
            
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
                print("Invalid key! Use W/A/S/D/SPACE/ESC only.", end=" ")
                continue
        
        if done:
            break
        
        # Execute action
        obs, shaped_reward, terminated, truncated, info = env.step(current_action)
        
        total_shaped_reward += shaped_reward
        step_count += 1
        
        # Render frame
        frame_rendered = render_current_frame()
        
        if not frame_rendered:
            print("⚠️ Could not render frame")
        
        # Print summary
        lives = env.prev_lives if hasattr(env, 'prev_lives') else '?'
        powered = "YES" if env.is_powered_up() else "NO"
        
        print(f"    → Reward: {shaped_reward:.3f} | Total: {total_shaped_reward:.2f}")
        print(f"      Lives: {lives} | Powered Up: {powered}")
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
