"""
Greedy Algorithm for Ms. Pacman using OCAtari
Uses object detection and priority-based decision making
Inspired by reward shaping wrapper's priorities and action space
"""
import gymnasium as gym
import ale_py
import numpy as np
from collections import deque
import time
import cv2
from ocatari.core import OCAtari
from ocatari.vision.utils import find_objects

# Register ALE environments
gym.register_envs(ale_py)


class GreedyPacmanAgent:
    """
    Greedy agent for Ms. Pacman that makes decisions based on:
    1. Immediate survival (avoid ghosts when not powered)
    2. PowerPill collection (high priority for power-ups)
    3. Ghost hunting (when powered up)
    4. Pellet collection (default behavior)
    
    Action Space (ALE MsPacman):
    0: NOOP, 1: UP, 2: RIGHT, 3: LEFT, 4: DOWN
    5: UPRIGHT, 6: UPLEFT, 7: DOWNRIGHT, 8: DOWNLEFT
    """
    
    # Priority weights (from reward shaping analysis)
    PRIORITY_SURVIVE = 100.0      # Highest: Don't die
    PRIORITY_HUNT_GHOST = 1000.0    # Chase ghosts when powered (INCREASED - very rewarding!)
    PRIORITY_POWERPILL = 50.0     # Get power-ups
    PRIORITY_PELLET = 10.0        # Collect pellets
    PRIORITY_EXPLORE = 1.0        # Explore when no clear target
    
    # Distance thresholds (from reward shaping wrapper)
    GHOST_DANGER_RADIUS = 22.0    # From GHOST_AVOID_RADIUS
    GHOST_CRITICAL_RADIUS = 13.2  # 60% of avoid radius (immediate danger)
    POWERPILL_COLLECTION_RADIUS = 30.0  # From POWERPILL_RADIUS
    PELLET_COLLECTION_RADIUS = 20.0
    
    # Action mappings
    ACTIONS = {
        'NOOP': 0,
        'UP': 1,
        'RIGHT': 2,
        'LEFT': 3,
        'DOWN': 4,
        'UPRIGHT': 5,
        'UPLEFT': 6,
        'DOWNRIGHT': 7,
        'DOWNLEFT': 8
    }
    
    # Direction vectors for basic actions
    ACTION_VECTORS = {
        0: (0, 0),    # NOOP
        1: (0, -1),   # UP
        2: (1, 0),    # RIGHT
        3: (-1, 0),   # LEFT
        4: (0, 1),    # DOWN
        # 5: (1, -1),   # UPRIGHT
        # 6: (-1, -1),  # UPLEFT
        # 7: (1, 1),    # DOWNRIGHT
        # 8: (-1, 1)    # DOWNLEFT
    }
    
    def __init__(self, enable_logging=True, log_interval=100):
        self.enable_logging = enable_logging
        self.log_interval = log_interval
        self.step_count = 0
        self.episode_count = 0
        
        # Tracking
        self.last_position = None
        self.position_history = deque(maxlen=10)
        self.last_action = 0
        self.stuck_counter = 0
        self.consecutive_stuck_steps = 0  # Track how long we've been stuck
        
    def is_powered_up(self, frame):
        """Detect if Pac-Man is powered up (blue ghosts visible)"""
        eatable_ghosts = find_objects(frame, [(66, 114, 194)], min_distance=1)
        return bool(eatable_ghosts)
    
    def get_objects_by_category(self, objects):
        """Organize objects by category"""
        categorized = {
            'player': None,
            'ghosts': [],
            'powerpills': [],
            'pellets': []
        }
        
        for obj in objects:
            category = getattr(obj, 'category', None)
            if category == 'Player':
                categorized['player'] = obj
            elif category == 'Ghost':
                categorized['ghosts'].append(obj)
            elif category == 'PowerPill':
                categorized['powerpills'].append(obj)
            elif category == 'Pellet':
                categorized['pellets'].append(obj)
        
        return categorized
    
    def calculate_distance(self, pos1, pos2):
        """Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_direction_to_target(self, current_pos, target_pos):
        """Get the general direction from current to target"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Normalize to get direction
        distance = self.calculate_distance(current_pos, target_pos)
        if distance == 0:
            return (0, 0)
        
        return (dx / distance, dy / distance)
    
    def find_best_action_for_direction(self, direction, avoid_last_action=False, force_different=False):
        """
        Find the best action to move in a given direction.
        
        Args:
            direction: (dx, dy) normalized direction vector
            avoid_last_action: Try to avoid repeating last action if stuck
            force_different: Force a completely different action (for severe stuck states)
        """
        dx, dy = direction
        
        # Calculate dot product with each action vector to find best match
        action_scores = []
        
        for action, (ax, ay) in self.ACTION_VECTORS.items():
            if action == 0:  # Skip NOOP for now
                continue
            
            # Dot product (how aligned is this action with target direction)
            score = dx * ax + dy * ay
            
            # Strong penalty for last action if we're stuck
            if avoid_last_action and action == self.last_action:
                score -= 5.0  # Much stronger penalty
            
            # Force completely different action when severely stuck
            if force_different and action == self.last_action:
                continue  # Skip last action entirely
            
            action_scores.append((action, score))
        
        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best action (or second best if forcing different)
        if action_scores:
            return action_scores[0][0]
        return self.ACTIONS['NOOP']
    
    def find_safe_direction(self, player_pos, ghosts, try_perpendicular=False):
        """
        Find direction away from dangerous ghosts.
        Returns direction vector to move away from nearest ghost.
        
        Args:
            player_pos: Current player position
            ghosts: List of ghost objects
            try_perpendicular: If True, try perpendicular direction (for stuck situations)
        """
        if not ghosts:
            return (0, 0)
        
        # Find nearest ghost
        nearest_ghost = min(ghosts, 
                          key=lambda g: self.calculate_distance(player_pos, (g.x, g.y)))
        ghost_pos = (nearest_ghost.x, nearest_ghost.y)
        
        # Calculate vector away from ghost
        dx = player_pos[0] - ghost_pos[0]
        dy = player_pos[1] - ghost_pos[1]
        
        # Normalize
        distance = self.calculate_distance(player_pos, ghost_pos)
        if distance == 0:
            return (0, 0)
        
        escape_dx = dx / distance
        escape_dy = dy / distance
        
        # If stuck, try perpendicular direction instead
        if try_perpendicular:
            # Rotate 90 degrees: (dx, dy) -> (-dy, dx) or (dy, -dx)
            # Try the perpendicular direction that moves more
            return (-escape_dy, escape_dx)
        
        return (escape_dx, escape_dy)
    
    def evaluate_targets(self, player_pos, categorized_objects, is_powered):
        """
        Evaluate and prioritize all possible targets.
        Returns: (target_obj, priority, target_type)
        """
        targets = []
        
        # 1. SURVIVAL: Check for dangerous ghosts
        if not is_powered and categorized_objects['ghosts']:
            for ghost in categorized_objects['ghosts']:
                distance = self.calculate_distance(player_pos, (ghost.x, ghost.y))
                
                if distance < self.GHOST_CRITICAL_RADIUS:
                    # Critical danger - maximum priority to escape
                    priority = self.PRIORITY_SURVIVE + (self.GHOST_CRITICAL_RADIUS - distance)
                    targets.append((ghost, priority, 'escape_ghost'))
                elif distance < self.GHOST_DANGER_RADIUS:
                    # Moderate danger - high priority
                    priority = self.PRIORITY_SURVIVE * 0.5 + (self.GHOST_DANGER_RADIUS - distance)
                    targets.append((ghost, priority, 'avoid_ghost'))
        
        # 2. POWERPILLS: High value targets
        for powerpill in categorized_objects['powerpills']:
            distance = self.calculate_distance(player_pos, (powerpill.x, powerpill.y))
            
            if distance < self.POWERPILL_COLLECTION_RADIUS:
                # Closer = higher priority
                priority = self.PRIORITY_POWERPILL * (1.0 - distance / self.POWERPILL_COLLECTION_RADIUS)
                targets.append((powerpill, priority, 'powerpill'))
        
        # 3. GHOST HUNTING: When powered up
        if is_powered and categorized_objects['ghosts']:
            for ghost in categorized_objects['ghosts']:
                distance = self.calculate_distance(player_pos, (ghost.x, ghost.y))
                
                # HIGH BASE PRIORITY for all ghosts when powered
                # Closer ghosts get bonus priority
                priority = self.PRIORITY_HUNT_GHOST * (0.5 + 0.5 * (1.0 - min(distance / 50.0, 1.0)))
                targets.append((ghost, priority, 'hunt_ghost'))
        
        # 4. PELLETS: Default food collection
        for pellet in categorized_objects['pellets']:
            distance = self.calculate_distance(player_pos, (pellet.x, pellet.y))
            
            if distance < self.PELLET_COLLECTION_RADIUS:
                priority = self.PRIORITY_PELLET * (1.0 - distance / self.PELLET_COLLECTION_RADIUS)
                targets.append((pellet, priority, 'pellet'))
        
        # Sort by priority (highest first)
        targets.sort(key=lambda x: x[1], reverse=True)
        
        return targets
    
    def check_if_stuck(self, current_pos):
        """Check if agent is stuck in the same area"""
        self.position_history.append(current_pos)
        
        if len(self.position_history) < 5:
            self.consecutive_stuck_steps = 0
            return False
        
        # Calculate variance of recent positions
        positions = np.array(list(self.position_history))
        variance = np.var(positions[:, 0]) + np.var(positions[:, 1])
        
        # Low variance = stuck (more sensitive threshold)
        is_stuck = variance < 15.0
        
        # Update consecutive stuck counter
        if is_stuck:
            self.consecutive_stuck_steps += 1
        else:
            self.consecutive_stuck_steps = 0
        
        return is_stuck
    
    def select_action(self, env):
        """
        Main greedy decision making function.
        
        Args:
            env: OCAtari environment
            
        Returns:
            action: Integer action to take
        """
        # Get current game state
        frame = env.getScreenRGB() if hasattr(env, 'getScreenRGB') else None
        objects = getattr(env, 'objects', [])
        
        # Organize objects
        categorized = self.get_objects_by_category(objects)
        
        if categorized['player'] is None:
            return self.ACTIONS['NOOP']
        
        player = categorized['player']
        player_pos = (player.x, player.y)
        
        # Check if powered up
        is_powered = self.is_powered_up(frame) if frame is not None else False
        
        # Log power-up state changes
        if not hasattr(self, '_last_powered'):
            self._last_powered = False
        if is_powered != self._last_powered:
            if self.enable_logging:
                print(f"\n*** POWER STATE CHANGE: {'POWERED UP!' if is_powered else 'Power ended'} ***\n")
            self._last_powered = is_powered
        
        # Check if stuck
        is_stuck = self.check_if_stuck(player_pos)
        
        # Evaluate all targets with priorities
        targets = self.evaluate_targets(player_pos, categorized, is_powered)
        
        # Decision making based on highest priority target
        if targets:
            best_target, priority, target_type = targets[0]
            target_pos = (best_target.x, best_target.y)
            
            if self.enable_logging and self.step_count % self.log_interval == 0:
                print(f"Step {self.step_count}: Target={target_type}, "
                      f"Priority={priority:.2f}, **POWERED={is_powered}**, "
                      f"Pos=({player_pos[0]:.0f},{player_pos[1]:.0f}), "
                      f"Stuck={is_stuck} (consecutive={self.consecutive_stuck_steps})")
            
            # IMMEDIATE RESPONSE TO STUCK STATE
            if self.consecutive_stuck_steps >= 3:
                # Severely stuck - force completely different action
                if target_type in ['escape_ghost', 'avoid_ghost']:
                    # Try perpendicular escape when stuck
                    safe_direction = self.find_safe_direction(player_pos, categorized['ghosts'], 
                                                             try_perpendicular=True)
                    action = self.find_best_action_for_direction(safe_direction, 
                                                                avoid_last_action=True,
                                                                force_different=True)
                else:
                    # For other targets, try random direction to break free
                    available_actions = [a for a in [1, 2, 3, 4, 5, 6, 7, 8] if a != self.last_action]
                    action = np.random.choice(available_actions) if available_actions else np.random.choice([1, 2, 3, 4])
                    
                if self.enable_logging:
                    print(f"  -> FORCING action change due to stuck! Action={action}")
            
            # Handle escape/avoidance separately
            elif target_type in ['escape_ghost', 'avoid_ghost']:
                # Move away from ghost
                safe_direction = self.find_safe_direction(player_pos, categorized['ghosts'],
                                                         try_perpendicular=(self.consecutive_stuck_steps > 0))
                action = self.find_best_action_for_direction(safe_direction, 
                                                            avoid_last_action=is_stuck,
                                                            force_different=(self.consecutive_stuck_steps >= 2))
            else:
                # Move toward target
                direction = self.get_direction_to_target(player_pos, target_pos)
                action = self.find_best_action_for_direction(direction, 
                                                            avoid_last_action=is_stuck,
                                                            force_different=(self.consecutive_stuck_steps >= 2))
        else:
            # No clear target - explore
            if is_stuck or self.consecutive_stuck_steps > 0:
                # Try a different action to get unstuck - exclude last action
                available_actions = [a for a in [1, 2, 3, 4] if a != self.last_action]
                action = np.random.choice(available_actions) if available_actions else np.random.choice([1, 2, 3, 4])
            else:
                # Continue last action or move right (default)
                action = self.last_action if self.last_action != 0 else self.ACTIONS['RIGHT']
        
        self.last_action = action
        self.step_count += 1
        
        return action
    
    def run_episode(self, env, max_steps=10000, render=False):
        """
        Run a single episode using greedy algorithm.
        
        Args:
            env: OCAtari environment
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            total_reward, steps, info
        """
        obs, info = env.reset()
        
        # Skip loading screen (64 NOOPs)
        for _ in range(64):
            obs, _, _, _, _ = env.step(0)
        
        total_reward = 0
        steps = 0
        done = False
        
        self.position_history.clear()
        self.last_action = 0
        self.stuck_counter = 0
        self.consecutive_stuck_steps = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Episode {self.episode_count + 1}")
        print(f"{'='*60}")
        
        # Setup rendering window if needed
        if render:
            cv2.namedWindow('Greedy Pacman Agent', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Greedy Pacman Agent', 640, 480)
        
        while not done and steps < max_steps:
            # Select action using greedy algorithm
            action = self.select_action(env)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                try:
                    # Get RGB frame from OCAtari
                    if hasattr(env, 'getScreenRGB'):
                        frame = env.getScreenRGB()
                    else:
                        frame = env.render()
                    
                    if frame is not None:
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Greedy Pacman Agent', frame_bgr)
                        
                        # Check for quit key
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nQuitting early...")
                            done = True
                            break
                except Exception as e:
                    print(f"Render error: {e}")
                
                time.sleep(0.03)  # Slow down to make it easier to watch
        
        if render:
            cv2.destroyAllWindows()
        
        self.episode_count += 1
        
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count} Complete")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Average Reward per Step: {total_reward/max(steps,1):.3f}")
        print(f"{'='*60}\n")
        
        return total_reward, steps, info


def test_greedy_agent(n_episodes=5, render=False):
    """
    Test the greedy agent on Ms. Pacman.
    
    Args:
        n_episodes: Number of episodes to run
        render: Whether to render the game
    """
    print("\n" + "="*60)
    print("Greedy Pacman Agent Test")
    print("="*60)
    
    # Create environment - always use rgb_array mode
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="human",
                  mode="vision")  # Use 'both' mode for accurate object detection
    
    # Create agent
    agent = GreedyPacmanAgent(enable_logging=True, log_interval=100)
    
    # Run episodes
    episode_rewards = []
    episode_steps = []
    
    for episode in range(n_episodes):
        reward, steps, info = agent.run_episode(env, render=render)
        episode_rewards.append(reward)
        episode_steps.append(steps)
    
    # Summary statistics
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Reward: {np.min(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.1f}")
    print("="*60 + "\n")
    
    env.close()
    
    return episode_rewards, episode_steps


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Test greedy Pacman agent')
    parser.add_argument('--episodes', '-e', type=int, default=3,
                        help='Number of episodes to run (default: 3)')
    parser.add_argument('--render', '-r', action='store_true',
                        help='Enable rendering (requires display)')
    parser.add_argument('--save-video', '-s', type=str, default=None,
                        help='Save gameplay to video file (e.g., greedy_pacman.mp4)')
    
    args = parser.parse_args()
    
   
    # Check if we can render
    if args.render:
        if 'DISPLAY' not in os.environ:
            print("WARNING: No display found. Disabling rendering.")
            print("Running in evaluation mode (no visualization).")
            args.render = False
    
    # Run test
    print(f"\nRunning greedy agent for {args.episodes} episodes")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    if args.save_video:
        print(f"Video will be saved to: {args.save_video}")
    print()
    
    rewards, steps = test_greedy_agent(n_episodes=args.episodes, render=args.render)
    
    print("\nGreedy agent testing complete!")
    print(f"Results: Rewards = {rewards}")
    print(f"\nUsage:")
    print("  Without display: python greedy_pacman_agent.py -e 5")
    print("  With display:    python greedy_pacman_agent.py -e 3 --render")
