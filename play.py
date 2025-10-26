"""
Play trained Pacman RL agents - Unified Player
Supports both baseline and enhanced models for all 4 algorithms
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN, PPO, A2C
try:
    from sb3_contrib import QRDQN
except ImportError:
    QRDQN = None

from ocatari.core import OCAtari
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
import sys
import os
import numpy as np
import cv2
import time
import argparse

# Check if preprocess module exists for enhanced mode
try:
    from preprocess import PreprocessFrame
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Register ALE environments
gym.register_envs(ale_py)

# Algorithm configurations
ALGORITHMS = {
    'dqn': {'class': DQN, 'name': 'DQN'},
    'ppo': {'class': PPO, 'name': 'PPO'},
    'a2c': {'class': A2C, 'name': 'A2C'},
    'rainbow': {'class': QRDQN if QRDQN else DQN, 'name': 'Rainbow/QR-DQN'}
}


def make_env_baseline():
    """Create baseline environment (standard Atari)"""
    env = gym.make("ALE/Pacman-v5",
                   render_mode="rgb_array",
                   frameskip=1)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env


def make_env_enhanced():
    """Create enhanced environment (OCAtari with preprocessing)"""
    if not ENHANCED_AVAILABLE:
        raise ImportError("PreprocessFrame not found. Make sure preprocess.py is available.")
    
    env = OCAtari("ALE/Pacman-v5",
                  render_mode="rgb_array",
                  mode="vision")
    env = PreprocessFrame(env, width=84, height=84, force_image=True)
    env = Monitor(env)
    return env


def load_model(algorithm, model_path, mode='baseline'):
    """Load a trained model"""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    algo_config = ALGORITHMS[algorithm]
    print(f"Loading {algo_config['name']} from {model_path}...")
    
    # Create environment based on mode
    if mode == 'enhanced':
        env = DummyVecEnv([make_env_enhanced])
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)
    else:  # baseline
        env = DummyVecEnv([make_env_baseline])
        env = VecFrameStack(env, n_stack=4)
    
    # Load model
    model = algo_config['class'].load(model_path, env=env)
    
    return model, env, algo_config['name']


def play_agent(algorithm, model_path, num_episodes=5, render=True, mode='baseline'):
    """Play the agent and optionally render gameplay"""
    print("=" * 60)
    print(f"Playing Pacman - {ALGORITHMS[algorithm]['name']}")
    print(f"Mode: {mode.upper()}")
    print("=" * 60)
    
    # Load model
    model, env, algo_name = load_model(algorithm, model_path, mode)
    
    print(f"Episodes: {num_episodes}")
    print("-" * 60)
    
    if render:
        window_name = f'Pacman - {algo_name} ({mode})'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0]
            steps += 1
            
            # Render if requested
            if render:
                try:
                    # Get the environment for rendering
                    original_env = env.envs[0].env
                    
                    # Navigate through wrappers
                    while hasattr(original_env, "env") and not hasattr(original_env, "render"):
                        original_env = original_env.env
                    
                    # Get frame
                    if hasattr(original_env, "getScreenRGB"):
                        frame = original_env.getScreenRGB()
                    else:
                        frame = original_env.render()
                    
                    if frame is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow(window_name, frame_bgr)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nQuitting early...")
                            done = True
                            break
                except Exception as e:
                    pass
                
                time.sleep(0.03)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}/{num_episodes}: Score = {total_reward:.0f}, Steps = {steps}")
    
    if render:
        cv2.destroyAllWindows()
    
    env.close()
    
    # Print statistics
    print("-" * 60)
    print(f"Average Score: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Best Score: {np.max(episode_rewards):.0f}")
    print(f"Worst Score: {np.min(episode_rewards):.0f}")
    print("=" * 60)
    
    return episode_rewards, episode_lengths


def compare_models(baseline_path, enhanced_path, algorithm, num_episodes=10):
    """Compare baseline vs enhanced model performance"""
    print("=" * 60)
    print(f"Comparing Baseline vs Enhanced - {ALGORITHMS[algorithm]['name']}")
    print("=" * 60)
    
    results = {}
    
    # Test baseline
    if os.path.exists(baseline_path):
        print("\n[1/2] Testing BASELINE model...")
        baseline_rewards, baseline_lengths = play_agent(
            algorithm, baseline_path, num_episodes, render=False, mode='baseline'
        )
        results['baseline'] = {
            'mean_reward': np.mean(baseline_rewards),
            'std_reward': np.std(baseline_rewards),
            'best_reward': np.max(baseline_rewards)
        }
    else:
        print(f"\nBaseline model not found: {baseline_path}")
    
    # Test enhanced
    if os.path.exists(enhanced_path):
        print("\n[2/2] Testing ENHANCED model...")
        enhanced_rewards, enhanced_lengths = play_agent(
            algorithm, enhanced_path, num_episodes, render=False, mode='enhanced'
        )
        results['enhanced'] = {
            'mean_reward': np.mean(enhanced_rewards),
            'std_reward': np.std(enhanced_rewards),
            'best_reward': np.max(enhanced_rewards)
        }
    else:
        print(f"\nEnhanced model not found: {enhanced_path}")
    
    # Print comparison
    if 'baseline' in results and 'enhanced' in results:
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"{'Mode':<15} {'Mean Score':<20} {'Best Score':<15}")
        print("-" * 60)
        
        baseline_mean = results['baseline']['mean_reward']
        baseline_std = results['baseline']['std_reward']
        baseline_best = results['baseline']['best_reward']
        print(f"{'Baseline':<15} {baseline_mean:>8.2f} ± {baseline_std:<8.2f} {baseline_best:>14.0f}")
        
        enhanced_mean = results['enhanced']['mean_reward']
        enhanced_std = results['enhanced']['std_reward']
        enhanced_best = results['enhanced']['best_reward']
        print(f"{'Enhanced':<15} {enhanced_mean:>8.2f} ± {enhanced_std:<8.2f} {enhanced_best:>14.0f}")
        
        improvement = ((enhanced_mean - baseline_mean) / baseline_mean) * 100
        print("-" * 60)
        print(f"Improvement: {improvement:+.2f}%")
        print("=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Pacman RL agents')
    parser.add_argument('--algorithm', '-a', type=str, choices=list(ALGORITHMS.keys()),
                        help='Algorithm: dqn, ppo, a2c, rainbow')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--mode', type=str, choices=['baseline', 'enhanced'], default='baseline',
                        help='Model mode: baseline or enhanced')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                        help='Number of episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare baseline vs enhanced')
    
    args = parser.parse_args()
    
    if args.compare and args.algorithm:
        # Compare mode
        baseline_path = f"models/baseline_{args.algorithm}/pacman_{args.algorithm}_baseline_final.zip"
        enhanced_path = f"models/enhanced_{args.algorithm}/pacman_{args.algorithm}_enhanced_final.zip"
        compare_models(baseline_path, enhanced_path, args.algorithm, args.episodes)
    elif args.algorithm and args.model:
        # Play specific model
        play_agent(args.algorithm, args.model, args.episodes, not args.no_render, args.mode)
    else:
        # Show usage
        print("Pacman RL Agent Player")
        print("\nUsage Examples:")
        print("  # Play baseline DQN")
        print("  python play.py -a dqn -m models/baseline_dqn/pacman_dqn_baseline_final.zip --mode baseline")
        print("\n  # Play enhanced PPO")
        print("  python play.py -a ppo -m models/enhanced_ppo/pacman_ppo_enhanced_final.zip --mode enhanced")
        print("\n  # Compare baseline vs enhanced")
        print("  python play.py -a dqn --compare -e 10")
        print("\nAvailable algorithms:", list(ALGORITHMS.keys()))
