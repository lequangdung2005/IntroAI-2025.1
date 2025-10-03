"""
Play and evaluate trained Pacman agents (DQN, PPO, A2C, Rainbow)
Load any trained model and watch it play or evaluate performance
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN, PPO, A2C
try:
    from sb3_contrib import QRDQN
except ImportError:
    QRDQN = None
    
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import time
import os
import numpy as np
import cv2

# Register ALE environments
gym.register_envs(ale_py)

# Algorithm configurations
ALGORITHMS = {
    'dqn': {
        'class': DQN,
        'default_path': 'models/dqn/pacman_dqn_final.zip',
        'name': 'DQN (Deep Q-Network)'
    },
    'ppo': {
        'class': PPO,
        'default_path': 'models/ppo/pacman_ppo_final.zip',
        'name': 'PPO (Proximal Policy Optimization)'
    },
    'a2c': {
        'class': A2C,
        'default_path': 'models/a2c/pacman_a2c_final.zip',
        'name': 'A2C (Advantage Actor-Critic)'
    },
    'rainbow': {
        'class': QRDQN if QRDQN else DQN,
        'default_path': 'models/rainbow/pacman_rainbow_final.zip',
        'name': 'Rainbow/QR-DQN'
    }
}

def make_env_render():
    """Create and wrap the Pacman environment for rendering"""
    env = gym.make("ALE/Pacman-v5", 
                   render_mode="rgb_array",
                   frameskip=1)
    env = AtariWrapper(env)
    env = Monitor(env)
    return env

def load_model(algorithm, model_path=None):
    """Load a trained model for the specified algorithm"""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(ALGORITHMS.keys())}")
    
    algo_config = ALGORITHMS[algorithm]
    
    if model_path is None:
        model_path = algo_config['default_path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading {algo_config['name']} from {model_path}...")
    
    # Create environment
    env = DummyVecEnv([make_env_render])
    env = VecFrameStack(env, n_stack=4)
    
    # Load model
    model = algo_config['class'].load(model_path, env=env)
    
    return model, env, algo_config['name']

def play_agent(algorithm, model_path=None, num_episodes=5, render=True):
    """Play the agent and optionally render gameplay"""
    print("=" * 60)
    print(f"Playing Pacman Agent")
    print("=" * 60)
    
    # Load model
    model, env, algo_name = load_model(algorithm, model_path)
    
    print(f"Algorithm: {algo_name}")
    print(f"Episodes: {num_episodes}")
    print("-" * 60)
    
    if render:
        cv2.namedWindow('Pacman Agent', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pacman Agent', 640, 480)
    
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
            # with open("debug.txt", "a") as f:
            #     f.write(f"Step {steps}: Action {action}, Reward {reward}, Done {done}\n")
            total_reward += reward[0]
            steps += 1
            
            # Render if requested
            if render:
                try:
                    original_env = env.envs[0].env
                    frame = original_env.render()
                    if frame is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Pacman Agent', frame_bgr)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nQuitting early...")
                            done = True
                            break
                except Exception:
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

def evaluate_all_algorithms(num_episodes=10):
    """Evaluate all trained algorithms and compare performance"""
    print("=" * 60)
    print("Evaluating All Algorithms")
    print("=" * 60)
    
    results = {}
    
    for algo in ALGORITHMS.keys():
        default_path = ALGORITHMS[algo]['default_path']
        
        if os.path.exists(default_path):
            print(f"\nEvaluating {ALGORITHMS[algo]['name']}...")
            try:
                rewards, lengths = play_agent(algo, num_episodes=num_episodes, render=False)
                results[algo] = {
                    'name': ALGORITHMS[algo]['name'],
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'mean_length': np.mean(lengths),
                    'best_reward': np.max(rewards)
                }
            except Exception as e:
                print(f"Error evaluating {algo}: {e}")
        else:
            print(f"\nSkipping {ALGORITHMS[algo]['name']} - model not found at {default_path}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean Score':<15} {'Best Score':<15}")
    print("-" * 60)
    
    for algo, result in sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
        print(f"{result['name']:<25} {result['mean_reward']:>8.2f} ± {result['std_reward']:<4.2f} {result['best_reward']:>14.0f}")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Play and evaluate Pacman RL agents')
    parser.add_argument('--algorithm', '-a', type=str, choices=list(ALGORITHMS.keys()),
                        help='Algorithm to use (dqn, ppo, a2c, rainbow)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                        help='Number of episodes to play')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (evaluation only)')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare all algorithms')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all algorithms
        evaluate_all_algorithms(num_episodes=args.episodes)
    elif args.algorithm:
        # Play specific algorithm
        play_agent(args.algorithm, args.model, args.episodes, render=not args.no_render)
    else:
        # Show help if no arguments
        print("Usage:")
        print("  Play specific algorithm:")
        print("    python play_all.py -a dqn -e 5")
        print("    python play_all.py -a ppo --model models/ppo/best/best_model.zip")
        print("  Compare all algorithms:")
        print("    python play_all.py --compare -e 10")
        print("\nAvailable algorithms:", list(ALGORITHMS.keys()))
