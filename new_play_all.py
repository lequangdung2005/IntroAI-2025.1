"""
Play and evaluate trained Pacman agents (DQN, PPO, A2C, Rainbow) using OCAtari
Load any trained model and watch it play or evaluate performance
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN, PPO, A2C
try:
    from sb3_contrib import QRDQN
except ImportError:
    QRDQN = None

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from ocatari.core import OCAtari
from preprocess import PreprocessFrame
import time
import os
import numpy as np
import cv2
from tabulate import tabulate

# Register ALE environments
gym.register_envs(ale_py)


def get_best_models(algorithm):
    """Get all model files from the best directory for the given algorithm"""
    best_dir = f"models/{algorithm}/best/"
    if not os.path.exists(best_dir):
        return []
    
    model_files = []
    for file in os.listdir(best_dir):
        if file.endswith('.zip'):
            model_files.append(os.path.join(best_dir, file))
    
    return sorted(model_files)

# Algorithm configurations
ALGORITHMS = {
    'dqn': {
        'class': DQN,
        'name': 'DQN (Deep Q-Network)'
    },
    'ppo': {
        'class': PPO,
        'name': 'PPO (Proximal Policy Optimization)'
    },
    'a2c': {
        'class': A2C,
        'name': 'A2C (Advantage Actor-Critic)'
    },
    'rainbow': {
        'class': QRDQN if QRDQN else DQN,
        'name': 'Rainbow/QR-DQN'
    }
}

def make_env_render():
    """Create and wrap the MsPacman environment using OCAtari for rendering"""
    env = OCAtari("ALE/MsPacman-v5",
                  render_mode="rgb_array",
                  mode="both")  # Changed to 'both' for accurate detection
    # Preprocess frames (grayscale 84x84) to match training environment
    env = PreprocessFrame(env, width=84, height=84, force_image=True)
    env = Monitor(env)
    return env

def load_model(algorithm, model_path=None):
    """Load a trained model for the specified algorithm"""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(ALGORITHMS.keys())}")

    algo_config = ALGORITHMS[algorithm]

    if model_path is None:
        # Get all models from best directory
        best_models = get_best_models(algorithm)
        if not best_models:
            raise FileNotFoundError(f"No models found in models/{algorithm}/best/")
        model_path = best_models[0]  # Use first model found
        print(f"Auto-selected model: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading {algo_config['name']} from {model_path}...")

    # Create environment with OCAtari
    env = DummyVecEnv([make_env_render])
    env = VecTransposeImage(env)  # Transpose (H,W,C) -> (C,H,W) for CNN
    env = VecFrameStack(env, n_stack=4)

    # Load model
    model = algo_config['class'].load(model_path, env=env)

    return model, env, algo_config['name']

def play_agent(algorithm, model_path=None, num_episodes=5, render=True):
    """Play the agent and optionally render gameplay"""
    print("=" * 60)
    print(f"Playing MsPacman Agent with OCAtari")
    print("=" * 60)

    # Load model
    model, env, algo_name = load_model(algorithm, model_path)

    print(f"Algorithm: {algo_name}")
    print(f"Episodes: {num_episodes}")
    print("-" * 60)

    if render:
        cv2.namedWindow('MsPacman Agent', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MsPacman Agent', 640, 480)

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
                    # Get the original OCAtari environment
                    original_env = env.envs[0].env
                    while hasattr(original_env, 'env'):
                        original_env = original_env.env

                    # Get RGB frame from OCAtari
                    if hasattr(original_env, 'getScreenRGB'):
                        frame = original_env.getScreenRGB()
                    else:
                        frame = original_env.render()

                    if frame is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('MsPacman Agent', frame_bgr)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nQuitting early...")
                            done = True
                            break
                except Exception as e:
                    print(f"Render error: {e}")
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

def play_all_models_for_algorithm(algorithm, num_episodes=5, render=False):
    """Play all models in the best directory for a specific algorithm"""
    print("=" * 60)
    print(f"Playing All {ALGORITHMS[algorithm]['name']} Models")
    print("=" * 60)
    
    best_models = get_best_models(algorithm)
    if not best_models:
        print(f"No models found for {algorithm} in models/{algorithm}/best/")
        return {}
    
    results = {}
    
    for i, model_path in enumerate(best_models):
        model_name = os.path.basename(model_path)
        print(f"\nModel {i+1}/{len(best_models)}: {model_name}")
        print("-" * 40)
        
        try:
            rewards, lengths = play_agent(algorithm, model_path, num_episodes, render)
            results[model_name] = {
                'path': model_path,
                'rewards': rewards,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'best_reward': np.max(rewards),
                'mean_length': np.mean(lengths)
            }
        except Exception as e:
            print(f"Error playing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print summary for this algorithm
    print(f"\n{'='*70}")
    print(f"SUMMARY - {ALGORITHMS[algorithm]['name']}")
    print(f"{'='*70}")
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        # Prepare data for tabulate
        table_data = []
        for model_name, result in sorted(valid_results.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
            table_data.append([
                model_name,
                f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                f"{result['best_reward']:.0f}",
                f"{result['mean_length']:.0f}"
            ])
        
        headers = ["Model Name", "Mean Score", "Best Score", "Avg Steps"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="left", numalign="right"))
    
    print("=" * 70)
    return results

def play_all_models_all_algorithms(num_episodes=5, render=False):
    """Play all models for all algorithms"""
    print("=" * 60)
    print("Playing All Models for All Algorithms")
    print("=" * 60)
    
    all_results = {}
    
    for algorithm in ALGORITHMS.keys():
        all_results[algorithm] = play_all_models_for_algorithm(algorithm, num_episodes, render)
    
    # Print overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON - BEST MODEL FROM EACH ALGORITHM")
    print(f"{'='*80}")
    
    algorithm_bests = {}
    for algorithm, models in all_results.items():
        valid_models = {k: v for k, v in models.items() if 'error' not in v}
        if valid_models:
            best_model = max(valid_models.items(), key=lambda x: x[1]['mean_reward'])
            algorithm_bests[algorithm] = {
                'name': ALGORITHMS[algorithm]['name'],
                'model_name': best_model[0],
                'mean_reward': best_model[1]['mean_reward'],
                'std_reward': best_model[1]['std_reward'],
                'best_reward': best_model[1]['best_reward']
            }
    
    if algorithm_bests:
        # Prepare data for tabulate
        table_data = []
        for i, (algorithm, result) in enumerate(sorted(algorithm_bests.items(), key=lambda x: x[1]['mean_reward'], reverse=True), 1):
            table_data.append([
                f"{i}",  # Rank
                result['name'],
                result['model_name'],
                f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                f"{result['best_reward']:.0f}"
            ])
        
        headers = ["Rank", "Algorithm", "Best Model", "Mean Score", "Best Score"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="left", numalign="right"))
    
    print("=" * 80)
    return all_results

def evaluate_all_algorithms(num_episodes=10):
    """Evaluate all trained algorithms and compare performance (using first model from each best directory)"""
    print("=" * 60)
    print("Evaluating All Algorithms with OCAtari")
    print("=" * 60)

    results = {}

    for algo in ALGORITHMS.keys():
        best_models = get_best_models(algo)
        
        if best_models:
            print(f"\nEvaluating {ALGORITHMS[algo]['name']}...")
            try:
                # Use first model from best directory
                rewards, lengths = play_agent(algo, model_path=best_models[0], num_episodes=num_episodes, render=False)
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
            print(f"\nSkipping {ALGORITHMS[algo]['name']} - no models found in models/{algo}/best/")

    # Print comparison table
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    if results:
        # Prepare data for tabulate
        table_data = []
        for i, (algo, result) in enumerate(sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True), 1):
            table_data.append([
                f"{i}",  # Rank
                result['name'],
                f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                f"{result['best_reward']:.0f}",
                f"{result['mean_length']:.0f}"
            ])
        
        headers = ["Rank", "Algorithm", "Mean Score", "Best Score", "Avg Steps"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="left", numalign="right"))

    print("=" * 70)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Play and evaluate MsPacman RL agents with OCAtari')
    parser.add_argument('--algorithm', '-a', type=str, choices=list(ALGORITHMS.keys()),
                        help='Algorithm to use (dqn, ppo, a2c, rainbow)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                        help='Number of episodes to play')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (evaluation only)')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare all algorithms (first model from each best directory)')
    parser.add_argument('--play-all-models', action='store_true',
                        help='Play all models in best directories for all algorithms')
    parser.add_argument('--play-all-for-algo', type=str, choices=list(ALGORITHMS.keys()),
                        help='Play all models for a specific algorithm')

    args = parser.parse_args()

    if args.play_all_models:
        # Play all models for all algorithms
        play_all_models_all_algorithms(num_episodes=args.episodes, render=not args.no_render)
    elif args.play_all_for_algo:
        # Play all models for specific algorithm
        play_all_models_for_algorithm(args.play_all_for_algo, num_episodes=args.episodes, render=not args.no_render)
    elif args.compare:
        # Compare all algorithms (first model from each)
        evaluate_all_algorithms(num_episodes=args.episodes)
    elif args.algorithm:
        # Play specific algorithm
        play_agent(args.algorithm, args.model, args.episodes, render=not args.no_render)
    else:
        # Show help if no arguments
        print("Usage:")
        print("  Play specific algorithm:")
        print("    python new_play_all.py -a dqn -e 5")
        print("    python new_play_all.py -a ppo --model models/ppo/best/best_model.zip")
        print("  Play all models for a specific algorithm:")
        print("    python new_play_all.py --play-all-for-algo dqn -e 3")
        print("  Play all models for all algorithms:")
        print("    python new_play_all.py --play-all-models -e 3 --no-render")
        print("  Compare all algorithms (first model from each):")
        print("    python new_play_all.py --compare -e 10")
        print("\nAvailable algorithms:", list(ALGORITHMS.keys()))

