"""
Watch a trained DQN agent play Pacman using Stable Baselines3
"""
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import time
import os
import numpy as np
import cv2

# Register ALE environments
gym.register_envs(ale_py)

def make_env_render():
    """Create and wrap the Pacman environment for rendering"""
    # Use rgb_array mode to avoid SDL issues
    env = gym.make("ALE/Pacman-v5", 
                   render_mode="rgb_array",  # Use RGB array instead of human
                   frameskip=1)
    
    # Wrap with Atari preprocessing (same as training)
    env = AtariWrapper(env)
    
    # Monitor wrapper for logging
    env = Monitor(env)
    
    return env

def play_agent(model_path="models/pacman_dqn_final.zip", num_episodes=5):
    """Load a trained model and watch it play"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("\nAvailable models in 'models/' directory:")
        if os.path.exists("models"):
            models = [f for f in os.listdir("models") if f.endswith(".zip")]
            if models:
                for model in models:
                    print(f"  - {model}")
                print(f"\nTry running: python play_pacman.py --model models/{models[0]}")
            else:
                print("  No trained models found!")
                print("\nPlease train a model first by running:")
                print("  python train_pacman.py")
        return
    
    print("=" * 60)
    print("Loading trained DQN agent for Pacman")
    print("=" * 60)
    print(f"Model: {model_path}")
    
    # Create environment for rendering
    env = DummyVecEnv([make_env_render])
    env = VecFrameStack(env, n_stack=4)
    
    # Load the trained model
    print("Loading model...")
    model = DQN.load(model_path, env=env)
    
    print(f"Starting playback for {num_episodes} episodes...")
    print("Press 'q' in the game window to quit early.")
    print("-" * 60)
    
    # Create OpenCV window for display
    cv2.namedWindow('Pacman DQN Agent', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pacman DQN Agent', 640, 480)
    
    # Play multiple episodes
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Let the model choose the action (deterministic=True for best action)
            action, _states = model.predict(obs, deterministic=True)
            
            # Perform the action
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            # Get the RGB frame from the environment
            try:
                # Get the original environment from the vectorized wrapper
                original_env = env.envs[0].env
                # Render the frame
                frame = original_env.render()
                if frame is not None:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # Display the frame
                    cv2.imshow('Pacman DQN Agent', frame_bgr)
                    
                    # Wait for 1ms and check if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nQuitting early...")
                        done = True
                        break
            except Exception as e:
                pass  # Continue without rendering if there's an issue
            
            # Small delay to make it watchable
            time.sleep(0.03)
        
        print(f"Episode {episode + 1}/{num_episodes}: Score = {total_reward:.0f}, Steps = {steps}")
    
    print("-" * 60)
    print("Playback finished!")
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    import sys
    
    # Default model path
    model_path = "models/pacman_dqn_final.zip"
    num_episodes = 5
    
    # Simple command-line argument parsing
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("Usage: python play_pacman.py [MODEL_PATH] [NUM_EPISODES]")
            print("\nExamples:")
            print("  python play_pacman.py")
            print("  python play_pacman.py models/pacman_dqn_final.zip")
            print("  python play_pacman.py models/pacman_dqn_final.zip 10")
            sys.exit(0)
        model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        num_episodes = int(sys.argv[2])
    
    # Watch the agent play
    play_agent(model_path, num_episodes)
