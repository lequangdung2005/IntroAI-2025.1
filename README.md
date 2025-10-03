# Pacman Reinforcement Learning - Algorithm Comparison

Train and compare 4 state-of-the-art RL algorithms on Pacman: **DQN, PPO, A2C, and Rainbow**.

## 🎮 Overview

This project implements and compares four reinforcement learning algorithms:

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| **DQN** | Value-based | Deep Q-Network with experience replay |
| **PPO** | Policy-based | Proximal Policy Optimization, stable training |
| **A2C** | Policy-based | Advantage Actor-Critic, efficient on-policy |
| **Rainbow** | Value-based | Combines multiple DQN improvements (QR-DQN) |

## 📦 Installation

### 1. Create Environment

```bash
conda create -n pacman python=3.10
conda activate pacman
```

### 2. Install Dependencies

```bash
pip install gymnasium stable-baselines3 ale-py torch tensorboard opencv-python numpy
pip install "gymnasium[atari,accept-rom-license]"
pip install sb3-contrib  # For Rainbow (QR-DQN)
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training Individual Algorithms

Train each algorithm separately:

```bash
# Train DQN
python train_dqn.py

# Train PPO
python train_ppo.py

# Train A2C
python train_a2c.py

# Train Rainbow
python train_rainbow.py
```

**Training parameters:**
- Total timesteps: 1,000,000
- Checkpoint frequency: Every 50,000 steps
- Models saved to: `models/{algorithm}/`
- Logs saved to: `logs/{algorithm}/`

### Monitor Training Progress

```bash
# View training progress in TensorBoard
tensorboard --logdir=./logs/

# View specific algorithm
tensorboard --logdir=./logs/dqn/
```

### Playing and Evaluation

**Play a specific algorithm:**

```bash
# Play DQN agent (with rendering)
python play_all.py -a dqn -e 5

# Play PPO agent
python play_all.py -a ppo -e 10

# Play specific model checkpoint
python play_all.py -a dqn --model models/dqn/pacman_dqn_100000_steps.zip
```

**Compare all algorithms:**

```bash
# Evaluate and compare all 4 algorithms (10 episodes each)
python play_all.py --compare -e 10
```

**Evaluate without rendering (faster):**

```bash
python play_all.py -a ppo -e 20 --no-render
```

## 📊 Algorithm Comparison

### DQN (Deep Q-Network)
- **Type:** Value-based
- **Pros:** Stable, proven performance on Atari games
- **Cons:** Sample inefficient, requires large replay buffer
- **Best for:** Games with discrete actions

### PPO (Proximal Policy Optimization)
- **Type:** Policy-based
- **Pros:** Very stable, works well with parallel environments
- **Cons:** Slower per-step than DQN
- **Best for:** Complex continuous/discrete action spaces

### A2C (Advantage Actor-Critic)
- **Type:** Policy-based
- **Pros:** Efficient, good for parallel training
- **Cons:** Less stable than PPO
- **Best for:** Fast training with multiple environments

### Rainbow (QR-DQN)
- **Type:** Value-based (enhanced)
- **Pros:** State-of-the-art performance, combines best DQN improvements
- **Cons:** More complex, requires more memory
- **Best for:** Maximum performance on Atari games

## 📁 Project Structure

```
.
├── train_dqn.py          # Train DQN agent
├── train_ppo.py          # Train PPO agent
├── train_a2c.py          # Train A2C agent
├── train_rainbow.py      # Train Rainbow/QR-DQN agent
├── play_all.py           # Unified play/evaluation script
├── requirements.txt      # Python dependencies
├── models/               # Saved models (created during training)
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   └── rainbow/
└── logs/                 # TensorBoard logs (created during training)
    ├── dqn/
    ├── ppo/
    ├── a2c/
    └── rainbow/
```

## 🎯 Expected Performance

After 1M timesteps of training:

| Algorithm | Avg Score | Training Time (GPU) |
|-----------|-----------|---------------------|
| DQN       | 800-1200  | 2-3 hours          |
| PPO       | 900-1400  | 3-4 hours          |
| A2C       | 700-1100  | 2-3 hours          |
| Rainbow   | 1000-1500 | 3-4 hours          |

*Note: Performance varies based on random seeds and training conditions*

## 🔧 Hyperparameter Tuning

Each training script includes optimized hyperparameters. To modify:

**DQN/Rainbow:**
- `buffer_size`: Replay buffer size (50000 default, increase if RAM allows)
- `learning_rate`: Learning rate (1e-4 default)
- `exploration_fraction`: Fraction of training for epsilon decay

**PPO:**
- `n_steps`: Steps per update per environment (128 default)
- `batch_size`: Minibatch size (256 default)
- `learning_rate`: Learning rate (2.5e-4 default)

**A2C:**
- `n_steps`: Steps per update (5 default)
- `learning_rate`: Learning rate (7e-4 default)

## 📈 Evaluation Metrics

The scripts track:
- **Episode reward:** Total score per game
- **Episode length:** Steps until game over
- **Training loss:** Model learning progress
- **Exploration rate:** (For DQN/Rainbow) Epsilon-greedy exploration

## 🐛 Troubleshooting

**"Namespace ALE not found":**
```bash
pip install ale-py "gymnasium[atari,accept-rom-license]"
```

**"Not enough memory for replay buffer":**
Reduce `buffer_size` in training script (e.g., from 100000 to 50000)

**SDL Renderer error:**
The play script uses OpenCV rendering to avoid SDL issues

**Import sb3_contrib error:**
```bash
pip install sb3-contrib
```

## 📚 References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [A2C/A3C Paper](https://arxiv.org/abs/1602.01783)
- [Rainbow Paper](https://arxiv.org/abs/1710.02298)

## 🤝 Contributing

Feel free to experiment with:
- Different hyperparameters
- Custom reward functions
- Other Atari games
- Advanced algorithms (SAC, TD3, etc.)

## 📝 License

MIT License - feel free to use for learning and research!
