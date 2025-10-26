# Pacman Reinforcement Learning with Reward Shaping

A comprehensive comparison of Deep RL algorithms on Atari Pacman, comparing baseline (standard rewards) vs enhanced training (custom reward shaping).

## 📋 Overview

This project implements and compares 4 state-of-the-art Deep Reinforcement Learning algorithms:
- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimization)
- **A2C** (Advantage Actor-Critic)
- **Rainbow** (QR-DQN with distributional RL)

Each algorithm is trained in two modes:
1. **Baseline**: Standard Atari preprocessing with game rewards only
2. **Enhanced**: Custom reward shaping with 10 reward components

## 🎯 Project Goals

1. **Implement** 4 modern RL algorithms using Stable Baselines3
2. **Compare** baseline vs enhanced training with reward shaping
3. **Analyze** the impact of domain-specific reward engineering
4. **Evaluate** which algorithms benefit most from reward shaping

## 🚀 Quick Start

```bash
# 1. Setup environment
conda create -n pacman python=3.10
conda activate pacman

# 2. Install dependencies
pip install gymnasium ale-py stable-baselines3 sb3-contrib ocatari opencv-python torch numpy

# 3. Train baseline DQN
cd baseline
python train_baseline_dqn.py

# 4. Train enhanced DQN (with reward shaping)
cd ../enhanced_algorithm
python train_enhanced_dqn.py

# 5. Compare results
cd ..
python play.py -a dqn --compare -e 10
```

See [QUICKSTART.md](QUICKSTART.md) for detailed usage.

## 📁 Project Structure

```
├── baseline/                          # Baseline training (standard rewards)
│   ├── train_baseline_dqn.py         # DQN baseline
│   ├── train_baseline_ppo.py         # PPO baseline
│   ├── train_baseline_a2c.py         # A2C baseline
│   └── train_baseline_rainbow.py     # Rainbow baseline
│
├── enhanced_algorithm/                # Enhanced training (reward shaping)
│   ├── enhanced_wrapper.py           # Custom reward wrapper (10 components)
│   ├── preprocess.py                 # OCAtari preprocessing
│   ├── train_enhanced_dqn.py         # DQN enhanced
│   ├── train_enhanced_ppo.py         # PPO enhanced
│   ├── train_enhanced_a2c.py         # A2C enhanced
│   ├── train_enhanced_rainbow.py     # Rainbow enhanced
│   ├── train_all_enhanced.sh         # Batch training script
│   └── README_ENHANCED.md            # Enhanced mode documentation
│
├── play.py                            # Unified player & comparison tool
├── QUICKSTART.md                      # Quick start guide
└── README.md                          # This file
```

## 🎮 Training Modes

### Baseline Mode

**Location:** `baseline/`

**Features:**
- Standard Atari preprocessing (AtariWrapper)
- Game rewards only (no custom shaping)
- Fair comparison baseline
- Faster training (no object detection)

**Environment:**
```python
env = gym.make("ALE/Pacman-v5")
env = AtariWrapper(env)  # Grayscale, resize, frame skip
env = VecFrameStack(env, n_stack=4)
```

### Enhanced Mode

**Location:** `enhanced_algorithm/`

**Features:**
- OCAtari object detection (ghosts, dots, power pellets)
- 10-component custom reward wrapper
- Domain-specific Pacman strategy
- Optimized for performance

**Environment:**
```python
env = OCAtari("ALE/Pacman-v5", mode="vision")
env = PreprocessFrame(env)  # Custom preprocessing
env = EnhancedPacmanRewardWrapper(env)  # Reward shaping
env = VecFrameStack(env, n_stack=4)
```

## 🧠 Enhanced Reward Components

The enhanced mode implements 10 reward components to guide learning:

| Component | Reward | Purpose |
|-----------|--------|---------|
| Score Bonus | +0.1 × score | Encourage high scores |
| Power Pellet Seeking | +5.0 | Move toward power pellets |
| Ghost Hunting | +50.0 | Eat vulnerable ghosts |
| Ghost Avoidance | +2.0 | Stay away from threats |
| Dot Collection | +1.0 | Efficient dot eating |
| Movement Bonus | +0.5 | Exploration reward |
| Corner Penalty | -1.0 | Avoid getting trapped |
| Survival Bonus | +0.1 | Stay alive longer |
| Death Penalty | -100.0 | Strong death discouragement |
| Level Completion | +200.0 | Beat the level |

**Total Enhanced Reward = Σ (component_rewards)**

See [enhanced_algorithm/README_ENHANCED.md](enhanced_algorithm/README_ENHANCED.md) for implementation details.

## 🤖 Algorithms

### DQN (Deep Q-Network)

**Type:** Value-based
**Paper:** [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602)

**Key Features:**
- Experience replay buffer
- Target network for stability
- ε-greedy exploration

**Hyperparameters:**
- Learning rate: 1e-4
- Buffer size: 50,000
- Batch size: 32
- Gamma: 0.99
- Target update: 1,000 steps

### PPO (Proximal Policy Optimization)

**Type:** Policy-based
**Paper:** [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)

**Key Features:**
- Clipped surrogate objective
- Parallel environment collection
- On-policy learning

**Hyperparameters:**
- Learning rate: 2.5e-4
- N steps: 128
- Batch size: 256
- N epochs: 4
- Clip range: 0.1

### A2C (Advantage Actor-Critic)

**Type:** Policy-based
**Paper:** [Asynchronous Methods for Deep RL (2016)](https://arxiv.org/abs/1602.01783)

**Key Features:**
- Actor-critic architecture
- Advantage estimation
- Fast synchronous updates

**Hyperparameters:**
- Learning rate: 7e-4
- N steps: 5
- Gamma: 0.99
- Entropy coefficient: 0.01

### Rainbow (QR-DQN)

**Type:** Value-based (Distributional)
**Paper:** [Rainbow: Combining Improvements in DRL (2017)](https://arxiv.org/abs/1710.02298)

**Key Features:**
- Quantile regression
- Distributional value learning
- Multiple DQN improvements

**Hyperparameters:**
- Learning rate: 1e-4
- Buffer size: 50,000
- Batch size: 32
- N quantiles: 200

## 📊 Training Configuration

All algorithms train with identical settings for fair comparison:

- **Total Timesteps:** 2,000,000
- **Evaluation Frequency:** Every 10,000 steps
- **Checkpoint Frequency:** Every 50,000 steps
- **Device:** CUDA (GPU)
- **Frame Stack:** 4 frames
- **Environment:** ALE/Pacman-v5

## 🎮 Usage

### Train Baseline Models

```bash
# Train individual algorithms
cd baseline
python train_baseline_dqn.py
python train_baseline_ppo.py
python train_baseline_a2c.py
python train_baseline_rainbow.py
```

### Train Enhanced Models

```bash
# Train individual algorithms
cd enhanced_algorithm
python train_enhanced_dqn.py
python train_enhanced_ppo.py
python train_enhanced_a2c.py
python train_enhanced_rainbow.py

# Or train all at once
bash train_all_enhanced.sh
```

### Play and Evaluate

```bash
# Play baseline DQN
python play.py -a dqn -m models/baseline_dqn/pacman_dqn_baseline_final.zip --mode baseline

# Play enhanced PPO
python play.py -a ppo -m models/enhanced_ppo/pacman_ppo_enhanced_final.zip --mode enhanced

# Compare baseline vs enhanced
python play.py -a dqn --compare -e 10
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir=./logs/

# View in browser
# Open http://localhost:6006
```

## 📈 Expected Results

### Performance Comparison (2M Timesteps)

| Algorithm | Baseline Score | Enhanced Score | Improvement |
|-----------|----------------|----------------|-------------|
| DQN | 800-1200 | 1200-1800 | +30-50% |
| PPO | 900-1400 | 1400-2200 | +40-60% |
| A2C | 700-1100 | 1100-1600 | +40-50% |
| Rainbow | 1000-1500 | 1500-2500 | +40-60% |

*Results vary based on random seed and hardware*

### Training Time (on RTX 3080)

- **DQN:** ~2-3 hours
- **PPO:** ~3-4 hours
- **A2C:** ~2-3 hours
- **Rainbow:** ~3-4 hours

## 🔬 Experimental Insights

### Reward Shaping Benefits

1. **Faster Learning:** Enhanced models reach higher scores earlier
2. **Better Strategy:** Learn Pacman-specific behaviors (ghost hunting, power pellet seeking)
3. **Higher Peak Performance:** Achieve higher maximum scores
4. **More Stable:** Less variance in performance

### Algorithm Comparison

1. **PPO:** Most stable, best with reward shaping
2. **Rainbow:** Best raw performance, complex
3. **DQN:** Solid baseline, reliable
4. **A2C:** Fastest training, less stable

### Reward Component Impact

Most impactful components:
- **Ghost Hunting** (+50): Dramatically improves scores
- **Death Penalty** (-100): Strong learning signal
- **Level Completion** (+200): Long-term goal

## 🛠️ Dependencies

```
gymnasium>=0.29.0          # RL environment interface
ale-py>=0.8.0             # Atari Learning Environment
stable-baselines3>=2.0.0  # RL algorithms (DQN, PPO, A2C)
sb3-contrib>=2.0.0        # Additional algorithms (Rainbow)
ocatari>=0.1.0            # Object detection for Atari
opencv-python>=4.8.0      # Rendering and visualization
torch>=2.0.0              # Neural network backend
numpy>=1.24.0             # Numerical computing
```

Install all:
```bash
pip install gymnasium ale-py stable-baselines3 sb3-contrib ocatari opencv-python torch numpy
```

## 📝 File Descriptions

### Training Scripts

- `train_baseline_*.py`: Standard training with game rewards only
- `train_enhanced_*.py`: Enhanced training with reward shaping
- `train_all_enhanced.sh`: Batch script to train all enhanced models

### Core Components

- `enhanced_wrapper.py`: Custom reward wrapper with 10 components
- `preprocess.py`: OCAtari preprocessing for enhanced mode
- `play.py`: Unified player and comparison tool

### Documentation

- `README.md`: This file (project overview)
- `QUICKSTART.md`: Quick start guide with examples
- `enhanced_algorithm/README_ENHANCED.md`: Enhanced mode details

## 🐛 Troubleshooting

### Memory Issues

**Problem:** "Not enough memory" or buffer size warnings

**Solution:**
```python
# Edit train script
buffer_size=30000  # Reduce from 50000
```

### CUDA Out of Memory

**Problem:** GPU memory insufficient

**Solution:**
```python
# Use CPU instead (slower)
device="cpu"  # In train script
```

### Import Errors

**Problem:** "Module not found"

**Solution:**
```bash
pip install gymnasium ale-py stable-baselines3 sb3-contrib ocatari opencv-python
```

### SDL Renderer Error

**Problem:** Can't initialize SDL display

**Solution:** Already handled! `play.py` uses OpenCV rendering
```bash
# If issues persist, disable rendering
python play.py -a dqn -m model.zip --no-render
```

### PreprocessFrame Not Found

**Problem:** Enhanced mode can't find preprocessing

**Solution:**
```bash
# Make sure you're in enhanced_algorithm/ folder
cd enhanced_algorithm
python train_enhanced_dqn.py
```

## 📚 References

### Papers

1. **DQN:** Mnih et al. (2013) - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
2. **A2C:** Mnih et al. (2016) - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
3. **PPO:** Schulman et al. (2017) - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
4. **Rainbow:** Hessel et al. (2017) - [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)

### Frameworks

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [OCAtari](https://github.com/k4ntz/OC_Atari)
- [ALE (Arcade Learning Environment)](https://github.com/mgbellemare/Arcade-Learning-Environment)

## 🤝 Contributing

Suggestions for improvement:
1. Add more reward components
2. Tune hyperparameters
3. Implement other algorithms (SAC, TD3)
4. Add curriculum learning
5. Implement multi-agent scenarios

## 📄 License

This project is for educational purposes (IntroAI 2025.1 course).

## 🎓 Course Information

**Course:** Introduction to Artificial Intelligence 2025.1
**Topic:** Deep Reinforcement Learning with Atari Games
**Focus:** Reward Shaping and Algorithm Comparison

## 🙏 Acknowledgments

- Stable Baselines3 team for excellent RL library
- OCAtari team for object detection tools
- OpenAI Gym/Gymnasium for environment interface
- Atari Learning Environment for game emulation

---

**Quick Links:**
- [Quick Start Guide](QUICKSTART.md)
- [Enhanced Mode Details](enhanced_algorithm/README_ENHANCED.md)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

**Status:** ✅ Ready to train and compare!
