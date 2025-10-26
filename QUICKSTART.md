# Quick Start Guide - Pacman RL with Reward Shaping

## 🚀 Quick Setup

```bash
# 1. Activate environment
conda activate pacman

# 2. Install packages
pip install gymnasium ale-py stable-baselines3 sb3-contrib ocatari opencv-python torch numpy

# 3. Choose your training mode:
#    - baseline: Standard Atari rewards (fair comparison)
#    - enhanced: Custom reward shaping (better performance)
```

## 🎯 Two Training Modes

### **Baseline Mode** (Standard Training)
- Uses standard Atari preprocessing only
- No custom reward shaping
- Fair comparison baseline
- Location: `baseline/` folder

### **Enhanced Mode** (Reward Shaping)
- 10-component custom reward wrapper
- OCAtari object detection
- Optimized for Pacman strategy
- Location: `enhanced_algorithm/` folder

## 🎮 Quick Start - Train & Play

### Option 1: Train Baseline (Standard Rewards)

```bash
# Train baseline DQN (2-3 hours on GPU)
cd baseline
python train_baseline_dqn.py

# Train baseline PPO (3-4 hours on GPU)
python train_baseline_ppo.py

# Train baseline A2C (2-3 hours on GPU)
python train_baseline_a2c.py

# Train baseline Rainbow (3-4 hours on GPU)
python train_baseline_rainbow.py
```

### Option 2: Train Enhanced (Reward Shaping)

```bash
# Train enhanced DQN
cd enhanced_algorithm
python train_enhanced_dqn.py

# Train enhanced PPO
python train_enhanced_ppo.py

# Train enhanced A2C
python train_enhanced_a2c.py

# Train enhanced Rainbow
python train_enhanced_rainbow.py

# Or train all enhanced algorithms at once
bash train_all_enhanced.sh
```

### Play Trained Models

```bash
# Play baseline DQN
python play.py -a dqn -m models/baseline_dqn/pacman_dqn_baseline_final.zip --mode baseline

# Play enhanced PPO
python play.py -a ppo -m models/enhanced_ppo/pacman_ppo_enhanced_final.zip --mode enhanced

# Play any checkpoint
python play.py -a dqn -m models/baseline_dqn/pacman_dqn_baseline_1000000_steps.zip --mode baseline
```

### Compare Baseline vs Enhanced

```bash
# Compare DQN: baseline vs enhanced (10 episodes each)
python play.py -a dqn --compare -e 10

# Compare PPO: baseline vs enhanced
python play.py -a ppo --compare -e 10

# Compare all 4 algorithms
python play.py -a dqn --compare -e 20
python play.py -a ppo --compare -e 20
python play.py -a a2c --compare -e 20
python play.py -a rainbow --compare -e 20
```

## 📊 Command Reference

### play.py Options

```bash
-a, --algorithm {dqn,ppo,a2c,rainbow}  Choose algorithm
-m, --model PATH                       Path to model file
--mode {baseline,enhanced}             Model mode
-e, --episodes NUM                     Number of episodes (default: 5)
--no-render                            Disable rendering (faster)
-c, --compare                          Compare baseline vs enhanced

# Examples:
python play.py -a dqn -m models/baseline_dqn/pacman_dqn_baseline_final.zip --mode baseline
python play.py -a ppo --compare -e 10
python play.py -a rainbow -m models/enhanced_rainbow/pacman_rainbow_enhanced_final.zip --mode enhanced --no-render
```

## 📁 Project Structure

```
baseline/                          # Standard training (no reward shaping)
├── train_baseline_dqn.py
├── train_baseline_ppo.py
├── train_baseline_a2c.py
└── train_baseline_rainbow.py

enhanced_algorithm/                # Enhanced training (reward shaping)
├── enhanced_wrapper.py            # 10-component reward wrapper
├── preprocess.py                  # OCAtari preprocessing
├── train_enhanced_dqn.py
├── train_enhanced_ppo.py
├── train_enhanced_a2c.py
├── train_enhanced_rainbow.py
├── train_all_enhanced.sh          # Batch training script
└── README_ENHANCED.md             # Enhanced mode documentation

play.py                            # Unified player (baseline & enhanced)
models/                            # Saved models
├── baseline_dqn/
├── baseline_ppo/
├── enhanced_dqn/
└── enhanced_ppo/
```

## 🎯 Algorithm Selection Guide

| Algorithm | Type | Pros | Cons | Best For |
|-----------|------|------|------|----------|
| **DQN** | Value-based | Stable, proven | Needs replay buffer | Baseline experiments |
| **PPO** | Policy-based | Very stable, parallelizable | Slower than A2C | Production use |
| **A2C** | Policy-based | Fast, efficient | Less stable | Quick experiments |
| **Rainbow** | Value-based | Best performance | Complex, memory-intensive | Maximum performance |

## 🧪 Enhanced Reward Components

The enhanced mode uses 10 reward components:

1. **Score Bonus** (0.1x): Raw game score
2. **Power Pellet Seeking** (+5): Move toward power pellets
3. **Ghost Hunting** (+50): Eat vulnerable ghosts
4. **Ghost Avoidance** (+2): Stay away from dangerous ghosts
5. **Smart Dot Collection** (+1): Efficient dot eating
6. **Movement Bonus** (+0.5): Exploration reward
7. **Corner Penalty** (-1): Avoid getting trapped
8. **Survival Bonus** (+0.1): Stay alive longer
9. **Death Penalty** (-100): Strong death discouragement
10. **Level Completion** (+200): Beat the level

See `enhanced_algorithm/README_ENHANCED.md` for details.

## 💡 Tips & Best Practices

1. **Start with Baseline**: Train baseline models first for comparison
2. **Memory Issues?** Reduce `buffer_size` in DQN/Rainbow (50000 → 30000)
3. **Quick Test?** Reduce `total_timesteps` (2000000 → 100000)
4. **Monitor Training**: Use TensorBoard: `tensorboard --logdir=./logs/`
5. **Compare Results**: Always use `--compare` flag to evaluate improvement
6. **GPU Required**: All scripts default to CUDA (modify if CPU-only)

## 🐛 Common Issues

**"Module not found" errors:**
```bash
pip install gymnasium ale-py stable-baselines3 sb3-contrib ocatari opencv-python
```

**"Not enough memory":**
```python
# Edit train script, reduce buffer_size
buffer_size=30000  # Instead of 50000
```

**"CUDA out of memory":**
```python
# Use CPU instead (slower)
device="cpu"  # In train script
```

**"PreprocessFrame not found" (enhanced mode only):**
```bash
# Make sure you're in enhanced_algorithm/ folder
cd enhanced_algorithm
```

**SDL Renderer error:**
```bash
# play.py uses OpenCV rendering (cv2), should work fine
# If issues persist, use --no-render flag
```

## 📈 Expected Results

### After 2M Timesteps

| Algorithm | Baseline (Standard) | Enhanced (Reward Shaping) | Improvement |
|-----------|---------------------|---------------------------|-------------|
| **DQN** | 800-1200 | 1200-1800 | +30-50% |
| **PPO** | 900-1400 | 1400-2200 | +40-60% |
| **A2C** | 700-1100 | 1100-1600 | +40-50% |
| **Rainbow** | 1000-1500 | 1500-2500 | +40-60% |

*Results vary based on random seed and hardware*

## 📚 Next Steps

1. **Train Baseline Models**: Start with `baseline/` to establish baseline
2. **Train Enhanced Models**: Move to `enhanced_algorithm/` for reward shaping
3. **Compare Performance**: Use `play.py --compare` to evaluate
4. **Analyze Results**: Review logs and TensorBoard data
5. **Fine-tune**: Adjust hyperparameters or reward weights

## 📖 More Information

- `README.md` - Full project documentation
- `enhanced_algorithm/README_ENHANCED.md` - Enhanced mode details
- `baseline/` - Standard training scripts
- `enhanced_algorithm/` - Reward shaping scripts
