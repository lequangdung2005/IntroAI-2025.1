# Enhanced Pacman RL Training with Reward Shaping

Train 4 RL algorithms (DQN, PPO, A2C, Rainbow) on **regular Pacman** with an **enhanced reward shaping wrapper** for improved performance.

## 🎯 What's New

### Enhanced Reward Wrapper
The `EnhancedPacmanRewardWrapper` provides **10 reward components** to boost learning:

1. **Score Increase Bonus** - Reward for collecting dots/pellets
2. **Power Pellet Seeking** - Encourage moving toward power pellets when vulnerable
3. **Ghost Hunting** - Big bonus for chasing ghosts when powered up
4. **Ghost Avoidance** - Penalty for being near ghosts when vulnerable
5. **Dot Collection** - Reward for moving toward nearest dots
6. **Movement Bonus** - Small bonus for moving (avoid standing still)
7. **Corner Penalty** - Penalize getting trapped in corners near ghosts
8. **Survival Bonus** - Small reward for staying alive each step
9. **Death Penalty** - Large penalty for losing a life
10. **Level Completion** - Huge bonus for completing a level

### Key Features
- ✅ Uses **OCAtari** for object detection (see ghosts, dots, power pellets)
- ✅ Changed from **MsPacman to regular Pacman**
- ✅ Sophisticated reward shaping for faster learning
- ✅ Logs reward components for analysis
- ✅ 2M timesteps for better convergence

## 📁 Files Created

```
enhanced_wrapper.py          # Enhanced reward shaping wrapper
train_enhanced_dqn.py        # Train DQN with enhanced rewards
train_enhanced_ppo.py        # Train PPO with enhanced rewards
train_enhanced_a2c.py        # Train A2C with enhanced rewards
train_enhanced_rainbow.py    # Train Rainbow with enhanced rewards
```

## 🚀 Training

### Train Individual Algorithms

```bash
# Train DQN
python train_enhanced_dqn.py

# Train PPO
python train_enhanced_ppo.py

# Train A2C
python train_enhanced_a2c.py

# Train Rainbow
python train_enhanced_rainbow.py
```

### Training Configuration

All algorithms are configured for **2M timesteps** (double the original):
- **DQN/Rainbow**: Buffer size 100K, learning starts at 50K steps
- **PPO**: 4 parallel environments, batch size 256
- **A2C**: 4 parallel environments, learning rate 7e-4
- **Device**: CUDA (GPU) for all algorithms

### Model Save Locations

```
models/enhanced_dqn/        # DQN checkpoints and final model
models/enhanced_ppo/        # PPO checkpoints and final model
models/enhanced_a2c/        # A2C checkpoints and final model
models/enhanced_rainbow/    # Rainbow checkpoints and final model

# Best models saved in:
models/enhanced_*/best/     # Best performing model based on evaluation
```

## 📊 Monitor Training

```bash
# View all enhanced training runs
tensorboard --logdir=./logs/

# View specific algorithm
tensorboard --logdir=./logs/enhanced_dqn/
tensorboard --logdir=./logs/enhanced_ppo/
```

## 🎮 Playing Trained Models

Update `play_all.py` to use enhanced models:

```python
ALGORITHMS = {
    'dqn': {
        'class': DQN,
        'default_path': 'models/enhanced_dqn/pacman_dqn_enhanced_final.zip',
        'name': 'DQN (Enhanced)'
    },
    # ... similar for other algorithms
}
```

Then play:
```bash
python play_all.py -a dqn -e 5
python play_all.py --compare -e 10
```

## 🔍 Reward Wrapper Details

### How It Works

The wrapper analyzes the game state using OCAtari's object detection:
- **Detects**: Player position, ghosts, dots, power pellets
- **Calculates**: Distances to objects
- **Checks**: Power-up state (are ghosts blue?)
- **Applies**: Multiple reward bonuses/penalties

### Example Rewards

```python
# When powered up and near ghost
if is_powered and ghost_distance < 20:
    reward += 15.0 / (distance + 1)  # Big bonus!

# When vulnerable and near ghost
if not is_powered and ghost_distance < 15:
    reward -= 20.0 / (distance + 1)  # Big penalty!

# Near power pellet (when vulnerable)
if near_powerpill and not is_powered:
    reward += 5.0 / (distance + 1)  # Seek power!
```

### Logging

Rewards are logged every 100 steps to `enhanced_reward_log.txt`:
```
Step 100: powered=False, base_reward=10.00, shaped_reward=15.23, score=120, lives=3
Step 200: powered=True, base_reward=200.00, shaped_reward=250.45, score=520, lives=3
```

## 📈 Expected Performance

With enhanced rewards, expect **better scores** and **faster learning**:

| Algorithm | Expected Avg Score | Training Time (GPU) |
|-----------|-------------------|---------------------|
| DQN       | 1200-1800        | 4-5 hours          |
| PPO       | 1400-2000        | 5-6 hours          |
| A2C       | 1100-1600        | 4-5 hours          |
| Rainbow   | 1500-2200        | 5-6 hours          |

*Performance should be significantly better than vanilla training!*

## 🔧 Customizing Rewards

Edit `enhanced_wrapper.py` to adjust reward weights:

```python
# Increase ghost avoidance penalty
if dist < 15 and not is_powered:
    shaped_reward -= 30.0 / (dist + 1)  # Changed from 20.0

# Increase level completion bonus
if terminated and current_lives >= self.prev_lives:
    shaped_reward += 500  # Changed from 200
```

## 🆚 Comparison with Original

| Feature | Original Training | Enhanced Training |
|---------|------------------|------------------|
| Game | MsPacman | **Pacman** |
| Reward | Sparse (game score only) | **Dense (10 components)** |
| Object Info | No | **Yes (OCAtari)** |
| Timesteps | 1M | **2M** |
| Strategy | Random exploration | **Guided by shaped rewards** |

## 🐛 Troubleshooting

**Import errors:**
```bash
pip install ocatari gymnasium ale-py stable-baselines3 sb3-contrib
```

**Memory issues:**
Reduce `buffer_size` in training scripts:
```python
buffer_size=50000,  # Instead of 100000
```

**CUDA not available:**
Change `device="cuda"` to `device="cpu"` in training scripts

## 💡 Tips for Best Results

1. **Monitor early performance** - Check TensorBoard after 100K steps
2. **Compare algorithms** - Try all 4 to see which works best
3. **Adjust rewards** - Tune the wrapper weights for your needs
4. **Use checkpoints** - Test intermediate models to track progress
5. **Train longer** - Consider 3M-5M steps for even better results

## 📚 Next Steps

1. Train all 4 algorithms
2. Compare performance using `play_all.py --compare`
3. Analyze reward logs in `enhanced_reward_log.txt`
4. Fine-tune reward weights based on results
5. Experiment with different hyperparameters

## 🤝 Key Advantages

- **Faster Learning**: Dense rewards guide the agent
- **Better Strategy**: Learns to chase ghosts when powered up
- **Safer Play**: Learns to avoid ghosts when vulnerable
- **Score Maximization**: Encouraged to collect dots efficiently
- **Survival Focus**: Balances aggression with safety

Good luck training! 🎮🚀
