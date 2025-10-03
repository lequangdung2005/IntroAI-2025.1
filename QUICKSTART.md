# Quick Start Guide - Pacman RL Algorithms

## 🚀 Quick Setup

```bash
# 1. Activate environment
conda activate pacman

# 2. Install packages
pip install -r requirements.txt

# 3. Train an algorithm (choose one)
python train_dqn.py      # DQN - Value-based, stable
python train_ppo.py      # PPO - Policy-based, very stable
python train_a2c.py      # A2C - Fast, efficient
python train_rainbow.py  # Rainbow - Best performance
```

## 🎮 Usage Examples

### Train Individual Algorithms

```bash
# Train DQN (2-3 hours on GPU)
python train_dqn.py

# Train PPO (3-4 hours on GPU)
python train_ppo.py
```

### Play and Watch

```bash
# Play DQN agent
python play_all.py -a dqn -e 5

# Play PPO agent
python play_all.py -a ppo -e 5

# Play specific checkpoint
python play_all.py -a dqn --model models/dqn/pacman_dqn_100000_steps.zip
```

### Compare All Algorithms

```bash
# Evaluate and compare all 4 algorithms
python play_all.py --compare -e 10
```

### Monitor Training

```bash
# View training progress in TensorBoard
tensorboard --logdir=./logs/
```

## 📊 Command Reference

### play_all.py Options

```bash
# Play specific algorithm
-a, --algorithm {dqn,ppo,a2c,rainbow}  Choose algorithm
-m, --model PATH                       Custom model path
-e, --episodes NUM                     Number of episodes (default: 5)
--no-render                            Disable rendering (faster evaluation)
-c, --compare                          Compare all algorithms

# Examples:
python play_all.py -a dqn -e 10
python play_all.py -a ppo --model models/ppo/best/best_model.zip
python play_all.py --compare -e 20 --no-render
```

## 📁 File Structure

```
train_dqn.py          # Train DQN
train_ppo.py          # Train PPO
train_a2c.py          # Train A2C
train_rainbow.py      # Train Rainbow
play_all.py           # Play/evaluate any algorithm
train_all.sh          # Train all algorithms sequentially (optional)
```

## 🎯 Algorithm Selection Guide

**Choose DQN if:**
- You want proven, stable performance
- You have enough RAM for replay buffer
- You prefer value-based methods

**Choose PPO if:**
- You want maximum stability
- You have multiple CPU cores
- You prefer policy-based methods

**Choose A2C if:**
- You want fast training
- You have limited resources
- You want to experiment quickly

**Choose Rainbow if:**
- You want best performance
- You have enough RAM and GPU
- You want state-of-the-art results

## 💡 Tips

1. **Memory Issues?** Reduce `buffer_size` in DQN/Rainbow scripts
2. **Training Too Slow?** Reduce `total_timesteps` for quick tests
3. **Compare Models?** Use `python play_all.py --compare`
4. **Monitor Progress?** Use TensorBoard: `tensorboard --logdir=./logs/`

## 🐛 Common Issues

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**"Not enough memory":**
Edit train script, reduce `buffer_size` from 50000 to 30000

**SDL Renderer error:**
play_all.py uses OpenCV rendering, should work fine

## 📈 Expected Results

After 1M timesteps:
- **DQN:** 800-1200 score
- **PPO:** 900-1400 score  
- **A2C:** 700-1100 score
- **Rainbow:** 1000-1500 score

*Results vary based on random seed*

## 📚 More Information

See `README_COMPARISON.md` for detailed documentation.
