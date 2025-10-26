#!/bin/bash
# Quick start script to train all enhanced algorithms
# Usage: bash train_all_enhanced.sh

echo "=========================================="
echo "Enhanced Pacman Training - All Algorithms"
echo "=========================================="
echo ""
echo "Training Configuration:"
echo "  - Game: Pacman (not MsPacman)"
echo "  - Environment: OCAtari with object detection"
echo "  - Reward: Enhanced with 10 reward components"
echo "  - Timesteps: 2,000,000 per algorithm"
echo "  - Algorithms: DQN, PPO, A2C, Rainbow"
echo ""
echo "Expected total time: 18-22 hours (with GPU)"
echo "=========================================="
echo ""
echo "Starting training automatically..."
echo ""

# Create log file
LOG_FILE="training_progress_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Train DQN
echo ""
echo "=========================================="
echo "[1/4] Training DQN with Enhanced Rewards"
echo "=========================================="
python train_enhanced_dqn.py 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: DQN training failed!" | tee -a "$LOG_FILE"
    exit 1
fi

# Train PPO
echo ""
echo "=========================================="
echo "[2/4] Training PPO with Enhanced Rewards"
echo "=========================================="
python train_enhanced_ppo.py 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: PPO training failed!" | tee -a "$LOG_FILE"
    exit 1
fi

# Train A2C
echo ""
echo "=========================================="
echo "[3/4] Training A2C with Enhanced Rewards"
echo "=========================================="
python train_enhanced_a2c.py 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: A2C training failed!" | tee -a "$LOG_FILE"
    exit 1
fi

# Train Rainbow
echo ""
echo "=========================================="
echo "[4/4] Training Rainbow with Enhanced Rewards"
echo "=========================================="
python train_enhanced_rainbow.py 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Rainbow training failed!" | tee -a "$LOG_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "All Training Completed Successfully!"
echo "=========================================="
echo ""
echo "Models saved in:"
echo "  - models/enhanced_dqn/"
echo "  - models/enhanced_ppo/"
echo "  - models/enhanced_a2c/"
echo "  - models/enhanced_rainbow/"
echo ""
echo "View training progress:"
echo "  tensorboard --logdir=./logs/"
echo ""
echo "Compare performance:"
echo "  python play_all.py --compare -e 10"
echo ""
echo "Full log saved to: $LOG_FILE"
echo ""
