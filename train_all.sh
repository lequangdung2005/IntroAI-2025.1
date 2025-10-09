#!/bin/bash
# Train all 4 algorithms sequentially
# Usage: bash train_all.sh

echo "=========================================="
echo "Training All Algorithms for Pacman"
echo "=========================================="
echo ""
echo "This will train 4 algorithms:"
echo "  1. DQN"
echo "  2. PPO"
echo "  3. A2C"
echo "  4. Rainbow"
echo ""
echo "Total estimated time: 10-15 hours (with GPU)"
echo "=========================================="
echo ""



# Train DQN
echo ""
echo "=========================================="
echo "Training DQN (1/4)"
echo "=========================================="
python train_dqn.py
if [ $? -ne 0 ]; then
    echo "DQN training failed!"
    exit 1
fi

# Train PPO
echo ""
echo "=========================================="
echo "Training PPO (2/4)"
echo "=========================================="
python train_ppo.py
if [ $? -ne 0 ]; then
    echo "PPO training failed!"
    exit 1
fi

# Train A2C
echo ""
echo "=========================================="
echo "Training A2C (3/4)"
echo "=========================================="
python train_a2c.py
if [ $? -ne 0 ]; then
    echo "A2C training failed!"
    exit 1
fi

# Train Rainbow
echo ""
echo "=========================================="
echo "Training Rainbow (4/4)"
echo "=========================================="
python train_rainbow.py
if [ $? -ne 0 ]; then
    echo "Rainbow training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo ""
echo "To compare results:"
echo "  python play_all.py --compare -e 10"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir=./logs/"
echo ""
