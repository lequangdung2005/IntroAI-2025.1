# Pacman DQN with Stable Baselines3

This project trains a Deep Q-Network (DQN) agent to play Pacman using Stable Baselines3 and Gymnasium.

## Installation

1. **Activate your conda environment:**
   ```bash
   conda activate pacman
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Atari ROMs (required for Pacman):**
   ```bash
   pip install "gymnasium[atari,accept-rom-license]"
   ```

## Usage

### Training the Agent

Train the DQN agent to play Pacman:

```bash
python train_pacman.py
```

The training will:
- Train for 1,000,000 steps (adjust in the code if needed)
- Save checkpoints every 50,000 steps in the `models/` directory
- Save the final model as `models/pacman_dqn_final.zip`
- Log training progress to `logs/` for TensorBoard

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir=./logs/
```

### Watching the Trained Agent

Watch your trained agent play Pacman:

```bash
python play_pacman.py
```

**Options:**
- Specify a different model:
  ```bash
  python play_pacman.py models/pacman_dqn_50000_steps.zip
  ```
- Watch more episodes:
  ```bash
  python play_pacman.py models/pacman_dqn_final.zip 10
  ```

## Project Structure

```
.
├── train_pacman.py       # Training script using Stable Baselines3 DQN
├── play_pacman.py        # Script to watch trained agent play
├── requirements.txt      # Python dependencies
├── models/              # Saved model checkpoints (created during training)
└── logs/                # TensorBoard logs (created during training)
```

## How It Works

### Training (`train_pacman.py`)
- Uses Stable Baselines3's DQN implementation
- Applies Atari preprocessing (grayscale, resize, frame stacking)
- Trains with optimized hyperparameters for Atari games
- Automatically uses GPU if available

### Playing (`play_pacman.py`)
- Loads a trained model from the `models/` directory
- Renders the game in a window so you can watch
- Displays scores and statistics for each episode

## Troubleshooting

**"Namespace ALE not found" error:**
Make sure you've installed the ALE package:
```bash
pip install ale-py "gymnasium[atari,accept-rom-license]"
```

**No trained models found:**
Train a model first with `python train_pacman.py`

**CUDA/GPU issues:**
The code automatically detects and uses GPU if available. If you have issues, you can force CPU by modifying the `device="auto"` parameter in `train_pacman.py` to `device="cpu"`.

## Training Tips

- **Initial training:** The agent will perform poorly at first. This is normal!
- **Training time:** Training takes several hours on a GPU, longer on CPU
- **Checkpoints:** Models are saved every 50,000 steps, so you can test early versions
- **Hyperparameters:** Adjust in `train_pacman.py` if needed

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster training)
