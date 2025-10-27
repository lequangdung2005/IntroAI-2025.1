# Shaping Reward Analysis

Analysis of shaping reward logs to find optimal parameters for Ms. Pac-Man RL training.

## 📁 Files

1. **analyze_shaping_rewards.py** - Main analysis script
2. **final_shaping_reward_params.json** - Recommended parameters (generated)
3. **ANALYSIS_RESULTS.txt** - Detailed results summary

## 🚀 Usage

```bash
# Run analysis
python3 analyze_shaping_rewards.py
```

This will:
- Load all `shaping_reward_*_log.txt` files
- Analyze correlations and distributions
- Generate `final_shaping_reward_params.json`

## 📊 Results

**Recommended Parameters:**
```
bonus_powerpill_reward        =  0.200
bonus_eating_ghost_reward     =  0.500  ⭐ MOST IMPORTANT
penalty_nearing_ghost_reward  =  0.250  (POSITIVE - log values already negative)
```

**Key Finding:** Eating ghost has the highest correlation (+0.141) with performance!

## 💻 How to Apply

```python
import json

# Load parameters
with open('final_shaping_reward_params.json', 'r') as f:
    params = json.load(f)

# Use in your environment
bonus_powerpill = params['bonus_powerpill_reward']        # 0.200
bonus_eating_ghost = params['bonus_eating_ghost_reward']  # 0.500
penalty_nearing = params['penalty_nearing_ghost_reward']  # 0.250 (positive, log values already negative)
```

## 📖 Details

See **ANALYSIS_RESULTS.txt** for full analysis including:
- Correlation table across all algorithms
- Distribution statistics
- Detailed rationale for each parameter
