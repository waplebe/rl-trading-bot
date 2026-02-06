"""
Configuration for RL Trading Bot
"""
import os

# ==========================================
# PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # CSV files are in the root folder
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# DATA
# ==========================================
# CSV files pattern (all NQ minute data)
CSV_FILES = [
    "USATECH.IDXUSD_Candlestick_1_M_BID_31.12.2012-20.07.2013.csv",
    "USATECH.IDXUSD_Candlestick_1_M_BID_20.07.2013-19.07.2016.csv",
    "USATECH.IDXUSD_Candlestick_1_M_BID_19.07.2016-18.07.2019.csv",
    "USATECH.IDXUSD_Candlestick_1_M_BID_19.07.2019-19.07.2020.csv",
    "USATECH.IDXUSD_Candlestick_1_M_BID_19.07.2020-19.07.2021.csv",
    "USATECH.IDXUSD_Candlestick_1_M_BID_19.07.2021-01.01.2022.csv",
    "USATECH.IDXUSD_Candlestick_1_M_BID_01.01.2022-19.01.2022.csv",
    "USATECH.IDXUSD_Candlestick_1_M_BID_19.07.2022-18.07.2025.csv",
]

# Train/Test split date
TRAIN_END_DATE = "2022-01-01"  # Train: 2012-2021, Test: 2022-2025

# ==========================================
# TRADING ENVIRONMENT
# ==========================================
# Window size - how many candles the agent "sees"
WINDOW_SIZE = 50

# Initial balance in USD
INITIAL_BALANCE = 100_000.0

# NQ point value (1 point = $20 for NQ futures)
POINT_VALUE = 20.0

# Commission per trade (round trip)
COMMISSION = 4.50

# Max position size (number of contracts)
MAX_POSITION = 1

# ==========================================
# REWARD SHAPING
# ==========================================
# Penalty for each trade (to avoid overtrading)
TRADE_PENALTY = -0.001

# Bonus for profitable trade close
PROFIT_BONUS = 2.0

# Penalty multiplier for losing trade close
LOSS_PENALTY = 1.0

# Max drawdown before episode ends (0.5 = 50% loss allowed during training)
MAX_DRAWDOWN_KILL = 0.50

# Episode max length (in steps) — shorter = faster learning cycles
EPISODE_MAX_STEPS = 5000

# ==========================================
# TRAINING
# ==========================================
# RL Algorithm: "PPO", "A2C", "DQN"
RL_ALGORITHM = "PPO"

# Total training timesteps
TOTAL_TIMESTEPS = 2_000_000  # Serious training run

# Learning rate
LEARNING_RATE = 3e-4

# Neural network architecture
NET_ARCH = [256, 256]  # Two hidden layers, 256 neurons each

# Batch size
BATCH_SIZE = 256

# Number of steps per update
N_STEPS = 4096

# Discount factor (how much future rewards matter)
GAMMA = 0.995

# GAE lambda (for PPO)
GAE_LAMBDA = 0.95

# Clip range (for PPO)
CLIP_RANGE = 0.2

# Entropy coefficient (encourages exploration — higher = more exploration)
ENT_COEF = 0.02

# Number of epochs per update
N_EPOCHS = 10

# ==========================================
# EVALUATION
# ==========================================
# Number of episodes to evaluate
EVAL_EPISODES = 5

# Resample to N-minute bars for faster training (None = keep 1min)
# Set to 5, 15, or 60 for faster experiments
RESAMPLE_MINUTES = 5
