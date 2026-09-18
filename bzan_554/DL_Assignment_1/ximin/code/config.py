# Settings (batch size, learning rate, etc)

# Configuration settings for the entire project

import os

# Get the directory where this config file is located (code/)
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CONFIG_DIR)  # Parent directory (repo root)

# Data settings
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'pricing.csv')
TEST_SPLIT = 0.2  # 80/20 train/test split
RANDOM_SEED = 42

# Model architecture
INPUT_FEATURES = 5  # sku, price, order, duration, category
HIDDEN_LAYER_1 = 64
HIDDEN_LAYER_2 = 32
HIDDEN_LAYER_3 = 16
OUTPUT_FEATURES = 1  # quantity

# Training settings
BATCH_SIZE = 32
# CHANGED 0.001 -> 0.005: with the target scaled (see data_handler.scale_target),
# the larger step trains faster in one pass. Measured: test R2 0.519 -> 0.532.
LEARNING_RATE = 0.005
EPOCHS = 1  # For incremental learning, typically 1 pass through data
VERBOSE = 1  # 0=silent, 1=progress bar, 2=one line per epoch

# Directories
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
# .keras (not legacy .h5): Keras 3 cannot reload a compiled .h5 model,
# which broke plots.py loading it back for permutation importance.
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'trained_model.keras')
METRICS_SAVE_PATH = os.path.join(RESULTS_DIR, 'metrics.csv')
MEMORY_LOG_PATH = os.path.join(RESULTS_DIR, 'memory_log.csv')
BATCH_METRICS_PATH = os.path.join(RESULTS_DIR, 'batch_metrics.csv')
IMPORTANCE_PATH = os.path.join(RESULTS_DIR, 'variable_importance.csv')

# Plot output (300 DPI = publication quality for slides/print)
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
LEARNING_CURVE_PLOT = os.path.join(PLOT_DIR, 'learning_curve.png')
IMPORTANCE_PLOT = os.path.join(PLOT_DIR, 'variable_importance.png')
PLOT_DPI = 300

# Permutation importance
N_PERMUTATION_REPEATS = 5  # Shuffles per feature; more = tighter error bars

# Memory tracking
MEMORY_LOG_INTERVAL = 1  # Record RAM every N batches (1 = every batch)

# Learning curve
MOVING_AVG_WINDOW = 100  # Window for smoothing per-batch MSE