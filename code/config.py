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
LEARNING_RATE = 0.001
EPOCHS = 1  # For incremental learning, typically 1 pass through data
VERBOSE = 1  # 0=silent, 1=progress bar, 2=one line per epoch

# Directories
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'trained_model.h5')
METRICS_SAVE_PATH = os.path.join(RESULTS_DIR, 'metrics.csv')