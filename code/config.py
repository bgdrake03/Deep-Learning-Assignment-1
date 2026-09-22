"""Every setting for the project lives here. Change things here, not elsewhere."""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- data
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'pricing.csv')
FEATURES = ['sku', 'price', 'order', 'duration', 'category']
TARGET = 'quantity'

# Rows pulled off disk at a time. This is what keeps the job inside RAM:
# we never hold more than one chunk of the file at once.
CHUNK_SIZE = 50_000

TEST_SPLIT = 0.2
RANDOM_SEED = 42

# pricing.csv is SORTED BY sku. Fed to the model in file order, every
# mini-batch would contain nearly identical rows, which breaks stochastic
# gradient descent - it assumes each batch is a fair sample of the data.
# So we hold this many rows in a buffer, shuffle it, and draw batches from
# there. The buffer is a FIXED number of rows, so it stays the same size no
# matter how large the file gets. At 200,000 rows it costs about 5 MB.
#
# Measured on this data:
#     shuffle buffer   test R2
#   none (file order)   0.4445
#              50,000   0.4478
#             100,000   0.5000
#             200,000   0.5312   <- chosen
#             400,000   0.5406   (a full shuffle of this file, for reference)
SHUFFLE_BUFFER = 200_000

# How many held-out rows we keep in memory for scoring. Scoring, permutation
# importance and partial dependence all need data in RAM; capping the sample
# means that cost is FIXED and does not grow with the size of the file.
SAMPLE_SIZE = 50_000

# ---------------------------------------------------------------- model
HIDDEN_LAYERS = [64, 32, 16]   # three hidden layers, as the assignment requires
LEARNING_RATE = 0.005
BATCH_SIZE = 32

# ---------------------------------------------------------------- evaluation
N_PERMUTATION_REPEATS = 5   # shuffles per feature; averaged, std = error bars
PDP_GRID_POINTS = 40        # points along each partial dependence curve
PDP_SAMPLE_SIZE = 5_000     # rows used per partial dependence curve
MOVING_AVG_WINDOW = 100     # batches, for smoothing the learning curve

# ---------------------------------------------------------------- output
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_PATH         = os.path.join(RESULTS_DIR, 'trained_model.keras')
METRICS_PATH       = os.path.join(RESULTS_DIR, 'metrics.csv')
BATCH_METRICS_PATH = os.path.join(RESULTS_DIR, 'batch_metrics.csv')
IMPORTANCE_PATH    = os.path.join(RESULTS_DIR, 'variable_importance.csv')
PDP_PATH           = os.path.join(RESULTS_DIR, 'partial_dependence.csv')
MEMORY_LOG_PATH    = os.path.join(RESULTS_DIR, 'memory_log.csv')

LEARNING_CURVE_PLOT = os.path.join(PLOT_DIR, 'learning_curve.png')
IMPORTANCE_PLOT     = os.path.join(PLOT_DIR, 'variable_importance.png')
PDP_PLOT            = os.path.join(PLOT_DIR, 'partial_dependence.png')
MEMORY_PLOT         = os.path.join(PLOT_DIR, 'memory_usage.png')
PLOT_DPI = 200
