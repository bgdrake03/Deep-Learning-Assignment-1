# Load and prepare data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *

def load_and_split_data():
    """
    Load CSV file, split into train/test sets, and scale features.

    Returns:
        X_train, X_test, y_train, y_test: Split and scaled data
    """
    # Load data
    df = pd.read_csv(DATA_FILE)

    # Separate features and target
    X = df.drop('quantity', axis=1)  # All columns except quantity
    y = df['quantity']  # Target column

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED
    )

    # Scale features (fit on training data only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert back to DataFrames to maintain consistency
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_test, y_train, y_test

# ============================================================================
# TARGET SCALING  (added to fix low R2)
#
# WHY: load_and_split_data() scales the FEATURES to mean 0 / std 1, but the
# target 'quantity' stays raw (mean 23, std 51, max 4165). Asking the network
# to output numbers that large from inputs near zero produces very large
# gradients, which saturate the sigmoid hidden layers and stall learning.
#
# Measured on this dataset: test R2 0.35 -> 0.52 from this change alone.
#
# NOTE: after scaling, model.predict() returns STANDARDIZED units, not
# quantities. Always call unscale_predictions() before computing R2/MSE or
# reporting results, or the metrics are meaningless.
# ============================================================================

def scale_target(y_train):
    """
    Standardize the training target to mean 0 / std 1.

    Mean and std come from TRAINING data only -- same no-leakage rule as the
    feature scaler's fit_transform/transform split.

    Args:
        y_train: Training target values

    Returns:
        y_scaled, mean, std: Scaled target plus the constants needed to undo it
    """
    mean = y_train.mean()
    std = y_train.std()
    return (y_train - mean) / std, mean, std


def unscale_predictions(y_pred, mean, std):
    """
    Convert standardized predictions back to real quantity units.

    Args:
        y_pred: Model output (standardized)
        mean, std: Values returned by scale_target()

    Returns:
        Predictions in original quantity units
    """
    return y_pred * std + mean


def create_mini_batches(X, y, batch_size=BATCH_SIZE):
    """
    Generator that yields mini-batches for incremental training.
    
    Args:
        X: Feature data
        y: Target data
        batch_size: Number of records per batch
        
    Yields:
        X_batch, y_batch: Mini-batches of specified size
    """
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        X_batch = X.iloc[i:i+batch_size].values
        y_batch = y.iloc[i:i+batch_size].values
        yield X_batch, y_batch

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Features shape: {X_train.shape}")