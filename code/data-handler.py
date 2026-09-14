# Load and prepare data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *

def load_and_split_data():
    """
    Load CSV file and split into train/test sets.
    
    Returns:
        X_train, X_test, y_train, y_test: Split data
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
    
    return X_train, X_test, y_train, y_test

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