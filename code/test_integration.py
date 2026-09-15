# Integration test - run on small data sample to verify pipeline

import numpy as np
import pandas as pd
from model import build_model
from data_handler import load_and_split_data, create_mini_batches
from evaluate import calculate_r2, calculate_mse, calculate_moving_average
from config import *
import time

def test_incremental_learning_small():
    """
    Test incremental learning on small data sample.
    Verifies all pieces work together before running on full dataset.
    """
    print("="*60)
    print("INTEGRATION TEST: Small Data Sample")
    print("="*60)

    # Load and split full data
    print("\n1. Loading data...")
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Use only first 1000 samples for testing
    sample_size = 1000
    X_train_small = X_train.iloc[:sample_size]
    y_train_small = y_train.iloc[:sample_size]
    X_test_small = X_test.iloc[:min(sample_size // 4, len(X_test))]
    y_test_small = y_test.iloc[:min(sample_size // 4, len(y_test))]

    print(f"   Training samples (small): {len(X_train_small)}")
    print(f"   Test samples (small): {len(X_test_small)}")

    # Build model
    print("\n2. Building model...")
    model = build_model()

    # Train on small sample
    print("\n3. Training on small sample (incremental batches)...")
    start_time = time.time()

    mse_history = []
    batch_count = 0

    for X_batch, y_batch in create_mini_batches(X_train_small, y_train_small, batch_size=32):
        metrics = model.train_on_batch(X_batch, y_batch)
        loss = metrics[0] if isinstance(metrics, list) else metrics
        mse_history.append(loss)
        batch_count += 1

        if batch_count % 5 == 0:
            print(f"   Batch {batch_count}: MSE = {loss:.4f}")

    training_time = time.time() - start_time
    print(f"   Total batches: {batch_count}")
    print(f"   Training time: {training_time:.2f}s")

    # Evaluate
    print("\n4. Evaluating on small sample...")
    y_train_pred = model.predict(X_train_small.values, verbose=0)
    y_test_pred = model.predict(X_test_small.values, verbose=0)

    # Flatten predictions for metric calculation
    y_train_pred = y_train_pred.flatten()
    y_test_pred = y_test_pred.flatten()

    r2_train = calculate_r2(y_train_small.values, y_train_pred)
    r2_test = calculate_r2(y_test_small.values, y_test_pred)
    mse_train = calculate_mse(y_train_small.values, y_train_pred)
    mse_test = calculate_mse(y_test_small.values, y_test_pred)

    print(f"\n   Training R²: {r2_train:.4f}")
    print(f"   Test R²: {r2_test:.4f}")
    print(f"   Training MSE: {mse_train:.4f}")
    print(f"   Test MSE: {mse_test:.4f}")

    # Test moving average
    print("\n5. Testing moving average (learning curve)...")
    if len(mse_history) > 10:
        moving_avg = calculate_moving_average(mse_history, window=10)
        print(f"   Original MSE history length: {len(mse_history)}")
        print(f"   Moving average (window=10) length: {len(moving_avg)}")
        print(f"   First 5 moving avg values: {moving_avg[:5]}")
        print(f"   Last 5 moving avg values: {moving_avg[-5:]}")

    print("\n" + "="*60)
    print("[SUCCESS] INTEGRATION TEST PASSED")
    print("="*60)
    print("\nNext steps:")
    print("  - Run main.py on full dataset")
    print("  - Monitor R² and MSE metrics")
    print("  - Check results in ../results/")

if __name__ == "__main__":
    test_incremental_learning_small()
