# Main training script

import numpy as np
from model import build_model
from data_handler import load_and_split_data, create_mini_batches
from evaluate import calculate_r2, calculate_mse, calculate_moving_average
from config import *
import pandas as pd
import time

def train_incremental():
    """
    Main training loop for incremental learning.
    Reads data in mini-batches and updates model on each batch.
    """
    print("="*60)
    print("STARTING INCREMENTAL LEARNING TRAINING")
    print("="*60)
    
    # Load data
    print("\n1. Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Build model
    print("\n2. Building neural network model...")
    model = build_model()
    model.summary()
    
    # Incremental training loop
    print("\n3. Starting incremental training...")
    start_time = time.time()
    
    mse_history = []
    batch_count = 0
    
    for X_batch, y_batch in create_mini_batches(X_train, y_train):
        # Train on this batch
        loss = model.train_on_batch(X_batch, y_batch)
        mse_history.append(loss)
        
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"   Processed {batch_count} batches...")
    
    training_time = time.time() - start_time
    
    # Evaluation
    print("\n4. Evaluating model...")
    y_train_pred = model.predict(X_train.values, verbose=0)
    y_test_pred = model.predict(X_test.values, verbose=0)
    
    r2_train = calculate_r2(y_train, y_train_pred)
    r2_test = calculate_r2(y_test, y_test_pred)
    mse_train = calculate_mse(y_train, y_train_pred)
    mse_test = calculate_mse(y_test, y_test_pred)
    
    print(f"\n   Training R²: {r2_train:.4f}")
    print(f"   Test R²: {r2_test:.4f}")
    print(f"   Training MSE: {mse_train:.4f}")
    print(f"   Test MSE: {mse_test:.4f}")
    print(f"   Training time: {training_time:.2f} seconds")
    
    # Save results
    print("\n5. Saving results...")
    model.save(MODEL_SAVE_PATH)
    
    results = {
        'metric': ['R2_train', 'R2_test', 'MSE_train', 'MSE_test', 'training_time'],
        'value': [r2_train, r2_test, mse_train, mse_test, training_time]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_SAVE_PATH, index=False)
    
    print(f"\n   Model saved to: {MODEL_SAVE_PATH}")
    print(f"   Metrics saved to: {METRICS_SAVE_PATH}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    train_incremental()