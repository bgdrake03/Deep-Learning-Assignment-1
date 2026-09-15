# Main training script

import numpy as np
from model import build_model
from data_handler import load_and_split_data, create_mini_batches
from evaluate import calculate_r2, calculate_mse, calculate_moving_average
from utils import MemoryTracker, get_memory_mb, get_system_memory_mb
from config import *
import pandas as pd
import time
from datetime import datetime

def train_incremental():
    """
    Main training loop for incremental learning.
    Reads data in mini-batches and updates model on each batch.
    """
    run_timestamp = datetime.now()
    tracker = MemoryTracker()
    tracker.log('start')

    sys_mem = get_system_memory_mb()

    print("="*60)
    print("STARTING INCREMENTAL LEARNING TRAINING")
    print(f"Run Date/Time: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System RAM: {sys_mem['total_mb']:.0f} MB total, "
          f"{sys_mem['available_mb']:.0f} MB available")
    print(f"Starting memory: {tracker.baseline_mb:.1f} MB")
    print("="*60)
    
    # Load data
    print("\n1. Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    tracker.log('after_data_load')
    print(f"   Memory: {get_memory_mb():.1f} MB")
    
    # Build model
    print("\n2. Building neural network model...")
    model = build_model()
    model.summary()
    tracker.log('after_model_build')
    print(f"   Memory: {get_memory_mb():.1f} MB")
    
    # Incremental training loop
    print("\n3. Starting incremental training...")
    start_time = time.time()
    
    mse_history = []
    batch_count = 0
    
    for X_batch, y_batch in create_mini_batches(X_train, y_train):
        metrics = model.train_on_batch(X_batch, y_batch)
        loss = metrics[0] if isinstance(metrics, list) else metrics
        mse_history.append(loss)

        batch_count += 1

        # Log RAM every MEMORY_LOG_INTERVAL batches
        if batch_count % MEMORY_LOG_INTERVAL == 0:
            tracker.log('batch', batch=batch_count)

        if batch_count % 100 == 0:
            print(f"   Processed {batch_count} batches... "
                  f"(memory: {get_memory_mb():.1f} MB)")
    
    training_time = time.time() - start_time
    tracker.log('after_training')

    train_mem = tracker.summary()
    print("")
    print(f"   Memory start: {train_mem['start_mb']:.1f} MB")
    print(f"   Memory peak:  {train_mem['peak_mb']:.1f} MB")
    print(f"   Memory end:   {train_mem['end_mb']:.1f} MB")
    print(f"   Memory growth: {train_mem['growth_mb']:+.1f} MB over {batch_count} batches")
    
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

    tracker.log('end')
    tracker.to_dataframe().to_csv(MEMORY_LOG_PATH, index=False)
    mem = tracker.summary()

    # Per-batch learning curve: raw MSE plus a moving average to smooth the noise.
    # np.convolve drops the first (window - 1) points, so pad them with NaN to
    # keep one row per batch.
    moving_avg = calculate_moving_average(mse_history, window=MOVING_AVG_WINDOW)
    padding = [np.nan] * (len(mse_history) - len(moving_avg))

    batch_metrics = pd.DataFrame({
        'batch': range(1, len(mse_history) + 1),
        'mse': mse_history,
        f'mse_moving_avg_{MOVING_AVG_WINDOW}': padding + list(moving_avg)
    })
    batch_metrics.to_csv(BATCH_METRICS_PATH, index=False)

    results = {
        'metric': ['run_date', 'run_time', 'R2_train', 'R2_test', 'MSE_train', 'MSE_test',
                   'training_time', 'memory_start_mb', 'memory_peak_mb', 'memory_end_mb',
                   'memory_growth_mb', 'memory_peak_training_mb', 'memory_growth_training_mb',
                   'MSE_first_batch', 'MSE_last_batch', 'n_batches'],
        'value': [
            run_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            run_timestamp.isoformat(),
            r2_train,
            r2_test,
            mse_train,
            mse_test,
            training_time,
            mem['start_mb'],
            mem['peak_mb'],
            mem['end_mb'],
            mem['growth_mb'],
            train_mem['peak_mb'],
            train_mem['growth_mb'],
            mse_history[0],
            mse_history[-1],
            len(mse_history)
        ]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_SAVE_PATH, index=False)
    
    print(f"\n   Model saved to: {MODEL_SAVE_PATH}")
    print(f"   Metrics saved to: {METRICS_SAVE_PATH}")
    print(f"   Memory log saved to: {MEMORY_LOG_PATH}")
    print(f"   Batch metrics saved to: {BATCH_METRICS_PATH}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    train_incremental()