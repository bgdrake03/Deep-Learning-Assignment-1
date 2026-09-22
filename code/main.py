"""Run the whole assignment, in the order the course covered it.

    1. stream the data off disk               assignment 0
    2. build the network                      assignments 2 and 4
    3. train it incrementally                 assignments 0 and 2
    4. score it with R2 and MSE               assignment 4
    5. variable importance                    assignment 1, Q1
    6. partial dependence plots               assignment 1, Q2
    7. learning curve, RAM, training time     this assignment

    uv run python code/main.py
"""

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')   # quiet TensorFlow startup

import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil

import plots
from config import *
from data import (compute_statistics, stream_train_batches, load_sample, unscale)
from metrics import (r2, mse, moving_average, permutation_importance,
                     partial_dependence)
from model import build_model


def memory_mb():
    """RAM this process is actually holding, in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def banner(text):
    print('\n' + '=' * 62)
    print(text)
    print('=' * 62)


def main():
    started_at = datetime.now()
    machine = psutil.virtual_memory()

    banner('INCREMENTAL LEARNING ON LARGE DATA')
    print(f'Run at       : {started_at:%Y-%m-%d %H:%M:%S}')
    print(f'Machine RAM  : {machine.total / 1024 ** 2:,.0f} MB')
    print(f'Data file    : {os.path.getsize(DATA_FILE) / 1024 ** 2:,.1f} MB on disk')
    print(f'Chunk size   : {CHUNK_SIZE:,} rows held at a time')

    # ---------------------------------------------------------------- 1. data
    banner('1. STREAMING THE DATA  (assignment 0)')
    print('Pass 1: accumulating mean and std from training rows only...')
    t0 = time.time()
    mean, std, n_train, n_test = compute_statistics()
    print(f'   {n_train:,} training rows, {n_test:,} test rows'
          f'   ({time.time() - t0:.1f}s)')
    print(f'   quantity: mean {mean[TARGET]:.2f}, std {std[TARGET]:.2f}'
          f'  -> standardised for training')
    print(f'   RAM after pass 1: {memory_mb():.1f} MB'
          '   (statistics only, no data kept)')

    # ---------------------------------------------------------------- 2. model
    banner('2. THE NETWORK  (assignments 2 and 4)')
    model = build_model()
    model.summary()

    # ---------------------------------------------------------------- 3. train
    banner('3. INCREMENTAL TRAINING  (assignments 0 and 2)')
    print(f'One pass over the data, updating after every {BATCH_SIZE} records.\n')

    batch_mse = []
    memory_log = []
    ram_before_training = memory_mb()
    t0 = time.time()

    for n_batches, (X_batch, y_batch) in enumerate(stream_train_batches(mean, std), start=1):
        batch_mse.append(model.train_on_batch(X_batch, y_batch))
        memory_log.append(memory_mb())

        if n_batches % 2000 == 0:
            print(f'   {n_batches * BATCH_SIZE:>9,} records   '
                  f'RAM {memory_log[-1]:6.1f} MB')

    training_time = time.time() - t0
    ram_after_training = memory_mb()

    # Loss came back in standardised units because that is what we trained on.
    # MSE scales by the square of the standard deviation.
    batch_mse = np.array(batch_mse) * std[TARGET] ** 2

    print(f'\n   {n_batches:,} batches, {n_batches * BATCH_SIZE:,} records'
          f' in {training_time:.1f}s')
    print(f'   RAM during training: {ram_before_training:.1f} MB -> '
          f'{ram_after_training:.1f} MB'
          f'  ({ram_after_training - ram_before_training:+.1f} MB)')
    print(f'   peak RAM: {max(memory_log):.1f} MB')

    # ---------------------------------------------------------------- 4. score
    banner('4. R2 AND MSE  (assignment 4)')
    X_test, y_test = load_sample(mean, std, n_test, 'test')
    X_train, y_train = load_sample(mean, std, n_train, 'train')
    print(f'Scoring on {len(y_train):,} training and {len(y_test):,} test rows'
          '  (bounded samples, so RAM does not grow with the file)\n')

    def predict(X):
        """Model output, converted back to real quantity units."""
        raw = model.predict(X, verbose=0, batch_size=8192).ravel()
        return unscale(raw, mean, std)

    r2_train, r2_test = r2(y_train, predict(X_train)), r2(y_test, predict(X_test))
    mse_train, mse_test = mse(y_train, predict(X_train)), mse(y_test, predict(X_test))

    print(f'   R2   train {r2_train:.4f}    test {r2_test:.4f}')
    print(f'   MSE  train {mse_train:,.1f}    test {mse_test:,.1f}')

    # ------------------------------------------------------------ 5. importance
    banner('5. VARIABLE IMPORTANCE  (assignment 1, Q1)')
    baseline, importance = permutation_importance(
        predict, X_test, y_test, FEATURES, N_PERMUTATION_REPEATS, RANDOM_SEED)
    print(f'Baseline test R2: {baseline:.4f}\n')
    for name, drop, spread in importance:
        print(f'   {name:<10} R2 drops {drop:+.4f}  (+/- {spread:.4f})')

    importance_df = (pd.DataFrame(importance,
                                  columns=['feature', 'importance_mean', 'importance_std'])
                     .sort_values('importance_mean', ascending=False)
                     .reset_index(drop=True))
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    # ---------------------------------------------------- 6. partial dependence
    banner('6. PARTIAL DEPENDENCE  (assignment 1, Q2)')
    print('Forcing each feature to a range of values and averaging the'
          ' predictions.\n')
    # X_test is still in file order, and the file is sorted by sku, so its
    # first rows are all low skus. Draw the subset at random so the curves
    # average over the whole population, not one corner of it.
    pick = np.random.default_rng(RANDOM_SEED).choice(
        len(X_test), min(PDP_SAMPLE_SIZE, len(X_test)), replace=False)
    X_pdp = X_test[pick]
    pdp_rows = []

    for col, name in enumerate(FEATURES):
        grid, average = partial_dependence(predict, X_pdp, col, PDP_GRID_POINTS)
        # grid is in standardised units; undo that so the x-axis reads in the
        # feature's real units, which is what makes the plot interpretable
        grid_real = grid * std[name] + mean[name]
        pdp_rows.append(pd.DataFrame({'feature': name,
                                      'value': grid_real,
                                      'avg_prediction': average}))
        print(f'   {name:<10} prediction ranges '
              f'{average.min():7.2f} to {average.max():7.2f}'
              f'   (spread {average.max() - average.min():6.2f})')

    pdp_df = pd.concat(pdp_rows, ignore_index=True)
    pdp_df.to_csv(PDP_PATH, index=False)

    # ---------------------------------------------------------------- 7. save
    banner('7. SAVING RESULTS')
    model.save(MODEL_PATH)

    smoothed = moving_average(batch_mse, MOVING_AVG_WINDOW)
    pd.DataFrame({
        'batch': np.arange(1, len(batch_mse) + 1),
        'records_seen': np.arange(1, len(batch_mse) + 1) * BATCH_SIZE,
        'mse': batch_mse,
        f'mse_moving_avg_{MOVING_AVG_WINDOW}':
            [np.nan] * (len(batch_mse) - len(smoothed)) + list(smoothed),
    }).to_csv(BATCH_METRICS_PATH, index=False)

    pd.DataFrame({'batch': np.arange(1, len(memory_log) + 1),
                  'records_seen': np.arange(1, len(memory_log) + 1) * BATCH_SIZE,
                  'memory_mb': memory_log}).to_csv(MEMORY_LOG_PATH, index=False)

    # Report SMOOTHED endpoints. A single batch is 32 records and far too noisy
    # to quote: the first and last individual batches can easily suggest the
    # model got worse when it did not.
    pd.DataFrame({
        'metric': ['run_datetime', 'n_train_rows', 'n_test_rows',
                   'R2_train', 'R2_test', 'MSE_train', 'MSE_test',
                   'training_time_sec', 'records_per_sec', 'n_batches',
                   'MSE_smoothed_start', 'MSE_smoothed_end',
                   'ram_before_training_mb', 'ram_after_training_mb',
                   'ram_growth_during_training_mb', 'ram_peak_mb',
                   'machine_ram_mb', 'data_file_mb'],
        'value': [started_at.strftime('%Y-%m-%d %H:%M:%S'), n_train, n_test,
                  r2_train, r2_test, mse_train, mse_test,
                  training_time, n_batches * BATCH_SIZE / training_time, n_batches,
                  smoothed[0], smoothed[-1],
                  ram_before_training, ram_after_training,
                  ram_after_training - ram_before_training, max(memory_log),
                  machine.total / 1024 ** 2,
                  os.path.getsize(DATA_FILE) / 1024 ** 2],
    }).to_csv(METRICS_PATH, index=False)

    # ---------------------------------------------------------------- 8. plots
    banner('8. PLOTS')
    for path in (plots.learning_curve(), plots.variable_importance(importance_df),
                 plots.partial_dependence_grid(pdp_df), plots.memory_usage()):
        print(f'   {path}')

    banner('DONE')
    print(f'Test R2 {r2_test:.4f}   training time {training_time:.1f}s   '
          f'RAM growth while training {ram_after_training - ram_before_training:+.1f} MB')


if __name__ == '__main__':
    main()
