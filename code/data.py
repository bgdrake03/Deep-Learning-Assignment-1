"""Reading the data without ever holding all of it in RAM.

This is the point of the assignment. pricing.csv stands in for a file too big
to load, so every function here reads it in chunks and keeps only running
totals or a bounded sample. Memory stays flat however large the file gets.

Concept from assignment 0: learn from a stream, one piece at a time.
"""

import numpy as np
import pandas as pd
from config import *

COLUMNS = FEATURES + [TARGET]


def _read_chunks():
    """Yield the CSV CHUNK_SIZE rows at a time. The only place we touch disk."""
    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        yield chunk


def _is_test(row_numbers):
    """Send about TEST_SPLIT of rows to the test set, decided by row number.

    A hash of the row number rather than a random draw, so a given row lands
    in the same set on every pass and every run. That lets us split the data
    without shuffling it or holding it in memory.
    """
    h = (row_numbers.astype(np.int64) * 2654435761) % 2 ** 32
    return (h / 2 ** 32) < TEST_SPLIT


def compute_statistics():
    """Pass 1: mean and std of every column, from TRAINING rows only.

    Standardising needs the mean and std of the whole training set, which we
    cannot get without seeing every row. So we make one pass accumulating
    three running totals - count, sum, sum of squares - and derive the
    statistics at the end. Nothing is kept but a handful of numbers.

    Using training rows only is the no-leakage rule: the test set must not
    influence anything the model sees.

    Returns:
        mean, std (pd.Series over COLUMNS), n_train, n_test
    """
    n_train = n_test = 0
    total = np.zeros(len(COLUMNS))
    total_sq = np.zeros(len(COLUMNS))
    start = 0

    for chunk in _read_chunks():
        rows = np.arange(start, start + len(chunk))
        start += len(chunk)
        test = _is_test(rows)

        train = chunk[COLUMNS].to_numpy(dtype=np.float64)[~test]
        n_train += len(train)
        n_test += int(test.sum())
        total += train.sum(axis=0)
        total_sq += (train ** 2).sum(axis=0)

    mean = total / n_train
    var = np.maximum(total_sq / n_train - mean ** 2, 1e-12)
    return (pd.Series(mean, index=COLUMNS),
            pd.Series(np.sqrt(var), index=COLUMNS),
            n_train, n_test)


def _scale_features(X, mean, std):
    return (X - mean[FEATURES].to_numpy(np.float32)) / std[FEATURES].to_numpy(np.float32)


def stream_train_batches(mean, std):
    """Pass 2: scaled, shuffled training mini-batches, read straight from disk.

    The target is standardised too. Left raw it has mean 23 and reaches 4165,
    and asking sigmoid layers to produce numbers that large saturates them and
    stalls learning. Predictions therefore come back in standardised units, so
    unscale() must be applied before any of them are scored or reported.

    The rows are shuffled through a bounded buffer before being handed out.
    pricing.csv is sorted by sku, so reading it in order would give every batch
    nearly identical rows - and stochastic gradient descent assumes each batch
    is a fair sample of the data. We cannot shuffle a file we cannot hold, so
    we fill a SHUFFLE_BUFFER-sized window, shuffle that, and empty it before
    filling the next one. Memory stays bounded.

    Yields:
        (X_batch, y_batch) ready for model.train_on_batch
    """
    rng = np.random.default_rng(RANDOM_SEED)
    buffered_X, buffered_y, n_buffered = [], [], 0
    start = 0

    def drain(parts_X, parts_y):
        X = np.concatenate(parts_X)
        y = np.concatenate(parts_y)
        order = rng.permutation(len(X))
        X, y = X[order], y[order]
        for i in range(0, len(X), BATCH_SIZE):
            yield X[i:i + BATCH_SIZE], y[i:i + BATCH_SIZE]

    for chunk in _read_chunks():
        rows = np.arange(start, start + len(chunk))
        start += len(chunk)
        keep = ~_is_test(rows)

        buffered_X.append(
            _scale_features(chunk[FEATURES].to_numpy(dtype=np.float32)[keep], mean, std))
        buffered_y.append(
            (chunk[TARGET].to_numpy(dtype=np.float32)[keep] - mean[TARGET]) / std[TARGET])
        n_buffered += int(keep.sum())

        if n_buffered >= SHUFFLE_BUFFER:
            yield from drain(buffered_X, buffered_y)
            buffered_X, buffered_y, n_buffered = [], [], 0

    if n_buffered:
        yield from drain(buffered_X, buffered_y)


def load_sample(mean, std, n_available, which):
    """Pass 3: a bounded random sample of rows, scaled, held in RAM.

    Scoring R2, permutation importance and partial dependence all need data in
    memory. Capping the sample at SAMPLE_SIZE makes that a fixed cost rather
    than one that grows with the file.

    Args:
        mean, std: statistics from compute_statistics()
        n_available: how many rows of this kind exist, also from pass 1
        which: 'train' or 'test'

    Returns:
        X scaled, y in ORIGINAL quantity units so scores are in real units
    """
    keep_prob = min(1.0, SAMPLE_SIZE / max(n_available, 1))
    rng = np.random.default_rng(RANDOM_SEED)
    Xs, ys = [], []
    start = 0

    for chunk in _read_chunks():
        rows = np.arange(start, start + len(chunk))
        start += len(chunk)
        wanted = _is_test(rows) if which == 'test' else ~_is_test(rows)
        pick = wanted & (rng.random(len(chunk)) < keep_prob)
        if pick.any():
            Xs.append(chunk[FEATURES].to_numpy(dtype=np.float32)[pick])
            ys.append(chunk[TARGET].to_numpy(dtype=np.float32)[pick])

    return _scale_features(np.concatenate(Xs), mean, std), np.concatenate(ys)


def unscale(y_scaled, mean, std):
    """Turn standardised model output back into real quantity units."""
    return y_scaled * std[TARGET] + mean[TARGET]
