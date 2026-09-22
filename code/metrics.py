"""Scoring the model, and working out what it learned.

    r2, mse                 how wrong the model is  (assignment 4)
    moving_average          smoothing for the learning curve
    permutation_importance  how much each feature matters  (assignment 1, Q1)
    partial_dependence      which direction and what shape  (assignment 1, Q2)

Nothing here knows about TensorFlow or about scaling. Each function that needs
predictions takes a `predict` function as an argument, so the same code works
on any model.
"""

import numpy as np


def r2(y_true, y_pred):
    """R-squared, written out from the formula in the assignment PDF:

        R2 = 1 - sum((y_i - yhat_i)^2) / sum((y_i - ybar)^2)

    The top is how wrong we are. The bottom is how wrong we would be if we
    ignored the inputs and always guessed the mean. So R2 is the fraction of
    the variation we managed to explain: 1 is perfect, 0 is no better than
    guessing the mean, and negative is worse than that.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_residual / ss_total


def mse(y_true, y_pred):
    """Mean squared error: average of the squared misses."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return np.mean((y_true - y_pred) ** 2)


def moving_average(values, window):
    """Average of the last `window` values, slid along the series.

    One batch of 32 records is far too noisy to read a trend from, so the
    learning curve plots this instead of the raw per-batch loss.
    """
    return np.convolve(values, np.ones(window) / window, mode='valid')


def permutation_importance(predict, X, y, feature_names, n_repeats, seed):
    """How much worse the model predicts when one feature is scrambled.

    Shuffling a column destroys the link between that feature and the target
    while leaving the column's own distribution untouched, so the only thing
    that changes is the relationship. If the model relied on that feature, R2
    falls a lot; if it ignored the feature, R2 barely moves.

    The model is never refit - we only ever call predict(). (Assignment 1, Q1.)

    Returns:
        baseline R2, and a list of (feature, mean drop, std of drops)
    """
    baseline = r2(y, predict(X))
    rng = np.random.default_rng(seed)
    results = []

    for col, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()          # never touch the real data
            rng.shuffle(X_shuffled[:, col])
            drops.append(baseline - r2(y, predict(X_shuffled)))
        results.append((name, float(np.mean(drops)), float(np.std(drops))))

    return baseline, results


def partial_dependence(predict, X, col, n_grid):
    """What the model predicts as one feature is swept across its range.

    For each value on the grid we force EVERY row to that value, leave every
    other column exactly as it really is, predict, and average the results.
    That answers "if the whole population had this value, what would the model
    predict on average?" - which isolates one feature's effect from the rest.

    Averaging over the real rows, rather than predicting a single average row,
    is what makes the answer describe the actual population. (Assignment 1, Q2.)

    Returns:
        grid of values tested, average prediction at each one
    """
    grid = np.linspace(X[:, col].min(), X[:, col].max(), n_grid)
    averages = np.empty(n_grid)

    for i, value in enumerate(grid):
        X_fake = X.copy()            # a throwaway copy, as always
        X_fake[:, col] = value       # force this column for every row
        averages[i] = predict(X_fake).mean()

    return grid, averages
