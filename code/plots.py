# Plots for the presentation: learning curve and variable importance.
#
# Run AFTER main.py -- this reads results/batch_metrics.csv and the saved
# model, it does not train anything.
#
#   uv run python code/plots.py

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # quiet TF startup noise

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # write files, don't try to open a window
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import r2_score

from config import *
from data_handler import load_and_split_data, scale_target, unscale_predictions

# Shared styling so both figures look like they belong together
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': True
})

BLUE = '#2C6FB5'
GREY = '#BFC9D4'


def plot_learning_curve():
    """
    Learning curve: instances learned (x) vs moving-average MSE (y).

    X-axis is batch number * BATCH_SIZE, i.e. how many records the model has
    actually seen -- that is what the assignment asks for, not batch count.

    Returns:
        str: Path to the saved figure
    """
    df = pd.read_csv(BATCH_METRICS_PATH)
    ma_col = f'mse_moving_avg_{MOVING_AVG_WINDOW}'

    instances = df['batch'] * BATCH_SIZE
    smoothed = df[ma_col]

    # A second, much wider window: MA(100) still carries a lot of batch noise,
    # so the wide line is what actually shows the trend.
    wide = MOVING_AVG_WINDOW * 10
    wide_ma = np.convolve(df['mse'], np.ones(wide) / wide, mode='valid')
    wide_x = instances.values[wide - 1:]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(instances, smoothed, color=GREY, linewidth=1.0,
            label=f'Moving average ({MOVING_AVG_WINDOW} batches)', zorder=1)
    ax.plot(wide_x, wide_ma, color=BLUE, linewidth=2.4,
            label=f'Moving average ({wide:,} batches)', zorder=2)

    ax.set_xlabel('Instances learned (records seen)')
    ax.set_ylabel('Mean squared error')
    ax.set_title('Incremental Learning Curve', fontweight='bold', pad=12)

    lo = np.nanpercentile(smoothed, 1)
    hi = np.nanpercentile(smoothed, 99)
    pad = (hi - lo) * 0.12
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xlim(0, instances.max())
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:,.0f}k'))

    ax.grid(axis='y', alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, loc='upper right')

    ax.annotate(f'Smoothed MSE: {wide_ma[0]:,.0f} at start to {wide_ma[-1]:,.0f} at end',
                xy=(0.02, 0.06), xycoords='axes fraction',
                fontsize=10, color='#555555')

    fig.savefig(LEARNING_CURVE_PLOT, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    return LEARNING_CURVE_PLOT


def compute_permutation_importance(model, X_test, y_test, y_mean, y_std,
                                   n_repeats=N_PERMUTATION_REPEATS):
    """
    Permutation importance: shuffle one feature, see how much R2 falls.

    Shuffling breaks the link between that feature and the target while
    leaving its distribution intact. A big R2 drop means the model relied on
    it; ~0 means it was not being used.

    Args:
        model: Trained Keras model
        X_test, y_test: Held-out data (features already scaled)
        y_mean, y_std: Target scaling constants, to undo before scoring
        n_repeats: Shuffles per feature (averaged, with std as error bars)

    Returns:
        pd.DataFrame: feature, importance_mean, importance_std, sorted desc
    """
    X = X_test.values.astype('float32')

    baseline = r2_score(
        y_test,
        unscale_predictions(model.predict(X, verbose=0, batch_size=4096).ravel(),
                            y_mean, y_std))
    print(f"   Baseline test R2: {baseline:.4f}")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for col_idx, name in enumerate(X_test.columns):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            rng.shuffle(X_shuffled[:, col_idx])  # break this feature only
            r2 = r2_score(
                y_test,
                unscale_predictions(
                    model.predict(X_shuffled, verbose=0, batch_size=4096).ravel(),
                    y_mean, y_std))
            drops.append(baseline - r2)

        rows.append({'feature': name,
                     'importance_mean': float(np.mean(drops)),
                     'importance_std': float(np.std(drops))})
        print(f"   {name:<10} importance {np.mean(drops):+.4f}")

    return (pd.DataFrame(rows)
              .sort_values('importance_mean', ascending=False)
              .reset_index(drop=True))


def plot_variable_importance(importance_df):
    """
    Horizontal bar chart of permutation importance, most important on top.

    Args:
        importance_df: Output of compute_permutation_importance()

    Returns:
        str: Path to the saved figure
    """
    df = importance_df.sort_values('importance_mean')  # bottom-up for barh

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df['feature'], df['importance_mean'],
                   xerr=df['importance_std'], color=BLUE,
                   error_kw={'ecolor': '#666666', 'capsize': 4, 'lw': 1})

    # Value label at the end of each bar
    span = df['importance_mean'].max() or 1.0
    for bar, value in zip(bars, df['importance_mean']):
        ax.text(bar.get_width() + span * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}', va='center', fontsize=10)

    ax.set_xlabel('Drop in test R² when feature is shuffled')
    ax.set_ylabel('Feature')
    ax.set_title('Variable Importance (Permutation)', fontweight='bold', pad=12)
    ax.set_xlim(0, span * 1.18)
    ax.grid(axis='x', alpha=0.3, linewidth=0.6)
    # figtext (not annotate) so the caption sits below the axis label
    # instead of colliding with it
    fig.text(0.5, -0.02,
             f'Averaged over {N_PERMUTATION_REPEATS} shuffles; '
             'error bars = 1 standard deviation',
             ha='center', fontsize=9, color='#555555')

    fig.savefig(IMPORTANCE_PLOT, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    return IMPORTANCE_PLOT


def main():
    print("=" * 60)
    print("GENERATING PRESENTATION PLOTS")
    print("=" * 60)

    print("\n1. Learning curve...")
    print(f"   Saved: {plot_learning_curve()}")

    print("\n2. Permutation importance...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    _, y_mean, y_std = scale_target(y_train)   # same constants main.py used
    model = keras.models.load_model(MODEL_SAVE_PATH, compile=False)  # inference only

    importance = compute_permutation_importance(model, X_test, y_test,
                                                y_mean, y_std)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    print(f"   Scores saved: {IMPORTANCE_PATH}")

    print("\n3. Variable importance plot...")
    print(f"   Saved: {plot_variable_importance(importance)}")

    print("\n" + "=" * 60)
    print("RANKED FEATURES")
    print("=" * 60)
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
