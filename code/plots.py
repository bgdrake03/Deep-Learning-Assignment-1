"""The four figures the assignment asks for.

    learning_curve          records learned vs moving-average MSE
    variable_importance     which features the model relies on
    partial_dependence_grid which direction and shape each feature has
    memory_usage            evidence that RAM stays flat while streaming

Each function reads a results file written by main.py and returns the path it
saved to, so main.py can just print them.
"""

import matplotlib

matplotlib.use('Agg')      # write files, never try to open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import *

plt.rcParams.update({
    'font.size': 14,          # these figures are read at slide size, not on paper
    'axes.titlesize': 17,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

TEAL, GREY, CORAL = '#0E7C7B', '#C5D9D8', '#C9502F'


def _thousands(value, _):
    return f'{value / 1000:,.0f}k'


def learning_curve():
    """MSE against the number of records the model has seen."""
    df = pd.read_csv(BATCH_METRICS_PATH)
    smoothed = df[f'mse_moving_avg_{MOVING_AVG_WINDOW}']

    # The 100-batch average is still noisy, so a much wider one is drawn on top
    # to show the actual trend.
    wide = MOVING_AVG_WINDOW * 10
    wide_mse = np.convolve(df['mse'], np.ones(wide) / wide, mode='valid')
    wide_x = df['records_seen'].values[wide - 1:]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['records_seen'], smoothed, color=GREY, lw=1,
            label=f'moving average, {MOVING_AVG_WINDOW} batches')
    ax.plot(wide_x, wide_mse, color=TEAL, lw=2.4,
            label=f'moving average, {wide:,} batches')

    ax.set_xlabel('Instances learned (records seen)')
    ax.set_ylabel('Mean squared error')
    ax.set_title('Incremental learning curve', fontweight='bold')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))

    low, high = np.nanpercentile(smoothed, [1, 99])
    ax.set_ylim(max(0, low - (high - low) * 0.1), high + (high - low) * 0.1)
    ax.set_xlim(0, df['records_seen'].max())
    ax.grid(axis='y', alpha=0.3)
    ax.legend(frameon=False)
    ax.annotate(f'smoothed MSE {wide_mse[0]:,.0f} at the start'
                f'  ->  {wide_mse[-1]:,.0f} at the end',
                xy=(0.02, 0.06), xycoords='axes fraction', color='#557575',
                fontsize=12)

    fig.savefig(LEARNING_CURVE_PLOT, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    return LEARNING_CURVE_PLOT


def variable_importance(importance_df):
    """Horizontal bars: how far test R2 falls when each feature is shuffled."""
    df = importance_df.sort_values('importance_mean')     # bottom-up for barh

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(df['feature'], df['importance_mean'], xerr=df['importance_std'],
                   color=TEAL, error_kw={'ecolor': '#666666', 'capsize': 4, 'lw': 1})

    span = max(df['importance_mean'].max(), 1e-9)
    for bar, value in zip(bars, df['importance_mean']):
        ax.text(bar.get_width() + span * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}', va='center', fontsize=12)

    ax.set_xlabel('Drop in test R-squared when the feature is shuffled')
    ax.set_title('Variable importance (permutation)', fontweight='bold')
    ax.set_xlim(0, span * 1.2)
    ax.grid(axis='x', alpha=0.3)
    fig.text(0.5, -0.04,
             f'averaged over {N_PERMUTATION_REPEATS} shuffles;'
             ' error bars are one standard deviation',
             ha='center', fontsize=11, color='#557575')

    fig.savefig(IMPORTANCE_PLOT, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    return IMPORTANCE_PLOT


def partial_dependence_grid(pdp_df):
    """One panel per feature: its value against the average prediction.

    All panels share a y-axis so the curves are directly comparable - a flat
    line really is flat relative to the others, not just rescaled.
    """
    features = list(pdp_df['feature'].unique())
    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.6),
                             sharey=True)

    for ax, name in zip(np.atleast_1d(axes), features):
        part = pdp_df[pdp_df['feature'] == name]
        ax.plot(part['value'], part['avg_prediction'], color=TEAL, lw=2)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(f'{name} (original units)')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)

    np.atleast_1d(axes)[0].set_ylabel('Average predicted quantity')
    fig.suptitle('Partial dependence: effect of each feature on predicted quantity',
                 fontweight='bold', y=1.04)
    fig.tight_layout()

    fig.savefig(PDP_PLOT, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    return PDP_PLOT


def memory_usage():
    """RAM held by the process as records stream through.

    This is the evidence for the whole assignment: a flat line means the data
    is never accumulating in memory, so the file could be any size at all.
    """
    df = pd.read_csv(MEMORY_LOG_PATH)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df['records_seen'], df['memory_mb'], color=TEAL, lw=1.2)
    ax.axhline(df['memory_mb'].iloc[0], color=CORAL, ls='--', lw=1,
               label=f'start: {df["memory_mb"].iloc[0]:.0f} MB')

    ax.set_xlabel('Instances learned (records seen)')
    ax.set_ylabel('Process RAM (MB)')
    ax.set_title('RAM usage during incremental training', fontweight='bold')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(frameon=False)

    growth = df['memory_mb'].iloc[-1] - df['memory_mb'].iloc[0]
    ax.annotate(f'growth over the whole run: {growth:+.1f} MB',
                xy=(0.02, 0.9), xycoords='axes fraction', color='#557575',
                fontsize=12)

    fig.savefig(MEMORY_PLOT, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    return MEMORY_PLOT
