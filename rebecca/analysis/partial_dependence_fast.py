# Fast Partial Dependence Plot (PDP) Analysis - Optimized version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_handler import load_and_split_data, scale_target, unscale_predictions

def compute_partial_dependence_fast(model, X_test, feature_idx, y_mean, y_std, n_samples=30, sample_fraction=0.2):
    """
    Compute partial dependence (fast version - uses sample of data).
    """
    # Use a sample for faster computation
    sample_size = int(len(X_test) * sample_fraction)
    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[sample_idx].copy()

    feature_vals_scaled = X_sample.iloc[:, feature_idx].values
    min_val = feature_vals_scaled.min()
    max_val = feature_vals_scaled.max()

    feature_range = np.linspace(min_val, max_val, n_samples)
    predictions = []

    for val in feature_range:
        X_modified = X_sample.copy()
        X_modified.iloc[:, feature_idx] = val
        pred_scaled = model.predict(X_modified.values, verbose=0)
        pred_actual = unscale_predictions(pred_scaled, y_mean, y_std)
        predictions.append(pred_actual.mean())

    return feature_range, np.array(predictions)

def create_combined_pdp_plot(model, X_test, y_mean, y_std, feature_names, output_dir='results/plots'):
    """
    Create one large figure with all PDPs (faster).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    results = {}

    for idx, feature_name in enumerate(feature_names):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        print(f"Computing PDP for {feature_name}...")
        feature_range, predictions = compute_partial_dependence_fast(
            model, X_test, idx, y_mean, y_std, n_samples=25, sample_fraction=0.3
        )

        results[feature_name] = {
            'feature_range': feature_range,
            'predictions': predictions
        }

        # Plot
        ax.plot(feature_range, predictions, 'b-', linewidth=3, label='PDP')
        ax.fill_between(feature_range, predictions, alpha=0.15, color='blue')
        ax.set_xlabel(f'{feature_name} (scaled)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Quantity', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Statistics
        trend = "↑" if predictions[-1] > predictions[0] else "↓"
        slope = (predictions[-1] - predictions[0]) / (feature_range[-1] - feature_range[0])
        change_pct = 100 * (predictions[-1] - predictions[0]) / predictions.mean() if predictions.mean() != 0 else 0

        stats_text = f'{trend} {change_pct:+.1f}%\nSlope: {slope:.4f}'
        ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                fontsize=9, family='monospace')

    # Title
    fig.suptitle('Partial Dependence Plots - All Features', fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(f'{output_dir}/pdp_combined.png', dpi=120, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/pdp_combined.png")
    plt.close()

    return results

def print_analysis(results, feature_names):
    """
    Print PDP analysis summary.
    """
    print("\n" + "="*80)
    print("PARTIAL DEPENDENCE PLOT (PDP) ANALYSIS SUMMARY")
    print("="*80)

    for feature_name in feature_names:
        if feature_name not in results:
            continue

        preds = results[feature_name]['predictions']
        feature_range = results[feature_name]['feature_range']

        # Metrics
        slope = (preds[-1] - preds[0]) / (feature_range[-1] - feature_range[0])
        change_pct = 100 * (preds[-1] - preds[0]) / preds.mean() if preds.mean() != 0 else 0
        pred_range = preds.max() - preds.min()

        # Non-linearity
        second_deriv = np.diff(np.diff(preds)) if len(preds) > 2 else np.array([0])
        nonlinearity = np.abs(second_deriv).mean()

        print(f"\n{feature_name.upper()}")
        print("-" * 80)
        print(f"  Relationship: {'Positive ↑' if slope > 0 else 'Negative ↓'}")
        print(f"  Change: {change_pct:+.1f}% across feature range")
        print(f"  Slope: {slope:.6f} (change per unit)")
        print(f"  Prediction Range: {preds.min():.2f} to {preds.max():.2f} (span: {pred_range:.2f})")
        print(f"  Non-linearity: {'Yes' if nonlinearity > 0.0001 else 'No'}")

        # Importance interpretation
        if abs(slope) < 0.001:
            importance = "VERY LOW"
        elif abs(slope) < 0.01:
            importance = "LOW"
        elif abs(slope) < 0.05:
            importance = "MODERATE"
        else:
            importance = "HIGH"

        print(f"  Estimated Importance: {importance}")

def main():
    print("="*80)
    print("PARTIAL DEPENDENCE PLOT (PDP) ANALYSIS - FAST VERSION")
    print("="*80)

    print("\nLoading model...")
    model = keras.models.load_model('results/trained_model.keras')

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    y_train_scaled, y_mean, y_std = scale_target(y_train)

    feature_names = list(X_test.columns)
    print(f"Features: {', '.join(feature_names)}")

    print("\nGenerating Partial Dependence Plots...")
    results = create_combined_pdp_plot(model, X_test, y_mean, y_std, feature_names)

    print_analysis(results, feature_names)

    print("\n" + "="*80)
    print("PDP ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nPlot saved to: results/plots/pdp_combined.png")

if __name__ == "__main__":
    main()
