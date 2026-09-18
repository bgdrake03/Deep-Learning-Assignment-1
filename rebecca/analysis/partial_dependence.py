# Partial Dependence Plot (PDP) Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_handler import load_and_split_data, scale_target, unscale_predictions

def compute_partial_dependence(model, X_test, feature_idx, feature_name, y_mean, y_std, n_samples=50):
    """
    Compute partial dependence for a single feature.

    Args:
        model: Trained Keras model
        X_test: Test data (scaled)
        feature_idx: Index of feature to vary
        feature_name: Name of feature
        y_mean, y_std: Target scaling parameters
        n_samples: Number of points along feature range

    Returns:
        feature_values, predictions: Arrays for plotting
    """
    # Get range of feature values (in original scale)
    # Note: X_test is already scaled, so we need to unscale to get original range
    # But for PDPs, we work with scaled space

    feature_vals_scaled = X_test.iloc[:, feature_idx].values
    min_val = feature_vals_scaled.min()
    max_val = feature_vals_scaled.max()

    # Create range of values for this feature
    feature_range = np.linspace(min_val, max_val, n_samples)

    # Compute predictions by varying one feature
    predictions = []

    for val in feature_range:
        # Create a copy of X_test with this feature set to val
        X_modified = X_test.copy()
        X_modified.iloc[:, feature_idx] = val

        # Get predictions (still in scaled units)
        pred_scaled = model.predict(X_modified.values, verbose=0)

        # Unscale predictions to actual quantity units
        pred_actual = unscale_predictions(pred_scaled, y_mean, y_std)

        # Average prediction across all samples
        predictions.append(pred_actual.mean())

    return feature_range, np.array(predictions)

def create_pdp_plots(model, X_test, y_test, y_mean, y_std, feature_names, output_dir='results/plots'):
    """
    Create PDPs for all features.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create a figure with subplots for all features
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    results = {}

    for idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        print(f"Computing PDP for {feature_name}...")

        # Compute partial dependence
        feature_range, predictions = compute_partial_dependence(
            model, X_test, idx, feature_name, y_mean, y_std, n_samples=50
        )

        results[feature_name] = {
            'feature_range': feature_range,
            'predictions': predictions
        }

        # Plot
        ax.plot(feature_range, predictions, 'b-', linewidth=2.5, label='PDP')
        ax.fill_between(feature_range, predictions, alpha=0.2, color='blue')
        ax.set_xlabel(f'{feature_name} (scaled)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Quantity', fontsize=11, fontweight='bold')
        ax.set_title(f'Partial Dependence: {feature_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Calculate and display trend
        trend = "Increasing" if predictions[-1] > predictions[0] else "Decreasing"
        slope = (predictions[-1] - predictions[0]) / (feature_range[-1] - feature_range[0])
        ax.text(0.02, 0.98, f'Trend: {trend}\nSlope: {slope:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)

    # Hide the extra subplot
    axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/partial_dependence_plots.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/partial_dependence_plots.png")
    plt.close()

    return results

def create_individual_pdp_plots(model, X_test, y_test, y_mean, y_std, feature_names, output_dir='results/plots'):
    """
    Create individual high-quality PDP for each feature.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for idx, feature_name in enumerate(feature_names):
        print(f"Creating individual PDP for {feature_name}...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute partial dependence
        feature_range, predictions = compute_partial_dependence(
            model, X_test, idx, feature_name, y_mean, y_std, n_samples=100
        )

        # Plot
        ax.plot(feature_range, predictions, 'b-', linewidth=3, label='Partial Dependence')
        ax.fill_between(feature_range, predictions, alpha=0.2, color='blue')
        ax.scatter([feature_range[0], feature_range[-1]],
                   [predictions[0], predictions[-1]],
                   color='red', s=100, zorder=5, label='Endpoints')

        ax.set_xlabel(f'{feature_name} (scaled feature value)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Quantity (units)', fontsize=12, fontweight='bold')
        ax.set_title(f'Partial Dependence Plot: {feature_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)

        # Add statistics
        trend = "Increasing" if predictions[-1] > predictions[0] else "Decreasing"
        slope = (predictions[-1] - predictions[0]) / (feature_range[-1] - feature_range[0])
        curvature = np.mean(np.diff(np.diff(predictions)))  # Measure non-linearity

        stats_text = f'''Relationship: {trend}
Slope: {slope:.4f}
Non-linearity: {abs(curvature):.4f}
Min Pred: {predictions.min():.2f}
Max Pred: {predictions.max():.2f}
Range: {predictions.max() - predictions.min():.2f}'''

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/pdp_{feature_name}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/pdp_{feature_name}.png")
        plt.close()

def analyze_pdps(results, feature_names):
    """
    Analyze and interpret PDP results.
    """
    print("\n" + "="*80)
    print("PARTIAL DEPENDENCE ANALYSIS SUMMARY")
    print("="*80)

    for feature_name in feature_names:
        if feature_name not in results:
            continue

        data = results[feature_name]
        preds = data['predictions']
        feature_range = data['feature_range']

        print(f"\n{feature_name.upper()}")
        print("-" * 80)

        # Calculate metrics
        trend = "Positive" if preds[-1] > preds[0] else "Negative"
        slope = (preds[-1] - preds[0]) / (feature_range[-1] - feature_range[0])
        max_pred = preds.max()
        min_pred = preds.min()
        pred_range = max_pred - min_pred

        # Non-linearity: second derivative
        second_deriv = np.diff(np.diff(preds))
        nonlinearity = np.mean(np.abs(second_deriv))

        print(f"Relationship Direction: {trend}")
        print(f"Slope (rate of change): {slope:.6f}")
        print(f"Prediction Range: {min_pred:.2f} to {max_pred:.2f} (span: {pred_range:.2f})")
        print(f"Non-linearity Index: {nonlinearity:.6f}")

        # Interpretation
        if abs(slope) < 0.01:
            importance = "LOW - Feature has minimal impact"
        elif abs(slope) < 0.1:
            importance = "MODERATE - Feature has some impact"
        else:
            importance = "HIGH - Feature strongly influences predictions"

        print(f"Estimated Importance: {importance}")

        if nonlinearity > 0.001:
            print(f"Non-linear Pattern Detected: YES (curvature = {nonlinearity:.6f})")
        else:
            print(f"Non-linear Pattern Detected: NO (linear relationship)")

def main():
    print("="*80)
    print("PARTIAL DEPENDENCE PLOT (PDP) ANALYSIS")
    print("="*80)

    # Load model
    print("\nLoading trained model...")
    model = keras.models.load_model('results/trained_model.keras')

    # Load data
    print("Loading and processing data...")
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Get target scaling parameters
    y_train_scaled, y_mean, y_std = scale_target(y_train)

    # Feature names
    feature_names = list(X_test.columns)
    print(f"Features: {feature_names}")

    # Create PDPs
    print("\nGenerating Partial Dependence Plots...")
    results = create_pdp_plots(model, X_test, y_test, y_mean, y_std,
                               feature_names, output_dir='results/plots')

    print("\nGenerating Individual High-Quality PDPs...")
    create_individual_pdp_plots(model, X_test, y_test, y_mean, y_std,
                                feature_names, output_dir='results/plots')

    # Analyze results
    analyze_pdps(results, feature_names)

    print("\n" + "="*80)
    print("PDP ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: results/plots/")
    print("  - partial_dependence_plots.png (all features)")
    print("  - pdp_<feature>.png (individual plots)")

if __name__ == "__main__":
    main()
