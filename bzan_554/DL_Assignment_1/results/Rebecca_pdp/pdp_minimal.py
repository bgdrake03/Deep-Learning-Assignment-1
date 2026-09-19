# Minimal PDP - Fast computation with immediate results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from data_handler import load_and_split_data, scale_target, unscale_predictions
from pathlib import Path

print("Loading model and data...")
model = keras.models.load_model('results/trained_model.keras')
X_train, X_test, y_train, y_test = load_and_split_data()
y_train_scaled, y_mean, y_std = scale_target(y_train)

# Use small sample for speed
sample_idx = np.random.choice(len(X_test), 500, replace=False)
X_sample = X_test.iloc[sample_idx]

features = list(X_test.columns)
print(f"Computing PDPs for: {features}")

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (ax, feat) in enumerate(zip(axes, features)):
    print(f"  {feat}...", end=" ")

    # Get feature range
    feat_min = X_sample.iloc[:, idx].min()
    feat_max = X_sample.iloc[:, idx].max()
    feat_vals = np.linspace(feat_min, feat_max, 20)

    # Compute predictions
    preds = []
    for val in feat_vals:
        X_mod = X_sample.copy()
        X_mod.iloc[:, idx] = val
        pred_scaled = model.predict(X_mod.values, verbose=0, batch_size=128)
        pred_unscaled = unscale_predictions(pred_scaled, y_mean, y_std)
        preds.append(pred_unscaled.mean())

    preds = np.array(preds)

    # Plot
    ax.plot(feat_vals, preds, 'b-', linewidth=3)
    ax.fill_between(feat_vals, preds, alpha=0.15, color='blue')
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Value (scaled)', fontsize=10)
    ax.set_ylabel('Predicted Quantity', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add trend
    slope = (preds[-1] - preds[0]) / (feat_vals[-1] - feat_vals[0])
    trend = "UP" if slope > 0 else "DOWN" if slope < 0 else "FLAT"
    ax.text(0.98, 0.05, f'{trend} {slope:.4f}', transform=ax.transAxes,
            ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            fontsize=9, family='monospace')

    print("[OK]")

axes[-1].set_visible(False)
fig.suptitle('Partial Dependence Plots - All Features', fontsize=14, fontweight='bold')
plt.tight_layout()
Path('results/plots').mkdir(exist_ok=True)
plt.savefig('results/plots/PDP_all_features.png', dpi=120, bbox_inches='tight')
print("\n[SAVED] results/plots/PDP_all_features.png")
plt.close()

print("\n" + "="*80)
print("PDP ANALYSIS COMPLETE")
print("="*80)
