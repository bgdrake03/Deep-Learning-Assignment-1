"""
make_pdp.py  --  Generates the PARTIAL DEPENDENCE PLOTS required by the assignment.

WHY THIS SCRIPT EXISTS
----------------------
Every other required figure (learning curve, variable importance) has already been
produced. Partial dependence plots are the one remaining required deliverable.

WHAT IT DOES
------------
For each feature j, it sweeps that feature across a grid of values while holding all
other features at their real observed values, then averages the model's prediction:

        PD_j(v) = (1/n) * SUM_i  f( x_i  with  x_i[j] := v )

That is the definition of partial dependence: the marginal effect of feature j on the
predicted quantity, averaging out every other feature.

HOW TO RUN
----------
    cd code/
    python make_pdp.py

Outputs into results/:
    pdp_grid.png          <- 2x2 grid, drop straight onto the PDP slide
    pdp_<feature>.png     <- one full-size plot per feature
    pdp_values.csv        <- the underlying numbers

RUNTIME: ~30-60 seconds.

NOTE ON THE SCALER
------------------
data_handler.py does not save the fitted StandardScaler. This script therefore rebuilds
it by re-running the identical split (same RANDOM_SEED=42, same TEST_SPLIT=0.2) and
re-fitting on the training rows only. Because the seed is fixed, the resulting transform
is bit-identical to the one used at training time. No test data touches the scaler fit.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_FILE, TEST_SPLIT, RANDOM_SEED, RESULTS_DIR, MODEL_SAVE_PATH
from data_handler import scale_target, unscale_predictions

# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------
FEATURES     = ["duration", "price", "category", "order", "sku"]  # importance order
GRID_POINTS  = 25      # values swept per feature
N_SAMPLE     = 5000    # background rows used for the average (5k is plenty and fast)
PLOT_FEATURES = ["duration", "price", "order", "category"]  # the 4 shown in the grid


def main():
    # ------------------------------------------------------------------------------
    # 1. Rebuild the exact training split and scaler
    # ------------------------------------------------------------------------------
    print("1. Loading data and rebuilding the training-time scaler...")
    df = pd.read_csv(DATA_FILE)
    X = df.drop("quantity", axis=1)
    y = df["quantity"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    scaler.fit(X_train_raw)          # fit on TRAIN ONLY -- no leakage
    cols = list(X.columns)

    # TARGET SCALING (added in the current code): main.py trained on a
    # standardized y, so the model emits STANDARDIZED units. Recover the same
    # train-only mean/std used at training time so predictions can be converted
    # back to real quantities. Without this the PDP y-axis is in std units and
    # every curve is silently wrong.
    _, y_mean, y_std = scale_target(y_train)
    print(f"   Target scaling constants: mean {y_mean:.2f}, std {y_std:.2f}")

    # ------------------------------------------------------------------------------
    # 2. Load the trained model
    # ------------------------------------------------------------------------------
    print(f"2. Loading trained model from {MODEL_SAVE_PATH} ...")
    model = keras.models.load_model(MODEL_SAVE_PATH, compile=False)

    # ------------------------------------------------------------------------------
    # 3. Background sample, drawn from the TEST set (unseen rows)
    # ------------------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X_test_raw), size=min(N_SAMPLE, len(X_test_raw)), replace=False)
    background = X_test_raw.iloc[idx].copy()
    print(f"3. Background sample: {len(background):,} rows from the test set")

    # ------------------------------------------------------------------------------
    # 4. Compute partial dependence
    # ------------------------------------------------------------------------------
    print("4. Computing partial dependence...")
    records = []
    pdp = {}

    for feat in FEATURES:
        # Grid across the 5th-95th percentile: avoids extrapolating into outliers
        lo, hi = np.percentile(X_train_raw[feat], [5, 95])
        if feat in ("category", "order"):
            # Integer-valued features: use actual observed values, not a float grid
            vals = np.unique(X_train_raw[feat])
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) > GRID_POINTS:
                vals = np.quantile(vals, np.linspace(0, 1, GRID_POINTS))
                vals = np.unique(np.round(vals))
        else:
            vals = np.linspace(lo, hi, GRID_POINTS)

        avg_preds = []
        for v in vals:
            tmp = background.copy()
            tmp[feat] = v                                 # override ONE feature
            tmp_scaled = scaler.transform(tmp[cols])      # same transform as training
            preds = unscale_predictions(
                model.predict(tmp_scaled, verbose=0, batch_size=4096).ravel(),
                y_mean, y_std)                            # back to quantity units
            avg_preds.append(preds.mean())                # average out the others
            records.append({"feature": feat, "value": v, "avg_prediction": preds.mean()})

        pdp[feat] = (np.asarray(vals), np.asarray(avg_preds))
        span = pdp[feat][1].max() - pdp[feat][1].min()
        print(f"   {feat:<10s} swept {len(vals):>3d} values | "
              f"predicted quantity ranges {span:7.2f} units")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, "pdp_values.csv"), index=False)

    # ------------------------------------------------------------------------------
    # 5. Plot -- 2x2 grid for the slide
    # ------------------------------------------------------------------------------
    print("5. Plotting...")
    mean_q = float(y_train.mean())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, feat in zip(axes.ravel(), PLOT_FEATURES):
        vals, preds = pdp[feat]
        ax.plot(vals, preds, linewidth=2.5, color="#1f6fb4",
                marker="o", markersize=3.5)
        ax.axhline(mean_q, color="grey", linestyle="--", linewidth=1,
                   label=f"mean quantity ({mean_q:.1f})")
        ax.set_title(f"Partial dependence: {feat}", fontweight="bold")
        ax.set_xlabel(f"{feat} (original units)")
        ax.set_ylabel("Predicted quantity")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Partial Dependence Plots - effect of each feature on predicted quantity",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = os.path.join(RESULTS_DIR, "pdp_grid.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"   saved {out}")

    # Individual plots
    for feat in FEATURES:
        vals, preds = pdp[feat]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(vals, preds, linewidth=2.5, color="#1f6fb4", marker="o", markersize=4)
        ax.axhline(mean_q, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"Partial dependence: {feat}", fontweight="bold")
        ax.set_xlabel(f"{feat} (original units)")
        ax.set_ylabel("Predicted quantity")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = os.path.join(RESULTS_DIR, f"pdp_{feat}.png")
        fig.savefig(p, dpi=170)
        plt.close(fig)
        print(f"   saved {p}")

    # ------------------------------------------------------------------------------
    # 6. Talking points for the slide
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("READ THIS OFF FOR THE SLIDE -- PD range = how much predicted quantity")
    print("moves as the feature sweeps its 5th-95th percentile:")
    print("=" * 68)
    for feat in FEATURES:
        _, preds = pdp[feat]
        rng_ = preds.max() - preds.min()
        direction = "increasing" if preds[-1] > preds[0] else "decreasing"
        print(f"  {feat:<10s} range {rng_:7.2f} units   overall {direction}")
    print("\nSanity check: this ordering should broadly match variable_importance.csv")
    print("(duration strongest, then price, then category/order/sku near zero).")


if __name__ == "__main__":
    main()
