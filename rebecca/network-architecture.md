# Neural Network Architecture

## Overview

**Model Type:** Sequential Feed-Forward Neural Network

**Purpose:** Incremental learning for product quantity prediction

---

## Input Layer

- **Features:** 5 (sku, price, order, duration, category)
- **Input shape:** (batch_size, 5)
- **Preprocessing:** StandardScaler (normalized to mean=0, std=1)

---

## Hidden Layers

### Hidden Layer 1

- **Number of neurons:** 64
- **Activation:** sigmoid
- **Parameters:** 5 x 64 + 64 (bias) = 384
- **Output shape:** (batch_size, 64)

### Hidden Layer 2

- **Number of neurons:** 32
- **Activation:** sigmoid
- **Parameters:** 64 x 32 + 32 (bias) = 2,080
- **Output shape:** (batch_size, 32)

### Hidden Layer 3

- **Number of neurons:** 16
- **Activation:** sigmoid
- **Parameters:** 32 x 16 + 16 (bias) = 528
- **Output shape:** (batch_size, 16)

---

## Output Layer

- **Number of neurons:** 1 (predicting quantity)
- **Activation:** linear (regression task - continuous value)
- **Parameters:** 16 x 1 + 1 (bias) = 17
- **Output shape:** (batch_size, 1)

---

## Model Summary

| Metric | Value |
|--------|-------|
| Total Parameters | 3,009 |
| Trainable Parameters | 3,009 |
| Non-trainable Parameters | 0 |
| Model Size | 11.75 KB |

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (learning_rate=0.001) |
| Loss Function | Mean Squared Error (MSE) |
| Monitoring Metric | Mean Absolute Error (MAE) |
| Training Method | Incremental (train_on_batch per 32-sample batch) |
| Batch Size | 32 |
| Epochs | 1 (full pass through data) |

---

## Architecture Rationale

1. **Sigmoid activation:** Non-linear, helps learn complex relationships between features and target
2. **Decreasing layer sizes (64 to 32 to 16):** Progressively compresses information and reduces overfitting
3. **Linear output:** Appropriate for regression tasks predicting continuous quantity values
4. **Small architecture:** Efficient for large datasets (400k samples), prevents memory issues
5. **Incremental training:** Allows processing huge datasets without loading all data into memory at once

---

## Performance Metrics

Dataset: 500k samples (400k training, 100k testing)

| Metric | Training | Test |
|--------|----------|------|
| R² Score | 0.3117 | 0.3235 |
| Mean Squared Error (MSE) | 1768.13 | 1676.84 |
| Training Time | ~48 seconds | - |

### Interpretation

- Test R² is higher than training R², indicating good generalization without overfitting
- Model explains approximately 32% of quantity variance
- Test MSE is lower than training MSE, confirming good model performance on unseen data
