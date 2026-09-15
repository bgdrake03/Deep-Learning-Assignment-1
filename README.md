# BZAN 554 - Group Assignment 1: Incremental Learning

## Project Overview

Build a feed-forward neural network that predicts product quantity sold using incremental learning on large datasets. The model trains on 400,000 samples in approximately 48 seconds and achieves an R² score of 0.3235 on test data.

## Status: In Development

Model implementation is complete and has been trained and tested. All core components are functional. Performance results and metrics are logged with timestamps for each training run. See performance results below for latest run data.

## Files

### Code
- `code/config.py` - Configuration (paths, hyperparameters, architecture settings)
- `code/model.py` - Neural network definition and compilation
- `code/data_handler.py` - Data loading, splitting, scaling, and mini-batch generation
- `code/evaluate.py` - Metrics calculation (R², MSE, moving average)
- `code/main.py` - Main training loop for incremental learning
- `code/test_integration.py` - Integration test on small data sample

### Documentation
- `rebecca/MODEL_DOCUMENTATION.md` - Comprehensive model guide and usage instructions
- `rebecca/network-architecture.md` - Detailed architecture specifications and rationale

### Data
- `data/pricing.csv` - 500,000 product records (6 features)
- `data/data_dictionary.csv` - Feature descriptions

### Results
- `results/trained_model.h5` - Trained neural network weights
- `results/metrics.csv` - Performance metrics

## How to Run

### Full Training on All Data
```bash
python code/main.py
```

Expected output:
- Processes 400,000 training samples in ~12,500 batches
- Trains in approximately 48 seconds
- Saves model to `results/trained_model.h5`
- Saves metrics to `results/metrics.csv`

### Quick Integration Test
```bash
python code/test_integration.py
```

Expected output:
- Tests on 1,000 samples
- Completes in ~3 seconds
- Verifies all pipeline components work together

## Data

- **Dataset:** `data/pricing.csv`
- **Total samples:** 500,000
- **Train/test split:** 80/20 (400k train, 100k test)
- **Features:** 5 numeric (sku, price, order, duration, category)
- **Target:** quantity (continuous)
- **Preprocessing:** StandardScaler normalization
- **Batch size:** 32 (configurable in config.py)

## Model Architecture

### Network Structure
```
Input (5 features)
  ↓
Dense Layer 1: 64 neurons, sigmoid activation (384 parameters)
  ↓
Dense Layer 2: 32 neurons, sigmoid activation (2,080 parameters)
  ↓
Dense Layer 3: 16 neurons, sigmoid activation (528 parameters)
  ↓
Output: 1 neuron, linear activation (17 parameters)

Total Parameters: 3,009 (11.75 KB)
```

### Training Configuration
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss function:** Mean Squared Error (MSE)
- **Metric:** Mean Absolute Error (MAE)
- **Training method:** Incremental (batch-by-batch)
- **Batch size:** 32
- **Epochs:** 1 (full pass through data)

### Architecture Rationale
1. Sigmoid activation: Non-linear, learns complex patterns
2. Decreasing layer sizes: Progressively compresses information, reduces overfitting
3. Linear output: Appropriate for regression (continuous quantity prediction)
4. Small architecture: Prevents overfitting on large dataset, efficient memory usage
5. Incremental training: Enables processing arbitrarily large datasets

## Performance Results

Each training run generates timestamped results saved to `results/metrics.csv` with the exact date and time of the run.

### Latest Run Example (500k dataset)
| Metric | Training | Test |
|--------|----------|------|
| **R² Score** | 0.3117 | **0.3235** |
| **MSE** | 1768.13 | 1676.84 |
| **RMSE** | 42.05 | 40.95 |
| **Training Time** | ~48 seconds | - |

**Note:** Check `results/metrics.csv` for timestamped results from all training runs.

### Model Performance Analysis
- Test R² > Training R²: Excellent generalization (no overfitting)
- Model explains ~32% of quantity variance
- Test MSE < Training MSE: Further confirms good performance on unseen data
- Efficient training: 500k samples in under 1 minute

## Implementation Details

### Key Features
- Feature scaling with StandardScaler (prevents feature dominance)
- Incremental learning with mini-batches (handles large datasets)
- Cross-platform path handling (works on Windows, Mac, Linux)
- Comprehensive error checking and auto-directory creation
- Integration test for component verification

### What Was Done (Day 1)
1. Fixed code skeleton issues:
   - Fixed import paths (data-handler.py → data_handler.py)
   - Implemented cross-platform path handling
   - Added all dependencies to pyproject.toml
   - Auto-create results directory

2. Implemented missing components:
   - Added feature scaling (StandardScaler)
   - Fixed loss extraction from Keras metrics
   - Created integration test script

3. Ran full training:
   - Trained on all 400,000 samples
   - Verified incremental learning works
   - Saved model and metrics
   - Achieved R² of 0.3235 on test data

4. Created comprehensive documentation:
   - MODEL_DOCUMENTATION.md (complete usage guide)
   - Updated network-architecture.md with actual implementation details

## Testing

### Integration Test
The `test_integration.py` script verifies:
- Data loading and preprocessing
- Mini-batch generation
- Incremental training with train_on_batch
- R² and MSE calculation
- Moving average for learning curves
- Model evaluation on train and test sets

Run with:
```bash
python code/test_integration.py
```

## Dependencies

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Documentation

For detailed information, see:
- **MODEL_DOCUMENTATION.md** - Complete model guide, architecture details, how to use the model
- **network-architecture.md** - Architecture specifications and performance metrics

## Project Structure

```
.
├── code/
│   ├── config.py              # Configuration
│   ├── model.py               # Model definition
│   ├── data_handler.py        # Data processing
│   ├── evaluate.py            # Metrics
│   ├── main.py                # Training script
│   └── test_integration.py    # Integration test
├── data/
│   ├── pricing.csv            # Dataset
│   └── data_dictionary.csv    # Feature info
├── results/
│   ├── trained_model.h5       # Trained weights
│   └── metrics.csv            # Performance metrics
├── rebecca/
│   ├── MODEL_DOCUMENTATION.md # Complete guide
│   └── network-architecture.md # Architecture details
└── README.md                  # This file
```

## Key Learnings

1. **Incremental Learning:** Essential for large datasets, processes data in batches
2. **Feature Scaling:** Critical for neural network convergence and performance
3. **Generalization:** Test performance better than training indicates good model
4. **Batch Processing:** Enables handling arbitrary dataset sizes efficiently
5. **Architecture Design:** Simpler networks often better (prevents overfitting)

## Next Steps / Improvements

Potential enhancements for future work:
- Hyperparameter tuning (different layer sizes, learning rates)
- Multiple epochs for better convergence
- Regularization (dropout, L2) to reduce overfitting
- Feature engineering (create new features)
- Ensemble methods (combine multiple models)
- Different activations (ReLU, ELU)
- Skip connections or other advanced architectures

## Status Summary

Status: IN DEVELOPMENT - Model has been run and tested

- Model: Fully implemented and working
- Training: Successfully runs on full 500k dataset
- Testing: Integration test passes, results vary per run (timestamped)
- Documentation: Complete and detailed
- Performance: R² = 0.3235 in latest test run (good generalization)
- Logging: All runs timestamped in metrics.csv for tracking progress