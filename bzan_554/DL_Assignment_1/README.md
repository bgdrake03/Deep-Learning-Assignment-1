# BZAN 554 - Group Assignment 1: Incremental Learning

## Project Overview

Build a feed-forward neural network that predicts product quantity sold using incremental learning on large datasets. The model trains on 400,000 samples in approximately 48 seconds and achieves an R² score of 0.3235 on test data.

## Status: In Development

Model implementation is complete and has been trained and tested. All core components are functional. Performance results and metrics are logged with timestamps for each training run. See performance results below for latest run data.

## Repository Contents

### Code (`code/` directory)
- `code/config.py` - Configuration settings (paths, hyperparameters, architecture, batch size, learning rate)
- `code/model.py` - Neural network definition and compilation using Keras Sequential API
- `code/data_handler.py` - Data loading, train/test splitting, feature scaling with StandardScaler, and mini-batch generation
- `code/evaluate.py` - Metrics calculation functions (R², MSE, moving average for learning curves)
- `code/main.py` - Main training script with incremental learning loop and result logging with timestamps
- `code/test_integration.py` - Integration test script for verifying all components work together on small dataset
- `code/utils.py` - Utility functions placeholder

### Documentation (`rebecca/` directory)
- `rebecca/model-documentation.md` - Comprehensive model guide including model architecture, training details, usage instructions, and performance analysis
- `rebecca/network-architecture.md` - Markdown-formatted architecture specifications with layer details, parameters, training configuration, and performance metrics

### Data (`data/` directory)
- `data/pricing.csv` - Main dataset with 500,000 product records and 6 features (sku, price, quantity, order, duration, category)
- `data/data_dictionary.csv` - Data dictionary describing all features

### Results (`results/` directory - created during training)
- `results/trained_model.h5` - Trained neural network weights saved in HDF5 format
- `results/metrics.csv` - Performance metrics from training runs, timestamped for tracking progress across multiple runs

### Project Files
- `README.md` - This file, comprehensive project documentation
- `pyproject.toml` - Python project configuration with dependencies (tensorflow, numpy, pandas, scikit-learn)
- `LICENSE` - Project license file
- `.gitignore` - Git ignore patterns
- `.python-version` - Python version specification

### Supporting Materials (`given-materials/` directory)
- `given-materials/Group_Assignment_1.pdf` - Assignment specification
- `given-materials/Groups_from_Zaretzki.xlsx` - Reference groups data

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

**Dataset Files:**
- `data/pricing.csv` - Main dataset containing 500,000 product records
- `data/data_dictionary.csv` - Data dictionary with feature descriptions

**Dataset Specifications:**
- **Total samples:** 500,000 records
- **Train/test split:** 80/20 (400,000 training, 100,000 test)
- **Features:** 5 numeric columns
  - `sku` - Product identifier
  - `price` - Product price
  - `order` - Order sequence number
  - `duration` - Time period
  - `category` - Product category
- **Target:** `quantity` (continuous numeric value)
- **Preprocessing:** StandardScaler normalization (mean=0, std=1)
- **Batch size:** 32 samples per batch (configurable in `code/config.py`)

## Assignment Materials

Reference materials provided with the assignment:
- `given-materials/Group_Assignment_1.pdf` - Official assignment specification and requirements
- `given-materials/Groups_from_Zaretzki.xlsx` - Reference group assignments data

These files provide the original assignment context and specifications.

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

**Python Version:** 3.9 or higher

**Required Packages:**
- `tensorflow>=2.13.0` - Deep learning framework and Keras API
- `numpy>=1.24.0` - Numerical computing library
- `pandas>=2.0.0` - Data manipulation and analysis
- `scikit-learn>=1.3.0` - Machine learning utilities (preprocessing, metrics)

**Installation:**

Option 1: Using pyproject.toml (recommended)
```bash
pip install -e .
```

Option 2: Direct package installation
```bash
pip install tensorflow>=2.13.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0
```

Option 3: Using requirements.txt (if available)
```bash
pip install -r requirements.txt
```

Configuration stored in `pyproject.toml`

## Documentation

For detailed information, see (all located in `rebecca/` directory):
- **model-documentation.md** - Comprehensive guide with model overview, architecture, training details, and usage instructions
- **network-architecture.md** - Markdown-formatted architecture specifications with layer details, parameters, training configuration, and performance metrics

## Project Structure

```
Deep-Learning-Assignment-1/
├── code/                      # Python source code
│   ├── config.py              # Configuration (paths, hyperparameters)
│   ├── model.py               # Neural network definition
│   ├── data_handler.py        # Data loading, splitting, scaling, batching
│   ├── evaluate.py            # Metrics calculation (R², MSE, etc.)
│   ├── main.py                # Main training script with timestamp logging
│   ├── test_integration.py    # Integration test on small sample
│   └── utils.py               # Utility functions
├── data/                      # Dataset files
│   ├── pricing.csv            # 500k product records (main dataset)
│   └── data_dictionary.csv    # Feature descriptions
├── results/                   # Training outputs (created during training)
│   ├── trained_model.h5       # Trained neural network weights
│   └── metrics.csv            # Timestamped performance metrics
├── rebecca/                   # Project documentation
│   ├── model-documentation.md # Comprehensive model guide
│   └── network-architecture.md # Architecture specifications (markdown)
├── given-materials/           # Assignment reference materials
│   ├── Group_Assignment_1.pdf # Assignment specification
│   └── Groups_from_Zaretzki.xlsx # Reference data
├── README.md                  # This file
├── pyproject.toml             # Python project configuration
├── LICENSE                    # Project license
├── .gitignore                 # Git ignore patterns
└── .python-version            # Python version specification
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