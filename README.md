# BZAN 554 - Group Assignment 1: Incremental Learning

## Project Overview
Build a feed-forward neural network that predicts product quantity sold using incremental learning on large data.

## Files
- `config.py` - All settings (batch size, learning rate, architecture, paths)
- `model.py` - Neural network definition
- `data_handler.py` - Load data and create mini-batches
- `evaluate.py` - Calculate metrics (R², MSE, etc.)
- `main.py` - Main training loop
- `utils.py` - Helper functions (to be added)

## How to Run
```bash
python main.py
```

## Data
- File: `../data/pricing.csv`
- Train/test split: 80/20
- Batch size: 32 (configurable in config.py)

## Model Architecture
- Input: 5 features
- Hidden layer 1: 64 neurons, sigmoid
- Hidden layer 2: 32 neurons, sigmoid
- Hidden layer 3: 16 neurons, sigmoid
- Output: 1 neuron (quantity prediction)

## Key Metrics
- R² (coefficient of determination)
- MSE (mean squared error)
- Training time
- Memory usage

## Next Steps (Day 2)
- Test code on small data subset
- Debug any issues
- Optimize batch size if needed