# Metrics and evaluation

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def calculate_r2(y_true, y_pred):
    """
    Calculate R² score.
    R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        r2: R² score
    """
    return r2_score(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        mse: Mean squared error
    """
    return mean_squared_error(y_true, y_pred)

def calculate_moving_average(values, window=100):
    """
    Calculate moving average (for learning curves).
    
    Args:
        values: List of values
        window: Window size for moving average
        
    Returns:
        moving_avg: Moving average values
    """
    return np.convolve(values, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    # Test the functions
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    
    print(f"R²: {calculate_r2(y_true, y_pred):.4f}")
    print(f"MSE: {calculate_mse(y_true, y_pred):.4f}")