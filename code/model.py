# Neural network definition

import tensorflow as tf
from tensorflow import keras
from config import *

def build_model():
    """
    Build a sequential neural network with 3 hidden layers (sigmoid activation)
    and incremental learning capability.
    
    Returns:
        model: Compiled Keras model ready for incremental training
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_FEATURES,)),
        keras.layers.Dense(HIDDEN_LAYER_1, 
                          activation='sigmoid'),
        keras.layers.Dense(HIDDEN_LAYER_2, 
                          activation='sigmoid'),
        keras.layers.Dense(HIDDEN_LAYER_3, 
                          activation='sigmoid'),
        keras.layers.Dense(OUTPUT_FEATURES)  # Linear activation for regression
    ])
    
    # Compile for incremental training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error for monitoring
    )
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()  # Print architecture summary