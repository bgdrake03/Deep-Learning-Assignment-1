"""The network: five inputs, three sigmoid hidden layers, one linear output.

The same shape we built by hand in assignment 2, written in Keras as in
assignment 4. Each Dense layer computes  activation(inputs @ W + b).

The output layer is LINEAR, not sigmoid: quantity runs from 1 to 4165, and a
sigmoid can only ever produce values between 0 and 1.
"""

import tensorflow as tf
from config import *


def build_model():
    layers = [tf.keras.layers.Input(shape=(len(FEATURES),), name='input')]
    for i, units in enumerate(HIDDEN_LAYERS, start=1):
        layers.append(tf.keras.layers.Dense(units, activation='sigmoid',
                                            name=f'hidden{i}'))
    layers.append(tf.keras.layers.Dense(1, activation='linear', name='output'))

    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse')
    return model


if __name__ == '__main__':
    build_model().summary()
