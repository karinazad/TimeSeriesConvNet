import sys
import os

import numpy as np
import keras
from keras import layers

from keras.optimizers import SGD


class CNN:
    def __init__(self):
        self.model = keras.Sequential()
        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     input_shape=(64, 64, 3)))
        self.model.add(layers.MaxPool2D())
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(layers.MaxPool2D())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=512, activation='relu'))
        self.model.add(layers.Dense(units=512, activation='relu'))
        self.model.add(layers.Dense(units=1, activation='sigmoid'))

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 100,
              ):
        self.model.compile(optimizer=SGD(), loss='mse', metrics=['accuracy'],)

        return self.model.fit(X, y, epochs=epochs)

    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray,):

        return self.model.evaluate(X,y)


