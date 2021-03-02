import sys
import os
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD


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

        self.history = None

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs=2500,
              steps_per_epoch=200,
              **kwargs):

        self.model.compile(optimizer=SGD(),
                           loss='mse',
                           metrics=['accuracy'], )

        self.history = self.model.fit(X,y,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      **kwargs)
    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 **kwargs):
        return self.model.evaluate(X, y, **kwargs)

    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(self.history.history['loss'])
        axes[0].set_title("Loss")
        # axes[0].legend()

        axes[1].plot(self.history.history['accuracy'])
        axes[1].set_title("Accuracy")
        # axes[1].legend()

        plt.suptitle(f"CNN history")
        plt.show()
