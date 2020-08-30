import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random as random

def main():
    mnist = keras.datasets.mnist

    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train , x_test = x_train / 255.0, x_test / 255.0

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(x_train,y_train,epochs = 5)

    model.evaluate(x_test,  y_test, verbose=2)

if __name__ == "__main__":
    main()