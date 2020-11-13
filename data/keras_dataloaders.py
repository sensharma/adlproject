from keras.datasets import mnist

import numpy as np

def keras_mnist_loader(indices):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test_rand = X_test[indices, :, :, None]
    y_test_rand = y_test[indices]
    return X_train, y_train, X_test, y_test, X_test_rand, y_test_rand
