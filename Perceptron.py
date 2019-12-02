import numpy as np
from DataOptions import getData, getDataX

MAX_ITERATION = 1650


class Perceptron:
    def __init__(self, trainx, trainy):
        self._trainx = trainx
        self._trainy = trainy
        self._eta = 1
        self._w = None

    def train(self, itr=150):
        x_data = getDataX(self._trainx)
        y_data = getData(self._trainy)
        w = np.zeros([3, x_data.shape[1]])
        for _ in range(itr):
            prev_w = w.copy()
            for x, y in zip(x_data, y_data):
                y_hat = np.argmax(np.dot(w, x))
                if int(y) != y_hat:
                    w[int(y), :] = prev_w[int(y), :] + self._eta * x
                    w[y_hat, :] = prev_w[y_hat, :] - self._eta * x
        self._w = w

    def predict(self, x):
        return np.argmax(np.dot(self._w, x))
