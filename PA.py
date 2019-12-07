import numpy as np
from DataOptions import getData, getDataX


class PA:
    def __init__(self, trainx, trainy):
        self._trainx = trainx
        self._trainy = trainy
        self._w = None

    def train(self, epochs=100):
        x_data = getDataX(self._trainx)
        y_data = getData(self._trainy)
        w = np.zeros([3, x_data.shape[1]])
        for _ in range(epochs):
            prev_w = w.copy()
            for x, y in zip(x_data, y_data):
                y_hat = np.argmax(np.dot(w, x))
                if int(y) != y_hat:
                    w[int(y), :] = prev_w[int(y), :] + self.tau(w, x, y, y_hat) * x
                    w[y_hat, :] = prev_w[y_hat, :] - self.tau(w, x, y, y_hat) * x
        self._w = w

    def predict(self, x):
        return np.argmax(np.dot(self._w, x))

    def tau(self, w, x, y, y_hat):
        temp = (-np.dot(w[int(y), :], x)) + np.dot(w[y_hat, :], x)
        l = max(0, 1 + temp)
        n = np.linalg.norm(x)
        n = n ** 2
        return l / (2 * n)
