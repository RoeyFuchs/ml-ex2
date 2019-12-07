import numpy as np
from DataOptions import getData, getDataX


class SVM:
    def __init__(self, trainx, trainy):
        self._trainx = trainx
        self._trainy = trainy
        self._eta = 0.0001
        self._lambda = 1.111
        self._w = None

    def train(self, epochs=100):
        x_data = getDataX(self._trainx)
        y_data = getData(self._trainy)
        w = np.zeros([3, x_data.shape[1]])
        lambda_eta = self._eta*self._lambda
        for _ in range(epochs):
            prev_w = w.copy()
            for x, y in zip(x_data, y_data):
                y_hat = np.argmax(np.dot(w, x))
                if int(y) != y_hat:
                    w[int(y), :] = (1-lambda_eta)*prev_w[int(y), :] + self._eta * x
                    w[y_hat, :] = (1-lambda_eta)*prev_w[y_hat, :] - self._eta * x
                    w[[i for i in range(w.shape[0]) if (i != y_hat and i != int(y))],:] *= (1-lambda_eta)
        self._w = w

    def predict(self, x):
        return np.argmax(np.dot(self._w, x))

