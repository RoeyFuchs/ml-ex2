from Utils import *


class Perceptron:
    def __init__(self, trainx, trainy):
        self._trainx = trainx
        self._trainy = trainy
        self._eta = 1
        self._w = None

    def train(self, epochs=350):
        x_data = get_data_x(self._trainx)
        y_data = get_data(self._trainy)
        x_data, y_data = shuffle_data(x_data, y_data)
        x_data, y_data, x_valid, y_valid = split(x_data, y_data)
        w = np.zeros([3, x_data.shape[1]])
        best_w = np.zeros([3, x_data.shape[1]])
        for _ in range(epochs):
            prev_w = w.copy()
            x_data, y_data = shuffle_data(x_data, y_data)
            for x, y in zip(x_data, y_data):
                y_hat = np.argmax(np.dot(w, x))
                if int(y) != y_hat:
                    w[int(y), :] = prev_w[int(y), :] + self._eta * x
                    w[y_hat, :] = prev_w[y_hat, :] - self._eta * x
            best_w = check_best(x_valid, y_valid, best_w, w)
        self._w = best_w

    def predict(self, x):
        return np.argmax(np.dot(self._w, x))


