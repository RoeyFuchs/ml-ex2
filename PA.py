from Utils import *


class PA:
    def __init__(self, trainx, trainy):
        self._trainx = trainx
        self._trainy = trainy
        self._w = None

    def train(self, epochs=150):
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
                y = int(y)
                arg_list = np.argsort(-np.dot(w, x))
                if arg_list[0] == y:  # if y_hat == y, we will take another y_hat
                    y_hat = arg_list[1]
                else:
                    y_hat = arg_list[0]
                if hinge_loss(w, x, y, y_hat) > 0:  # check if need to update values
                    t = self.tau(w, x, y, y_hat)
                    w[y, :] = prev_w[y, :] + t * x
                    w[y_hat, :] = prev_w[y_hat, :] - t * x
            best_w = check_best(x_valid, y_valid, best_w, w)
        self._w = best_w

    def predict(self, x):
        return np.argmax(np.dot(self._w, x))

    def tau(self, w, x, y, y_hat):
        n = np.linalg.norm(x)
        n = n ** 2
        return hinge_loss(w, x, y, y_hat) / (2 * n)
