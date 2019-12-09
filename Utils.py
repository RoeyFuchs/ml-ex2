import numpy as np


def replace_nominal(arr):
    arr[arr == "M"] = 0.4
    arr[arr == "F"] = 0.8
    arr[arr == "I"] = 2
    return arr


def change_sign(arr):  # have to do it due to u2 server
    arr[arr[:, 0] == 0.4, 0] = -0.4
    arr[arr[:, 0] == 2, 0] = -2
    return arr


def get_data(file_name):
    a = np.genfromtxt(file_name, delimiter=',', dtype='str')
    a = replace_nominal(a)  # change M,F,I to numbers
    a = a.astype(np.float)
    return a


def get_data_x(file_name):
    a = get_data(file_name)
    a = change_sign(a)
    for i in range(a.shape[1]):  # normalize - Z-Score
        a[:, i] = (a[:, i] - np.mean(a[:, i])) / np.std(a[:, i])
    return a


def shuffle_data(x_data, y_data):
    p = np.random.permutation(x_data.shape[0])
    x_data = x_data[p, :]
    y_data = y_data[p]
    return x_data, y_data


# split the data 80-20 - data set and validation set
def split(x_data, y_data):
    num = int(x_data.shape[0] * 0.2)
    x_valid = x_data[0:num, :]
    x_data = x_data[num:, :]
    y_valid = y_data[0:num]
    y_data = y_data[num:]
    return x_data, y_data, x_valid, y_valid


# hinge loss function for multi-class
def hinge_loss(w, x, y, y_hat):
    temp = (-np.dot(w[y, :], x)) + np.dot(w[y_hat, :], x)
    return max(0, 1 + temp)


# check, for 2 w's, who's better for the validation set
def check_best(x_valid, y_valid, best_w, w):
    # for w
    w_err = 0
    for x, y in zip(x_valid, y_valid):
        if y != np.argmax(np.dot(w, x)):
            w_err += 1
    w_best_err = 0
    # for w_best
    for x, y in zip(x_valid, y_valid):
        if y != np.argmax(np.dot(best_w, x)):
            w_best_err += 1
    if w_err <= w_best_err:
        return w.copy()
    return best_w
