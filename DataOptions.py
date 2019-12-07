import numpy as np


def replaceNominal(arr):
    arr[arr == "M"] = 0.4
    arr[arr == "F"] = 0.8
    arr[arr == "I"] = 2
    return arr


def changeSign(arr): # have to do it because of the u2
    arr[arr[:, 0] == 0.4, 0] = -0.4
    arr[arr[:, 0] == 2, 0] = -2
    return arr


def getData(file_name):
    a = np.genfromtxt(file_name, delimiter=',', dtype='str')
    a = replaceNominal(a)  # change M,F,I to numbers
    a = a.astype(np.float)
    return a


def getDataX(file_name):
    a = getData(file_name)
    a = changeSign(a)
    for i in range(a.shape[1]):  # normalize - Z-Score
        a[:, i] = (a[:, i] - np.mean(a[:, i])) / np.std(a[:, i])
    return a
