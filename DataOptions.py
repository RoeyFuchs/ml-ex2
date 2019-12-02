import numpy as np


def replaceNominal(arr):
    arr[arr == "M"] = -0.4
    arr[arr == "F"] = 0.8
    arr[arr == "I"] = -2
    return arr


def getData(file_name):
    a = np.genfromtxt(file_name, delimiter=',', dtype='str')
    a = replaceNominal(a)
    a = a.astype(float)
    return a


def getDataX(file_name):
    a = np.genfromtxt(file_name, delimiter=',', dtype='str')
    a = replaceNominal(a)
    a = a.astype(float)

    for i in range(a.shape[1]):  # normalize
        a[:, i] = (a[:, i] - np.mean(a[:,i]))/np.std(a[:, i])

    return a
