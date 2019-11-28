import numpy as np


def replaceNominal(arr):
    arr[arr == "M"] = 1
    arr[arr == "F"] = 4
    arr[arr == "I"] = 8
    return arr


def getData(file_name):
    a = np.genfromtxt(file_name, delimiter=',', dtype='str')
    a = replaceNominal(a)
    a = a.astype(float)
    return a
