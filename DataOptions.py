import numpy as np


def replaceNominal(arr):
    arr[arr == "M"] = 0.1
    arr[arr == "F"] = 0.4
    arr[arr == "I"] = 0.8
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
    '''b = np.zeros([a.shape[0], 3])
    b[:,0] = a[:, 1]
    b[:,1] = a[:, 2]
    b[:,2] = a[:, 7]'''
    return a

