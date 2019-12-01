import sys
import threading
from DataOptions import *
import numpy as np
from scipy.stats.stats import pearsonr

import datetime

from Perceptron import Perceptron
from PA import PA


def main():
    # findCor(getDataX(sys.argv[1]), getData(sys.argv[2]))

    print(datetime.datetime.now())
    A = PA(sys.argv[1], sys.argv[2])

    perceptron_thread = threading.Thread(target=A.train)
    perceptron_thread.start()
    perceptron_thread.join()
    print(datetime.datetime.now())
    test(A, sys.argv[1], sys.argv[2])
    '''
    s = 50
    ans = np.zeros([50, 2])
    i = 0
    while (s < 2500):
        A.train(s)
        ans[i] = s, test(A, sys.argv[1], sys.argv[2])
        i += 1
        s += 100
    '''


def test(alg, q, ans):
    q = getDataX(q)
    ans = getData(ans)
    err = 0
    for i in range(q.shape[0]):
        if alg.predict(q[i]) != int(ans[i]):
            # print("predict = " + str(alg.predict(q[i]))+ ", ans[i] = "+ str(ans[i]))
            err += 1
            # print("error")
    print("ERROR = " + str(err) + "/" + str(q.shape[0]) + " = " + str(err / q.shape[0]))
    return err


def findCor(a, b):
    for i in range(a.shape[1]):
        print(str(i) + " = " + str(pearsonr(a[:, i], b)))


if __name__ == "__main__":
    main()
