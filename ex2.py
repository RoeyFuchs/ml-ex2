import sys
import threading
from DataOptions import *
import numpy as np

import datetime

from Perceptron import Perceptron


def main():
    print(datetime.datetime.now())
    A = Perceptron(sys.argv[1], sys.argv[2])

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
    q = getData(q)
    ans = getData(ans)
    err = 0
    for i in range(q.shape[0]):
        if alg.predict(q[i]) != int(ans[i]):
            # print("predict = " + str(alg.predict(q[i]))+ ", ans[i] = "+ str(ans[i]))
            err += 1
            # print("error")
    print("ERROR = " + str(err) + "/" + str(q.shape[0]) + " = " + str(err / q.shape[0]))
    return err


if __name__ == "__main__":
    main()



