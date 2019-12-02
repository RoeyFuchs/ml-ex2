import sys
from DataOptions import *
import datetime

from Perceptron import Perceptron
from PA import PA
from SVM import SVM


def main():
    print(datetime.datetime.now())

    A = Perceptron(sys.argv[1], sys.argv[2])
    A.train()


    C = PA(sys.argv[1], sys.argv[2])
    C.train()



    B = SVM(sys.argv[1], sys.argv[2])
    B.train()
    result(A,B,C,sys.argv[3])

    print(datetime.datetime.now())

    '''print("perceptron test: ")
    test(A, sys.argv[1], sys.argv[2])

    print("SVM test: ")
    test(B, sys.argv[1], sys.argv[2])
    print("PA test: ")
    test(C, sys.argv[3], sys.argv[2])'''


def result(perceptron, svm, pa, file):
    q = getDataX(file)
    for i in range(q.shape[0]):
        perceptron_yhat = perceptron.predict(q[i])
        svm_yhat = svm.predict(q[i])
        pa_yhat = pa.predict(q[i])
        print(f"perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}")



def test(alg, q, ans):
    ans = getData(ans)
    err = 0
    for i in range(q.shape[0]):
        if alg.predict(q[i]) != int(ans[i]):
            # print("predict = " + str(alg.predict(q[i]))+ ", ans[i] = "+ str(ans[i]))
            err += 1
            # print("error")
    #print("ERROR = " + str(err) + "/" + str(q.shape[0]) + " = " + str(err / q.shape[0]))
    success = (1-(err / q.shape[0]))*100
    print("success = " + str((1-(err / q.shape[0]))*100) + "%")
    return success


if __name__ == "__main__":
    main()
