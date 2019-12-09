import sys
from Utils import *
from Perceptron import Perceptron
from PA import PA
from SVM import SVM

def main():
    perc = Perceptron(sys.argv[1], sys.argv[2])
    perc.train()
    svm = SVM(sys.argv[1], sys.argv[2])
    svm.train()
    pa = PA(sys.argv[1], sys.argv[2])
    pa.train()

    result(perc,svm,pa,sys.argv[3])


def result(perceptron, svm, pa, file):
    q = get_data_x(file)
    for i in range(q.shape[0]):
        perceptron_yhat = perceptron.predict(q[i])
        svm_yhat = svm.predict(q[i])
        pa_yhat = pa.predict(q[i])
        print(f"perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}")

if __name__ == "__main__":
    main()
