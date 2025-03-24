from ANFIS import GaussianMF, BellMF, SigmoidMF, TriangularMF

n_sets = 4
uod = [-1, 1]

def gaussian():
    mf = GaussianMF(n_sets, uod)

def bell():
    mf = BellMF(n_sets, uod)

def sigmoid():
    mf = SigmoidMF(n_sets, uod)

def triangular():
    mf = TriangularMF(n_sets, uod)

if __name__ == '__main__':
    gaussian()
    bell()
    sigmoid()
    triangular()
