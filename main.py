import numpy , random , math
from scipy.optimize import minimize
import matplotlib . pyplot as plt

N = 5 # temporary, nr of datapoints
T = numpy.zeros(N) # classes (value -1 or 1)
X = numpy.zeros((N, N)) # Vector of vectors
C = 2 # Slack value
B = [(0, C) for b in range(N)]
start = numpy.zeros(N)


def compute_P():
    P = numpy.zeros((N, N))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            P[i,j] = T[i] * T[j] * kernel_linear(X[i], X[j])
    return P

P = compute_P() # Precomputed t[i]*t[j] * kernel_linear(X[i], X[j] for efficiency.


def kernel_linear(x_i, x_j): 
    return numpy.dot(x_i.T, x_j)

def objective(alpha):
    sum1 = sum1 = 1/2 * numpy.dot(alpha.T, numpy.dot(P, alpha))
    sum2 = numpy.sum(alpha)
    return sum1 - sum2



input()
