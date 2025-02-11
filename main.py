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

def get_SV(alpha):
    SV = []
    alpha_nonzero = []
    T_nonzero = []
    for i in range(alpha.size):
        if alpha[i] >= math.pow(10, -5): # threshold since in reality the values in alpha won't be exactly 0, just very small.
            alpha_nonzero.append(alpha[i])
            SV.append(X[i])
            T_nonzero.append(T[i])

    return numpy.array(alpha_nonzero), numpy.array(SV), numpy.array(T_nonzero)



ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']
alpha_nonzero, SV, T_nonzero = get_SV(alpha)
 
def ind(s): # s is the point we want to classify.
    sum = 0 
    for i in range(alpha_nonzero.size):
        sum += alpha_nonzero[i] * T_nonzero[i] * kernel_linear(s, SV[i]) - b

input()
