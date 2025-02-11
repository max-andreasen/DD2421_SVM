import numpy , random , math
from scipy.optimize import minimize
import matplotlib . pyplot as plt


t = [] # classes (value -1 or 1)
X = [] # Vector of vectors
P = [[]] # Precomputed t[i]*t[j] * kernel_linear(X[i], X[j] for efficiency.

def compute_P():
    P = [[]]
    return numpy.array(P)

def kernel_linear(x_i, x_j): 
    return numpy.dot(x_i.T, x_j)

def objective(alpha):
    sum1 = 0
    sum2 = 0
    for i in range(len(alpha)):
        sum2 = alpha[i]
        for j in range(len(alpha)):
            sum1 += alpha[i]*alpha[j]* P[i][j]

    sum1 /= 2
    return sum1 - sum2

    


    pass

N = 0 # Number of training samples
C = 2 # Slack value
B = [(0, C) for b in range(N)]

start = numpy.zeros(N)

print(kernel(numpy.array([1, 2]), numpy.array([3, 4])))
input()
