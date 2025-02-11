import numpy , random , math
from scipy.optimize import minimize
import matplotlib . pyplot as plt


def kernel_linear(x_i, x_j): 
    return numpy.dot(x_i.T, x_j)

N = 0 # Number of training samples
C = 2 # Slack value
B = [(0, C) for b in range(N)]

start = numpy.zeros(N)

print(kernel(numpy.array([1, 2]), numpy.array([3, 4])))
input()
