import numpy , random , math
from scipy.optimize import minimize
import matplotlib . pyplot as plt

N = 3 # Number of training samples
C = 2 # Slack value
B = [(0, C) for b in range(N)] # Boundary for alpha values
T = [1, -1, 1] # T represents class 1 or -1

start = numpy.zeros(N) # Starting alpha vector, initialized to zeroes. 

# Check condition for running the SVM, Eq. 10
def zerofun(alpha_vector): 
    sum = 0
    for i in range(len(alpha_vector)):
        sum += (numpy.dot(alpha_vector[i], T[i]))

    return sum


constraints = {"type": "eq", "fun":zerofun}