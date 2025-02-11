import numpy , random , math
from scipy.optimize import minimize
import matplotlib . pyplot as plt

N = 0 # Number of training samples
C = 2 # Slack value
B = [(0, C) for b in range(N)]

start = numpy.zeros(N)

print("Hello")
input()
