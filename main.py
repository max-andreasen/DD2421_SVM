import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from data_gen_and_plot import generate_data, plot_classes

numpy.random.seed(100)
N, X, T, classA, classB = generate_data()
C = 2 # Slack value
B = [(0, C) for b in range(N)]
start = numpy.zeros(N)

# change the string below:
kernel_type = "RBF"

def kernel(x_i, x_j):
    match kernel_type:
        case "linear":
            return kernel_linear(x_i, x_j)
        case "polynomial":
            return kernel_polynomial(x_i, x_j, 2)
        case "RBF":
            return kernel_RBF(x_i, x_j, 0.75)
        case _:
            print("Incorrect kernel type!")

def kernel_linear(x_i, x_j): 
    return numpy.dot(x_i.T, x_j)

def kernel_polynomial(x_i, x_j, p):
    return math.pow( numpy.dot(x_i.T, x_j) +1, p )

def kernel_RBF(x_i, x_j, sigma):
    return math.pow(math.e, -(numpy.linalg.norm(x_i - x_j)**2)/(2*sigma**2))

def compute_P():
    P = numpy.zeros((N, N))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            P[i,j] = T[i] * T[j] * kernel(X[i], X[j])
    return P

P = compute_P() # Precomputed t[i]*t[j] * kernel(X[i], X[j] for efficiency.

def objective(alpha):
    sum1 = sum1 = 1/2 * numpy.dot(alpha.T, numpy.dot(P, alpha))
    sum2 = numpy.sum(alpha)
    return sum1 - sum2

# Check condition for running the SVM, Eq. 10
def zerofun(alpha_vector): 
    sum = 0
    for i in range(len(alpha_vector)):
        sum += (alpha_vector[i] * T[i]) # could convert to numpy.dot()

    return sum


constraints = {"type": "eq", "fun":zerofun}

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


# Calculates the b value
def calc_b(s, t_s, alpha_vector):
    sum = 0
    for i in range(len(alpha_vector)):
        sum += (alpha_vector[i]*T[i]*kernel(s, X[i]))
    
    return sum - t_s

ret = minimize(objective, start, bounds=B, constraints=constraints)
alpha = ret['x']
alpha_nonzero, SV, T_nonzero = get_SV(alpha)
b = calc_b(SV[0], T[0] ,alpha) # call with any support vector.

def ind(s): 
    sum = 0 
    for i in range(alpha_nonzero.size):
        sum += alpha_nonzero[i] * T_nonzero[i] * kernel(s, SV[i])
    return sum - b



plot_classes(classA, classB, ind)
