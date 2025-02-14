import numpy
import random
import matplotlib.pyplot as plt

def generate_data():
    classA = numpy.concatenate((
        numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
        numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
    ))

    classB = numpy.random.randn(20, 2) * 0.2 + [0, -0.5]

    inputs = numpy.concatenate((classA, classB))

    targets = numpy.concatenate((
        numpy.ones(classA.shape[0]),
        -numpy.ones(classB.shape[0])
    ))

    N = inputs.shape[0]  # Number of rows (samples)

    # Randomly reorder the data (from sample code)
    permute = list(range(N))
    random.shuffle(permute)

    inputs = inputs[permute, :]
    targets = targets[permute]

    return N, inputs, targets, classA, classB


def plot_classes(classA, classB, ind):
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
            'b. ')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
            'r. ')
    plt.axis('equal')
    plt.savefig('svmplot.pdf')

    # plot decision boundary
    
    xgrid=numpy.linspace(-5, 5)
    ygrid=numpy.linspace(-4, 4)
    grid=numpy.array([[ind(numpy.array([x, y])) for x in xgrid] for y in ygrid ] )
    contour = plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0) , colors =('red' , 'black', 'blue'), linewidths=(1, 3, 1))
    plt.title("Decicion boundary around data points")
    #plt.clabel(contour)
    plt.legend(['Class -1', 'Class +1'], loc='best')
    plt.show()
    
    