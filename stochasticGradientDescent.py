## Here we are writing a simple Stochastic Gradient Descent Algorithm
## First we will generate some training examples assuming  a model
## Next step we will train a model on the examples using Stochastic Gradient Descent Algorithm

import numpy as np

## generate training examples
goldenWeight = np.array([2,3, 4])
numExamples = 1000
dimensions = len(goldenWeight)
def generateExample():
    x = np.random.randn(dimensions)
    y = np.dot(x, goldenWeight) + np.random.randn()
    #print (x, y)
    return (x,y)

examples = [ generateExample() for _ in xrange(numExamples) ]

##Stochastic Gradient descent Algorithm
## Regression problem - squared loss without regularization


def stochasticLoss( w, i):
    x, y = examples[i]
    return ((np.dot(x, w) - y)*x)**2 

def stochasticGradient(w, i):
    x, y = examples[i]
    return ((np.dot(x, w) - y)*x)*2


def stochasticGradientDescent(examples, d, lossF, gradientF):
    w = np.zeros(d)
    iterations = 1000
    stepSize = 0.05
    for i in xrange(iterations):
        for j in xrange(len(examples)):
            lossVal = lossF(w, j)
            gradient = gradientF(w, j)
            w = w - stepSize * gradient
        #print "Iteration %s Loss is %s and gradient is %s and w is %s" % ( i, lossVal, gradient, w )
        print "Iteration %s  and w is %s" % ( i, w )
    return w

w = stochasticGradientDescent(examples, dimensions, stochasticLoss, stochasticGradient)
print w

