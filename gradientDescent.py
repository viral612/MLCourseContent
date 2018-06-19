## Here we are writing a simple Gradient Descent Algorithm
## First we will generate some training examples assuming  a model
## Next step we will train a model on the examples using Gradient Descent 

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



##gradient descent Algorithm
## Regression problem - squared loss without regularization

def loss(points, w):
    lossVal = sum((np.dot(w,x)-y)**2 for x,y in points)/numExamples
    return lossVal

def gradientLoss( points, w):
    return sum( 2*(np.dot(w,x)-y)*x for x,y in points)/numExamples

def gradientDescent(points, d):
    w = np.zeros(d)
    iterations = 1000
    stepSize = 0.01
    for i in xrange(iterations):
        lossVal = loss(points, w)
        gradient = gradientLoss(points, w)
        w = w - stepSize * gradient
        print "Iteration %s Loss is %s and gradient is %s and w is %s" % ( i, lossVal, gradient, w )
    return w

w = gradientDescent(examples, dimensions)
print w
