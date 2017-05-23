import numpy as np

def gradientDescent(x, y, theta, alpha=0.01, numIterations=1000 ):
    xTrans = np.transpose(x)
    m = y.shape
    costo = 1.0
    i = 0
    while i<numIterations and costo > 0.0001:
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)

    #    print("Iteration %d | Cost: %f" % (i, cost[0]))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
        i +=1
        costo = cost[0] 
        
    return theta
