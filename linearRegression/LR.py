'''
https://towardsdatascience.com/basic-linear-regression-algorithm-in-python-for-beginners-c519a808b5f8
'''
import numpy as np
import pandas as pd
import os
# Scientific and vector computation for python
import numpy as np

def cost_calc(theta, X, y):
    m=len(X)
    return (1/2*m) * np.sum((hypothesis(theta, X) - y)**2)

def hypothesis(theta, X):
    return theta[0] + theta[1]*X

def gradient_descent(theta, X, y, epoch, alpha):
    cost = []
    i = 0
    m=len(X)
    while i < epoch:
        hx = hypothesis(theta, X)
        theta[0] -= alpha*(sum(hx-y)/m)
        theta[1] -= (alpha * np.sum((hx - y) * X))/m
        cost.append(cost_calc(theta, X, y))
        i += 1
    print("Theta:: ",theta)
    return theta, cost


def predict(theta, X, y, epoch, alpha):
    theta, cost = gradient_descent(theta, X, y, epoch, alpha)
    return hypothesis(theta, X), cost, theta
# %matplotlib inline
import matplotlib.pyplot as plt


def main():
    df = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
    m = len(df)
    theta = [0,0]
    y_predict, cost, theta = predict(theta, df[:, 0], df[:, 1], 1500, 0.01)
    plt.figure()
    plt.scatter(df[:, 0], df[:, 1], label = 'Original y')
    plt.scatter(df[:, 0], y_predict, label = 'predicted y')
    plt.legend(loc = "upper left")
    plt.xlabel("input feature")
    plt.ylabel("Original and Predicted Output")
    plt.show()
    plt.figure()
    plt.scatter(df[:, 0], df[:, 1])
    plt.show()

if __name__ == '__main__':
    main()