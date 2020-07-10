

import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# library written for this exercise providing additional functions for assignment submission, and others
# import utils 


def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data 
    points and gives the figure axes labels of population and profit.
    
    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.
    
    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.    
    
    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You 
    can also set the marker edge color using the `mec` property.
    """
    fig = pyplot.figure()  # open a new figure
    
    # ====================== YOUR CODE HERE ======================= 
    pyplot.plot(x, y, 'bo', ms=10, mec='k')
    pyplot.ylabel('Profitssssssssssssssssssssssssssss in $10,000')
    pyplot.xlabel('Population of City in 101,0s')
    # pyplot.show()

def computeCost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already 
        appended to the features so we have n+1 columns.
    
    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).
    
    theta : array_like
        The parameters for the regression function. This is a vector of 
        shape (n+1, ).
    
    Returns
    -------
    J : float
        The value of the regression cost function.
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. 
    You should set J to the cost.
    """
    
    # initialize some useful values
    m = y.size  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
    
    # ====================== YOUR CODE HERE =====================
    h = np.dot(X, theta)
    
    J = (1/(2 * m)) * np.sum(np.square(h - y))
    
    # ===========================================================
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.
    
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).
    
    y : arra_like
        Value at given features. A vector of shape (m, ).
    
    theta : array_like
        Initial values for the linear regression parameters. 
        A vector of shape (n+1, ).
    
    alpha : float
        The learning rate.
    
    num_iters : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)

        # =====================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history


def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    # ================================================================
    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    h = np.dot(X, theta)
    
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    
    # ==================================================================
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================

        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history
    
def main():
    # Read comma separated data
    data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m = y.size  # number of training examples
    A = np.eye(2)
    plotData(X, y)
    X = np.stack([np.ones(m), X], axis=1)
    J = computeCost(X, y, theta=np.array([0.0, 0.0]))
    print('With theta = [0, 0] \nCost computed = %.2f' % J)
    print('Expected cost value (approximately) 32.07\n')
    # further testing of the cost function
    J = computeCost(X, y, theta=np.array([-1, 2]))
    print('With theta = [-1, 2]\nCost computed = %.2f' % J)
    print('Expected cost value (approximately) 54.24')
    # initialize fitting parameters
    theta = np.zeros(2)

    # some gradient descent settings
    iterations = 1500
    alpha = 0.01

    theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
    print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
    print('Expected theta values (approximately): [-3.6303, 1.1664]')
    # plot the linear fit
    plotData(X[:, 1], y)
    pyplot.plot(X[:, 1], np.dot(X, theta), '-')
    pyplot.legend(['Training data', 'Linear regression'])
    # pyplot.show()
    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5], theta)
    print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))
    predict2 = np.dot([1, 7], theta)
    print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))
    # grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # Fill out J_vals
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i, j] = computeCost(X, y, [theta0, theta1])
            
    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T

    # surface plot
    fig = pyplot.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    pyplot.title('Surface')

    # contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    ax = pyplot.subplot(122)
    pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
    pyplot.title('Contour, showing minimum')
    pyplot.show()
    #####################################################
    # Load data
    data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.size

    # print out some data points
    #print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
    #print('-'*26)
    #for i in range(10):
    #    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
    # call featureNormalize on the loaded data
    X_norm, mu, sigma = featureNormalize(X)

    print('Computed mean:', mu)
    print('Computed standard deviation:', sigma)
    X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

    """
    Instructions
    ------------
    We have provided you with the following starter code that runs
    gradient descent with a particular learning rate (alpha). 

    Your task is to first make sure that your functions - `computeCost`
    and `gradientDescent` already work with  this starter code and
    support multiple variables.

    After that, try running gradient descent with different values of
    alpha and see which one gives you the best result.

    Finally, you should complete the code at the end to predict the price
    of a 1650 sq-ft, 3 br house.

    Hint
    ----
    At prediction, make sure you do the same feature normalization.
    """
    # Choose some alpha value - change this
    alpha = 0.1
    num_iters = 400

    # init theta and run gradient descent
    theta = np.zeros(3)
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')

    # Display the gradient descent's result
    print('theta computed from gradient descent: {:s}'.format(str(theta)))

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ======================= YOUR CODE HERE ===========================
    # Recall that the first column of X is all-ones. 
    # Thus, it does not need to be normalized.

    X_array = [1, 1650, 3]
    X_array[1:3] = (X_array[1:3] - mu) / sigma
    price = np.dot(X_array, theta)   # You should change this

    # ===================================================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))

if __name__ == '__main__':
    main()