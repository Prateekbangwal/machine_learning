import pandas as pd
import numpy as np


class AdalineGd(object):
    """
    Adaptive Linear Neuron Classifier.

    :parameter:
    ___________
    eta : float Learning rate (between 0.0 and 1.0)
    n_iter : int passes over the training dataset
    random_state : random number generator seed for random weight initialization

    :attributes:
    _________
    w_ : 1d-array
    weight after fitting
    cost_: list
    sum of squares cost function value in each epoch
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data

        :param X: {array-like} , shape = [n_examples, n_features]
        :param y: {array-like}, shape = [n_features]
        :return: self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc= 0.0, scale= 0.01, size= 1+ X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
            return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


