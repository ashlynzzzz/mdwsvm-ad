import numpy as np
import cvxpy as cp

class one_class_svm():
    '''
    This class aims to do one-class svm and label anomalous vectors as +1

    Variables: 
        X:  training data matrix of interest (d by n) where n is the number of training samples and p is the number of features
        y:  training labels
        v:  hyperparameter within (0,1)
        K:  kernel, the default is a radius-based Gaussian kernel
    '''
    def __init__(self, X, y, v, K = lambda x, y: np.exp(-np.linalg.norm(x - y)**2/2)):
        self.X = X
        self.y = y
        self.v = v
        self.K = K
        self.alpha, self.rho = self.train()

    def train(self):
        '''
        Output:
        alpha, rho:     for f(x) = B.T@x + beta
        '''
        return alpha, rho

    def test(data):
        '''
        Input:
        data:   data for evaluation
        
        Output:
        y:      predicting labels for X
        '''
        return y
