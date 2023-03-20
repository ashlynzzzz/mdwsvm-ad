import numpy as np
import cvxpy as cp

class mdwsvm():
    '''
    This class aims to reimplement MDWSVM to do multiclass classification

    Variables: 
        X:      training data matrix of interest (d by n) where n is the number of training samples and p is the number of features
        y:      training labels
        W:      vertices matrix for all classes
        C:      constraint hyperparameter on B which is the coefficient in f
        alp:    weighting parameter, the default value of 0.5
    '''
    def __init__(self, X, y, W, C, alp = 0.5):
        self.X = X
        self.y = y
        self.W = W
        self.C = C
        self.alpha = alp
        self.B, self.beta = self.train()

    def train(self):
        '''
        Output:
        B, beta:     for f(x) = B.T@x + beta
        '''
        # intermediate: r, ita, xi
        return B, beta

    def test(data):
        '''
        Input:
        data:   data for evaluation

        Output:
        y:      predicting labels for data
        '''
        return y