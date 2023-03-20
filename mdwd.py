import numpy as np
import cvxpy as cp

class mdwd():
    '''
    This class aims to reimplement MDWD to do multiclass classification

    Variables: 
        X:  training data matrix of interest (d by n) where n is the number of training samples and p is the number of features
        y:  training labels
        W:  vertices matrix for all classes
        C:  constraint hyperparameter on B which is the coefficient in f
    '''
    def __init__(self, X, y, W, C):
        self.X = X
        self.y = y
        self.W = W
        self.C = C
        self.B, self.beta = self.train()

    def train(self):
        '''
        Output:
        B, beta:    for f(x) = B.T@x + beta
        '''
        return B, beta

    def test(data):
        '''
        Input:
        data:   data for evaluation

        Output:
        y:      predicting labels for data
        '''
        return y
