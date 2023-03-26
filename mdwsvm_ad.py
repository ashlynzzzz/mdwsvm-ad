import numpy as np
import cvxpy as cp

class mdwsvm_ad:
    '''
    This class aims to reimplement MDWSVM to do multiclass classification

    Variables: 
            X:      training data matrix of interest (d by n) where n is the number of training samples and p is the number of features
            y:      training labels
            W:      vertices matrix for all classes: K by (K+1)
            C:      constraint hyperparameter on B which is the coefficient in f
            alp:    weighting parameter, the default value of 0.5
            v:      hyperparameter within (0,1)
            K:      kernel, the default is a radius-based Gaussian kernel
    '''

    def __init__(self, X, y, W, C, v, alp = 0.5, K = lambda x, y: np.exp(-np.linalg.norm(x - y)**2/2)):
        self.X = X
        self.y = y
        self.W = W
        self.C = C
        self.alp = alp
        self.v = v
        self.K = K
        self.B, self.beta = self.fit()

    def fit(self):
        # TODO

        return d, e        

    def predict(self, data):
        '''
        Input:
            data:  data matrix for prediction (d by n)

        Output:
            y:  predicted labels for data
        '''
        # TODO
        return y