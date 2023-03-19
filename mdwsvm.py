import numpy as np
import cvxpy as cp

def mdwsvm(X, W, C, alp = 0.5):
    '''
    This function aims to reimplement MDWSVM to do multiclass classification

    Input: 
        X:      Data Matrix of interest (d by n) where n is the number of training samples and p is the number of features
        W:      vertices matrix for all classes
        C:      constraint hyperparameter on B which is the coefficient in f
        alp:    weighting parameter, the default value of 0.5
        
    Output: 
        y:  labels for training data
        f:  f(x) = x * B + beta_0 AMSVM decision boundary
        f0: f0(x) = x * B + beta_d DWD decision boundary
        classifier: argmax_j <f(x), W_j> MDWSVM classifier for predicting future data
    '''
    #TODO: minimization
    # input: X, W, C, alp
    # intermediate: r, ita, xi
    # output: B, beta_0, beta_d

    # classification
    f = lambda x: 0 #TODO
    f0 = lambda x: 0 #TODO
    classifier = lambda x: 0 #TODO
    y = classifier(X)
    return y, f, f0, classifier