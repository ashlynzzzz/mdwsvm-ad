import numpy as np
import cvxpy as cp

def mdwsvm_ad(X, W, phi, v, alp = 0.5):
    '''
    This function aims to do MDWSVM-AD that can detect anomalous data in the test data

    Input: 
        X:      Data Matrix of interest (d by n) where n is the number of training samples and p is the number of features
        W:      vertices matrix for all classes
        K:      kernel, the default is a radius-based Gaussian kernel
        alp:    weighting parameter, the default value of 0.5
        
    Output: 
        y:  labels for training data
        classifier: argmax_j <f(x), W_j> MDWSVM-AD classifier for predicting future data
    '''
    #TODO: minimization -- next week
    # input: X, W, phi, v, alp
    # intermediate: r, ita, xi
    # output: B, beta_0, beta_d

    # classification
    f = lambda x: 0 #TODO may use in classifier
    classifier = lambda x: 0 #TODO
    y = classifier(X)
    return y, classifier