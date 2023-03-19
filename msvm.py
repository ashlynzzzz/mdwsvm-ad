import numpy as np
import cvxpy as cp

def msvm(X, W, C):
    '''
    This function aims to reimplement MSVM to do multiclass classification

    Input: 
        X:  ata Matrix of interest (p by n) where n is the number of training samples and p is the number of features
        W:  vertices matrix for all classes
        C:  constraint hyperparameter on B which is the coefficient in f
        
    Output: 
        y:  labels for training data
        f:  argmax_j <f(x), W_j> MSVM classifier for predicting future data where f(x) = x * B + beta
    '''
    #TODO: minimization
    # input: X, W, C
    # intermediate: xi
    # output: B, beta

    # classification
    f = lambda x: 0 #TODO
    y = f(X)
    return y, f