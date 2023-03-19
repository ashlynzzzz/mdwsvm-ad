import numpy as np
import cvxpy as cp

def mdwd(X, W, C):
    '''
    This function aims to reimplement MDWD to do multiclass classification

    Input: 
        X:  Data Matrix of interest (d by n) where n is the number of training samples and p is the number of features
        W:  vertices matrix for all classes
        C:  constraint hyperparameter on B which is the coefficient in f
        
    Output: 
        y:  labels for training data
        f:  argmax_j <f0(x), W_j> MSVM classifier for predicting future data where f0(x) = x * B + beta_d
    '''
    #TODO: minimization
    # input: X, W, C
    # intermediate: r, ita
    # output: B, beta_0

    # classification
    f0 = lambda x: 0 #TODO
    y = f0(X)
    return y, f0