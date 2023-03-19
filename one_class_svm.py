import numpy as np
import cvxpy as cp

def one_class_svm(X, v, K = lambda x, y: np.exp(-np.linalg.norm(x - y)**2/2)):
    '''
    This function aims to do one-class svm and label anomalous vectors as +1

    Input: 
        X:  Data Matrix of interest (p by n) where n is the number of training samples and p is the number of features
        v:  hyperparameter within (0,1)
        K:  kernel, the default is a radius-based Gaussian kernel

    Output: 
        y:  labels predicting anomalous or not for training data
        f:  classifier to predict labels for future data
    '''

    n = X.shape[1]
    # TODO: minimize the dual problem
    # solution: alpha


    # classification
    rho = 0 #TODO #given by alpha and K
    f = lambda x: 0 #TODO
    y = 0 #TODO
    return y, f