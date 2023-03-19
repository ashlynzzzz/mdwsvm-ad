import numpy as np
from mdwsvm import mdwsvm
from one_class_svm import one_class_svm

def hybrid(X, v, W, C, alp = 0.5, K = lambda x, y: np.exp(-np.linalg.norm(x - y)**2/2)):
    '''
    This function aims to do hybrid version of MDWSVM and 1-class SVM for comparison

    Input: 
        X:      Data Matrix of interest (d by n) where n is the number of training samples and p is the number of features
        for one_class_svm:
        v:      hyperparameter within (0,1)
        K:      kernel, the default is a radius-based Gaussian kernel
        for mdwsvm:
        W:      vertices matrix for all classes
        C:      constraint hyperparameter on B which is the coefficient in f
        alp:    weighting parameter, the default value of 0.5
        
    Output: 
        y_train:    labels for training data
        y_test:     labels for test data
    '''
    #TODO: training
    #first use 1-class svm to find anomalous data
    #second perform mdwsvm on the rest data

    #TODO: testing
    # get test label
    return y_train, y_test