import numpy as np
from mdwsvm import mdwsvm
from one_class_svm import one_class_svm

def hybrid(X_train, y_train, X_test, v, W, C, alp = 0.5, K = lambda x, y: np.exp(-np.linalg.norm(x - y)**2/2)):
    '''
    This function aims to do hybrid version of MDWSVM and 1-class SVM for comparison

    Input: 
        X_train:    traing data matrix of interest (d by n) where n is the number of training samples and p is the number of features
        y_train:    training labels
        X_test:     test data matrix

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
    m1 = one_class_svm(X_test, v, K)
    y_anomaly = m1.test()
    # train mdwsvm
    m2 = mdwsvm(X_train, y_train, W, C, alp)
    y_train = m2.test(X_train)
    
    #second perform mdwsvm on the rest data
    X_test_normal = 0 # extract non_anomaly samples
    y_normal = m2.test(X_test_normal) 
    y_test = 0 # should rearrange y_anomaly and y_normal to become one array

    return y_test