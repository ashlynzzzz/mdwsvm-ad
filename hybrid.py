import numpy as np
from mdwsvm import mdwsvm
from one_class_svm import one_class_svm

def hybrid(X):
    '''
    This function aims to do hybrid version of MDWSVM and 1-class SVM for comparison

    Input: 
        X:      Data Matrix of interest (p by n) where n is the number of training samples and p is the number of features
        
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