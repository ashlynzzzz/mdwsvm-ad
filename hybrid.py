import numpy as np
from mdwsvm import mdwsvm
from one_class_svm import one_class_svm

# def hybrid(X_train, y_train, X_test, v, W, C, alp = 0.5, K = lambda x, y: np.exp(-np.linalg.norm(x - y)**2/2)):
def hybrid(X_train, y_train, X_test, v, W, C, K, alp = 0.5):
    '''
    This function aims to do hybrid version of MDWSVM and 1-class SVM for comparison

    Input: 
        X_train:    traing data matrix of interest (d by n) where n is the number of training samples and d is the number of features
        y_train:    training labels
        X_test:     test data matrix (d by m)

        for one_class_svm:
        v:      hyperparameter within (0,1)
        K:      kernel, the default is a radius-based Gaussian kernel

        for mdwsvm:  
        W:      vertices matrix for all classes: (K-1) by K
        C:      constraint hyperparameter on B which is the coefficient in f
        alp:    weighting parameter, the default value of 0.5
        
    Output: 
        y_train:    labels for training data
        y_test:     labels for test data
    '''
    #TODO: training
    #Initialize y_test
    y_test = np.zeros(X_test.shape[1], dtype=int)
    #first use 1-class svm to find anomalous data
    model1 = one_class_svm(X_test, v, K)
    y_anomaly = model1.predict()
    y_test[y_anomaly == -1] = -1 # Use -1 to denote the label of anomalous data
    # train mdwsvm
    model2 = mdwsvm(X_train, y_train, W, C, alp)
    
    #second perform mdwsvm on the rest data
    index_normal = np.where(y_anomaly != -1)[0]
    X_test_normal = X_test[:,index_normal] # extract non_anomaly samples
    y_normal = model2.predict(X_test_normal) 
    y_test[index_normal] = y_normal # should rearrange y_anomaly and y_normal to become one array

    return y_test