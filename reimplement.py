import numpy as np
from mdwsvm import mdwsvm
from msvm import msvm
from mdwd import mdwd
from metric import within_class_error

prob = [1/3, 1/2, 2/3] # varying probabilities for class 1
dim = [10, 100, 500, 1000]
err = np.zeros((4,3,3)) # 4-dim, 3 methods, 3 prob
for i in range(3):
    p = prob[i]
    for j in range(4):
        d = dim[j]
        #TODO: generate 3-class simulation data
        X_train = 0 # d-dim training dataset
        y_train = 0 # labels for training data
        X_test = 0 # d-dim test data
        y_test = 0 # labels for test data

        # MDWSVM
        #TODO: use cross validation to choose C based on X_train
        
        #TODO: use the optimal C to train X_train and get the final classifier

        #TODO: perform the final classifier on X_test
        err[j,0,i] = 0 # store the value for error

        # MSVM

        err[j,1,i] = 0

        # MDWD

        err[j,2,i] = 0

#TODO: plot