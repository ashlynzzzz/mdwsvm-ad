import numpy as np
from mdwsvm import mdwsvm
from msvm import msvm
from mdwd import mdwd
from vertices_generator import vertices
from metric import within_class_error
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def cross_validation(c_values, num_folds, size, w, X_train, X_test, y_test, y_train, method_obj):
    '''
    This function is to generate the best c value
    
    Input:
    c_values: list of c_values
    num_folds: number of folers
    size: size of training data
    w: vertices matrix
    X_train: training data
    y_train: training label
    X_test: testing data
    y_test: testing label
    method_obj: estimator object

    Output:
    error:  the error for the best c estimator using testing data
    '''
    
    folder_size = int(size / num_folds)
    # Loop over each value of c and perform cross-validation
    best_c = 0
    best_score = -1
    
    for c in c_values:
        scores = np.zeros(5)
        
        # Perform cross-validation and calculate the average score
        for i in range(num_folds):
            # Get testing set    
            testx = X_train[:, i*folder_size:(i+1)*folder_size]
            testy = y_train[i*folder_size:(i+1)*folder_size]
            # Get training set    
            trainx = np.hstack((X_train[:, 0:(i)*folder_size], X_train[:, (i+1)*folder_size:size]))
            trainy = np.hstack((y_train[0:(i)*folder_size], y_train[(i+1)*folder_size:size]))
            # Create method object
            method = method_obj(trainx, trainy, w, c)
            
            pred_y = method.predict(testx)
            score = 1 - within_class_error(y_ture = testy, y_pred = pred_y)
            scores[i] = score
            
        # Check if the current value of c is the best so far
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_c = c
            best_score = avg_score

    # use the optimal C to train X_train and get the final classifier
    method_666 = method_obj(X_train, y_train, w, best_c)
    
    # perform the final classifier on X_test
    pred_y = method_666.predict(X_test)
    error = within_class_error(y_ture = y_test, y_pred = pred_y)   # store the value for error
        
    return error
    

prob = [1/3, 1/2, 2/3] # varying probabilities for class 1
dim = [2, 10, 100, 500, 1000]
err = np.zeros((5,3,3)) # 5-dim, 3 methods, 3 prob

for i in range(3):
    p1 = prob[i]    # prob of class 1
    p2 = (1 - p1) / 2   # prob of class 2,3
    size = 300   # size of training data set
    size1 = int(300 * p1)    # size of class 1
    size2 = int(300 * p2)    # size of class 2,3
    sigma = 0.55
    
    for j in range(5):
        d = dim[j]
        
        # Define the centers of the three groups on the unit ball
        u1 = np.concatenate(([1], np.zeros(d-1)))
        u2 = np.concatenate(([0], [1], np.zeros(d-2)))
        u3 = np.concatenate(([0], np.zeros(d-2), [1]))
        
        # Generate the dataset
        X_train = np.zeros((d, size))  # d-dim training dataset, each column is an obervation
        y_train = np.zeros(size, dtype=int)    # labels for training data
        X_test = np.zeros((d, 10 * size)) # d-dim test data
        y_test = np.zeros(10 * size, dtype=int) # labels for test data
        
        # Generate training dataset class1
        for i in range(size1):
            # Class1
            X_train[:, i] = np.random.normal(u1, sigma, size = d)
            y_train[i] = 0
        
        # Generate training dataset class2 and class3
        for i in range(size2):
            # Class2
            X_train[:, i + size1] = np.random.normal(u2, sigma, size = d)
            y_train[i + size1] = 1
            
            # Class3
            X_train[:, i + size1 + size2] = np.random.normal(u3, sigma, size = d)
            y_train[i + size1 + size2] = 2


        # Generate testing dataset class1
        for i in range(10 * size1):
            # Class1
            X_test[:, i] = np.random.normal(u1, sigma, size = d)
            y_test[i] = 1
        
        # Generate testing dataset class2 and class3
        for i in range(10 * size2):
            # Class2
            X_test[:, i + 10 * size1] = np.random.normal(u2, sigma, size = d)
            y_test[i + 10 * size1] = 2
            
            # Class3
            X_test[:, i + 10 * (size1 + size2)] = np.random.normal(u3, sigma, size = d)
            y_test[i + 10 * (size1 + size2)] = 3

        # Shuffle the data
        X_train, y_train = shuffle(X_train.T, y_train, random_state=42)
        X_train = X_train.T
        
        
        # Use cross validation to choose C for MDWSVM based on X_train
        # Define values for cross_validation
        c_values = [2**-3, 2**-2, 2**-1, 1, 2, 4, 8, 16, 32, 64, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12]
        w = vertices(3)
        
        # MDWSVM
        err[j,0,i] = cross_validation(c_values, 5, size, w, X_train, X_test, y_test, y_train, mdwsvm)

        # msvm
        err[j,1,i] = cross_validation(c_values, 5, size, w, X_train, X_test, y_test, y_train, msvm)

        # mdwd
        err[j,2,i] = cross_validation(c_values, 5, size, w, X_train, X_test, y_test, y_train, mdwd)


#TODO: plot

fig, axs = plt.subplots(nrows=3, ncols=1)

axs[0].set_title('p = 1/3')
axs[0].plot(dim, err[:,0,0], color = 'black', linestyle = '-')
axs[0].plot(dim, err[:,1,0], color = 'red', linestyle = '--')
axs[0].plot(dim, err[:,2,0], color = 'blue', linestyle = '-.')

axs[1].set_title('p = 1/2')
axs[1].plot(dim, err[:,0,1], color = 'black', linestyle = '-')
axs[1].plot(dim, err[:,1,1], color = 'red', linestyle = '--')
axs[1].plot(dim, err[:,2,1], color = 'blue', linestyle = '-.')

axs[2].set_title('p = 2/3')
axs[2].plot(dim, err[:,0,2], color = 'black', linestyle = '-')
axs[2].plot(dim, err[:,1,2], color = 'red', linestyle = '--')
axs[2].plot(dim, err[:,2,2], color = 'blue', linestyle = '-.')

plt.tight_layout()
plt.show()