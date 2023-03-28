import numpy as np
from metric import within_class_error

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
    c_best: the best c for this method on this dimention
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
    
    
    return error, best_c