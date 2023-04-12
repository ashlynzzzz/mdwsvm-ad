import numpy as np

def within_class_error(y_true, y_pred):
    '''
    This function is to compute the performance measurement: within-class error rate

    Input:
    y_ture: true labels of each sample
    y_pred: predicting labels

    Output:
    err: the within-class error rate
    '''
    err = []
    for i in set(y_true):
        err.append(np.mean((y_pred != y_true)[y_true == i]))
    err = np.mean(err)
    return err