import numpy as np

def vertex(k, q):
    '''
    This function is to generate k-class vertices in R^{k-1}
    
    Input:
    k:  number of classes
    q:  the q-th vertex

    Output:
    u:  vertex vector for q-th class
    '''
    if q == 0:
        u = np.zeros(k-1)
        if k >= 2:
            u[k-2] = -1
    else:
        u = np.concatenate((vertex(k-1, q-1) * np.sqrt(k*(k-2)),[1]))
        u = u / (k-1)

    return u

def vertices_v2(k):
    W = np.zeros((k-1,k))
    for i in range(k):
        W[:,i] = vertex(k, i)
    for i in range(k-1):
        W[:,i] = (W[:,i] + W[:,k-1]) * np.sqrt((k-1)/(k-2)) / np.sqrt(2)
    return W