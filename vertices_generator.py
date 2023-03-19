import numpy as np

def vertices(k):
    '''
    This function is to generate class vertices in R^{k-1}
    
    Input:
    k:  number of classes

    Output:
    W:  matrix of vertices where each column is a vertex for one class
    '''
    u = np.random.randm(k-1)
    u = u / np.linalg.norm(u) # random unit vector
    W = np.zeros((k,k-1))
    for i in range(k):
        if i == 0:
            W[:,i] = (k-1)**(-1/2) * u
        else:
            e = np.zeros(k-1)
            e[i] = 1
            W[:,i] = -(1+np.sqrt(k))/(k-1)**(2/3) * u + np.sqrt(k/(k-1)) * e
    return W