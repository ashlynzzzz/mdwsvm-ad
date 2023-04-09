import numpy as np

def Gaussian_kernel(X1, X2, sigma2=1.0):
    '''
    Gaussian kernel. Computes a covariance matrix from points in X1 and X2.
    
    Args:
        X1: data matrix 1 (d x n).
        X2: data matrix 2 (d x m).
        sigma2: sigma**2

    Returns:
        Covariance matrix (n x m).
    '''
    sqdist = np.sum(X1**2, 0).reshape(-1, 1) + np.sum(X2**2, 0) - 2 * np.dot(X1.T, X2)
    return np.exp(-0.5 / sigma2 * sqdist)