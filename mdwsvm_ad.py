import numpy as np
import cvxpy as cp

class mdwsvm_ad:
    '''
    This class aims to reimplement MDWSVM to do multiclass classification

    Variables: 
            X:      training data matrix of interest (d by n) where n is the number of training samples and dw is the number of features
            y:      training labels
            W:      vertices matrix for all classes: K by (K+1)
            C:      constraint hyperparameter on B which is the coefficient in f
            alp:    weighting parameter, the default value of 0.5
            v:      hyperparameter within (0,1)
            K:      kernel function defined in matrix form, the default is a radius-based Gaussian kernel
    '''

    def __init__(self, X, y, W, C, v, K, alp = 0.5):
        self.X = X
        self.y = y
        self.W = W
        self.C = C
        self.alp = alp
        self.v = v
        self.K = K
        self.d, self.e = self.fit()

    def fit(self):
        # TODO
        _, n = self.X.shape
        k, _ = self.W.shape
        W_y = self.W[:,self.y]
        _, counts = np.unique(self.y, return_counts=True)
        N_y = counts[self.y]
        # Calculate the train-train kernel matrix.
        G = self.K(self.X, self.X)

        d = cp.Variable(n)
        e = cp.Variable(n)
        f = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.sum([cp.quad_form(cp.multiply((d+e), W_y[j]), G) for j in range(k)])
                                -2 * cp.sum(cp.sqrt(self.C*self.alp*(d-f) / (self.v*N_y))))
        
        constraints = [W_y @ d == 0,
                       W_y @ e == 0,
                       cp.sum(e) == self.C,
                       # d - f > 0,
                       d - f >= 1e-16, 
                       e >= 0,
                       f >= 0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=1)

        return d.value, e.value        

    def predict(self, data):
        '''
        Input:
            data:  data matrix for prediction (d by m)

        Output:
            y:  predicted labels for data
        '''
        # TODO
        # Compute the train-test kernel matrix
        G = self.K(self.X, data)

        g = self.d + self.e
        W = self.W
        W_y = self.W[:,self.y]
        y = np.argmax(W.T @ ((np.diag(g) @ W_y.T).T @ G), axis=0)

        return y