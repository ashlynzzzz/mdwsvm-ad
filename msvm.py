import numpy as np
import cvxpy as cp

class msvm:
    '''
    This class aims to reimplement MSVM to do multiclass classification

    Variables:
        X:      training data matrix of interest (d by n) where n is the number of training samples and d is the number of features
        y:      training labels
        W:      vertices matrix for all classes
        C:      constraint hyperparameter on B which is the coefficient in f
    '''

    def __init__(self, X, y, W, C):
        self.X = X
        self.y = y
        self.W = W
        self.C = C
        self.B, self.beta = self.fit()

    def fit(self):
        d, n = self.X.shape
        k = len(np.unique(self.y))
        W_y = self.W[:,self.y]

        B = cp.Variable((d, k-1))
        beta = cp.Variable((k-1, 1))
        xi = cp.Variable(n)

        # Objective function for MSVM
        objective = cp.Minimize(cp.sum(xi))
        # Constraints
        constraints = [cp.diag((B.T @ self.X + beta).T @ W_y) + xi >= 1,
                       xi >= 0, 
                       cp.sum([cp.power(cp.pnorm(B[:,i], p=2), 2) for i in range(k-1)]) <= self.C]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return B.value, beta.value

    def predict(self, data):
        '''
        Input:
            data:   data matrix for prediction (d by n)

        Output:
            y:      predicted labels for data
        '''
        # classification
        f = self.B.T @ data + self.beta
        y = np.argmax(self.W.T @ f, axis=0)
        return y