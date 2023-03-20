import numpy as np
import cvxpy as cp


class msvm:
    '''
    This class aims to reimplement MSVM to do multiclass classification
    '''

    def fit(self, X, y, W, C):
        '''
        This function aims to train MSVM

        Input: 
            X:  training data matrix of interest (d by n) where n is the number of training samples and d is the number of features
            y:  training labels, should be an ndarray
            W:  vertices matrix for all classes
            C:  constraint hyperparameter on B which is the coefficient in f
        '''

        #TODO: minimization
        # input: X, y, W, C
        # output: B, beta
        self.W = W
        d, n = X.shape
        k = len(np.unique(y))
        W_y = W[:,y]

        B = cp.Variable((d, k-1))
        beta = cp.Variable((k-1, 1))
        xi = cp.Variable(n)

        # Objective function for MSVM
        objective = cp.Minimize(cp.sum(xi))
        # Constraints
        constraints = [cp.diag((B.T @ X + beta).T @ W_y) + xi >= 1,
                       xi >= 0, 
                       cp.sum([cp.power(cp.pnorm(B[:,i], p=2), 2) for i in range(k-1)]) <= C]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.B = B.value
        self.beta = beta.value

    def predict(self, X):
        '''
        This function aims to do prediction

        Input:
            X:  data matrix for prediction (d by n)

        Output:
            y:  predicted labels
        '''
        # classification
        f = self.B.T @ X + self.beta
        y = np.argmax(self.W.T @ f, axis=0)
        return y