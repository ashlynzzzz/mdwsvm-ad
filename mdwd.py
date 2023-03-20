import numpy as np
import cvxpy as cp

class mdwd:
    '''
    This class aims to reimplement MDWD to do multiclass classification
    '''

    def fit(self, X, y, W, C):
        '''
        This function aims to train MDWD

        Input: 
            X:  training data matrix of interest (d by n) where n is the number of training samples and d is the number of features
            y:  training labels, should be an ndarray
            W:  vertices matrix for all classes
            C:  constraint hyperparameter on B which is the coefficient in f
        '''

        self.W = W
        d, n = X.shape
        k = len(np.unique(y))
        W_y = W[:,y]

        B = cp.Variable((d, k-1))
        beta = cp.Variable((k-1, 1))
        r = cp.Variable(n)
        eta = cp.Variable(n)

        # Objective function for MSVM
        objective = cp.Minimize(cp.sum(cp.power(r, -1) + eta))
        # Constraints
        constraints = [r == cp.diag((B.T @ X + beta).T @ W_y) + eta,
                       r >= 0,
                       eta >= 0,
                       cp.sum([cp.power(cp.pnorm(B[:,i], p=2), 2) for i in range(k-1)]) <= C]
        prob = cp.Problem(objective, constraints)
        # prob.solve(solver=cp.ECOS)
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

        f = self.B.T @ X + self.beta
        y = np.argmax(self.W.T @ f, axis=0)
        return y
