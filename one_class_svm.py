import numpy as np
import cvxpy as cp

class one_class_svm():
    '''
    This class aims to do one-class svm and label anomalous vectors as +1

    Variables: 
        X:  data matrix of interest (d by n) where n is the number of training samples and d is the number of features
        v:  hyperparameter within (0,1)
        K:  kernel function defined in matrix form, the default is a radius-based Gaussian kernel
    '''
    def __init__(self, X, v, K):
        self.X = X
        self.v = v
        self.K = K
        self.G = self.K(self.X, self.X)
        self.alpha, self.rho = self.fit()

    def fit(self):
        '''
        Output:
        alpha, rho
        '''
        _, n = self.X.shape

        alpha = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.quad_form(alpha, self.G))

        constraints = [alpha <= 0,
                       alpha >= -1 / (self.v*n),
                       cp.sum(alpha) == -1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        index = np.argmax((alpha.value != 0) & (alpha.value != (-1/(self.v*n))))
        x = self.X[:,index].reshape(-1,1)
        rho = np.dot(alpha.value, self.K(self.X, x)).item()

        return alpha.value, rho

    def predict(self):
        '''
        Output:
        y:      predicting labels for data
        Note:   -1 for anomalies
        '''
        y = - np.sign(self.G.T @ self.alpha - self.rho)

        return y.flatten()
