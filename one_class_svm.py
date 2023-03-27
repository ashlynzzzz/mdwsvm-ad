import numpy as np
import cvxpy as cp

class one_class_svm():
    '''
    This class aims to do one-class svm and label anomalous vectors as +1

    Variables: 
        X:  data matrix of interest (d by n) where n is the number of training samples and d is the number of features
        v:  hyperparameter within (0,1)
        K:  kernel, the default is a radius-based Gaussian kernel
    '''
    def __init__(self, X, v, K = lambda x, y: np.exp(-np.linalg.norm(x - y)**2/2)):
        self.X = X
        self.v = v
        self.K = K
        self.alpha, self.rho = self.fit()

    def fit(self):
        '''
        Output:
        alpha, rho:     for f(x) = B.T@x + beta
        '''
        d, n = self.X.shape
        # Calculate the Gram matrix.
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                G[i,j] = self.K(self.X[:,i], self.X[:,j])

        alpha = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.quad_form(alpha, G))

        constraints = [alpha >= 0,
                       alpha <= 1 / (self.v*n),
                       cp.sum(alpha) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        index = np.argmax((alpha.value != 0) & (alpha.value != (1/(self.v*n))))
        x = self.X[:,index]
        rho = np.sum(np.array([alpha.value[j] * self.K(self.X[:,j], x) for j in range(n)]))

        return alpha.value, rho

    def predict(self, data):
        '''
        Input:
        data:   data for evaluation (d by m)

        Output:
        y:      predicting labels for data
        '''
        # Compute the train-test kernal matrix
        _, n = self.X.shape
        _, m = data.shape
        G = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                G[i,j] = self.K(self.X[:,i], data[:,j])
        
        y = np.sign(G.T @ self.alpha - self.rho)

        return y.flatten()
