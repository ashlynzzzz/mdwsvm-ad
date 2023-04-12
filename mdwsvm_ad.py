import numpy as np
import cvxpy as cp

class mdwsvm_ad:
    '''
    This class aims to reimplement MDWSVM to do multiclass classification

    Variables: 
            X:      training data matrix of interest (d by n) where n is the number of training samples and d is the number of features
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
        _, n = self.X.shape
        self.k, _ = self.W.shape
        W_y = self.W[:,self.y]
        _, counts = np.unique(self.y, return_counts=True)
        self.N_y = counts[self.y]
        # Calculate the train-train kernel matrix.
        G = self.K(self.X, self.X)

        d = cp.Variable(n)
        e = cp.Variable(n)
        f = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.sum([cp.quad_form(cp.multiply((d+e), W_y[j]), G) for j in range(self.k)])
                                -2 * cp.sum(cp.sqrt(self.C*self.alp*(d-f) / (self.v*self.N_y))))
        
        constraints = [W_y @ d + (self.C - cp.sum(d))*self.W[:,-1] == 0,
                       W_y @ e + (self.C - cp.sum(e))*self.W[:,-1] == 0,
                       e <= self.C * (1-self.alp) / (self.v * self.N_y),
                       d <= self.C * self.alp / (self.v * self.N_y),
                       d - f >= 1e-16, 
                       e >= 0,
                       f >= 0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=True)

        return d.value, e.value        

    def predict(self, data):
        '''
        Input:
            data:  data matrix for prediction (d by m)

        Output:
            y:  predicted labels for data
        '''
        # Compute the train-test kernel matrix
        G = self.K(self.X, data)

        g = self.d + self.e
        W = self.W
        W_y = self.W[:,self.y]

        constant = np.zeros(self.k)
        index = np.where((self.e > 0) & (self.e < (1-self.alp)*self.C/(self.v*self.N_y)))[0]
        for i in range(self.k):
            selected_index = index[np.argmax(self.y[index] == i)]
            selected_G = self.K(self.X, self.X[:,selected_index].reshape(-1,1))
            constant[i] = - W[:,i].T @ (np.diag(g) @ W_y.T).T @ selected_G

        constant = np.concatenate([constant,[0]]).reshape((-1,1))
        y = np.argmax(W.T @ ((np.diag(g) @ W_y.T).T @ G) + constant, axis=0)

        return y