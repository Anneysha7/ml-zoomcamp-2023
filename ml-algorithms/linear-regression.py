import numpy as np

'''
Math
X.w = y
Xt.X.w = Xt.y
(Xt.X)'.(Xt.X).w = (Xt.X)'.Xt.y         # where (Xt.X) is the gram matrix
w = (Xt.X)'.Xt.y
'''

'''
Steps:
- Initialise weight as 0
- Initialise bias as 0
'''


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        # lr is the learning rate
        # greater the lr, more accurate the fit
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iter in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = 1/(n_samples) * np.dot(X.T, (y_pred-y))
            db = 1/(n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    

    def mse(y_test, predictions):
        return np.mean((y_test - predictions)**2)