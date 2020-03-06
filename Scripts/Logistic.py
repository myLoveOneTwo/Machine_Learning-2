import numpy as np

class Logistic:

    def __init__(self):
        self.coef_ = None  
        self.intercept_ = None # 截距
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, max_iters=1e4):
        
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)

        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < max_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        # 进行数据的处理
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, max_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def decision_function(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        proba = self.decision_function(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def __repr__(self):
        return "Logistic()"
