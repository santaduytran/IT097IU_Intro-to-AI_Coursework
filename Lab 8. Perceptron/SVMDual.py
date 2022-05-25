import numpy as np
import random as rnd

class SVMDual():
    def __init__(self, max_iter=100, kernel_type='linear', C=1.0, epsilon=0.001, gamma=1.0):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic,
            'rbf': self.kernel_rbf
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.proba = []

    def fit(self, X, y):
        # n : number of samples (100)
        n = X.shape[0]
        print('shape changed: ', n)
        # we have alpha per sample of training set. Initially set to zeros
        alpha = np.zeros(n)
        self.w = np.zeros(n)
        self.b = np.zeros(n)
        print('alpha changed: ', alpha.shape)

        # pick the kernel user selected
        kernel = self.kernels[self.kernel_type]

        iteration = 0
        while True:
            iteration += 1

            # saving the copy of alpha from previous iteration
            alpha_prev = np.copy(alpha)

            # going through all the samples in one iteration
            for j in range(0, n):

                # selecting random sample index where i is not equal to j
                i = self.get_rnd_int(0, n - 1, j) # Get random int i~=j

                x_i = X[i, :]
                x_j = X[j, :]

                y_i = y[i]
                y_j = y[j]

                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)

                if k_ij == 0:
                    continue

                # select alpha of i and j from the alpha array to calculate L and H
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w
                )

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j)) / k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = max(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alpha[j])

            # Terminating condition: reacting convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            # Terminating condition: Reaching max iterations
            if iteration >= self.max_iter:
                print('Iteration number exceeded the max of %d iteration' % (self.max_iter))
                return
        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)

        return self

    def predict(self, X):
        return self.h(X, self.w, self.b)

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(alpha * y, X)

    # Predictions
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if (y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_j + alpha_prime_i - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a, b, z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt = cnt + 1
        return i

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)

    def kernel_rbf(self,x1, x2):
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))

    # Printing out the parameters of SVM
    def print_info(self):
        print('======== PRINT SVM INFO ========')
        print('C: ', self.C)
        print('max_iter: ', self.max_iter)
        print('epsilon: ', self.epsilon)
        print('kernel_type: ', self.kernel_type)

    def predict_proba(self, X):
        for xi in X:
            if (self.predict(xi) <= 0):
                self.proba.append([1, 0])
            else:
                self.proba.append([0, 1])
        return self.proba