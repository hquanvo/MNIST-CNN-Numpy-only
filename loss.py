import numpy as np


class SoftmaxLoss():

    def __init__(self, input_dim, output_dim, lammy = 0.04):
        # 1568 x 10
        self.W = np.random.randn(input_dim, output_dim) / 255
        # 10 x 1
        self.b = np.zeros((output_dim, 1))
        self.lammy = lammy

    def forward_pass(self, X):
        # number of examples, height, width, number of filter
        n, h, w, c = X.shape

        X_reshaped = X.reshape(X.shape[0], -1)

        Z = ((X_reshaped @ self.W).T + self.b).T

        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        cache = (X_reshaped, Z)

        return probs, cache

    def compute_cross_entropy_loss(self, probs, labels):
        n, _ = probs.shape
        # L2 regularization
        loss = np.sum(-np.log(probs[np.arange(n), labels])) / n + self.lammy * np.linalg.norm(self.W) / 2

        accuracy = np.sum(np.argmax(probs, axis=1) == labels)/n

        return loss, accuracy

    def backprop(self, probs, labels, cache):
        n, _ = probs.shape
        (X_reshaped, Z) = cache
        one_hot_labels = np.zeros_like(probs)
        one_hot_labels[np.arange(n), labels] = 1
        dA_dZ = np.zeros_like(probs)

        for i in range(n):
            exp_Z_i = -np.exp(Z[i])
            S = np.sum(exp_Z_i)
            dA_dZ[i, :] = -exp_Z_i[labels[i]] * -exp_Z_i / (S ** 2)
            dA_dZ[i, labels[i]] = exp_Z_i[labels[i]] * (S - exp_Z_i[labels[i]]) / (S ** 2)

        dL_dA = (probs - one_hot_labels)/n
        # dL/dW = dL/dA * dA/dZ * dZ/dW

        dZ_dX = self.W
        dZ_dW = X_reshaped

        dL_dZ = dL_dA


        dL_dW = dZ_dW.T @ dL_dZ + self.lammy * self.W
        dL_db = 1/n * np.sum(dL_dZ, axis = 0, keepdims= True).T
        dL_dX = dL_dZ @ dZ_dX.T

        return dL_dW, dL_db, dL_dX

    def update_parameters(self, dL_dw, dL_db, learning_rate = 0.01):
        self.W = self.W - learning_rate * dL_dw
        self.b = self.b - learning_rate * dL_db

        # print("Softmax params:")
        # print(self.W)
        # print(self.b)

