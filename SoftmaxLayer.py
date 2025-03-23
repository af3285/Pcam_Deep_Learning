import numpy as np
from Layer import Layer

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        self.previous_input = None
        self.previous_output = None
    def forward(self, dataIn):
        self.previous_input = dataIn
        exp_vals = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))
        self.previous_output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.previous_output
    def gradient(self):
        N, D = self.previous_input.shape
        grad = np.zeros((N, D, D))
        for i in range(N):
            S = self.previous_output[i]
            grad[i] = np.diag(S) - np.outer(S, S)
        return grad
    def backward(self, gradIn):
        gradOut = np.einsum('nij,nj->ni', self.gradient(), gradIn)
        return gradOut