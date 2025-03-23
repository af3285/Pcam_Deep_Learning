import numpy as np
from Layer import Layer

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        #ints
        super().__init__()
        self.previous_input = None
        self.previous_output = None
    def forward(self, dataIn):
        #Storing vals for gradient
        self.previous_input = dataIn
        self.previous_output = 1 / (1 + np.exp(-dataIn))
        return self.previous_output
    def gradient(self):
        N, D = self.previous_input.shape
        grad_diag = self.previous_output * (1 - self.previous_output)  # Derivative of sigmoid
        grad = np.zeros((N, D, D))
        for i in range(N):
            grad[i] = np.diag(grad_diag[i])
        return grad
    def backward(self, gradIn):
        grad_diag = self.previous_output * (1 - self.previous_output)
        gradOut = gradIn * grad_diag
        self.gradOut = gradOut
        return gradOut
    def sigmoid_derivative(self, output):
        # Derivative of sigmoid: sigma(x) * (1 - sigma(x))
        return output * (1 - output)
    def gradient2(self):
        # Flattening
        grad_diag = self.previous_output * (1 - self.previous_output)
        return grad_diag
    def backward2(self, gradIn):
        grad_diag = self.previous_output * (1 - self.previous_output)
        # Hadamard product
        gradOut = gradIn * grad_diag
        return gradOut
