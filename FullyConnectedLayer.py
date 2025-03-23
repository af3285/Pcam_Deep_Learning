import numpy as np
from Layer import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        self.weights = np.random.randn(sizeIn, sizeOut) * np.sqrt(1. / sizeIn)  # Xavier initialization
        self.biases = np.zeros((1, sizeOut))
        self.is_training = True
     

    def setTraining(self, is_training):
        self.is_training = is_training  

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def getBiases(self):
        return self.biases

    def setBiases(self, biases):
        self.biases = biases

    def forward(self, dataIn):
        self.previous_input = dataIn
        self.output = np.dot(dataIn, self.weights) + self.biases
        return self.output

    def gradient(self):
        N, D = self.previous_input.shape
        grad_input = np.zeros((N, D))
        grad_input[:] = self.weights.T 
        return grad_input

    def backward(self, gradIn):
        N = gradIn.shape[0]
        gradW = np.dot(self.previous_input.T, gradIn) / N
        gradB = np.sum(gradIn, axis=0, keepdims=True) / N
        gradX = np.dot(gradIn, self.weights.T)
        #Storing gradients for weight update
        self.gradW = gradW
        self.gradB = gradB
        return gradX

    def updateWeights(self, dJdH, LR):
        N = dJdH.shape[0]
        gradW = (self.previous_input.T @ dJdH) / N
        gradB = np.sum(dJdH, axis=0) / N  
        #Updating weights and biases
        self.weights -= LR * gradW
        self.biases -= LR * gradB

    def setPrevIn(self, dataIn):  # Added for testing update weight functionality
        self.previous_input = dataIn
