import numpy as np
from Layer import Layer

class CrossEntropy(Layer): #Cross Entorpy objective function
    def __init__(self):
        super().__init__()
    def eval(self, Y, Yhat):
        epsilon = 1e-15  # To prevent ln(0)
        Yhat = np.clip(Yhat, epsilon, 1 - epsilon)
        return -np.mean(np.sum(Y * np.log(Yhat), axis=1))
    def gradient(self, Y, Yhat):
        epsilon = 1e-15
        Yhat = np.clip(Yhat, epsilon, 1 - epsilon)
        return -Y / Yhat
    def forward(self, X):#dummy function to bypass requirements
        return X