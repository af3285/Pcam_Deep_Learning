import numpy as np
from Layer import Layer
class FlatteningLayer(Layer):
    def __init__(self):
        self.prevIn = None
        self.prevOut = None

    def forward(self, X):
        self.prevIn = X
        self.prevOut =  X.flatten(order='F').reshape(1, -1)
        return self.prevOut
    
    def backward(self, gradIn):
        np.array(gradIn)
        return gradIn.reshape(self.prevIn.shape, order='F')#Needs to be ordered just like the .flatten func
    def gradient(self):
        return self.prevIn.reshape(-1)