import numpy as np
from Layer import Layer
from scipy import stats
class InputLayer(Layer):
    def __init__( self , dataIn ):
        #ints and finds mean and std
        super().__init__()
        self.std=stats.tstd(dataIn)#my np.std was giving me a funky output so I used scipy here instead
        self.mean=np.mean(dataIn,axis=0)
        self.std = np.where(self.std == 0, 1, self.std)
    def forward(self , dataIn ):
        #zscores input matrix
        zscore=(dataIn-np.mean(dataIn))/np.std(dataIn)
        return zscore
    def gradient(self):
        return np.eye(self.mean.shape[0])