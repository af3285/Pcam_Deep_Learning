import numpy as np
from Layer import Layer
class ConvolutionalLayer(Layer):
    def __init__(self, kernel_size, num_kernels):
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        #inting kernals 
        self.kernels = np.random.randn(self.num_kernels, kernel_size, kernel_size) * np.sqrt(2. / (kernel_size * kernel_size))
        self.prevIn = None 
        self.prevOut = None 

    @staticmethod
    def crossCorrelate2D(kernel, X):
        if len(kernel.shape) >= 3:
            _,kernel_height, kernel_width = kernel.shape
        else:
            kernel_height, kernel_width = kernel.shape
        X_height, X_width = X.shape
        output_height = X_height - kernel_height + 1
        output_width = X_width - kernel_width + 1
        result = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                result[i, j] = np.sum(X[i:i+kernel_height, j:j+kernel_width] * kernel)
        return result

    def setKernels(self, kernels):
        self.kernels = kernels.astype(np.float64)

    def getKernels(self):
        return self.kernels

    def forward(self, X):
        #Preventing errors
        if len(X.shape) == 2: 
            X = np.expand_dims(X, axis=0)
        self.prevIn = X.astype(np.float64)  # Saving for backprop
        outputs = []
        for i in range(X.shape[0]):
            output = self.crossCorrelate2D(self.kernels, X[i])  #Perform cross-correlation across all images
            outputs.append(output)
        self.prevOut = np.array(outputs)#storing djdf
        return self.prevOut

    def gradient(self):
        return np.zeros_like(self.kernels)

    def updateKernels(self, dJdF,learning_rate):
        #Makes interger type compatible for lab 7
        dJdF = dJdF.astype(np.float64)
        #inting
        gradKernels = np.zeros_like(self.kernels)
        #Checks if input is one or multiple batches
        if len(self.prevIn.shape) == 3:
            batch_size = 1
            prevIn_single = self.prevIn[0] 
        else: 
            batch_size = self.prevIn.shape[0]
            prevIn_single = None  #ignores the single attribute since this is for multiple batches
        #For one batch
        if batch_size == 1:
            for y in range(dJdF.shape[1]): 
                for x in range(dJdF.shape[2]):
                    patch = prevIn_single[y:y + self.kernel_size, x:x + self.kernel_size]
                    gradKernels += patch * dJdF[0, y, x]
        #for multiple
        else:
            for b in range(batch_size):  
                for y in range(dJdF.shape[1]):  
                    for x in range(dJdF.shape[2]):  
                        patch = self.prevIn[b, y:y + self.kernel_size, x:x + self.kernel_size]
                        gradKernels += patch * dJdF[b, y, x] #may need to modify this for future projects 
        self.kernels -= learning_rate * gradKernels  # Perform gradient descent update


    def backward(self, dJdF):
        # Check the dimensions of the gradient with respect to the output
        if len(dJdF.shape) == 3:
            batch_size, output_height, output_width = dJdF.shape
            num_kernels = self.kernels.shape[0]  # Get the number of kernels
        else:
            batch_size, num_kernels, output_height, output_width = dJdF.shape
            input_height, input_width = self.prevIn.shape[1], self.prevIn.shape[2]

        # Initialize the gradient of the input (dJdX) to zero
        dJdX = np.zeros_like(self.prevIn)

        # Loop over each image in the batch
        for b in range(batch_size):
            for k in range(num_kernels):
                kernel = self.kernels[k]

                # Loop over each position in the output (height, width)
                for y in range(output_height):
                    for x in range(output_width):
                        # Extract the patch of the input corresponding to the current position
                        patch = self.prevIn[b, y:y + self.kernel_size, x:x + self.kernel_size]
                        
                        # Compute the gradient for this patch
                        dJdX[b, y:y + self.kernel_size, x:x + self.kernel_size] += dJdF[b, y, x] * kernel

        return dJdX

