import numpy as np
from Layer import Layer
class MaxPoolLayer(Layer):
    def __init__(self, pool_size, stride):
        super().__init__()  # Inherit from Layer
        self.pool_size = pool_size
        self.stride = stride
        self.prevIn = None
        self.prevOut = None
        self.indices = None

    def forward(self, X):
        if len(X.shape) != 3:
            raise ValueError("Input X must have 3 dimensions: (batch_size, height, width)")
        self.prevIn = X
        batch_size, height, width = X.shape
        pool_size = self.pool_size
        stride = self.stride
        if (height - pool_size) % stride != 0 or (width - pool_size) % stride != 0:
            raise ValueError(f"dimensions {height}x{width} are not compatible with pool size {pool_size} and stride {stride}")
        out_height = (height - pool_size) // stride + 1
        out_width = (width - pool_size) // stride + 1
        output = np.zeros((batch_size, out_height, out_width))
        self.indices = np.zeros_like(output, dtype=int)
        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    start_h, start_w = h * stride, w * stride
                    end_h, end_w = start_h + pool_size, start_w + pool_size
                    pool_region = X[i, start_h:end_h, start_w:end_w]
                    output[i, h, w] = np.max(pool_region)
                    self.indices[i, h, w] = np.argmax(pool_region)
        self.prevOut = output
        return output
    
    def backward(self, gradIn):
        gradOut = np.zeros_like(self.prevIn)
        batch_size, out_height, out_width = gradIn.shape
        
        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    start_h, start_w = h * self.stride, w * self.stride
                    max_idx = self.indices[i, h, w]  # Index of the max value in the pool region
                    max_h, max_w = divmod(max_idx, self.pool_size)  # Convert flat index to 2D index
                    gradOut[i, start_h + max_h, start_w + max_w] = gradIn[i, h, w]  # Backpropagate the gradient to the max value
        return gradOut  # Gradient with respect to the input
    def gradient(self):
        return self.indices  # Gradient will be based on the indices of the max values