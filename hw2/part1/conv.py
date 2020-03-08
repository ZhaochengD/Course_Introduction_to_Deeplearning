# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        
        self.x = x
        batchSize = x.shape[0]
        inputSize = x.shape[2]
        self.inputsize = x.shape[2]
        outputsize = (inputSize - self.kernel_size + self.stride) // self.stride
    
        self.out = np.zeros((batchSize, self.out_channel, outputsize))
        index = 0
        for time in range(0, inputSize - self.kernel_size + 1, self.stride):
            self.out[:,:,index] = np.tensordot(x[:, :, time:time+self.kernel_size], self.W, axes=([1,2],[1,2])) + self.b
            index += 1

        return self.out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        inputSize  = self.x.shape[2]
        
        self.dx = np.zeros(self.x.shape)
        index = 0
        for time in range(0, inputSize - self.kernel_size + 1, self.stride):
            self.dW += np.tensordot(delta[:, :, index], self.x[:,:,time:time+self.kernel_size], axes=([0],[0]))
            self.dx[:, :, time:time+self.kernel_size] += np.tensordot(np.transpose(delta, (1,0,2))[:,:,index], self.W, axes=([0],[0]))
            index += 1
        self.db += np.sum(delta,axis=(0,2))
        
        return self.dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c * self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)
