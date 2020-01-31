"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        temphidden = [input_size] + hiddens + [output_size]

            #self.linear_layers = [Linear(input_size, hiddens[0], weight_init_fn, bias_init_fn)]
        self.linear_layers = [Linear(temphidden[i], temphidden[i+1], weight_init_fn, bias_init_fn) for i in range(len(temphidden) - 1)]
            #self.linear_layers.append(Linear(hiddens[-1], output_size, weight_init_fn, bias_init_fn))
        #else:
            #self.linear_layers = [Linear(input_size, output_size, weight_init_fn, bias_init_fn)]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(hiddens[i]) for i in range(num_bn_layers)]
            
    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        for LayderIdx in range(self.nlayers):
            x = self.linear_layers[LayderIdx](x)
            if LayderIdx + 1 <= self.bn:
                if self.train_mode:
                    x = self.bn_layers[LayderIdx](x)
                else:
                    x = self.bn_layers[LayderIdx](x, eval=True)

            x = self.activations[LayderIdx](x) 
        self.output = x
        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for linearObj in self.linear_layers:
            linearObj.dW.fill(0.0)
            linearObj.db.fill(0.0)
        if self.bn:
            for bnObj in self.bn_layers:
                bnObj.dbeta.fill(0.0)
                bnObj.dgamma.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
            self.linear_layers[i].W += self.linear_layers[i].momentum_W
            
            self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
            self.linear_layers[i].b += self.linear_layers[i].momentum_b

        # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma -= self.lr * self.bn_layers[i].dgamma
                self.bn_layers[i].beta -= self.lr * self.bn_layers[i].dbeta

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        l = self.criterion(self.output, labels)
        d = self.criterion.derivative()
        for LayderIdx in range(self.nlayers-1, -1, -1):
            d = d * self.activations[LayderIdx].derivative()
            if LayderIdx+1 <= self.num_bn_layers:
                d = self.bn_layers[LayderIdx].backward(np.copy(d))
            d = self.linear_layers[LayderIdx].backward(np.copy(d))

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...
        temp_loss = []
        temp_error = []
        for b in range(0, len(trainx), batch_size):
            mlp.zero_grads()
            mlp.forward(trainx[b: b + batch_size])
            
            temp_loss.append(mlp.total_loss(trainy[b: b + batch_size]))
            temp_error.append(mlp.error(trainy[b: b + batch_size]))
            
            mlp.backward(trainy[b: b + batch_size])
            mlp.step()
                
        temp_val_loss = []
        temp_val_error = []               
        for b in range(0, len(valx), batch_size):
            mlp.forward(valx[b: b + batch_size])
            
            temp_val_loss.append(mlp.total_loss(valy[b: b + batch_size]))
            temp_val_error.append(mlp.error(valy[b: b + batch_size]))
            # Val ...

        validation_losses[e] = np.mean(temp_val_loss)
        validation_errors[e] = np.sum(temp_val_error)

    return (training_losses, training_errors, validation_losses, validation_errors)
