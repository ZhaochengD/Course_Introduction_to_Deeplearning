# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """
        self.x = x
        m = self.x.shape[1]
        if eval:
            self.norm = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.gamma * self.norm + self.beta
            return self.out


        self.mean = self.x.mean(axis=0).reshape(1, m)
        self.var = self.x.var(axis=0).reshape(1, m)
        self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.gamma * self.norm + self.beta

        # Update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        m = 1 / delta.shape[0]
        varEps = self.var + self.eps
        n = delta.shape[1]

        dnorm = delta * self.gamma
        self.dbeta = delta.sum(axis=0).reshape(1, n)
        self.dgamma = (delta * self.norm).sum(axis=0).reshape(1, n)
        dsigma2 = - 1/2 * ((dnorm * (self.x - self.mean) * np.power(varEps, -3/2))).sum(axis=0).reshape(1, n)
        dmu = - ((dnorm * np.power(varEps, -1/2))).sum(axis=0) - 2 * m * dsigma2 * (self.x - self.mean).sum(axis=0)
        dx = dnorm * np.power(varEps, -1/2) + dsigma2 * ((2 * m) * (self.x - self.mean)) + dmu * m
        return dx
