import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h
        

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.z = self.z_act(np.dot(self.Wzh, h) + np.dot(self.Wzx, x))
        self.r = self.r_act(np.dot(self.Wrh, h) + np.dot(self.Wrx, x))
        self.h_tilda = self.h_act(np.dot(self.Wh, (self.r * h)) + np.dot(self.Wx, x))
        h_t = (1 - self.z) * h + self.z * self.h_tilda
        

        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilda.shape == (self.h, )
        assert h_t.shape == (self.h, )

        return h_t


    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        new_x = np.matrix(self.x)
        new_hidden = np.matrix(self.hidden)

        d1 = (1 - self.z) * delta
        d2 = self.hidden * delta
        d3 = self.h_tilda * delta
        d4 = - d2
        d5 = d3 + d4
        d6 = self.z * delta
        d7 = d5 * self.z_act.derivative()
        d8 = d6 * self.h_act.derivative()
        d9 = d8.dot(self.Wx)
        d10 = d8.dot(self.Wh)
        d11 = d7.dot(self.Wzx)
        d12 = d7.dot(self.Wzh)
        d14 = d10 * self.r
        d15 = d10 * self.hidden
        d16 = d15 * self.r_act.derivative()
        d13 = d16.dot(self.Wrx)
        d17 = d16.dot(self.Wrh)

        dx = d9 + d11 + d13
        dh = d12 + d14 + d1 + d17

        self.dWrx = d16.T.dot(new_x)
        self.dWzx = d7.T.dot(new_x)
        self.dWx = d8.T.dot(new_x)
        self.dWrh = d16.T.dot(new_hidden)
        self.dWzh = d7.T.dot(new_hidden)
        self.dWh = d8.T.dot(np.matrix(self.r * self.hidden))
        
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
