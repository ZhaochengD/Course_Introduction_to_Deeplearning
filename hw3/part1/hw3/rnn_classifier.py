import numpy as np
import sys

sys.path.append('mytorch')
from rnn_cell import *
from linear import *

# RNN Phoneme Classifier
class RNN_Phoneme_Classifier(object):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        ### TODO: Understand then uncomment this code :)
        self.rnn = [RNN_Cell(input_size, hidden_size) if i == 0 else
                     RNN_Cell(hidden_size, hidden_size) for i in range(num_layers)]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """
        Initialize weights

        Parameters
        ----------
        rnn_weights:
        [[W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
         [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1], ...]

        linear_weights:
        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(*linear_weights)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):

        """
        RNN forward, multiple layers, multiple time steps

        Parameters
        ----------
        x : (batch_size, seq_len, input_size)
            Input
        h_0 : (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits : (batch_size, output_size)
        Output logits
        """

        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        for t in range(seq_len):
            for l in range(len(self.rnn)):
                if l == 0:
                    hidden[l,:,:] = self.rnn[l](x[:,t,:], self.hiddens[t][l,:,:])
                else:
                    hidden[l,:,:] = self.rnn[l](hidden[l-1,:,:], self.hiddens[t][l,:,:])
            self.hiddens.append(hidden.copy())
        logits = self.output_layer(self.hiddens[-1][-1,:,:])

        return logits

    def backward(self, delta):

        """
        RNN Back Propagation Through Time (BPTT)

        Parameters
        ----------
        delta : (batch_size, hidden_size)
        gradient w.r.t. the last time step output dY(seq_len-1)

        Returns
        -------
        dh_0 : (num_layers, batch_size, hidden_size)
        gradient w.r.t. the initial hidden states
        """

        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        for t in range(seq_len-1, -1, -1):
            for l in range(self.num_layers-1, -1, -1):
                h = self.hiddens[t+1][l,:,:]
                h_prev_l = self.x[:,t,:] if l == 0 else self.hiddens[t+1][l-1,:,:]
                h_prev_t = self.hiddens[t][l,:,:]

                dxt, dh[l] = self.rnn[l].backward(dh[l], h, h_prev_l, h_prev_t)
                dh[l-1] = dh[l-1] if l == 0 else dh[l-1] + dxt                

        dh_0 = np.copy(dh / batch_size)
        return dh_0

        raise NotImplementedError
