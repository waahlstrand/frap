import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader

class LSTM_to_FFNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout = 0, n_layers = 2):
        """A simple LSTM for training on FRAP recovery curves, taking 1D data. Uses 
        custom number of layers for deeper training to perform regression on three parameters.
        
        Arguments:
            input_size {int} -- The input dimension of the data (recovery curves: 1)
            hidden_size {int} -- Number of hidden units in the data
            batch_size {int} -- Number of sequences to process
            output_size {int} -- The dimension of the output from regression
            n_layers {int} -- Number of hidden LSTM layers in the RNN
        """
        
        super(LSTM_to_FFNN, self).__init__()

        # Model attributes
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        #self.batch_size     = batch_size
        self.output_size    = output_size
        self.dropout        = dropout
        self.n_layers       = n_layers

        # Define model components
        self.LSTM   = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout = self.dropout)
        self.dense  = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def initialize_hidden_state(self, batch_size):
        """Initializes the hidden state of an LSTM at t = 0 as zero. Hidden size
        is (number of layers, batch size, hidden size).
        
        Returns:
            (torch.Tensor, torch.Tensor) -- A tuple of zero tensors of dimension (number of layers, batch size, hidden size)
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        (h, c) = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device), torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))

        return (h, c)

    def forward(self, x):
        """The recurrent feedforward propagation of the network.
        
        Arguments:
            x {torch.Tensor} -- The recovery curve batch with dimensions (sequence length, batch, input size)
        
        Returns:
            torch.Tensor -- Recovery curve parameter estimate with dimensions (output size)
        """
        batch_size = x.shape[0]

        # Initialize the hidden state for timestep zero
        hidden = self.initialize_hidden_state(batch_size)

        # Assert that x has dim (sequence length, batch size, input size)
        output, hidden = self.LSTM(x.view(-1, batch_size, self.input_size), hidden)

        output = self.dense(output[-1])
        output = self.dense(output)

        y = self.linear(output)

        return y