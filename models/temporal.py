import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Curves(nn.Module):

    def __init__(self, batch_size, n_hidden = 16, shape = (31, 110)):

        super(Curves, self).__init__()

        self.batch_size         = batch_size
        self.output_size        = 3
        self.shape              = shape
        self.sequence_length    = shape[1]
        self.n_hidden           = n_hidden

        self.body       = nn.Sequential(nn.Linear(31*110, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 16*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(16*self.n_hidden, 16*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(16*self.n_hidden, 4*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(4*self.n_hidden, 3)
                                        )   
    
    def forward(self, x):


        x = torch.flatten(x, start_dim=1)

        x = self.body(x)


        return x



class CNN1d(nn.Module):

    def __init__(self, batch_size, n_hidden = 32, n_filters = 32, shape = (1, 110)):

        super(CNN1d, self).__init__()

        self.batch_size         = batch_size
        self.output_size        = 3
        self.shape              = shape
        self.sequence_length    = shape[1]
        self.n_filters          = n_filters
        self.n_hidden           = n_hidden

        self.body       = nn.Sequential(nn.Conv1d(1, out_channels=self.n_filters, kernel_size=2, stride=1),
                                        nn.BatchNorm1d(self.n_filters),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=3),
                                        nn.Conv1d(in_channels=self.n_filters, out_channels=2*self.n_filters, kernel_size=2),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=2*self.n_filters, out_channels=2*self.n_filters, kernel_size=2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(2*self.n_filters),
                                        nn.MaxPool1d(kernel_size=3),
                                        )   

        flat_size       = self._get_conv_output_size(shape)

        self.flatten = torch.flatten

        self.head       = nn.Sequential(nn.Linear(flat_size, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 2*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(2*self.n_hidden, self.output_size))
    
    def forward(self, x):


        x = self.body(x)

        x = self.flatten(x, start_dim=1)

        y = self.head(x)

        return y

    def _forward_through_body(self, x):
        
        x = self.body(x)

        return x

    def _get_conv_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

class FC(nn.Module):

    def __init__(self, batch_size, n_hidden = 16, shape = (1, 110)):
        super(FC, self).__init__()

        self.batch_size         = batch_size
        self.output_size        = 3
        self.shape              = shape
        self.sequence_length    = shape[1]
        self.n_hidden           = n_hidden

        self.flatten = torch.flatten

        self.head   = nn.Sequential(nn.Linear(self.sequence_length, 8*self.n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(8*self.n_hidden, 16*self.n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(16*self.n_hidden, 16*self.n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(16*self.n_hidden, 8*self.n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(8*self.n_hidden, self.output_size))
    
    def forward(self, x):

        x = self.flatten(x, start_dim=1)

        y = self.head(x)

        return y


class LSTM_to_FFNN(nn.Module):

    def __init__(self, hidden_size, input_size=1, output_size=3, dropout = 0, n_layers = 2):
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
        #self.dense1 = nn.Linear(self.hidden_size, 512)
        #self.dense2 = nn.Linear(512, 1024)
        #self.dense3 = nn.Linear(1024, 128)
        #self.linear = nn.Linear(128, self.output_size)
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

        output = F.relu(self.dense(output[-1]))
        output = F.relu(self.dense(output))

        #output = F.relu(self.dense1(output[-1]))
        #output = F.relu(self.dense2(output))
        #output = F.relu(self.dense3(output))

        y = self.linear(output)

        return y

