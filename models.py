import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import output_size_from_conv
from layers import Convolution1D
#from torch.utils.data import Dataset, DataLoader


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

class CNN1D(nn.Module):

    def __init__(self, n_filters = 64, n_hidden = 16):
        
        super(CNN1D, self).__init__()

        #self.sequence_length    = sequence_length
        self.input_size         = 1
        self.output_size        = 3
        self.sequence_length    = 110
        self.n_filters          = n_filters
        self.n_hidden           = n_hidden

        self.kernel_size         = 2
        self.maxpool_kernel_size = 3
        self.stride              = 1


        self.conv1 = Convolution1D(self.input_size, 
                                   self.n_filters, 
                                   self.sequence_length, 
                                   self.kernel_size, 
                                   self.maxpool_kernel_size)

        self.conv2 = Convolution1D(self.n_filters, 
                                   2*self.n_filters, 
                                   self.conv1.maxpool_size, 
                                   self.kernel_size, 
                                   self.maxpool_kernel_size)


        self.flatten = torch.flatten

        self.linear1 = nn.Linear(self.conv2.output_size(), 4*self.n_hidden)
        self.linear2 = nn.Linear(4*self.n_hidden, self.n_hidden)
        self.linear3 = nn.Linear(self.n_hidden, self.output_size)

    def forward(self, x):

        # Preprocess and assert correct dimensions
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)

        # First convolutional layer 
        x = self.conv1(x)

        # Second convolutional layer
        x = self.conv2(x)

        # Flatten the data, except batch-dimension
        x = self.flatten(x, start_dim=1, end_dim=2)

        # Fully connected layer
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y = self.linear3(x)

        return y

class FFNN(nn.Module):
    
    def __init__(self):

        super(FFNN, self).__init__()

        self.fc1 = nn.Linear(110, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, 3)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        y = self.fc6(x)

        return y

class CNN3D(nn.Module):

    def __init__(self, n_filters, n_hidden):

        super(CNN3D, self).__init__()

        self.in_channels        = 1
        self.out_channels       = n_filters
        self.width              = 256
        self.height             = self.width
        self.depth              = 100

        self.n_hidden           = n_hidden

        self.kernel_size         = 2
        self.maxpool_kernel_size = 3
        self.stride              = 1

        self.conv = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)

    def forward(self, x):
        """An implementation of the torch.nn forward operation.
        
        Arguments:
            x {torch.Tensor} -- A tensor of shape (N, C_in, D, H, W), where N is the batch size, C_in the number of
            in channels, D, H, W is depth, height, width respectively. This is equivalent to N video sequences of length D
            with C_in channels.
        """
        x = self.conv(x)

        return x
