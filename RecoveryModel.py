import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RecoveryModel(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, output_size, n_layers = 2):
        """A simple LSTM for training on FRAP recovery curves, taking 1D data. Uses 
        custom number of layers for deeper training to perform regression on three parameters.
        
        Arguments:
            input_size {int} -- The input dimension of the data (recovery curves: 1)
            hidden_size {int} -- Number of hidden units in the data
            batch_size {int} -- Number of sequences to process
            output_size {int} -- The dimension of the output from regression
            n_layers {int} -- Number of hidden LSTM layers in the RNN
        """
        
        super(RecoveryModel, self).__init__()

        # Model attributes
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.batch_size     = batch_size
        self.output_size    = output_size

        self.n_layers       = n_layers

        # Define model components
        self.LSTM   = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def initialize_hidden_state(self):
        """Initializes the hidden state of an LSTM at t = 0 as zero. Hidden size
        is (number of layers, batch size, hidden size).
        
        Returns:
            (torch.Tensor, torch.Tensor) -- A tuple of zero tensors of dimension (number of layers, batch size, hidden size)
        """
        (h, c) = (torch.zeros(self.n_layers, self.batch_size, self.hidden_size), torch.zeros(self.n_layers, self.batch_size, self.hidden_size))

        return (h, c)

    def forward(self, x):
        """The recurrent feedforward propagation of the network.
        
        Arguments:
            x {torch.Tensor} -- The recovery curve batch with dimensions (sequence length, batch, input size)
        
        Returns:
            torch.Tensor -- Recovery curve parameter estimate with dimensions (output size)
        """

        # Initialize the hidden state for timestep zero
        hidden = self.initialize_hidden_state()

        # Assert that x has dim (sequence length, batch size, input size)
        output, hidden = self.LSTM(x.view(-1, self.batch_size, self.input_size), hidden)

        y = self.linear(output[-1])

        #print(y.shape)

        return y


class RecoveryDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        super(RecoveryDataset, self).__init__()

        self.n_samples_train    = 2**16
        self.n_samples_val      = 2**14
        self.sequence_length    = 110
        self.target_length      = 3

        # Load all data
        x_train = np.fromfile(root_dir + "/x_train.bin", dtype = np.float32)
        y_train = np.fromfile(root_dir + "/y_train.bin", dtype = np.float32)
        x_val   = np.fromfile(root_dir + "/x_val.bin", dtype = np.float32)
        y_val   = np.fromfile(root_dir + "/y_val.bin", dtype = np.float32)
        

        # Concatenate all data 
        x_train = np.reshape(x_train, (self.n_samples_train, self.sequence_length))
        y_train = np.reshape(y_train, (self.n_samples_train, self.target_length))

        x_val = np.reshape(x_val, (self.n_samples_val, self.sequence_length))
        y_val = np.reshape(y_val, (self.n_samples_val, self.target_length))

        self.inputs  = np.vstack((x_train, x_val))
        self.targets = np.vstack((y_train, y_val))

    def __len__(self):
            
        return len(self.inputs[:,0])

    def __getitem__(self, idx):

        return self.inputs[idx, :], self.targets[idx, :]





dataset = RecoveryDataset(root_dir = "rcs")
dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=4)
