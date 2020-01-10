import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Split(nn.Module):
    """Temporal neural network model which splits the input recovery curve into pre- and post-bleach data, and then
    passes the new inputs to two separate fully connected networks. These are then merged and passed to a final fully 
    connected network.
    """

    def __init__(self, batch_size, n_hidden = 32, n_filters=32, shape = (1, 110)):
        """Initializes a Split neural network.
        
        Keyword Arguments:
            n_hidden {int} -- Number of base hidden units. (default: {32})
            shape {tuple} -- Shape of the recovery curve data. (default: {(1, 110)})
        """

        super(Split, self).__init__()

        self.batch_size         = batch_size
        self.output_size        = 3
        self.shape              = shape
        self.sequence_length    = shape[1]
        self.n_prebleach        = 10
        self.n_postbleach       = 100
        self.n_filters          = n_filters
        self.n_hidden           = n_hidden


        # Pre-bleach neural network
        self.prebleach_nn = nn.Sequential(nn.Linear(self.n_prebleach, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 2*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(2*self.n_hidden, self.n_hidden))

        # Post-bleach neural network
        self.postbleach_nn = nn.Sequential(nn.Linear(self.n_postbleach, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 8*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(8*self.n_hidden, 2*self.n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(2*self.n_hidden, 2*self.n_hidden))

        # Inference neural network
        self.head       = nn.Sequential(nn.Linear(3*self.n_hidden, 8*self.n_hidden),
                                         nn.ReLU(),
                                         nn.Linear(8*self.n_hidden, 2*self.n_hidden),
                                         nn.ReLU(),
                                         nn.Linear(2*self.n_hidden, 3))
    
    def forward(self, x):

        # Remove channel dimension
        x = torch.squeeze(x, 1)
        
        # Separate input into pre-bleach, post-bleach
        prebleach = x[:,0:10]
        postbleach = x[:,10:110]

        y1 = self.prebleach_nn(prebleach)
        y2 = self.postbleach_nn(postbleach)

        # Merge the two networks
        y = torch.cat((y1, y2), dim=1)

        y = self.head(y)


        return y


class CNN1d(nn.Module):
    """Temporal neural network model which passes the recovery curve through a sequence of 1D CNNs.
    """

    def __init__(self, batch_size, n_hidden = 32, n_filters = 32, shape = (1, 110)):
        """Initializes a CNN1d neural network
        
        Arguments:
            batch_size {int} -- Batch size used when training
        
        Keyword Arguments:
            n_hidden {int} -- Number of base hidden units used in the final fully connected network. (default: {32})
            n_filters {int} -- Number of filters in the 1D CNNs. (default: {32})
            shape {tuple} -- Shape of the recovery curve data. (default: {(1, 110)})
        """

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

        # Get the size of data after the convolutions
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
        """Pass through the 1D CNN body.
        
        Arguments:
            x {torch.Tensor} -- Input recovery curve data.
        
        Returns:
            tuple -- A shape tuple after passing the data through the body.
        """
        
        x = self.body(x)

        return x

    def _get_conv_output_size(self, shape):
        """Calculates the size (number of elements except for the batch dimension) of the data after passing it through a 1D CNN network.
        
        Arguments:
            shape {tuple} -- Input shape of the data.
        
        Returns:
            int -- Number of elements in data except for the batch dimension.
        """
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

class FC(nn.Module):
    """Temporal neural network model which passes the full recovery curve to a fully connected neural network.
    """

    def __init__(self, n_hidden = 16, shape = (1, 110)):
        """Initializes a FC neural network.
        
        Keyword Arguments:
            n_hidden {int} -- Number of base hidden units. (default: {16})
            shape {tuple} -- Shape of the recovery curve data. (default: {(1, 110)})
        """

        super(FC, self).__init__()

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
        
        # Remove channel dimension
        x = self.flatten(x, start_dim=1)

        y = self.head(x)

        return y
