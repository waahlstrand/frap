import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import alexnet
from models.convlstm import ConvLSTM
from models.layers import Conv3dBlock, Conv2dBlock


class Tratt(nn.Module):
    """DEPRECATED: USE ONLY TO TEST OLD RESULTS
    A downsampling convolutional neural network which extracts a FRAP sequence as a univariate sequence and outputs
    an estimate of the FRAP parameters.
    """

    def __init__(self, batch_size, shape=(1, 110, 256, 256)):
        """Initializes a Tratt module. The network is structured as a body of sequential convolutional layers which downsample
        the FRAP sequence frame-wise, which is fed to a 1D convolutional layer for noise reduction. The final head layer is a fully 
        connected network performing inference.
        
        Arguments:
            batch_size {int} -- The number of samples in a mini-batch. Necessary for automatic size detection in the network.
        
        Keyword Arguments:
            shape {tuple} -- Shape of the FRAP sequence data. (default: {(1, 110, 256, 256)})
        """

        super(Tratt, self).__init__()

        self.batch_size     = batch_size
        self.output_size    = 3
        self.shape          = shape

        self.in_channels     = self.shape[1]
        self.height          = 256
        self.width           = self.height

        self.body = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 4, stride=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3, stride=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

        )

        self.neck   = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, stride=1),
                                    nn.BatchNorm1d(1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2),
                                    )

        
        self.flatten    = torch.flatten
        flatten_size = self._get_conv_output_size(self.shape)


        self.head = nn.Sequential(
            nn.Linear(flatten_size, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, self.output_size)
        )

    def forward(self, x):

        x = x.squeeze()

        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        x = self.neck(x)
        x = self.flatten(x, start_dim=1)
        y = self.head(x)

        return y
    
    def _forward_through_body(self, x):
        """Pass through the body of the network.
        
        Arguments:
            x {torch.Tensor} -- Input recovery curve data.
        
        Returns:
            tuple -- A shape tuple after passing the data through the body.
        """
        
        x = x.squeeze()
        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        x = self.neck(x)

        return x

    def _get_conv_output_size(self, shape):
        """Calculates the size (number of elements except for the batch dimension) of the data after passing it through the CNNs.
        
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


class ConvFundo(nn.Module):
    """A ConvLSTM neural network which downsamples the FRAP sequence frame-wise and feeds it to a ConvLSTM. Struggles with
    overfitting and requires very many parameters.
    """

    def __init__(self, batch_size, input_shape=(1, 110, 256, 256)):
        """Initializes a ConvLSTM module. The network is structured as a body of sequential convolutional layers which downsample
        the FRAP sequence frame-wise, where the output is fed to a convolutional LSTM. The final head layer is a 
        fully connected network performing inference.
        
        Arguments:
            batch_size {int} -- The number of samples in a mini-batch. Necessary for automatic size detection in the network.
        
        Keyword Arguments:
            shape {tuple} -- Shape of the FRAP sequence data. (default: {(1, 110, 256, 256)})
        """

        super(ConvFundo, self).__init__()

        self.batch_size         = batch_size
        self.output_size        = 3
        self.input_shape        = input_shape
        self.in_channels        = self.input_shape[0]
        self.time_channels      = self.input_shape[1]
        self.hidden_channels    = 32
        self.n_layers           = 2
        self.n_convlstm         = 1

        self.body = nn.Sequential(Conv2dBlock(self.time_channels, self.time_channels, kernel_size=2, stride=1),
                                  nn.MaxPool2d(kernel_size=2),
                                  Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                  nn.MaxPool2d(kernel_size=2),
                                  Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                  Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                  nn.MaxPool2d(kernel_size=2),
                                  Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                  Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                  nn.MaxPool2d(kernel_size=2),
                                  Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                  Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                  nn.MaxPool2d(kernel_size=2)
                                )

        shape               = self._get_body_output_shape(input_shape)

        self.convlstm       = ConvLSTM(self.in_channels, self.hidden_channels, kernel_size = 3, n_layers = self.n_layers, shape=shape)

        self.avgpool        = nn.AdaptiveAvgPool3d(output_size=(self.time_channels,1,1))

        self.flatten        = torch.flatten
        self.fc             = nn.Sequential(nn.Linear(2*110*32, 4096),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5),
                                      nn.Linear(4096, 4096),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5),
                                      nn.Linear(4096, 3))

    def forward(self, x):

        # Downsampler
        x = x.squeeze()
        x = self.body(x)
        x = x.unsqueeze(1)

        # Encoder
        # Extract all layer outputs and concatenate

        xs = self.convlstm(x)
        
        xs = torch.cat(xs, dim=1)
        x = xs    

        # Regressor
        #x = self.neck(x)
        x = self.avgpool(x)
        x = self.flatten(x, 1)
        x = self.fc(x)

        return x



    def _forward_through_body(self, x):
        """Pass through the body of the network.
        
        Arguments:
            x {torch.Tensor} -- Input recovery curve data.
        
        Returns:
            tuple -- A shape tuple after passing the data through the body.
        """
        x = x.squeeze()
        x = self.body(x)
        x = x.unsqueeze(1)

        return x

    def _forward_through_neck(self, x):
        """Pass through the neck of the network.
        
        Arguments:
            x {torch.Tensor} -- Input recovery curve data.
        
        Returns:
            tuple -- A shape tuple after passing the data through the body.
        """
        #x = x.squeeze()
        x = self.neck(x)
        #x = x.unsqueeze(1)

        return x


    def _get_body_output_shape(self, shape):
        """Calculates the shape of the data after passing it through the CNNs.
        
        Arguments:
            shape {tuple} -- Input shape of the data.
        
        Returns:
            int -- Number of elements in data except for the batch dimension.
        """
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = list(output.shape)
            size.pop(0)

            return size

    def _get_body_output_size(self, shape):
        """Calculates the size (number of elements except for the batch dimension) of the data after passing it through the CNNs.
        
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


class Downsampler(nn.Module):
    """A downsampling convolutional neural network which extracts a FRAP sequence as a univariate sequence and outputs
    an estimate of the FRAP parameters.
    """

    def __init__(self, batch_size, shape=(1, 110, 256, 256)):
        """Initializes a Downsampler module. The network is structured as a body of sequential convolutional layers which downsample
        the FRAP sequence frame-wise, which is fed to a 1D convolutional layer for noise reduction. The final head layer is a fully 
        connected network performing inference.
        
        Arguments:
            batch_size {int} -- The number of samples in a mini-batch. Necessary for automatic size detection in the network.
        
        Keyword Arguments:
            shape {tuple} -- Shape of the FRAP sequence data. (default: {(1, 110, 256, 256)})
        """

        super(Downsampler, self).__init__()

        self.batch_size     = batch_size
        self.output_size    = 3
        self.shape          = shape

        self.in_channels     = self.shape[1]
        self.height          = 256
        self.width           = self.height

        self.body = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 4, stride=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3, stride=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )



        self.neck   = nn.Sequential(nn.Conv1d(1, 32, kernel_size=3, stride=1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2),
                                    )

        
        self.flatten    = torch.flatten
        flatten_size = self._get_conv_output_size(self.shape)


        self.head = nn.Sequential(
            nn.Linear(flatten_size, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, self.output_size)
        )

    def forward(self, x):

        x = x.squeeze()

        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        x = self.neck(x)
        x = self.flatten(x, start_dim=1)
        y = self.head(x)

        return y
    
    def _forward_through_body(self, x):
        """Pass through the body of the network.
        
        Arguments:
            x {torch.Tensor} -- Input recovery curve data.
        
        Returns:
            tuple -- A shape tuple after passing the data through the body.
        """
        
        x = x.squeeze()
        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        x = self.neck(x)

        return x

    def _get_conv_output_size(self, shape):
        """Calculates the size (number of elements except for the batch dimension) of the data after passing it through the CNNs.
        
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