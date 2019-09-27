import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import output_size_from_conv
from models.layers import Convolution1D



class Tratt(nn.Module):

    def __init__(self, batch_size, shape=(1, 110, 256, 256)):

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
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3),
            #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            #nn.BatchNorm2d(self.in_channels),
            #nn.ReLU(),
        )
        
        self.flatten    = torch.flatten
        flatten_size = self._get_conv_output_size(self.shape)


        self.head = nn.Sequential(
            nn.Linear(flatten_size, 512), 
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, self.output_size)
        )
    
    def _forward_through_body(self, x):
        
        x = x.squeeze()
        y = self.body(x)

        return y

    def _get_conv_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

    def forward(self, x):

        x = x.squeeze()

        x = self.body(x)

        x = self.flatten(x, start_dim=1)
        y = self.head(x)

        return y


class TopHeavyTratt(nn.Module):

    def __init__(self, batch_size, shape=(1, 110, 256, 256)):

        super(TopHeavyTratt, self).__init__()

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
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            #nn.BatchNorm2d(self.in_channels),
            #nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(32, 16, kernel_size=3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            #nn.MaxPool3d(kernel_size=2)
        )
        
        self.flatten    = torch.flatten
        flatten_size = self._get_conv_output_size(self.shape)


        self.head = nn.Sequential(
            nn.Linear(flatten_size, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, self.output_size)
        )
    
    def _forward_through_body(self, x):
        
        # Reshape for 2D CNNs
        x = x.squeeze()
        x = self.body(x)

        # Reshape again for 3D CNNs
        x = x.unsqueeze(1)

        y = self.neck(x)

        return y

    def _get_conv_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

    def forward(self, x):

        in_shape = x.shape

        # Reshape for 2D CNNs
        x = x.squeeze()
        x = self.body(x)

        # Reshape again for 3D CNNs
        x = x.unsqueeze(1)

        x = self.neck(x)

        x = self.flatten(x, start_dim=1)
        y = self.head(x)

        return y


class ConvLSTM(nn.Module):

    def __init__(self, n_channels, n_hidden_channels, kernel_size, n_layers=1, shape=(1, 110, 256, 256)):

        super(ConvLSTM, self).__init__()
        self.n_layers           = n_layers
        self.n_input_channels   = n_channels
        self.n_hidden_channels  = n_hidden_channels
        self.kernel_size        = kernel_size
        self.shape              = shape
        self.sequence_length    = self.shape[1]


        self.layers = []

        # TODO: Implement multiple layers with different number of filters
        for idx in range(self.n_layers):
            if idx == 0: 
                n_channels = self.n_input_channels
            else: 
                n_channels = self.n_hidden_channels

            self.layers.append(ConvLSTMCell(n_channels, self.n_hidden_channels, self.kernel_size, self.shape))

    def forward(self, x, h = None):

        for layer in self.layers:
            ###############################
            for t in range(self.sequence_length):
                ##################################
                ##################################


        return xs, hs



class ConvLSTMCell(nn.Module):

    def __init__(self, n_channels, n_hidden_channels, kernel_size, shape=(1, 110, 256, 256)):

        super(ConvLSTMCell, self).__init__()

        self.shape = shape

        self.n_channels         = n_channels
        self.n_hidden_channels  = n_hidden_channels
        self.channel_axis       = 1
        self.activation         = F.relu

        if isinstance(kernel_size, list):
            self.input_kernel, self.hidden_kernel = kernel_size
        else:
            self.input_kernel    = kernel_size
            self.hidden_kernel   = kernel_size

        self.conv           = nn.Conv2d(self.n_channels,        self.n_hidden_channels, self.input_kernel,  bias=True)
        self.recurrent_conv = nn.Conv2d(self.n_hidden_channels, self.n_hidden_channels, self.hidden_kernel, bias=False)



    def forward(self, x, state):

        h, c = state

        #updated = torch.cat([x, h], dim=self.channel_axis)

        # Convolved input states
        x_i = self.conv(x)
        x_f = self.conv(x)
        x_c = self.conv(x)
        x_o = self.conv(x)

        # Convolved hidden states
        h_i = self.recurrent_conv(h)
        h_f = self.recurrent_conv(h)
        h_c = self.recurrent_conv(h)
        h_o = self.recurrent_conv(h)

        # Calculate gate values
        i = self.activation(x_i + h_i)
        f = self.activation(x_f + h_f)
        c = f * c + i * F.tanh(x_c + h_c)
        o = self.activation(x_o + h_o)
        h = o * F.tanh(c)

        return h, (h, c)


