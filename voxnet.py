import torch
import torch.nn as nn
from utils import output_size_from_conv

class VoxNet(nn.Module):
    
    def __init__(self, batch_size, output_size, input_size = (1, 110, 256, 256)):

        super(VoxNet, self).__init__()
        self.batch_size  = batch_size
        self.output_size = output_size
        self.input_size  = input_size
        self.length      = self.input_size[2]
        
        # Body of the neural network
        self.body = nn.Sequential(VoxNetLayer(in_channels=1, out_channels=32, kernel_size=5, stride=2),
                                  nn.Dropout3d(p=0.2),
                                  VoxNetLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                                  nn.MaxPool3d(2),
                                  nn.Dropout3d(p=0.3))

        self.flatten    = torch.flatten

        flatten_size    = self._get_conv_output_size(self.input_size)

        # Regression head of the network
        self.head = nn.Sequential(nn.Linear(flatten_size, 128),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.4),
                                  nn.Linear(128, self.output_size))

    def _forward_through_body(self, x):

        y = self.body(x)

        return y

    def forward(self, x):

        # Body of the neural network
        x = self.body(x)

        # Regression head of the network
        x = self.flatten(x, start_dim=1)
        y = self.head(x)

        return y

    def _get_conv_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size


class VoxNetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(VoxNetLayer, self).__init__()

        self.conv   = nn.Conv3d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride)
                                
        self.lrelu  = nn.LeakyReLU()

    def forward(self, x):

        y = self.lrelu(self.conv(x))

        return y


