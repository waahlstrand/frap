import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from models.layers import Convolution1D



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
            #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            #nn.BatchNorm2d(self.in_channels),
            #nn.ReLU(),
        )

        self.neck   = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3),
                                    nn.BatchNorm1d(1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2),
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
        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        y = self.neck(x)

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
        x = x.squeeze().unsqueeze(1)
        x = self.neck(x)
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


class BottomHeavyTratt(nn.Module):

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