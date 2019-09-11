import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import output_size_from_conv

class Convolution1D(nn.Module):

    def __init__(self, in_channels, out_channels, sequence_length, conv_kernel, maxpool_kernel, stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(Convolution1D, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.conv_kernel    = conv_kernel
        self.maxpool_kernel = maxpool_kernel
        self.length         = sequence_length

        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.groups         = groups
        self.bias           = bias


        self.conv = nn.Conv1d(self.in_channels, 
                              self.out_channels, 
                              self.conv_kernel, 
                              stride=self.stride, 
                              padding=self.padding, 
                              dilation=self.dilation, 
                              groups=self.groups, 
                              bias=self.bias)

        self.batch_norm = nn.BatchNorm1d(self.out_channels)
        self.maxpool1   = nn.MaxPool1d(self.maxpool_kernel)

        # Calculate the sizes from each convolution for the forward pass
        self.conv_size      = output_size_from_conv(self.length, 
                                                    self.conv_kernel)

        self.maxpool_size   = output_size_from_conv(self.conv_size, 
                                                    self.maxpool_kernel, 
                                                    stride=self.maxpool_kernel)

        self.output_size    = self.out_channels*self.maxpool_size

    def forward(self, x):

        x = F.relu(self.batch_norm(self.conv(x)))

        x = self.maxpool1(x)

        return x