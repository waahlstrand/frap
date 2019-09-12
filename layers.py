import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import output_size_from_conv


class BasicResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        super(BasicResNetBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               stride=stride, 
                               padding=dilation, 
                               groups=groups, 
                               bias=False, 
                               dilation=dilation)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               stride=stride, 
                               padding=dilation, 
                               groups=groups, 
                               bias=False, 
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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


    def forward(self, x):

        x = F.relu(self.batch_norm(self.conv(x)))

        x = self.maxpool1(x)

        return x

    def output_size(self):

        return self.out_channels*self.maxpool_size
