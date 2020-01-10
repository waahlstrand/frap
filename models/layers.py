import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from utils import output_size_from_conv



class Conv3dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):

        super(Conv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):

        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)