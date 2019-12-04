import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from utils import output_size_from_conv


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, pool_features):

        super(InceptionBlock, self).__init__()

        self.branch1x1      = Conv3dBlock(in_channels, 4, kernel_size=1)

        self.branch5x5_1    = Conv3dBlock(in_channels, 3, kernel_size=1)
        self.branch5x5_2    = Conv3dBlock(3, 4, kernel_size=(1,5,5), padding=(0,2,2))

        self.branch3x3dbl_1 = Conv3dBlock(in_channels, 4, kernel_size=1)
        self.branch3x3dbl_2 = Conv3dBlock(4, 6, kernel_size=(1,3,3), padding=(0,1,1))
        self.branch3x3dbl_3 = Conv3dBlock(6, 6, kernel_size=(1,3,3), padding=(0,1,1))

        self.branch_pool = Conv3dBlock(in_channels, pool_features, kernel_size=1)

    def forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool3d(x, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


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