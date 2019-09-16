import torch
import torch.nn as nn

class VoxNet(nn.Module):
    
    def __init__(self, output_size):

        super(VoxNet, self).__init__()

        self.output_size = output_size
        
        # Body of the neural network
        self.layer1     = VoxNetLayer(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.drop1      = nn.Dropout3d(p=0.2)
        self.layer2     = VoxNetLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.maxpool    = nn.MaxPool3d(2)
        self.drop2      = nn.Dropout3d(p=0.3)
        self.flatten    = torch.flatten

        # Regression head of the network
        self.head = torch.nn.Sequential(torch.nn.Linear(, 128),
                                        torch.nn.ReLU()
                                        torch.nn.Dropout(p=0.4),
                                        torch.nn.Linear(128, self.output_size))

    def forward(self, x):

        # Body of the neural network
        x = self.layer1(x)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.drop2(x)

        
        # Regression head of the network
        x = self.flatten(x, start_dim=1, end_dim=3)
        y = self.head(x)

        return y


class VoxNetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride)

        super(VoxNetLayer, self).__init__()

        self.conv   = nn.Conv3d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride)
                                
        self.lrelu  = nn.LeakyReLU()

    def forward(self, x):

        y = self.lrelu(self.conv(x))

        return y


