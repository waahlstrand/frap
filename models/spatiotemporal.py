import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import alexnet
from models.resnet import resnet183d
from models.convlstm import ConvLSTM
from models.layers import InceptionBlock, Conv3dBlock, Conv2dBlock
#from models.ndrplz import ConvLSTM


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

        #self.body = nn.Sequential(
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 7, stride=3),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 3),
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3, stride=3),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 3),
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    #nn.MaxPool2d(kernel_size = 3),
        #    #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2),
        #    #nn.BatchNorm2d(self.in_channels),
        #    #nn.ReLU(),
        #    #nn.MaxPool2d(kernel_size = 2),
        #    #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2),
        #    #nn.BatchNorm2d(self.in_channels),
        #    #nn.ReLU(),
        #    #nn.MaxPool2d(kernel_size = 2),
        #    #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
        #    #nn.BatchNorm2d(self.in_channels),
        #    #nn.ReLU(),
        #)

        self.neck   = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, stride=1),
                                    nn.BatchNorm1d(1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2),
                                    )

        
        self.flatten    = torch.flatten
        flatten_size = self._get_conv_output_size(self.shape)


        #self.head = nn.Sequential(
        #    nn.Linear(flatten_size, 512), 
        #    nn.ReLU(),
        #    #nn.Dropout(p=0.4),
        #    nn.Linear(512, 128), 
        #    nn.ReLU(),
        #    nn.Linear(128, self.output_size)
        #)

        self.head = nn.Sequential(
            nn.Linear(flatten_size, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, self.output_size)
        )
    
    def _forward_through_body(self, x):
        
        x = x.squeeze()
        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        x = self.neck(x)

        return x

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

class FourierTratt(nn.Module):

    def __init__(self, batch_size, shape=(1, 110, 256, 256)):

        super(FourierTratt, self).__init__()

        self.batch_size     = batch_size
        self.output_size    = 3
        self.shape          = shape

        self.in_channels     = self.shape[1]
        self.height          = 256
        self.width           = self.height

        self.body = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3, stride=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2, stride=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 1),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 1),
            #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
            #nn.BatchNorm2d(self.in_channels),
            #nn.ReLU(),
        )

        #self.body = nn.Sequential(
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3, stride=1),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 2),
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2, stride=1),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 2),
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 2),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 2),
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 1),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 1),
        #    nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 1),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 1),
        #    #nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3),
        #    #nn.BatchNorm2d(self.in_channels),
        #    #nn.ReLU(),
        #)

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

class Fundo(nn.Module):

    def __init__(self, batch_size, shape=(1, 110, 256, 256)):

        super(Fundo, self).__init__()

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
            nn.MaxPool2d(kernel_size = 2)
        )

        self.neck   = nn.Sequential(nn.Conv1d(3, 3, kernel_size=3),
                                    nn.BatchNorm1d(3),
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
    
    def _forward_through_body(self, x, rcs, var):
        
        x = x.squeeze()
        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        rcs = rcs.unsqueeze(1)
        var = rcs

        x = torch.cat([x, rcs, var], dim=1)
        y = self.neck(x)

        return y

    def _get_conv_output_size(self, shape):
        
        with torch.no_grad():

            X = torch.rand(self.batch_size, *shape)
            rcs = torch.rand(self.batch_size, shape[1])

            output = self._forward_through_body(X, rcs, rcs)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

    def forward(self, x, rcs, var):

        x = x.squeeze()

        x = self.body(x)
        x = x.squeeze().unsqueeze(1)
        rcs = rcs.unsqueeze(1)
        var = var.unsqueeze(1)
        x = torch.cat([x, rcs, var], dim=1)
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
            #nn.Dropout(p=0.4),
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


class Net(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 110, 256, 256)):

        super(Net, self).__init__()

        self.batch_size     = batch_size
        self.output_size    = 3
        
        self.conv3d1    = nn.Conv3d(1, 1, kernel_size = (3, 1, 1), stride=(3, 1, 1))
        self.pool1      = nn.MaxPool3d(kernel_size=3)
        self.norm1      = nn.BatchNorm3d(1)
        shape           = self._get_conv3d1_output_size(input_shape)

        self.convlstm   = ConvLSTM(1, 32, kernel_size = 3, n_layers = 1, shape=shape)
        shape[0] = 1*32
        

        self.conv3d2    = nn.Conv3d(1*32, 1, kernel_size = 3, stride=2)
        self.norm2      = nn.BatchNorm3d(1)
        size            = self._get_conv3d2_output_size(shape)
        
        self.flatten    = torch.flatten
        self.linear1    = nn.Linear(size, 128)
        self.linear2    = nn.Linear(128, 64)
        self.linear3    = nn.Linear(64, self.output_size)


    def forward(self, x):

        # Encoder
        x = self.conv3d1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Extract all layer outputs and concatenate
        xs = self.convlstm(x)
        x = torch.cat(xs, dim=1)
        
        # Decoder
        x = self.conv3d2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        y = self.linear3(x)

        return y



    def _forward_through_conv3d1(self, x):
        
        x = self.conv3d1(x)
        y = self.pool1(x)

        return y

    def _forward_through_conv3d2(self, x):
        
        y = self.conv3d2(x)

        return y


    def _get_conv3d1_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_conv3d1(input)
            size = list(output.shape)
            size.pop(0)

            return size

    def _get_conv3d2_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_conv3d2(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

class FourierNet(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 110, 50, 50)):

        super(FourierNet, self).__init__()

        self.batch_size     = batch_size
        self.output_size    = 3
        self.input_shape    = list(input_shape)

        self.convlstm   = ConvLSTM(1, 32, kernel_size = 3, n_layers = 2, shape=self.input_shape)
        self.input_shape[0] = 2*32
        

        self.conv3d2    = nn.Conv3d(2*32, 1, kernel_size = 3, stride=2)
        self.pool2      = nn.MaxPool3d(kernel_size=(3,2,2))
        self.norm2      = nn.BatchNorm3d(1)
        size            = self._get_conv3d2_output_size(self.input_shape)
        
        self.flatten    = torch.flatten
        self.linear1    = nn.Linear(size, 512)
        self.linear2    = nn.Linear(512, 256)
        self.linear3    = nn.Linear(256, self.output_size)


    def forward(self, x):

        # Encode
        # Extract all layer outputs and concatenate
        xs = self.convlstm(x)
        x = torch.cat(xs, dim=1)
        
        # Decoder
        x = self.conv3d2(x)
        x = self.pool2(x)
        x = self.norm2(x)

        x = torch.relu(x)
        x = self.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        y = self.linear3(x)

        return y


    def _forward_through_conv3d2(self, x):
        
        x = self.conv3d2(x)
        y = self.pool2(x)

        return y


    def _get_conv3d2_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_conv3d2(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

class ConvFundo(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 110, 256, 256)):

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

        shape = self._get_body_output_shape(input_shape)

        self.convlstm       = ConvLSTM(self.in_channels, self.hidden_channels, kernel_size = 3, n_layers = self.n_layers, shape=shape)
        #self.backward_convlstm      = ConvLSTM(self.in_channels, self.hidden_channels, kernel_size = 3, n_layers = self.n_layers, shape=shape)

        #self.neck = nn.Sequential(Conv3dBlock(self.n_layers*self.n_convlstm*self.hidden_channels, 
        #                                      self.n_layers*self.n_convlstm*self.hidden_channels//2, 
        #                                      kernel_size=(3, 1, 1), stride=1),
        #                        nn.MaxPool3d(kernel_size=(2,1,1))
        #                        )

        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=(54,5,5))
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(self.time_channels,1,1))

        self.flatten = torch.flatten
        self.fc       = nn.Sequential(nn.Linear(2*110*32, 4096),
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

        ## Reverse the order in time
        ## Pass to backward convlstm
        #xs_flip = self.backward_convlstm(x.flip([2]))

        #xs_flip = torch.cat(xs_flip, dim=1) 

        ## Concatenate along channel dimension        
        #x = torch.cat([xs, xs_flip], dim=1)  

        # Regressor
        #x = self.neck(x)
        x = self.avgpool(x)
        x = self.flatten(x, 1)
        x = self.fc(x)

        return x



    def _forward_through_body(self, x):
        
        x = x.squeeze()
        x = self.body(x)
        x = x.unsqueeze(1)

        return x

    def _forward_through_neck(self, x):
        
        #x = x.squeeze()
        x = self.neck(x)
        #x = x.unsqueeze(1)

        return x


    def _get_body_output_shape(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = list(output.shape)
            size.pop(0)

            return size

    def _get_body_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

    def _get_neck_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_neck(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size


        
class I2D(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 110, 256, 256)):

        super(I2D, self).__init__()

        self.batch_size     = batch_size
        self.output_size    = 3
        self.input_shape    = input_shape
        in_channels    = self.input_shape[0]
        out_channels   = in_channels

        # Initial downsampling net
        self.body = nn.Sequential(nn.Conv3d(1, 2, kernel_size=(1,7,7)),
                                 nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),
                                 #nn.Conv3d(2, 4, kernel_size=(1,1,1)),
                                 nn.Conv3d(2, 8, kernel_size=(1,3,3)),
                                 nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2))
                                 )

        self.test       = nn.Sequential(InceptionBlock(8, 8),
                                        InceptionBlock(22, 16),
                                        nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2)),
                                        InceptionBlock(30, 16),
                                        InceptionBlock(30, 16),
                                        nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2)),
                                        InceptionBlock(30, 32),
                                        InceptionBlock(46, 32),
                                        nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2)),
                                        InceptionBlock(46, 48),
                                        InceptionBlock(62, 48),
                                        nn.AdaptiveAvgPool3d(output_size=(1,3,3)),
                                        nn.Conv3d(62, 16, kernel_size=(1,1,1))
                                        )

        self.neck       = nn.Sequential(InceptionBlock(32, 8),
                                        InceptionBlock(232, out_channels),
                                        nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2)),
                                        InceptionBlock(in_channels, out_channels),
                                        InceptionBlock(in_channels, out_channels),
                                        InceptionBlock(in_channels, out_channels),
                                        InceptionBlock(in_channels, out_channels),
                                        nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2)),
                                        InceptionBlock(in_channels, out_channels),
                                        InceptionBlock(in_channels, out_channels),
                                        nn.AdaptiveAvgPool3d(output_size=(1,7,7)),
                                        nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1))
                                        )
        
        self.flatten    = torch.flatten

        self.head       = nn.Linear(16*3*3, self.output_size)
        


    def forward(self, x):

        x = self.body(x)
        x = self.test(x)
        x = self.flatten(x, 1)
        x = self.head(x)
        # = self.neck(x)


        return x


class C3D(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 110, 256, 256)):
        super(C3D, self).__init__()

        self.batch_size     = batch_size
        self.output_size    = 3
        self.input_shape    = input_shape
        in_channels    = self.input_shape[0]
        out_channels   = in_channels
        # Initial downsampling net
        self.body = nn.Sequential(Conv3dBlock(1, 4, kernel_size=(3,3,3), stride=(1,1,1)),
                                  nn.MaxPool3d(kernel_size=(1,2,2)),
                                  Conv3dBlock(4, 8, kernel_size=(3,3,3), stride=(1,1,1)),
                                  nn.MaxPool3d(kernel_size=(2,2,2)),
                                  Conv3dBlock(8, 8, kernel_size=(3,3,3), stride=(1,1,1)),
                                  Conv3dBlock(8, 8, kernel_size=(3,3,3), stride=(1,1,1)),
                                  nn.MaxPool3d(kernel_size=(2,2,2)),
                                  Conv3dBlock(8, 16, kernel_size=(3,3,3), stride=(1,1,1)),
                                  Conv3dBlock(16, 16, kernel_size=(3,3,3), stride=(1,1,1)),
                                  nn.MaxPool3d(kernel_size=(2,2,2)),
                                  Conv3dBlock(16, 16, kernel_size=(3,3,3), stride=(1,1,1)),
                                  Conv3dBlock(16, 16, kernel_size=(3,3,3), stride=(1,1,1)),
                                  nn.MaxPool3d(kernel_size=(2,2,2))
                                )
        
        self.flatten    = torch.flatten
        self.head       = nn.Linear(16*3*4*4, self.output_size)
        


    def forward(self, x):

        x = self.body(x)
        #x = self.test(x)
        x = self.flatten(x, 1)
        x = self.head(x)
        # = self.neck(x)


        return x


class LRCN(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 110, 256, 256)):

        super(LRCN, self).__init__()

        self.batch_size         = batch_size
        self.output_size        = 3
        self.input_shape        = input_shape
        self.in_channels        = self.input_shape[0]
        self.time_channels      = self.input_shape[1]
        self.hidden_channels    = 32
        self.n_layers           = 2
        self.n_convlstm         = 1


        self.alexnet = alexnet(pretrained=True)

        #self.features = nn.Sequential(
        #    nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=3, stride=2),
        #    nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=3, stride=2),
        #    nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=3, stride=2),
        #)
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        
        self.lstm1 = nn.LSTM(3, 16, num_layers=1)
        self.lstm2 = nn.LSTM(16, 3, num_layers=1)
        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=(self.time_channels,1,1))

        
        for child in self.alexnet.children():
            for param in child.parameters():
                param.requires_grad = False

        self.alexnet.classifier = nn.Sequential(nn.Dropout(),
                                                nn.Linear(256 * 6 * 6, 4096),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(),
                                                nn.Linear(4096, 4096),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(4096, self.output_size)
                                                )

        self.fc = nn.Linear(110*3, 3)

    def forward(self, x):

        frames = []

        for frame in x.split(1, dim=2):

            frame = frame.squeeze(dim=1)
            frame = torch.cat((frame, frame, frame), dim=1)
            frame = self.alexnet(frame)

            #frame = self.features(frame)
            #frame = self.avgpool(frame)
#
            ##print(frame.shape)
            #frame = torch.flatten(frame, 1)
            #frame = self.classifier(frame)
            frames.append(frame)

        x = torch.stack(frames, dim=1)

        x = self.lstm1(x)
        x = self.lstm2(x[0])
        x = x[0]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = torch.mean(x, 1)

        return x


class TimeSampled(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 110, 256, 256)):

        super(TimeSampled, self).__init__()

        self.batch_size         = batch_size
        self.output_size        = 3
        self.input_shape        = input_shape
        self.in_channels        = self.input_shape[0]
        self.time_channels      = self.input_shape[1]
        self.height             = self.input_shape[-1]
        self.width              = self.input_shape[-2]
        self.hidden_channels    = 16
        self.n_layers           = 1
        self.n_convlstm         = 1

        self.step_size          = 10
        self.index_to_select    = [0, *range(10, 30), *range(29,self.time_channels, self.step_size)]
        #self.index_to_select    = [*range(0, 110)]
        if torch.cuda.is_available():
            self.index_to_select    = torch.tensor(self.index_to_select).cuda(device=0)

        else: 
            self.index_to_select    = torch.tensor(self.index_to_select).cpu()

        self.sequence_length    = len(self.index_to_select)
        self.time_channels      = self.sequence_length


        self.body = nn.Sequential(Conv2dBlock(self.time_channels, self.time_channels, kernel_size=5, stride=1),
                                nn.MaxPool2d(kernel_size=3),
                                #Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                #nn.MaxPool2d(kernel_size=2),
                                #Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                #Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                #nn.MaxPool2d(kernel_size=2),
                                #Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                #Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                #nn.MaxPool2d(kernel_size=2),
                                #Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                #Conv2dBlock(self.time_channels, self.time_channels, kernel_size=3, stride=1),
                                #nn.MaxPool2d(kernel_size=2)
                                )

        #shape = self._get_body_output_shape(input_shape)
        shape = (self.in_channels, self.sequence_length, self.width, self.height)

        self.convlstm       = ConvLSTM(self.in_channels, self.hidden_channels, kernel_size = 3, n_layers = self.n_layers, shape=shape)
        #self.backward_convlstm      = ConvLSTM(self.in_channels, self.hidden_channels, kernel_size = 3, n_layers = self.n_layers, shape=shape)

        #self.neck = nn.Sequential(Conv3dBlock(self.n_layers*self.n_convlstm*self.hidden_channels, 
        #                                      self.n_layers*self.n_convlstm*self.hidden_channels//2, 
        #                                      kernel_size=(3, 1, 1), stride=1),
        #                        nn.MaxPool3d(kernel_size=(2,1,1))
        #                        )

        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=(54,5,5))
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(self.sequence_length,1,1))

        self.flatten = torch.flatten
        self.fc       = nn.Linear(self.hidden_channels*self.sequence_length, 3)

    def forward(self, x):

        # Sample in time
        x = torch.index_select(x, dim=2, index=self.index_to_select)
        #x = x.unsqueeze(1)

        # Downsampler
        #x = x.squeeze()
        #x = self.body(x)
        #x = x.unsqueeze(1)
        # Encoder
        # Extract all layer outputs and concatenate

        xs = self.convlstm(x)
        
        xs = torch.cat(xs, dim=1)
        x = xs    

        ## Reverse the order in time
        ## Pass to backward convlstm
        #xs_flip = self.backward_convlstm(x.flip([2]))

        #xs_flip = torch.cat(xs_flip, dim=1) 

        ## Concatenate along channel dimension        
        #x = torch.cat([xs, xs_flip], dim=1)  

        # Regressor
        #x = self.neck(x)
        x = self.avgpool(x)
        x = self.flatten(x, 1)
        x = self.fc(x)

        return x



    def _forward_through_body(self, x):

        x = torch.index_select(x, dim=2, index=self.index_to_select.cpu())
        
        x = x.squeeze()
        x = self.body(x)
        x = x.unsqueeze(1)

        return x

    def _forward_through_neck(self, x):
        
        #x = x.squeeze()
        x = self.neck(x)
        #x = x.unsqueeze(1)

        return x


    def _get_body_output_shape(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = list(output.shape)
            size.pop(0)

            return size

    def _get_body_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_body(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size

    def _get_neck_output_size(self, shape):
        
        with torch.no_grad():

            input = torch.rand(self.batch_size, *shape)
            output = self._forward_through_neck(input)
            size = output.data.view(self.batch_size, -1).size(1)

            return size