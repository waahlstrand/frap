import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvLSTM(nn.Module):
    """A ConvLSTM module, similar to an LSTM module, but with convolutional gates instead of linear ones.
    """

    def __init__(self, n_input_channels, n_hidden_channels, kernel_size, n_layers=1, shape=(1, 110, 256, 256)):

        super(ConvLSTM, self).__init__()
        self.n_layers           = n_layers
        self.n_input_channels   = n_input_channels
        self.n_hidden_channels  = n_hidden_channels
        self.kernel_size        = kernel_size
        self.shape              = shape
        self.sequence_length    = self.shape[1]


        self.layers = []

        for idx in range(self.n_layers):
            if idx == 0: 
                n_channels = self.n_input_channels
            else: 
                n_channels = self.n_hidden_channels

            self.layers.append(ConvLSTMCell(n_channels, self.n_hidden_channels, self.kernel_size, self.shape[-1]))

        # Make sure the layers are found on the device
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, h = None):
        
        self.batch_size = x.shape[0]

        #states  = []
        outputs = []

        for layer in self.layers:

            # Initialize all layers/cells
            h, c = layer._initialize_hidden_state(self.batch_size)

            # Save outputs
            layer_states = []

            for t in range(self.sequence_length):

                # Loop through all the frames in the sequence
                frame = x[:, :, t, :, :]
                
                h, c = layer(frame, (h, c))
                
                layer_states.append(h)

            hs = torch.stack(layer_states, dim=2)

            # Send output sequence from layer as next layer sequence
            x  = hs

            #states.append((h, c))
            outputs.append(hs)

        return outputs




class ConvLSTMCell(nn.Module):
    """The cell of a ConvLSTM. The number of cells signifies the depth/number of layers of the LSTM.
    """

    def __init__(self, n_channels, n_hidden_channels, kernel_size, size):

        super(ConvLSTMCell, self).__init__()

        
        self.n_channels         = n_channels
        self.n_hidden_channels  = n_hidden_channels
        self.channel_axis       = 1
        self.size = size

        if isinstance(kernel_size, list):
            self.input_kernel, self.hidden_kernel = kernel_size
        else:
            self.input_kernel    = kernel_size
            self.hidden_kernel   = kernel_size
        
        self.padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels=self.n_channels + self.n_hidden_channels,
                              out_channels=4 * self.n_hidden_channels,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=False)

        self.norm    = nn.BatchNorm2d(4 * self.n_hidden_channels)
        


    def forward(self, x, state):

        h, c = state
        
        cat = torch.cat([x, h], dim=1) 
        
        convolution = self.norm(self.conv(cat))

        input_sum, forget_sum, output_sum, state_sum = torch.split(convolution, self.n_hidden_channels, dim=1) 

        i = torch.sigmoid(input_sum)
        f = torch.sigmoid(forget_sum)
        o = torch.sigmoid(output_sum)
        g = torch.tanh(state_sum)

        c = f * c + i * g
        h = o * torch.tanh(c)
        

        return h, c


    def _initialize_hidden_state(self, batch_size):
        
        if torch.cuda.is_available():
            h = torch.zeros(batch_size, self.n_hidden_channels, self.size, self.size).cuda()
            c = torch.zeros(batch_size, self.n_hidden_channels, self.size, self.size).cuda()
        else:
            h = torch.zeros(batch_size, self.n_hidden_channels, self.size, self.size)
            c = torch.zeros(batch_size, self.n_hidden_channels, self.size, self.size)

        return h, c
