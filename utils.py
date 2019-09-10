import numpy as np


def output_size_from_conv(in_size, kernel_size, stride = 1, padding = 0, dilation = 1):
    
    size = np.floor( (in_size+ 2*padding-dilation*(kernel_size-1)-1)/stride +1 )
    
    return int(size)