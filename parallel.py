"""Parallell implementation of neural network training using pytorch. Calls a Matlab script as
as parallel process, generating a batch of training data.
"""
import torch
import torch.nn as nn
import numpy as np
import matlab.engine
from torch.utils.data import Dataset, DataLoader

class MatlabGenerator(torch.utils.data.Dataset):

    def __init__(self, batch_size):
        
        self.batch_size = batch_size
        self.n_channels = 1
        self.height     = 256
        self.width      = 256
        self.depth      = 110
        self.n_params   = 3

        print("Initializing session. Entering MATLAB.")
        self.engine = self.initialize_session()
        self.engine.addpath(r'~/Documents/MATLAB/frappe',nargout=0)
        self.engine.addpath(r'~/magnusro/frap_ann/frap_matlab',nargout=0)

        print("Back in python. MATLAB session started.")

        self.batch  = None

    @staticmethod
    def initialize_session(self):

        return matlab.engine.start_matlab()

    @staticmethod
    def initialize_pool(self):
        
        print("Initializing MATLAB parallel pool.")
        self.engine.initialize_pool(nargout=0)

    @staticmethod
    def kill_pool(self):

        print("Killing MATLAB parallel pool.")
        self.engine.kill_pool(nargout=0)

    def generate_batch(self):

        matlab_batch = np.ndarray(self.engine.generate_batch_for_python(nargout=2))

        X = np.reshape(matlab_batch[0], (self.batch_size, self.n_channels, self.depth, self.width, self.height))
        y = np.reshape(matlab_batch[1], (self.batch_size, self.n_params))

        self.X = X
        self.y = y


    def __len__(self):

        return self.batch_size

    def __getitem__(self, idx):

        return X[idx, :, :, :, :], y[idx, :]



def start_matlab_pool(engine):

    engine.initialize_pool(nargout=0)

# Define an output queue
def python_wrapper(engine):
    print("Inside function! Entering matlab")
    result =  engine.matlab_func(nargout=1)
    print("Got out of matlab!")
    return result

#print("Starting test")
#print("Starting Matlab pool")
#eng.addpath(r'~/Documents/MATLAB/frappe',nargout=0)
#start_matlab_pool(eng)
#print(eng.eval("number_of_workers"))
#print("Entering first function")
#result1 = python_wrapper(eng)
#print("Entering second function")
#result2 = python_wrapper(eng)
#print(result1, result2)

generator = MatlabGenerator()

generator.initialize_pool()
generator.generate_batch()
print(generator.batch[1])
generator.kill_pool()