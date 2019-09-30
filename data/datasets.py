import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matlab.engine
import logging

class RecoveryDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        super(RecoveryDataset, self).__init__()

        self.n_samples_train    = 2**16
        self.n_samples_val      = 2**14
        self.sequence_length    = 110
        self.target_length      = 3

        # Load all data
        x_train = np.fromfile(root_dir + "/x_train.bin", dtype = np.float32)
        y_train = np.fromfile(root_dir + "/y_train.bin", dtype = np.float32)
        x_val   = np.fromfile(root_dir + "/x_val.bin", dtype = np.float32)
        y_val   = np.fromfile(root_dir + "/y_val.bin", dtype = np.float32)
        

        # Concatenate all data 
        x_train = np.reshape(x_train, (self.n_samples_train, self.sequence_length))
        y_train = np.reshape(y_train, (self.n_samples_train, self.target_length))

        x_val = np.reshape(x_val, (self.n_samples_val, self.sequence_length))
        y_val = np.reshape(y_val, (self.n_samples_val, self.target_length))

        self.inputs  = np.vstack((x_train, x_val))
        self.targets = np.vstack((y_train, y_val))

    def __len__(self):
            
        return len(self.inputs[:,0])

    def __getitem__(self, idx):

        return self.inputs[idx, :], self.targets[idx, :]


class RecoveryTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        super(RecoveryTrainingDataset, self).__init__()

        self.n_samples_train    = 2**16
        self.n_samples_val      = 2**14
        self.sequence_length    = 110
        self.target_length      = 3

        # Load all data
        x_train = np.fromfile(root_dir + "/x_train.bin", dtype = np.float32)
        y_train = np.fromfile(root_dir + "/y_train.bin", dtype = np.float32)
        

        # Concatenate all data 
        x_train = np.reshape(x_train, (self.n_samples_train, self.sequence_length))
        y_train = np.reshape(y_train, (self.n_samples_train, self.target_length))

        self.inputs  = x_train
        self.targets = y_train

    def __len__(self):
            
        return len(self.inputs[:,0])

    def __getitem__(self, idx):

        return self.inputs[idx, :], self.targets[idx, :]

class RecoveryValidationDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        super(RecoveryValidationDataset, self).__init__()

        self.n_samples_train    = 2**16
        self.n_samples_val      = 2**14
        self.sequence_length    = 110
        self.target_length      = 3

        # Load all data
        x_val   = np.fromfile(root_dir + "/x_val.bin", dtype = np.float32)
        y_val   = np.fromfile(root_dir + "/y_val.bin", dtype = np.float32)
        

        x_val = np.reshape(x_val, (self.n_samples_val, self.sequence_length))
        y_val = np.reshape(y_val, (self.n_samples_val, self.target_length))

        self.inputs  = x_val
        self.targets = y_val

    def __len__(self):
            
        return len(self.inputs[:,0])

    def __getitem__(self, idx):

        return self.inputs[idx, :], self.targets[idx, :]


class SpatiotemporalDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(SpatiotemporalDataset, self).__init__()

        self.shape      = (1, 110, 256, 256)
        self.n_params   = 3
        self.length     = 4096
        
        #X = np.fromfile("/home/sms/magnusro/frap_ann/generate_clean_data/x_1.bin", dtype = np.float32)
        #y = np.fromfile("/home/sms/magnusro/frap_ann/generate_clean_data/y_1.bin", dtype = np.float32)

        #self.X = np.reshape(X, (-1, *self.shape))
        #self.y = np.reshape(y, (-1, self.n_params))

    def __len__(self):
            
        return self.length

    def __getitem__(self, idx):


        X_path = os.path.join("/home/sms/vws/data/spatiotemporal", "x_"+str(idx+1)+".npy")
        y_path = os.path.join("/home/sms/vws/data/spatiotemporal", "y_"+str(idx+1)+".npy")


        if os.path.exists(X_path) and os.path.exists(y_path):

            X = np.load(X_path)
            y = np.load(y_path)

            return X, y
        else:
            raise IndexError("Index out of bounds.")
        

class MatlabGenerator(torch.utils.data.Dataset):

    def __init__(self, batch_size, noise_level=0.1, n_workers = 32):
        
        self.batch_size     = batch_size
        self.n_workers      = n_workers
        self.noise_level    = noise_level
        self.n_channels     = 1
        self.height         = 256
        self.width          = 256
        self.depth          = 110
        self.n_params       = 3

        logging.info("Initializing session. Entering MATLAB...")
        self.engine = self.initialize_session()
        self.engine.addpath(r'~/Documents/MATLAB/frappe',nargout=0)
        self.engine.addpath(r'~/magnusro/frap_ann/frap_matlab',nargout=0)

        logging.info("- Back in python. MATLAB session started.")

        self.batch  = None

    #@staticmethod
    def initialize_session(self):

        return matlab.engine.start_matlab()

    #@staticmethod
    def initialize_pool(self):
        
        logging.info("Initializing MATLAB parallel pool...")
        self.engine.initialize_pool(nargout=0)
        logging.info("- Pool initialized.")

    #@staticmethod
    def kill_pool(self):

        logging.info("Killing MATLAB parallel pool...")
        self.engine.kill_pool(nargout=0)
        logging.info("- Pool killed.")

    def generate_batch(self):

        success = self.engine.generate_batch_for_python(self.noise_level, self.batch_size, self.n_workers, nargout=1)

        X = np.fromfile(r'x.bin', dtype=np.float32)
        y = np.fromfile(r'y.bin', dtype=np.float32)

        #print("Batch generated! Reshaping values.")
        #y = np.asarray(matlab_batch[1], dtype=np.float32)
        #print("y loaded to np")
        #X = np.asarray(matlab_batch[0], dtype=np.float32)
        #print("X loaded to np")

        X = np.reshape(X, (self.batch_size, self.n_channels, self.depth, self.width, self.height))
        y = np.reshape(y, (self.batch_size, self.n_params))
        
        self.X = X
        self.y = y


    def __len__(self):

        return self.batch_size

    def __getitem__(self, idx):

        return self.X[idx, :, :, :, :], self.y[idx, :]


#data = MatlabGenerator(32, 16)
#data.initialize_pool()
#data.generate_batch()
#data.kill_pool()