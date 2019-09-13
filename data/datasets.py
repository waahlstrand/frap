import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
