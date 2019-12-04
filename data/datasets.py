import torch
import torch.nn as nn
from torch.distributions import uniform
from torch.utils.data import Dataset, DataLoader

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2

import utils

import os
import re
import shutil
import random
import logging

import matlab.engine

class RecoveryDataset(torch.utils.data.Dataset):
    """A pre-generated dataset of 4096 recovery curves, collected as files separated in x,y and train, val, test.
    The data is then pooled to support custom validation.
    The dataset supports batch training with torch.utils.data.Dataloader.

    """

    def __init__(self, root_dir):
        """Initializes the recovery dataset given a root directory.
        
        Arguments:
            root_dir {str} -- A root directory containing the recovery curve data.
        """
        super(RecoveryDataset, self).__init__()

        self.n_samples_train    = 2**16
        self.n_samples_val      = 2**14
        self.sequence_length    = 110
        self.target_length      = 3
        self.n_curves           = 31

        # Load all data
        x_train = np.fromfile(root_dir + "/x_train.bin", dtype = np.float32)
        y_train = np.fromfile(root_dir + "/y_train.bin", dtype = np.float32)
        x_val   = np.fromfile(root_dir + "/x_val.bin", dtype = np.float32)
        y_val   = np.fromfile(root_dir + "/y_val.bin", dtype = np.float32)
        

        # Concatenate all data 
        x_train = np.reshape(x_train, (self.n_samples_train, self.n_curves, self.sequence_length))
        y_train = np.reshape(y_train, (self.n_samples_train, self.target_length))

        x_val = np.reshape(x_val, (self.n_samples_val, self.n_curves, self.sequence_length))
        y_val = np.reshape(y_val, (self.n_samples_val, self.target_length))

        self.inputs  = np.vstack((x_train, x_val))
        self.targets = np.vstack((y_train, y_val))

    def __len__(self):
            
        return len(self.inputs[:,:, 0])

    def __getitem__(self, idx):

        return {"X": self.inputs[idx, :, :], "y": self.targets[idx, :]}



class SpatiotemporalDataset(torch.utils.data.Dataset):

    def __init__(self, directory="/home/sms/vws/data/spatiotemporal/"):
        super(SpatiotemporalDataset, self).__init__()

        self.shape      = (1, 110, 256, 256)
        self.n_channels     = 1
        self.height         = 256
        self.width          = 256
        self.depth          = 110
        self.n_params   = 3
        self.length     = 4096
        self.directory  = directory
        
        #X = np.fromfile("/home/sms/magnusro/frap_ann/generate_clean_data/x_1.bin", dtype = np.float32)
        #y = np.fromfile("/home/sms/magnusro/frap_ann/generate_clean_data/y_1.bin", dtype = np.float32)

        #self.X = np.reshape(X, (-1, *self.shape))
        #self.y = np.reshape(y, (-1, self.n_params))

    def __len__(self):
            
        return self.length

    def __getitem__(self, idx):

        # List all files in paired order
        x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'x+.*\.bin', f)]
        y_files =  ["y_"+f.partition("_")[2] for f in x_files]

        X_path = os.path.join(self.directory, "data", x_files[idx])
        y_path = os.path.join(self.directory, "data", y_files[idx])
        y = np.fromfile(y_path, dtype=np.float32)
        y = np.reshape(y, (self.n_params))
        X = np.fromfile(X_path, dtype=np.float32)
        X = np.reshape(X, (self.n_channels, self.depth, self.width, self.height))

        batch = {"X": X,
                "y": y}

        return batch


class TemporalDataset(torch.utils.data.Dataset):
    """A pre-generated dataset of 2^16 recovery curves, collected as single files x, y.
    The dataset supports batch training with torch.utils.data.Dataloader.
    """

    def __init__(self, directory="/home/sms/vws/data/temporal/data"):
        """Initializes the recovery dataset given a root directory.
        
        Arguments:
            root_dir {str} -- A root directory containing the recovery curve data.
        """
        
        super(TemporalDataset, self).__init__()

        self.shape      = (1, 110)
        self.n_channels     = 31
        self.depth          = 110
        self.n_params       = 3
        #self.length     = 2 ** 16
        self.directory  = directory
        
        # List all files in paired order
        self.x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'x+.*\.bin', f)]
        self.y_files =  ["y_"+f.partition("_")[2] for f in self.x_files]

        self.length = len(self.x_files)

    def __len__(self):
        """Returns the number of samples in the dataset.
        
        Returns:
            int -- Number of samples in the dataset
        """
            
        return self.length

    def __getitem__(self, idx):
        """Returns a single sample by index in the directory.
        
        Arguments:
            idx {int} -- An integer index, must be 0 =< idx < length
        
        Returns:
            dict -- A dict of samples; "X" denoting features and "y" denoting targets.
        """

        X_path = os.path.join(self.directory, "data", self.x_files[idx])
        y_path = os.path.join(self.directory, "data", self.y_files[idx])
        y = np.fromfile(y_path, dtype=np.float32)
        y = np.reshape(y, (self.n_params))
        X = np.fromfile(X_path, dtype=np.float32)
        X = np.reshape(X, (self.n_channels, self.depth))

        batch = {"X": X,
                "y": y}

        return batch

        

class MatlabGenerator(torch.utils.data.Dataset):

    def __init__(self, batch_size, directory, transform=False, mode="spatiotemporal", noise_level=0.1, n_workers = 32):
        
        self.batch_size     = batch_size
        self.n_workers      = n_workers
        self.noise_level    = noise_level
        self.n_channels     = 1
        self.height         = 256
        self.width          = 256
        self.depth          = 110
        self.n_params       = 3
        self.crop_size      = 20
        self.mode           = mode
        self.directory      = directory
        self.transform      = transform 
        self.transformer    = Multiplier(batch_size)
        #self.transformer    = Occlusion(frequency=0.5)

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

        if self.mode == "fourier":
            success = self.engine.generate_fourier_batch_for_python(self.noise_level, self.crop_size, self.batch_size, self.directory, self.n_workers, nargout=1)    
        else: 
            success = self.engine.generate_batch_for_python(self.noise_level, self.batch_size, self.directory, self.n_workers, nargout=1)

        y = np.fromfile(os.path.join(self.directory, 'y.bin'), dtype=np.float32)
        y = np.reshape(y, (self.batch_size, self.n_params))

        self.y = y
        self.X      = None
        self.rcs    = None
        self.var    = None


        if self.mode == "spatiotemporal":
        
            X = np.fromfile(os.path.join(self.directory, 'x.bin'), dtype=np.float32)
            X = np.reshape(X, (self.batch_size, self.n_channels, self.depth, self.width, self.height))

            if self.transform:
                self.X = self.transformer(X)
            else:
                self.X = X

        if self.mode == "fourier":
        
            X = np.fromfile(os.path.join(self.directory, 'x.bin'), dtype=np.float32)
            X = np.reshape(X, (self.batch_size, self.n_channels, self.depth, self.crop_size, self.crop_size))
            self.X = X

        elif self.mode == "temporal":

            X = np.fromfile(os.path.join(self.directory, 'rcs.bin'), dtype=np.float32)
            X = np.reshape(X, (self.batch_size, self.n_channels, self.depth))
            self.X = X

        elif self.mode == "both":

            rcs = np.fromfile(os.path.join(self.directory, 'rcs.bin'), dtype=np.float32)
            rcs = np.reshape(rcs, (self.batch_size, 1, self.depth))

            var = np.fromfile(os.path.join(self.directory, 'var.bin'), dtype=np.float32)
            var = np.reshape(var, (self.batch_size, 1, self.depth))

            self.X      = torch.cat((rcs, var), dim=1)
        
        elif self.mode == "multichannel":

            self.n_channels = 2

            X = np.fromfile(os.path.join(self.directory, 'x.bin'), dtype=np.float32)
            X = np.reshape(X, (self.batch_size, self.n_channels, self.depth, self.width, self.height))



            if self.transform:
                self.X = self.transformer(X)
            else:
                self.X = X
           


        elif self.mode == "all":
            X   = np.fromfile(os.path.join(self.directory, 'x.bin'), dtype=np.float32)
            X = np.reshape(X, (self.batch_size, self.n_channels, self.depth, self.width, self.height))

            rcs = np.fromfile(os.path.join(self.directory, 'rcs.bin'), dtype=np.float32)
            rcs = np.reshape(rcs, (self.batch_size, self.depth))

            var = np.fromfile(os.path.join(self.directory, 'var.bin'), dtype=np.float32)
            var = np.reshape(var, (self.batch_size, self.depth))

            self.X      = X
            self.rcs    = rcs
            self.var    = var

    def __len__(self):

        return self.batch_size

    def __getitem__(self, idx):
        
        if self.mode == "spatiotemporal" or self.mode == "fourier":
            batch = {"X": self.X[idx, :, :, :, :],
                    "y": self.y[idx, :]}

        elif self.mode == "temporal" or self.mode == "both":
            batch = {"X": self.X[idx, :, :],
                    "y": self.y[idx, :]}

        elif self.mode == "all":
            batch = {"X": self.X[idx, :, :, :, :],
                    "rcs": self.rcs[idx, :],
                    "var": self.var[idx, :],
                    "y": self.y[idx, :]}

        return batch


class MixedMatlabGenerator(torch.utils.data.Dataset):

    def __init__(self, dataset_size, batch_size, directory, transform=False, mode="spatiotemporal", noise_level=0.1, n_workers = 32):
        
        self.batch_size     = batch_size
        self.dataset_size   = dataset_size
        self.n_workers      = n_workers
        self.noise_level    = noise_level
        self.n_channels     = 1
        self.height         = 256
        self.width          = 256
        self.depth          = 110
        self.n_params       = 3
        self.crop_size      = 20
        self.mode           = mode
        self.directory      = directory
        self.transform      = transform 
        self.transformer    = Multiplier(batch_size)
        #self.transformer    = Occlusion(frequency=0.5)

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

        success = self.engine.generate_batch_for_python(self.noise_level, self.dataset_size, self.directory, self.n_workers, nargout=1)

        y = np.fromfile(os.path.join(self.directory, 'data', 'y.bin'), dtype=np.float32)
        y = np.reshape(y, (self.dataset_size, self.n_params))

        self.y = y
        self.X      = None
        self.rcs    = None
        self.var    = None


        if self.mode == "spatiotemporal":
        
            X = np.fromfile(os.path.join(self.directory, 'data', 'x.bin'), dtype=np.float32)
            X = np.reshape(X, (self.dataset_size, self.n_channels, self.depth, self.width, self.height))

            if self.transform:
                self.X = self.transformer(X)
            else:
                self.X = X


        elif self.mode == "temporal":

            X = np.fromfile(os.path.join(self.directory, 'data', 'rcs.bin'), dtype=np.float32)
            X = np.reshape(X, (self.dataset_size, self.n_channels, self.depth))
            self.X = X

        elif self.mode == "both":

            rcs = np.fromfile(os.path.join(self.directory, 'data', 'rcs.bin'), dtype=np.float32)
            rcs = np.reshape(rcs, (self.dataset_size, 1, self.depth))

            var = np.fromfile(os.path.join(self.directory, 'data', 'var.bin'), dtype=np.float32)
            var = np.reshape(var, (self.dataset_size, 1, self.depth))

            self.X      = torch.cat((rcs, var), dim=1)
        

            if self.transform:
                self.X = self.transformer(X)
            else:
                self.X = X
           

        elif self.mode == "all":
            X   = np.fromfile(os.path.join(self.directory, 'data', 'x.bin'), dtype=np.float32)
            X = np.reshape(X, (self.dataset_size, self.n_channels, self.depth, self.width, self.height))

            rcs = np.fromfile(os.path.join(self.directory, 'data', 'rcs.bin'), dtype=np.float32)
            rcs = np.reshape(rcs, (self.dataset_size, self.depth))

            var = np.fromfile(os.path.join(self.directory, 'data', 'var.bin'), dtype=np.float32)
            var = np.reshape(var, (self.dataset_size, self.depth))

            self.X      = X
            self.rcs    = rcs
            self.var    = var

    def __len__(self):

        return self.dataset_size

    def __getitem__(self, idx):
        
        if self.mode == "spatiotemporal":
            batch = {"X": self.X[idx, :, :, :, :],
                    "y": self.y[idx, :]}

        elif self.mode == "temporal" or self.mode == "both":
            batch = {"X": self.X[idx, :, :],
                    "y": self.y[idx, :]}

        elif self.mode == "all":
            batch = {"X": self.X[idx, :, :, :, :],
                    "rcs": self.rcs[idx, :],
                    "var": self.var[idx, :],
                    "y": self.y[idx, :]}

        return batch



class TransferFromFiles(torch.utils.data.Dataset):

    def __init__(self, dataset_size, batch_size, directory, shape, transform=False, mode="spatiotemporal", noise_level=0.1, n_workers = 32,):
        
        self.batch_size     = batch_size
        self.dataset_size   = dataset_size
        self.n_workers      = n_workers
        self.noise_level    = noise_level
        self.n_channels     = 1
        self.height         = shape[-1]
        self.width          = shape[-2]
        self.depth          = shape[-3]
        self.n_params       = 3
        self.crop_size      = 20
        self.mode           = mode
        self.directory      = directory
        self.transform      = transform 
        self.multiple_files = True

        self.maximum        = np.array([np.log10(1e-9/np.power((7.5e-07),2)), 1.0, 0.95]).astype(np.float32)
        self.minimum        = np.array([np.log10(1e-12/np.power((7.5e-07),2)), 0.5, 0.5]).astype(np.float32)
        #self.transformer    = Occlusion(frequency=0.5)

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

        success = self.engine.generate_batch(self.noise_level, self.dataset_size, self.directory, self.n_workers, 0, self.multiple_files, nargout=1)

    def augment_batch(self, fraction=0.05):


        x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'x_+.*\.bin', f)]

        n_to_add = int(fraction*self.dataset_size)

        success = self.engine.generate_batch(self.noise_level, n_to_add, self.directory, self.n_workers, 1, 1, nargout=1)

        # Get temporary files
        x_temp_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'xtemp+.*\.bin', f)]
        y_temp_files =  ["ytemp_"+f.partition("_")[2] for f in x_temp_files]

        # Get random fraction of data to change
        x_files_slice = random.sample(x_files, n_to_add)
        y_files_slice =  ["y_"+f.partition("_")[2] for f in x_files_slice]



        for i in range(len(x_files_slice)):
            
            x_old_path = os.path.join(self.directory, "data", x_temp_files[i])
            x_new_path = os.path.join(self.directory, "data", x_files_slice[i])
            
            y_old_path = os.path.join(self.directory, "data", y_temp_files[i])
            y_new_path = os.path.join(self.directory, "data", y_files_slice[i])
            
            shutil.copy(x_old_path,x_new_path)
            shutil.copy(y_old_path, y_new_path)



    def __len__(self):

        return self.dataset_size

    def __getitem__(self, idx):

        # List all files in paired order
        x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'x_+.*\.bin', f)]
        y_files =  ["y_"+f.partition("_")[2] for f in x_files]

        
        if self.mode == "spatiotemporal":

            X_path = os.path.join(self.directory, "data", x_files[idx])
            y_path = os.path.join(self.directory, "data", y_files[idx])

            y = np.fromfile(y_path, dtype=np.float32)
            y = np.reshape(y, (self.n_params))

            X = np.fromfile(X_path, dtype=np.float32)
            X = np.reshape(X, (self.n_channels, self.depth, self.width, self.height))

            #y = utils.minmax(y, minimum=self.minimum, maximum=self.maximum)

            batch = {"X": X,
                    "y": y}

        return batch

        

class Occlusion(object):

    def __init__(self, frequency=0.7, percent=0.1, size_percent=0.01):
        sometimes = lambda aug: iaa.Sometimes(frequency, aug)
        self.seq = iaa.Sequential( sometimes (iaa.CoarseDropout(percent, size_percent=size_percent)))
        
    def __call__(self, X):

        image   = np.squeeze(X)

        image = self.seq(images=image)

        return np.reshape(image, (X.shape))


class Multiplier(object):

    def __init__(self, batch_size, lower_bound=0.7, upper_bound=1.3):

        self.batch_size = batch_size
        self.lb         = lower_bound
        self.ub         = upper_bound
        
    def __call__(self, X):

        distribution = uniform.Uniform(torch.Tensor([self.lb]),torch.Tensor([self.ub]))
        multiplier = distribution.sample(torch.Size([self.batch_size]))

        return multiplier[:, None, None, None].numpy() * X