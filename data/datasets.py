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
    """A torch.utils.data.Dataset that given a directory collects binary spatio-temporal FRAP data and returns numpy arrays.
    """

    def __init__(self, directory="/home/sms/vws/data/spatiotemporal/", regex=r'x_+.*\.bin'):
        """Initializes a SpatiotemporalDataset from a directory of simulated binary FRAP files.
        
        Keyword Arguments:
            directory {str} -- Path of the directory with a subfolder /data/ with simulated data. (default: {"/home/sms/vws/data/spatiotemporal/"})
            regex {regexp} -- regexp object to find all files in directory. (default: {r'x_+.*\.bin'})
        """
        super(SpatiotemporalDataset, self).__init__()

        self.shape      = (1, 110, 256, 256)
        self.n_channels     = 1
        self.height         = 256
        self.width          = 256
        self.depth          = 110
        self.n_params   = 3
        self.directory  = directory

        # List all files in paired order
        self.x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.search(regex, f)]
        self.y_files =  ["y_"+f.partition("_")[2] for f in self.x_files]

        self.length = len(self.x_files)

    def __len__(self):
        """The dataset size, i.e. the number of files in directory.
        
        Returns:
            int -- The number of files in the directory.
        """
            
        return self.length

    def __getitem__(self, idx):
        """Indexing function for the dataset. Indexes the files as ordered in the directory, since order is unimportant.
        Opens the binary files and returns a numpy array of the FRAP data in order (channels, time, width, height).
        
        Arguments:
            idx {int} -- Sample index to extract.
        
        Returns:
            dict -- A dictionary with elements *X*, *y*, the sample FRAP data and parameter targets respectively.
        """


        X_path = os.path.join(self.directory, "data", self.x_files[idx])
        y_path = os.path.join(self.directory, "data", self.y_files[idx])

        # Extract data as numpy arrays
        y = np.fromfile(y_path, dtype=np.float32)
        y = np.reshape(y, (self.n_params))
        X = np.fromfile(X_path, dtype=np.float32)
        X = np.reshape(X, (self.n_channels, self.depth, self.width, self.height))

        batch = {"X": X,
                "y": y}

        return batch


class ExperimentalDataset(torch.utils.data.Dataset):
    """A torch.utils.data.Dataset of experimental FRAP files given a directory.
    """

    def __init__(self, directory, mode = "pixel", regex=r'x_+.*\.bin'):
        """Initializes an ExperimentalDataset from a directory of binary experimental FRAP files.
        
        Arguments:
            directory {string} -- Path of the directory with a subfolder /data/ with experimental data.
            mode {string} -- Setting whether to return pixel data or recovery curve data. (default: "pixel")
        
        Keyword Arguments:
            regex {regexp} -- regexp object to find all files in directory. (default: {r'x_+.*\.bin'})
        """
        super(ExperimentalDataset, self).__init__()

        self.shape      = (1, 110, 256, 256)
        self.n_channels     = 1
        self.height         = 256
        self.width          = 256
        self.depth          = 110
        self.mode           = mode
        self.n_params       = 3
        self.directory      = directory

        # List all files in paired order
        self.x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.search(regex, f)]

        self.length = len(self.x_files)
        

    def __len__(self):
        """The dataset size, i.e. the number of files in directory.
        
        Returns:
            int -- The number of files in the directory.
        """
            
        return self.length

    def __getitem__(self, idx):
        """Indexing function for the dataset. Indexes the files as ordered in the directory, since order is unimportant.
        Opens the binary files and returns a numpy array of the FRAP data in order (channels, time, width, height) for
        pixel data or (channels, time) for recovery curves.
        
        Arguments:
            idx {int} -- Sample index to extract.
        
        Returns:
            dict -- A dictionary with elements *X*, *file*, the sample FRAP data and file name respectively.
        """

        X_path = os.path.join(self.directory, "data", self.x_files[idx])
        X = np.fromfile(X_path, dtype=np.float32)

        # Reshape the data
        if self.mode == "pixel" or self.mode == "px" or self.mode == "spatiotemporal":
            X = np.reshape(X, (self.n_channels, self.depth, self.width, self.height), order="F")
        elif self.mode == "rc" or self.mode == "temporal":
            X = np.reshape(X, (self.n_channels, self.depth), order="F")

        batch = {"X": X, "file": self.x_files[idx]}

        return batch


class TemporalDataset(torch.utils.data.Dataset):
    """A torch.utils.data.Dataset that given a directory collects binary spatio-temporal FRAP data and returns numpy arrays.
    """

    def __init__(self, prefix = "x", directory="/home/sms/vws/data/temporal/data"):
        """Initializes a TemporalDataset from a directory of binary simulated FRAP files.
        
        Arguments:
            prefix {str} -- Pre-fix for the recovery curve files, often "rc" or "x". (default: "x")
            directory {str} -- A root directory containing the recovery curve data.
        """
        
        super(TemporalDataset, self).__init__()
        self.shape      = (1, 110)
        self.n_channels     = 1
        self.depth          = 110
        self.n_params       = 3
        self.directory  = directory
        
        # List all files in paired order
        if prefix == "x":
            regex = r'x+.*\.bin'
        elif prefix == "rc":
            regex = r'rc+.*\.bin'
        
        self.x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(regex, f)]
        self.y_files =  ["y_"+f.partition("_")[2] for f in self.x_files]

        self.length = len(self.x_files)

    def __len__(self):
        """Returns the number of samples in the dataset.
        
        Returns:
            int -- Number of samples in the dataset
        """
            
        return self.length

    def __getitem__(self, idx):
        """Indexing function for the dataset. Indexes the files as ordered in the directory, since order is unimportant.
        Opens the binary files and returns a numpy array of the FRAP data in order (channels, time).
        
        Arguments:
            idx {int} -- Sample index to extract.
        
        Returns:
            dict -- A dictionary with elements *X*, *y*, the sample FRAP data and parameter targets respectively.
        """

        X_path = os.path.join(self.directory, "data", self.x_files[idx])
        y_path = os.path.join(self.directory, "data", self.y_files[idx])

        # Extract the data as numpy arrays
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



class FileMatlabGenerator(torch.utils.data.Dataset):
    """A torch.utils.data.Dataset of binary files generated by a MATLAB script."""

    def __init__(self, dataset_size, directory, shape, run_dir=r'~/Documents/MATLAB/frappe', scripts_dir=r'~/magnusro/frap_ann/frap_matlab', noise_level=0.2, n_workers = 32):
        """Initializes the Dataset generator object.
        
        Arguments:
            dataset_size {int} -- The number of samples to generate.
            directory {string} -- The destination folder path to put the data. Appends a '/data/' to the path.
            shape {tuple} -- A tuple of shape (channels, time, width, height).
        
        Keyword Arguments:
            run_dir {regexp} -- A regexp path object with the directory of the generating script (default: {r'~/Documents/MATLAB/frappe'})
            scripts_dir {regexp} -- A regexp path object with the directory of the FRAP simulation scripts (default: {r'~/magnusro/frap_ann/frap_matlab'})
            noise_level {float} -- The noise level used when generating the FRAP data. (default: {0.2})
            n_workers {int} -- The number of workers in the MATLAB parallel pool. (default: {32})
        """
        
        self.dataset_size   = dataset_size
        self.n_workers      = n_workers
        self.noise_level    = noise_level
        self.n_channels     = 1
        self.height         = shape[-1]
        self.width          = shape[-2]
        self.depth          = shape[-3]
        self.n_params       = 3
        self.directory      = directory
        self.multiple_files = True

        ########### Initialize the MATLAB engine ###############
        logging.info("Initializing session. Entering MATLAB...")
        self.engine = self.initialize_session()

        # Add paths to scripts
        self.engine.addpath(run_dir,nargout=0)
        self.engine.addpath(scripts_dir,nargout=0)

        logging.info("- Back in python. MATLAB session started.")


        self.batch  = None

    def initialize_session(self):
        """Initializes an instance of the MATLAB engine from Python. Any calls to the MATLAB session must be done via this object.
        
        Returns:
            matlab.engine -- A MATLAB engine object.
        """

        return matlab.engine.start_matlab()

    def initialize_pool(self):
        """Utility function to manually initialize the MATLAB parallel pool.
        """
        
        logging.info("Initializing MATLAB parallel pool...")
        self.engine.initialize_pool(nargout=0)
        logging.info("- Pool initialized.")

    def kill_pool(self):
        """Utility function to manually kill the MATLAB parallel pool.
        """

        logging.info("Killing MATLAB parallel pool...")
        self.engine.kill_pool(nargout=0)
        logging.info("- Pool killed.")

    def generate_batch(self):
        """Generates a batch (dataset) of binary FRAP files from MATLAB.
        """

        success = self.engine.generate_batch(self.noise_level, self.dataset_size, self.directory, self.n_workers, 0, self.multiple_files, nargout=1)

    def augment_batch(self, fraction=0.05):
        """Augments a batch with a fraction of new data. A random fraction of the files are overwritten with new data.
        
        Keyword Arguments:
            fraction {float} -- The fraction of files to update. (default: {0.05})
        """
        n_to_add = int(fraction*self.dataset_size)

        # Collect all the files with a regular expression
        x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'x_+.*\.bin', f)]

        # Generate a set of MATLAB files.
        success = self.engine.generate_batch(self.noise_level, n_to_add, self.directory, self.n_workers, 1, 1, nargout=1)

        # Use temporary files in the dataset, not used when training
        x_temp_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'xtemp+.*\.bin', f)]
        y_temp_files =  ["ytemp_"+f.partition("_")[2] for f in x_temp_files]

        # Get random fraction of data to change
        x_files_slice = random.sample(x_files, n_to_add)
        y_files_slice =  ["y_"+f.partition("_")[2] for f in x_files_slice]


        # For each sample, overwrite the temporary file names with the new data file names.
        for i in range(len(x_files_slice)):
            
            x_old_path = os.path.join(self.directory, "data", x_temp_files[i])
            x_new_path = os.path.join(self.directory, "data", x_files_slice[i])
            
            y_old_path = os.path.join(self.directory, "data", y_temp_files[i])
            y_new_path = os.path.join(self.directory, "data", y_files_slice[i])
            
            shutil.copy(x_old_path,x_new_path)
            shutil.copy(y_old_path, y_new_path)



    def __len__(self):
        """Gets the number of samples in the dataset.
        
        Returns:
            int -- Dataset size, i.e. the number of samples in the dataset.
        """

        return self.dataset_size

    def __getitem__(self, idx):
        """Indexing function for the dataset. Indexes the files as ordered in the directory, since order is unimportant.
        Opens the binary files and returns a numpy array of the FRAP data in order (channels, time, width, height).
        
        Arguments:
            idx {int} -- Sample index to extract.
        
        Returns:
            dict -- A dictionary with elements *X*, *y*, the sample FRAP data and target parameters respectively.
        """

        # List all files in paired order
        x_files =  [f for f in os.listdir(os.path.join(self.directory, "data")) if re.match(r'x_+.*\.bin', f)]
        y_files =  ["y_"+f.partition("_")[2] for f in x_files]

        # Get the full file paths
        X_path = os.path.join(self.directory, "data", x_files[idx])
        y_path = os.path.join(self.directory, "data", y_files[idx])

        ######## Extract binary data as numpy arrays ##############
        y = np.fromfile(y_path, dtype=np.float32)
        y = np.reshape(y, (self.n_params))

        X = np.fromfile(X_path, dtype=np.float32)
        X = np.reshape(X, (self.n_channels, self.depth, self.width, self.height))

        batch = {"X": X,
                "y": y}

        return batch

        

class Occlusion(object):
    """An augmentation object to use with the imgaug library. Adds a random rectangular occlusion to each image in
    a sequence.
    """

    def __init__(self, frequency=0.7, p=0.1, size_percent=0.01):
        """Initializes an occlusion augmentation object.
        
        Keyword Arguments:
            frequency {float} -- The frequency of occlusions in a sequence. (default: {0.7})
            p {float} -- The probability of any pixel being dropped (i.e. set to zero) in  
                         the lower-resolution dropout mask. (default: {0.1})
            size_percent {float} -- The size of the lower resolution image from which to sample the dropout  
                                    mask *in percent* of the input image.  
                                    Note that this means that *lower* values of this parameter lead to  
                                    *larger* areas being dropped (as any pixel in the lower resolution  
                                    image will correspond to a larger area at the original resolution).   (default: {0.01})
        """
        sometimes = lambda aug: iaa.Sometimes(frequency, aug)
        self.seq = iaa.Sequential( sometimes (iaa.CoarseDropout(p, size_percent=size_percent)))
        
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