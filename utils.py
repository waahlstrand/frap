import numpy as np
import json
import logging
import os
import ast
from data.datasets import *
from models.spatiotemporal import *
from models.temporal import *
from trainer import Trainer, Mixed, OnlineTrainer, Incrementer

class Configuration(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
    """
    
    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        """Construct nested AttrDicts from nested dictionaries or from a JSON file.
        
        Arguments:
            data {dict, JSON} -- Either a dict or JSON path name.
        
        Returns:
            Configuration -- A Configuration object.
        """
        if isinstance(data, str) and ".json" in data:
            f = open(data)
            data = json.load(f)
            f.close()

            return Configuration({key: Configuration.from_nested_dict(data[key]) for key in data})
        elif not isinstance(data, dict):
            return data
        else:
            return Configuration({key: Configuration.from_nested_dict(data[key]) for key in data})


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
        In general, it is useful to have a logger so that every output to the terminal is saved
        in a permanent file. Here we save it to `model_dir/train.log`.
        Example:
        ```
        logging.info("Starting training...")
        ```
        
    Arguments:
        log_path {[type]} -- The destination directory where to log.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def str_to_bool(s):
    """Converts a string with a boolean value to a bool.
    
    Arguments:
        s {str} -- A string of boolean value.
    
    Raises:
        ValueError: Raised if the string value is not "True" or "False".
    
    Returns:
        bool -- Returns the value of the boolean string.
    """
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 


def minmax(x, minimum, maximum):
    """A function to for min-max normalization, such that the output is constrained
    in the interval [0,1].
    
    Arguments:
        x {np.array} -- A numpy array object with the data.
        minimum {double} -- Minimum of the data
        maximum {double} -- Maximum of the data
    
    Returns:
        np.array -- Min-max normalized data
    """

    y = (x-minimum)/(maximum-minimum)

    return y

def inverse_minmax(y, minimum, maximum):
    """A function to retrieve the original data normalized by min-max.
    
    Arguments:
        y {np.array} -- Min-max normalized data
        minimum {double} -- Minimum of the data
        maximum {double} -- Maximum of the data

    Returns:
        np.array -- Original numpy array with the data
    """

    x = y * (maximum-minimum) + minimum

    return x


def get_trainer(trainer_name, model, config, criterion, optimizer, dataset, model_dir):
    """A utility function for getting a Trainer object by name. Returns a Trainer object used to
    wrap the training of neural networks.
    
    Arguments:
        trainer_name {string} -- Name of the Trainer object
        model {nn.Module} -- A PyTorch module with parameters to be optimized
        criterion {nn.loss} -- Loss function to optimize with respect to
        optimizer {nn.optimizer} -- Optimizer object
        dataset {torch.utils.data.Dataset} -- A dataset object to use with dataloaders
        model_dir {str} -- Directory path to the model
    
    Raises:
        NotImplementedError: Thrown if the Trainer name is incorrect or the Trainer object does not exist.
    
    Returns:
        Trainer -- Trainer object
    """

    if trainer_name == "trainer":
        trainer = Trainer(model, config, criterion, optimizer, dataset, model_dir)
    elif trainer_name == "approximator" or trainer_name == "online":
        trainer = OnlineTrainer(model, config, criterion, optimizer, dataset, model_dir)
    elif trainer_name == "mixed":
        trainer = Mixed(model, config, criterion, optimizer, dataset, model_dir)
    elif trainer_name == "incrementer":
        trainer = Incrementer(model, config, criterion, optimizer, dataset, model_dir)
    else:
        raise NotImplementedError("Trainer not available.")

    return trainer

def get_optimizer(model, optimizer_name, params):
    """A utility function for getting an optimizer for PyTorch neural networks.
    
    Arguments:
        model {nn.Module} -- A PyTorch module with parameters to be optimized
        optimizer_name {String} -- The name of the optimizer, either "adam" or "sgd"
        params {Configuration} -- A Configuration object with parameter attributes
    
    Returns:
        torch.optim -- A PyTorch optimizer object
    """

    if optimizer_name == "adam":
        optimizer   = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=params.decay, amsgrad=False)
    elif optimizer_name == "sgd":
        optimizer   = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, nesterov=True, weight_decay=params.decay)
    elif optimizer_name == "rmsprop":
        optimizer   = torch.optim.RMSprop(model.parameters(), lr=params.lr, alpha=0.99, eps=1e-08, weight_decay=params.decay, momentum=0, centered=False)
    else:
        raise NotImplementedError("Optimizer not available.")

    return optimizer


def get_dataset(source, data_path, directory, mode, use_transform, params):
    """Utility function for getting the desired FRAP dataset. Divided into "temporal", 
    "spatiotemporal" and "generate", they generate train-test split PyTorch datasets and a 
    generator of unique training data from Matlab, respectively.
    
    Arguments:
        source {str} -- The source of the data, either
                        - "temporal" returns a recovery curve dataset from a directory
                        - "spatiotemporal" returns a FRAP pixel dataset from a directory
                        - "generate"
                        - "mixed"
                        - "files_single" returns a dataset of given size as a single binary file.
                        - "files_split" returns a dataset of given size as one binary file for every sample. Suitable for big data.

        data_path {str} -- Path to the dataset samples
        directory {str} -- Path to the target directory of the generator
        mode {str} -- Either "temporal", "spatiotemporal", "all" or "Fourier", the type of data generated.
        params {Configuration} -- A Configuration dict of parameters, such as batch size.
    
    Returns:
        torch.nn.utils.Dataset -- A dataset to be loaded to a PyTorch Dataloader.
    """

    
    if source == "temporal":

        dataset = TemporalDataset(directory = data_path)

    elif source == "spatiotemporal":
        
        dataset = SpatiotemporalDataset(directory= data_path)

    elif source == "generate":

        dataset = MatlabGenerator(batch_size=params.batch_size, 
                                  directory=directory, 
                                  mode=mode, 
                                  transform=use_transform,
                                  noise_level=params.noise_level, 
                                  n_workers=params.batch_size)

    elif source == "mixed":

        dataset = MixedMatlabGenerator(dataset_size = params.dataset_size,
                                       batch_size=params.batch_size, 
                                        directory=directory, 
                                        mode=mode, 
                                        transform=use_transform,
                                        noise_level=params.noise_level, 
                                        n_workers=params.batch_size)

    elif source == "fromfiles" or source == "files_single":
        shape = ast.literal_eval(params.shape)


        dataset = FileMatlabGenerator(dataset_size = params.dataset_size,
                                        directory=directory, 
                                        transform=use_transform,
                                        noise_level=params.noise_level, 
                                        n_workers=params.batch_size,
                                        shape=shape
                                        )

    elif source == "mixed_separate" or source == "files_split":
            shape = ast.literal_eval(params.shape)

            training = FileMatlabGenerator(dataset_size = params.dataset_size,
                                        directory=os.path.join(directory, "train"), 
                                        transform=use_transform,
                                        noise_level=params.noise_level, 
                                        n_workers=32,
                                        shape=shape)
            
            validation = FileMatlabGenerator(dataset_size = int((1-params.train_fraction)*params.dataset_size),
                                        directory=os.path.join(directory, "val"), 
                                        transform=use_transform,
                                        noise_level=params.noise_level, 
                                        n_workers=32,
                                        shape=shape)

            dataset = (training, validation)


    return dataset

def get_model(model_name, params):
    """A utility function for fetching the desired PyTorch model to train.
    
    Arguments:
        model_name {str} -- The name of the model.
        params {Configuration} -- A Configuration dict of parameters to the model.
    
    Raises:
        NotImplementedError: Raised if the model name supplied is not implemented.
    
    Returns:
        torch.nn.Module -- Returns a PyTorch module.
    """

    # Interpret the shape parameter, given as a string, into a tuple.
    shape = ast.literal_eval(params.shape)

    if model_name == "cnn1d":
        model = CNN1d(batch_size = params.batch_size, n_filters=params.n_filters, n_hidden=params.n_hidden)
    elif model_name == "split":
        model = Split(n_hidden=params.n_hidden, shape=shape)
    elif model_name == "fc":
        model = FC(n_hidden=params.n_hidden)
    elif model_name == "convfundo":
        model = ConvFundo(params.batch_size, input_shape=shape)
    elif model_name == "tratt":
        model = Tratt(params.batch_size, shape=shape)
    elif model_name == "tratt2convlstm":
            model = Tratt2ConvLSTM(params.batch_size, shape=shape)
    elif model_name == "top_heavy_tratt":
        model = TopHeavyTratt(params.batch_size, shape=shape)
    elif model_name == "fundo":
        model = Fundo(params.batch_size, shape=shape)
    elif model_name == "downsampler":
        model = Downsampler(params.batch_size, shape=shape)
    elif model_name == "timesampled":
        model = TimeSampled(params.batch_size, input_shape=shape)
    elif model_name == "tratt_pretrained":
        state = torch.load("/home/sms/vws/frappe/saved/longtime/2k/States/states")
        model = Tratt(batch_size=16)
        model.load_state_dict(state)
        model.train()
    else:
        raise NotImplementedError("Model not implemented.")


    return model


def get_model_and_optimizer(model_name, optimizer_name, pretrain, params):
    """A utility function to get both model and optimizer, with support for starting from 
    a pretrained checkpoint state.
    
    Arguments:
        model_name {str} -- The name of the model
        optimizer_name {str} -- The name of the optimizer
        pretrain {Configuration} -- A Configuration dict with settings for the pretraining
        params {Configuration} -- A Configuration dict of parameters to the model and optmizer
    
    Returns:
        torch.nn.Module -- Returns a PyTorch module.
    """

    pretrain_from_checkpoint = str_to_bool(pretrain.from_checkpoint)

    # Load template model and optimizer
    model = get_model(model_name, params)
    optimizer = get_optimizer(model, optimizer_name, params)

    if pretrain_from_checkpoint:

            # Get full path to states
            path = pretrain.path

            # Load checkpoint data
            checkpoint = torch.load(path)

            # Apply to model and optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


            #epoch = checkpoint['epoch']
            #loss = checkpoint['loss']
    

    return model, optimizer