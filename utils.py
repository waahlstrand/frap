import numpy as np
import json
import logging
import ast
from data.datasets import *
from models.spatiotemporal import *
from models.temporal import *
from models.resnet import resnet18

class Configuration(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
    """
    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        """ Construct nested AttrDicts from nested dictionaries. """
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
    Args:
        log_path: (string) where to log
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

def get_dataloaders(mode, data_path, validation):

    if mode == "rcs":
        if validation:
            train = RecoveryTrainingDataset(data_path)
            val = RecoveryValidationDataset(data_path)

            return train, val

        else:
            train = RecoveryTrainingDataset(data_path)

            return train, None
    
    elif mode == "spatiotemporal":
        
        train = SpatiotemporalDataset()

        return train, None

def get_optimizer(model, optimizer_name, params):

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    elif optimizer_name == "sgd":
        optimizer   = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, nesterov=True)

    return optimizer


def get_dataset(source, data_path, directory, mode, use_transform, params):
    """Utility function for fetching the desired FRAP dataset. Divided into "temporal", 
    "spatiotemporal" and "generate", they generate train-test split PyTorch datasets and a 
    generator of unique training data from Matlab, respectively.
    
    Arguments:
        source {str} -- Either "temporal", "spatiotemporal" or "generate".
        data_path {str} -- Path to the dataset samples
        directory {str} -- Path to the target directory of the generator
        mode {str} -- Either "temporal", "spatiotemporal", "all" or "Fourier", the type of data generated.
        params {Configuration} -- A Configuration dict of parameters, such as batch size.
    
    Returns:
        torch.nn.utils.Dataset -- A dataset to be loaded to a PyTorch Dataloader.
    """

    
    if source == "temporal":

        dataset = RecoveryTrainingDataset(data_path)

    elif source == "spatiotemporal":
        
        dataset = SpatiotemporalDataset()

    elif source == "generate":

        dataset = MatlabGenerator(batch_size=params.batch_size, 
                                  directory=directory, 
                                  mode=mode, 
                                  transform=use_transform,
                                  noise_level=params.noise_level, 
                                  n_workers=params.batch_size)

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
    elif model_name == "fc":
        model = FC(batch_size = params.batch_size, n_hidden=params.n_hidden)
    elif model_name == "lstm":
        model = LSTM_to_FFNN(hidden_size = params.n_hidden)
    elif model_name == "resnet18":
        model = resnet18(in_channels=1, dimension=3, num_classes=3)
    elif model_name == "voxnet":
        model = vx.VoxNet(params.batch_size, 3)
    elif model_name == "tratt":
        model = Tratt(params.batch_size, shape=shape)
    elif model_name == "top_heavy_tratt":
        model = TopHeavyTratt(params.batch_size, shape=shape)
    elif model_name == "carl":
        model = Carl(params.batch_size, input_shape=shape)
    elif model_name == "Net":
        model = Net(params.batch_size, input_shape=shape)
    elif model_name == "fundo":
        model = Fundo(params.batch_size, shape=shape)
    elif model_name == "fouriernet":
        model = FourierNet(params.batch_size, input_shape=shape)
    elif model_name == "fouriertratt":
        model = FourierTratt(params.batch_size, shape=shape)
    else:
        raise NotImplementedError("Model not implemented.")

    return model