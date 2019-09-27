import numpy as np
import json
import logging
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

def output_size_from_conv(in_size, kernel_size, stride = 1, padding = 0, dilation = 1):
    
    size = np.floor( (in_size+ 2*padding-dilation*(kernel_size-1)-1)/stride +1 )
    
    return int(size)

def str_to_bool(s):
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

def get_dataset(source, data_path, params):

    
    if source == "temporal":

        dataset = RecoveryTrainingDataset(data_path)

    elif source == "spatiotemporal":
        
        dataset = SpatiotemporalDataset()

    elif source == "generate":

        dataset = MatlabGenerator(batch_size=params.batch_size, noise_level=params.noise_level, n_workers=params.batch_size)

    return dataset

def get_model(model_name, params):

    if model_name == "cnn1d":
        model = CNN1d(n_filters=params.n_filters, n_hidden=params.n_hidden)
    elif model_name == "resnet18":
        model = resnet18(in_channels=params.n_channels, dimension=2, num_classes=3)
    elif model_name == "voxnet":
        model = vx.VoxNet(params.batch_size, 3)
    elif model_name == "tratt":
        model = Tratt(params.batch_size)
    elif model_name == "top_heavy_tratt":
        model = TopHeavyTratt(params.batch_size)
    elif model_name == "filterer":
        model = Filterer(params.batch_size)
    else:
        raise NotImplementedError("Model not implemented.")

    return model