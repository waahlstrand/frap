import argparse
import os
import torch
import torch.nn as nn
import utils
from datetime import datetime
import logging

from models.temporal import CNN1d
from models.spatiotemporal import Tratt, TopHeavyTratt
from models import voxnet as vx
from models import resnet as rs
from trainer import Trainer

parser = argparse.ArgumentParser()

#parser.add_argument("--data_path", default='data', help='Path to the data.')
parser.add_argument('--config', default='config.json', help="Name of .json file")
#parser.add_argument('--job_name', default='training', help="Name of job file.")
#parser.add_argument("--verbose", default = 'True', help="Print log to terminal.")


def train(config, model_dir):
    # Record time
    now = datetime.now()

    # Set seed
    if config.cuda: 
        torch.cuda.manual_seed(2222)
    else:
        torch.manual_seed(2222)

    params      = config.params
    data_path   = config.data_path
    mode        = config.mode
    model_name  = config.model_name

    n_epochs    = params.n_epochs
    lr          = params.lr
    momentum    = params.momentum
    n_filters   = params.n_filters
    n_hidden    = params.n_hidden
    batch_size  = params.batch_size

    use_val     = utils.str_to_bool(config.validation)

    ############### INITIALISE MODEL ######################
    if model_name == "cnn1d":
        model = CNN1d(n_filters=n_filters, n_hidden=n_hidden)
    elif model_name == "resnet1d":
        model = resnet18(in_channels=1, dimension=1, num_classes=3)
    elif model_name == "voxnet":
        model = vx.VoxNet(batch_size, 3)
    elif model_name == "tratt":
        model = Tratt(batch_size)
    elif model_name == "top_heavy_tratt":
        model = TopHeavyTratt(batch_size)

    # Define a loss function. reduction='none' is elementwise loss, later summed manually
    criterion = nn.MSELoss(reduction='none')

    # Define an optimizer
    #ptimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    ############## GET DATALOADERS ########################
    # Get dataset of recovery curves
    #dataset = RecoveryDataset(data_path)
    logging.info("Loading the datasets...")
    training, validation = utils.get_dataloaders(mode, data_path, use_val)
    logging.info("- Loading complete.")

    # Initialize a Regressor training object
    logging.info("Initializing trainer object...")    
    trainer = Trainer(model, config, criterion, optimizer, training, validation, model_dir)
    logging.info("- Initialization complete.")

    ################ TRAIN THE MODEL ######################
    logging.info("Starting training for {} epoch(s)".format(n_epochs))
    trainer.train()
    logging.info("Training complete.")

    torch.save(trainer.model, os.path.join(model_dir, now.strftime("%Y%m%d-%H%M") + ".pt"))


if __name__ == '__main__':

    # Parse arguments to program
    args = parser.parse_args()

    config_name = args.config
    config_path = os.path.join("config/", config_name)

    # Extract parameters
    config = utils.Configuration.from_nested_dict(config_path)
    
    job_name    = config.job_name
    model_dir = os.path.join("saved/", job_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    
    ########## START TRAINING ###########
    train(config, model_dir)


