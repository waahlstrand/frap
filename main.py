import argparse
import os
import torch
import torch.nn as nn
import utils
from datetime import datetime
import logging

from models import CNN1D
from trainer import Trainer

parser = argparse.ArgumentParser()

#parser.add_argument("--data_path", default='data', help='Path to the data.')
parser.add_argument('--config_name', default='config.json', help="Name of .json file")
parser.add_argument('--job_name', default='training', help="Name of job file.")
parser.add_argument("--verbose", default = 'True', help="Print log to terminal.")


def train(config, model_dir, verbose):
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

    n_epochs    = params.n_epochs
    lr          = params.lr
    momentum    = params.momentum
    n_filters   = params.n_filters
    n_hidden    = params.n_hidden

    logs        = utils.str_to_bool(config.logging)
    use_val     = utils.str_to_bool(config.validation)

    ############### INITIALISE MODEL ######################
    model = CNN1D(n_filters=n_filters, n_hidden=n_hidden)

    # Define a loss function. reduction='none' is elementwise loss, later summed manually
    criterion = nn.MSELoss(reduction='none')

    # Define an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

    ############## GET DATALOADERS ########################
    # Get dataset of recovery curves
    #dataset = RecoveryDataset(data_path)
    logging.info("Loading the datasets...")
    training, validation = utils.get_dataloaders(mode, data_path, use_val)
    logging.info("- Loading complete.")

    # Initialize a Regressor training object
    logging.info("Initializing trainer object...")    
    trainer = Trainer(model, config, criterion, optimizer, training, validation, verbose)
    logging.info("- Initialization complete.")

    ################ TRAIN THE MODEL ######################
    logging.info("Starting training for {} epoch(s)".format(n_epochs))
    trainer.train()
    logging.info("Training complete.")

    torch.save(trainer.model, os.path.join(model_dir, now.strftime("%Y%m%d-%H%M") + ".pt"))


if __name__ == '__main__':

    # Parse arguments to program
    args = parser.parse_args()

    config_name = args.config_name
    job_name    = args.job_name
    verbose     = utils.str_to_bool(args.verbose)

    model_dir = os.path.join("saved/", job_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = os.path.join("config/", config_name)

    # Extract parameters
    config = utils.Configuration.from_nested_dict(config_path)

    # Initialize logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    
    ########## START TRAINING ###########
    train(config, model_dir, verbose)


