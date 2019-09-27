import argparse
import os
import torch
import torch.nn as nn
import utils
from datetime import datetime
import logging
from trainer import Approximator

parser = argparse.ArgumentParser()

#parser.add_argument("--data_path", default='data', help='Path to the data.')
parser.add_argument('--config', default='test.json', help="Name of .json file")
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
    source      = config.source
    model_name  = config.model_name

    n_epochs    = params.n_epochs
    lr          = params.lr
    momentum    = params.momentum
    batch_size  = params.batch_size

    use_val     = utils.str_to_bool(config.validation)

    ############### INITIALISE MODEL ######################
    model       = utils.get_model(model_name, params)

    # Define a loss function. reduction='none' is elementwise loss, later summed manually
    criterion   = nn.MSELoss(reduction='none')

    #writer = SummaryWriter()
    # Define an optimizer
    optimizer   = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    ############## GET DATALOADERS ########################
    # Get dataset of recovery curves
    #dataset = RecoveryDataset(data_path)
    logging.info("Loading the datasets...")
    dataset = utils.get_dataset(source, data_path, params)
    logging.info("- Loading complete.")

    # Initialize a Regressor training object
    logging.info("Initializing trainer object...")    
    trainer = Approximator(model, config, criterion, optimizer, dataset, model_dir)
    logging.info("- Initialization complete.")

    ################ TRAIN THE MODEL ######################
    logging.info("Starting training for {} epoch(s)...".format(n_epochs))
    trainer.train()
    logging.info("- Training complete.")

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


