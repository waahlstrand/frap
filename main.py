import argparse
import os
import torch
import torch.nn as nn
import utils
from datetime import datetime
import logging

# Parse the arguments to the main program
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='test.json', help="Name of .json file")



def train(config, model_dir):
    """Given a Configuration object containing training settings and hyperparameters, the train method launches a Trainer instance
    which trains a neural network model.
    
    Arguments:
        config {Configuration} -- Configuration object of settings, from a JSON file.
        model_dir {string} -- Path to the target directory of logs and results
    
    Returns:
        loss {double} -- The final validation or training loss, depending on the Trainer object.
    """

    # Record time
    now = datetime.now()

    # Set seed
    if config.cuda: 
        torch.cuda.manual_seed(2222)
    else:
        torch.manual_seed(2222)

    params          = config.params
    pretrain        = config.pretrain
    data_path       = config.data_path
    source          = config.source
    mode            = config.mode
    trainer_name    = config.trainer_name
    model_name      = config.model_name
    optimizer_name  = config.optimizer_name

    n_epochs    = params.n_epochs
    lr          = params.lr
    momentum    = params.momentum
    batch_size  = params.batch_size

    use_transform   = utils.str_to_bool(config.transform)
    use_val         = utils.str_to_bool(config.validation)

    # Define a loss function. reduction='none' is elementwise loss, later summed manually
    criterion   = nn.MSELoss(reduction='none')

    ############### INITIALISE MODEL AND OPTIMIZER ######################
    # Define a model and optimizer pair
    model, optimizer = utils.get_model_and_optimizer(model_name, optimizer_name, pretrain, params)

    ############## GET DATALOADERS ########################
    # Get dataset of recovery curves
    logging.info("Loading the datasets...")
    dataset = utils.get_dataset(source, data_path, model_dir, mode, use_transform, params)
    logging.info("- Loading complete.")

    # Initialize a Regressor training object
    logging.info("Initializing trainer object...")    
    trainer = utils.get_trainer(trainer_name, model, config, criterion, optimizer, dataset, model_dir)
    logging.info("- Initialization complete.")

    ################ TRAIN THE MODEL ######################
    logging.info("Starting training for {} epoch(s)...".format(n_epochs))
    trainer.train()
    logging.info("- Training complete.")

    torch.save(trainer.model, os.path.join(model_dir, now.strftime("%Y%m%d-%H%M") + ".pt"))

    return trainer.loss


if __name__ == '__main__':

    # Parse arguments to program
    args = parser.parse_args()

    # Get configuration JSON file
    config_name = args.config
    config_path = os.path.join("config/", config_name)

    # Extract settings from JSON to a Configuration object
    config = utils.Configuration.from_nested_dict(config_path)
    
    job_name    = config.job_name
    model_dir = os.path.join("saved/", job_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    
    ########## START TRAINING ###########
    loss = train(config, model_dir)


