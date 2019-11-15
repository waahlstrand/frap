import numpy as np
import torch
import utils
import itertools
import os
import json
import logging
from main import train


if __name__ == '__main__':

    # Parameter sets
    ##### CONSTANT PARAMETERS #####
    n_epochs        = [100]
    n_datasets      = [2]
    dataset_size    = [32]
    batch_size      = [64]
    shape           = ["(1, 110, 256, 256)"]
    noise_level     = [0.1]
    train_fraction  = [0.7]

    ##### SEARCHABLE PARAMETERS #####
    decay           = [0, 0.5]
    lr              = [1e-5, 1e-4, 1e-3]
    momentum        = [1e-8, 0.5, 0.99]
    #kernel_size     = [2, 3]
    #channel         = [1, 16, 32]


    params      = {'n_epochs': n_epochs,
                   'n_datasets': n_datasets,
                   'dataset_size': dataset_size,
                   'batch_size': batch_size,
                   'shape': shape,
                   'decay': decay,
                   'lr': lr, 
                   'momentum': momentum, 
                   #'kernel_size': kernel_size, 
                   #'channels': channel,
                   'noise_level': noise_level,
                   'train_fraction': train_fraction}

    config_permutations =  [dict(zip(params, v)) for v in itertools.product(*params.values())]

    jobs = []

    for params in config_permutations:

        # Main configuration
        main_config = {
        "job_name": "tuning",
        "model_name": "tratt",
        "trainer_name": "trainer",
        "optimizer_name": "sgd",
        'params': params,
        "verbose": "True",
        "cuda": "True",
        "gpu": 0,
        "source": "spatiotemporal",
        "tensorboard": "True",
        "data_path": "",
        "validation": "True",
        "clip": 0,
        "mode": "spatiotemporal",
        "transform":  "False"
        }

        jobs.append(main_config)


    ########## START TRAINING ###########
    losses = []
    i = 0
    for job in jobs:
        
        i = i + 1
        # Extract parameters
        config = utils.Configuration.from_nested_dict(job)

        job_name    = config.job_name
        model_dir = os.path.join("saved/", job_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Initialize logger
        utils.set_logger(os.path.join(model_dir, 'train_'+str(i)+'.log'))

        logging.info("Starting new job...")  
        loss = train(config, model_dir)

        # Print data to file and save
        job.update({'loss': loss.data.tolist()})

        with open(os.path.join(model_dir,'config'+str(i)+'.json'), 'w') as fp:
            json.dump(job, fp)        


        losses.append(loss)
        
        logging.info("- Job finished.")  

    print(losses)
