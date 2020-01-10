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
    n_epochs        = [850]
    n_datasets      = [1]
    dataset_size    = [65536]
    batch_size      = [16]
    shape           = ["(1, 110)"]
    noise_level     = [0.2]
    train_fraction  = [0.7]

    ##### SEARCHABLE PARAMETERS #####
    decay           = [0]
    lr              = [1e-5, 1e-4]
    momentum        = [1e-8, 0.99]
    n_filters       = [16, 128]
    n_hidden        = [32, 64]
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
                   'n_filters': n_filters,
                   'n_hidden': n_hidden,
                   'noise_level': noise_level,
                   'train_fraction': train_fraction}

    config_permutations =  [dict(zip(params, v)) for v in itertools.product(*params.values())]

    jobs = []

    for params in config_permutations:

        # Main configuration
        main_config = {

        "pretrain": {
        "from_checkpoint": "False",
        "path": ""
        },

        "job_name": "tuning_cnn1d",
        "model_name": "cnn1d",
        "trainer_name": "trainer",
        "optimizer_name": "sgd",
        'params': params,
        "verbose": "True",
        "cuda": "True",
        "gpu": 0,
        "source": "temporal",
        "tensorboard": "True",
        "data_path": "/home/sms/vws/data/temporal/",
        "validation": "True",
        "clip": 0,
        "mode": "temporal",
        "transform":  "False"
        }

        jobs.append(main_config)

        print(params)


    ########## START TRAINING ###########
    losses = []
    i = 0
    for job in jobs[12:-1]:
        
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

        with open(os.path.join(model_dir,'config'+str(i)+'.json'), 'w') as fp:
            json.dump(job, fp)        


        losses.append(loss)
        
        logging.info("- Job finished.")  

    #print(losses)
