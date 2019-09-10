import torch
import torch.nn as nn
from ray import tune
from models import CNN1D
from trainer import Trainer 
from datasets import RecoveryTrainingDataset, RecoveryValidationDataset

from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
import numpy as np



def train(config):

    training = RecoveryTrainingDataset("/home/vws/Documents/python/frappe/rcs")
    validation = RecoveryValidationDataset("/home/vws/Documents/python/frappe/rcs")

    params  = config["params"]
    model   = CNN1D(params["n_filter"], params["n_hidden"])

    criterion = config["criterion"]


    trainer = Trainer(model, 
                      criterion, 
                      torch.optim.SGD(model.parameters(), 
                                      lr=params["lr"], 
                                      momentum=params["momentum"], 
                                      nesterov=True), 
                      config, 
                      training, 
                      validation)

    trainer.train()

    tune.track.log(mean_loss=trainer.loss)

N_EPOCHS        = 100
TRAIN_FRACTION  = 0.8
BATCH_SIZE      = 128

config = {"device": 1, # 0 means CPU, 1 the first GPU, 2 second GPU
          "criterion": nn.MSELoss(reduction='none'),
          "settings": {"train_size": TRAIN_FRACTION,
                       "epochs": N_EPOCHS,
                       "batch_size": BATCH_SIZE},
          "logs": None,
          "params": {"n_hidden": tune.grid_search([16, 32, 64]),
                     "n_filter": tune.grid_search([16, 32, 64]),
                     "lr": tune.grid_search([0.001, 0.01, 0.1]),
                     "momentum": tune.uniform(0,1)
                     }
        }

analysis = tune.run(train, config=config)