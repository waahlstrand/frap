import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
from trainer import Trainer
from datasets import RecoveryDataset, RecoveryTrainingDataset, RecoveryValidationDataset
from models import LSTM_to_FFNN, CNN1D, FFNN

import os
from datetime import datetime

######################################################
now = datetime.now()
# Directory for Tensorboard logs
log_dir = "logs/" + now.strftime("%Y%m%d-%H%M") + "/"

# Directory for saved models
saved_dir = "saved/" + now.strftime("%Y%m%d-%H%M") + ".pt"


######################################################
# Define constants
INPUT_SIZE      = 1
OUTPUT_SIZE     = 3
N_EPOCHS        = 10000
SEQUENCE_LENGTH = 110
TRAIN_FRACTION  = 0.8

# Define hyperparameters
#hidden_size         = 64
#n_layers            = 2
#dropout             = 0.5
batch_size          = 256
learning_rate       = 0.0001
#clip                = 5
epochs              = range(0, N_EPOCHS)

# Get dataset of recovery curves
dataset = RecoveryDataset("new_data")
training = RecoveryTrainingDataset("new_data")
validation = RecoveryValidationDataset("new_data")

# Load a model. CNN1D is completely hard-coded in its hyperparameters
model = CNN1D(SEQUENCE_LENGTH, INPUT_SIZE, OUTPUT_SIZE)

# Define a loss function. reduction='none' signifies the loss for each element, not
# averaged or summed
criterion = nn.MSELoss(reduction='none')

# Define an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7, nesterov=True)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)

# Preliminary storing configuration in dictionary, to be moved to json
config = {"device": 1, # 0 means CPU, 1 the first GPU, 2 second GPU
          "settings": {"train_size": TRAIN_FRACTION,
                       "epochs": N_EPOCHS,
                       "batch_size": batch_size},
            "logs": {"log_dir": log_dir}}

######################################################
# Train the model
print("Initializing training object.")
trainer = Trainer(model, criterion, optimizer, config, training, validation)

print("Starting training.")
trainer.train()
print("Training complete.")

# Save model for inference, plotting, et c.
# Inference is done by 
# >> torch.load("model_path")
# >> y = model(x)
# x is a batch of data
torch.save(trainer.model, saved_dir)
