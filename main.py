import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from attrdict import AttrDict

import time
from RecoveryModel import RecoveryModel, RecoveryDataset, fit, validate, predict

import os
from datetime import datetime

now = datetime.now()
log_dir = "logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
model_dir = "models/" + now.strftime("%Y%m%d-%H%M%S") + "/"

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
cuda_available = torch.cuda.is_available()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if cuda_available:
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

# Tensorboard
writer = SummaryWriter(log_dir)


# Define constants
INPUT_SIZE  = 1
OUTPUT_SIZE = 3
N_EPOCHS    = 50


# Define hyperparameters
hidden_size     = 64
n_layers        = 1
batch_size      = 256
learning_rate   = 0.0001
clip            = 0
epochs          = range(0, N_EPOCHS)

# Get dataset of recovery curves
dataset = RecoveryDataset("rcs")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# Create model
model = RecoveryModel(INPUT_SIZE, hidden_size, batch_size, OUTPUT_SIZE, n_layers=n_layers)
model = model.to(device)

# Define loss and optimizer
criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#train= tqdm.tqdm(trainloader)
train = trainloader

data = {"train": trainloader,
        "val": valloader}

options = AttrDict({"optimizer": optimizer,
                    "criterion": criterion,
                    "device": device})


print("Starting the training loop...")

for epoch in epochs:

    epoch_time = time.time()

    ######## TRAIN THE MODEL ########
    loss = fit(model, data, options = options)

    ######## VALIDATE MODEL #########
    validation_loss = validate(model, data, options = options)

    # Print the data to Tensorboard
    timing = time.time()-epoch_time
    writer.add_scalars(now.strftime("%Y%m%d-%H%M%S"), {"training": loss,
                                                       "validation": validation_loss}, epoch)
    #writer.add_scalar("Time/train", timing, epoch)
    
    print('Epoch:  %d | Loss: %.4f | Validation: %.4f | Time: %.4f' % (epoch, loss, validation_loss, timing))


# Save the model for later use
torch.save(model.state_dict(), model_dir)





# for epoch in epochs:
#     start_time = time.time()

#     # Zero grad with the model
#     model.zero_grad()

#     running_loss = 0

#     for i, batch in enumerate(train):

#         ########### TRAINING #############
#         X = batch[0]
#         target = batch[1]

#         # Zero the gradient from previous computation
#         optimizer.zero_grad()

#         # Feed forward
#         y = model(X)

#         # Evaluate loss
#         loss = criterion(y, target)

#         # Backpropagate
#         loss.backward()

#         # Should we clip the gradient? 
#         # nn.utils.clip_grad_norm_(model.parameters(), clip)        
#         optimizer.step()
        
#         running_loss += loss.detach().item()

#         ############ VALIDATION #############
#         with torch.no_grad():
#             pass

#     timing = time.time()-start_time

#     writer.add_scalar("Loss/train", running_loss, epoch)
#     writer.add_scalar("Time/train", timing, epoch)
    
#     print('Epoch:  %d | Loss: %.4f | Time: %.4f' % (epoch, running_loss, timing))


writer.close()



        