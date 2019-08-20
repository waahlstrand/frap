import torch
import torch.nn as nn
import numpy as np
import tqdm

from RecoveryModel import RecoveryModel, RecoveryDataset

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
cuda_available = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if cuda_available:
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")



# Define constants
INPUT_SIZE  = 1
OUTPUT_SIZE = 3
N_EPOCHS    = 100


# Define hyperparameters
hidden_size     = 128
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
model = RecoveryModel(INPUT_SIZE, hidden_size, batch_size, OUTPUT_SIZE)

# Define loss and optimizer
criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#train= tqdm.tqdm(trainloader)
train = trainloader
loop = tqdm.tqdm(epochs)
for epoch in loop:

    # Zero grad with the model
    model.zero_grad()

    running_loss = 0


    for X, target in train:

        ########### TRAINING #############

        # Zero the gradient from previous computation
        optimizer.zero_grad()

        # Feed forward
        y = model(X)

        # Evaluate loss
        loss = criterion(y, target)

        # Backpropagate
        loss.backward()

        # Should we clip the gradient? 
        # nn.utils.clip_grad_norm_(model.parameters(), clip)        
        optimizer.step()

        running_loss += loss.item()

        ############ VALIDATION #############
        with torch.no_grad():
            pass
    
    loop.set_description('Epoch {}/{}'.format(epoch + 1, N_EPOCHS))
    loop.set_postfix(loss=running_loss.item()/batch_size)





        