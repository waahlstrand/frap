import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class RecoveryDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        super(RecoveryDataset, self).__init__()

        self.n_samples_train    = 2**16
        self.n_samples_val      = 2**14
        self.sequence_length    = 110
        self.target_length      = 3

        # Load all data
        x_train = np.fromfile(root_dir + "/x_train.bin", dtype = np.float32)
        y_train = np.fromfile(root_dir + "/y_train.bin", dtype = np.float32)
        x_val   = np.fromfile(root_dir + "/x_val.bin", dtype = np.float32)
        y_val   = np.fromfile(root_dir + "/y_val.bin", dtype = np.float32)
        

        # Concatenate all data 
        x_train = np.reshape(x_train, (self.n_samples_train, self.sequence_length))
        y_train = np.reshape(y_train, (self.n_samples_train, self.target_length))

        x_val = np.reshape(x_val, (self.n_samples_val, self.sequence_length))
        y_val = np.reshape(y_val, (self.n_samples_val, self.target_length))

        self.inputs  = np.vstack((x_train, x_val))
        self.targets = np.vstack((y_train, y_val))

    def __len__(self):
            
        return len(self.inputs[:,0])

    def __getitem__(self, idx):

        return {"sample" :self.inputs[idx, :], "target":self.targets[idx, :]}




def fit(model, data, options):

    model.train()

    train   = data["train"]

    model.zero_grad()
    running_loss = 0


    for i, batch in enumerate(train):

        X = batch["sample"].to(options.device)
        y = batch["target"].to(options.device)

        options.optimizer.zero_grad()

        # Feed forward the data
        prediction = model(X)

        # Calculate the MSE loss
        loss = options.criterion(prediction, y)

        # Backpropagate the loss
        loss.backward()

        # Should we clip the gradient? 
        # nn.utils.clip_grad_norm_(model.parameters(), clip)        
        options.optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), options.clip)
        
        running_loss += loss.detach()

    return running_loss/len(train.dataset)


def validate(model, data, options):

    # Set evaluation mode, equivalent but faster than model.eval()
    model.eval()

    model.zero_grad()

    with torch.no_grad():

        val = data["val"]

        running_loss = 0

        for i, batch in enumerate(val):

            X = batch["sample"].to(options.device)
            y = batch["target"].to(options.device)

            # Feed forward the data
            prediction = model(X)

            # Calculate the MSE loss
            loss = options.criterion(prediction, y)
            
            running_loss += loss.detach()

        return running_loss/len(val.dataset)

def predict(model, data):

    # Initialize evaluation mode
    model.eval(True)

    # Make single prediction from data
    prediction = model(data)

    model.eval(False)

    return prediction


dataset = RecoveryDataset(root_dir = "rcs")
dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=4)
