

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import data.datasets


import numpy as np
import pandas as pd
import h5py
import hdf5storage

from models.spatiotemporal import Tratt


import time

def test(model, device_string, loader, criterion=nn.MSELoss(reduction='none')):
    """A method to summarize the test error of a PyTorch model.
    
    Arguments:
        model {nn.Module} -- A trained PyTorch model
        decive {str} -- Either "cuda:0" or "cpu"
        loader {torch.util.data.Dataloader} -- A Dataloader from a test Dataset
    
    Keyword Arguments:
        criterion {nn.loss} -- A loss criterion (default: {nn.MSELoss(reduction='none')})
    
    Returns:
        dict -- Returns a dict with the total loss and parameter-wise loss
    """
        

    full_loss       = 0
    element_loss    = 0
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(loader):

            device = torch.device(device_string)

            X = batch["X"].to(device)
            y = batch["y"].to(device)

            # Feed forward the data
            prediction = model(X)

            # Calculate the MSE loss
            loss = criterion(prediction, y)
                    
            full_loss += torch.sum(loss.detach(), (0, 1))
            element_loss += torch.sum(loss.detach(), 0)


    result = {"loss": full_loss/(3*len(loader.dataset)), "param": element_loss/len(loader.dataset)}

    return result


if __name__ == "__main__":

    test_set = data.datasets.SpatiotemporalDataset("/home/sms/vws/data/test/")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=8)

    state = torch.load("/home/sms/vws/frappe/saved/longtime/2k/States/states")
    downsampler = Tratt(batch_size=8)
    downsampler.load_state_dict(state)

    result = test(downsampler, "cpu", test_loader)

    print("Downsampler", result)

    convlstm = torch.load("/home/sms/vws/frappe/saved/longtime_convlstm/2k/20191118-1125.pt")
    result = test(convlstm, "cuda:0", test_loader)

    print("ConvLSTM", result)