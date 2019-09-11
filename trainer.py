import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, model, criterion, optimizer, config, dataset, validation):
        
        self.config     = config
        self.model      = model
        self.dataset    = dataset
        self.validation = validation
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.loss       = 0

        settings = self.config["settings"]
        self.logs     = self.config["logs"]
        self.verbose  = self.config["verbose"]

        self.batch_size = settings["batch_size"]
        self.epochs = range(1, settings["epochs"]+1)

        self.train_loader, self.val_loader = self._split_data(self.dataset, self.validation, settings["train_size"])

        if self.logs:
            self.writer = SummaryWriter(self.logs["log_dir"])

    def _train_epoch(self, epoch):

        self.model.train()

        full_loss       = 0
        element_loss    = 0

        for i, (X, y) in enumerate(self.train_loader):

            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Feed forward the data
            prediction = self.model(X)

            # Calculate the MSE loss
            loss = self.criterion(prediction, y)

            # Backpropagate the loss
            loss.mean().backward()

            # Should we clip the gradient? 
            # nn.utils.clip_grad_norm_(model.parameters(), clip)        
            self.optimizer.step()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), options.clip)
            
            full_loss += torch.sum(loss.detach(), (0, 1))
            element_loss += torch.sum(loss.detach(), 0)

        result = {"loss": full_loss/(self.model.output_size*len(self.train_loader.dataset)), "param": element_loss/len(self.train_loader.dataset)}
        
        return result

    def _validate_epoch(self, epoch):

        self.model.eval()

        full_loss       = 0
        element_loss    = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(self.val_loader):

                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                # Feed forward the data
                prediction = self.model(X)

                # Calculate the MSE loss
                loss = self.criterion(prediction, y)

                # Should we clip the gradient? 
                # nn.utils.clip_grad_norm_(model.parameters(), clip)        
                self.optimizer.step()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), options.clip)
                
                full_loss += torch.sum(loss.detach(), (0, 1))
                element_loss += torch.sum(loss.detach(), 0)


        result = {"loss": full_loss/(self.model.output_size*len(self.val_loader.dataset)), "param": element_loss/len(self.val_loader.dataset)}

        return result

    def train(self):

        self.device     = self._configure_device(self.config["device"])

        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     self.model = nn.DataParallel(self.model)

        self.model      = self.model.to(self.device)

        for epoch in self.epochs:

            training_result     = self._train_epoch(epoch)
            validation_result   = self._validate_epoch(epoch)


            if self.logs:
                self.writer.add_scalars("Loss", {"training": training_result["loss"], 
                                                 "validation": validation_result["loss"]}, epoch)

                self.writer.add_scalars("Parameter loss", {"D": training_result["param"][0], 
                                                           "C": training_result["param"][1], 
                                                           "alpha": training_result["param"][2]}, epoch)

            ############### MANUAL PRINTING #####################
            if self.verbose:
                print('                                                      ')

                print('Epoch:  %d | Loss: %.4f | Validation: %.4f' % (epoch, 
                                                                        training_result["loss"], 
                                                                        validation_result["loss"]))

                print('MSE for   | D: %.4f | C: %.4f | alpha: %.4f' % (training_result["param"][0], 
                                                                        training_result["param"][1], 
                                                                        training_result["param"][2]))
                print('_____________________________________________________')

        self.loss = validation_result["loss"] 
        
        if self.logs:
            self.writer.close()

    def _configure_device(self, device_id):

        cuda_available = torch.cuda.is_available()

        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        #os.environ["CUDA_VISIBLE_DEVICES"]="1"

        # If we have a GPU available, we'll set our device to GPU. 
        # We'll use this device variable later in our code.
        if cuda_available and device_id > 0:
            n_gpu = torch.cuda.device_count()
            
            print("GPU is available. Using GPU.")
            device = torch.device("cuda:"+str(device_id-1))

        else:
            print("GPU not available. Using CPU.")
            device = torch.device("cpu")

        return device

    def _split_data(self, dataset, validation, train_size):
        n_train = int(train_size * len(self.dataset))
        n_val   = len(self.dataset) - n_train

        if validation is None:
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        else:
            train_dataset = dataset
            val_dataset   = validation

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)


        return train_loader, val_loader



