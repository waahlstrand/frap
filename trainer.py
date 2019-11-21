#import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import shutil
import utils
from torch.utils.data import Dataset, DataLoader

class BaseTrainer:

    def __init__(self, model, config, criterion, optimizer, dataset, model_dir):

        self.config     = config

        self.model      = model
        self.dataset    = dataset
        self.criterion  = criterion
        self.optimizer  = optimizer

        self.loss       = []
        self.parameters = 3

        self.save_path  = os.path.join(model_dir, "states")

        self.tensorboard = utils.str_to_bool(self.config.tensorboard)

        self.cuda       = utils.str_to_bool(self.config.cuda)
        self.gpu        = str(self.config.gpu)
        self.verbose    = utils.str_to_bool(self.config.verbose)
        self.validation = utils.str_to_bool(self.config.validation)
        self.clip       = self.config.clip
        self.mode       = self.config.mode

        self.batch_size     = self.config.params.batch_size
        self.n_epochs       = self.config.params.n_epochs
        self.train_fraction = self.config.params.train_fraction

        
        self.epochs = range(1, self.n_epochs+1)

        self.train_loader, self.val_loader = self._split_data(self.dataset, self.validation, self.train_fraction)

        if self.tensorboard:
            tensorboard_dir = os.path.join(model_dir, "logs/")

            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            else:
                try:
                    shutil.rmtree(tensorboard_dir)
                except OSError as e:
                    print ("Error: %s - %s." % (e.filename, e.strerror))



            self.writer = SummaryWriter(tensorboard_dir)

    def _run_epoch(self, epoch, training, loader):
        
        if training:
            self.model.train()
        else:
            self.model.eval()

        full_loss       = 0
        element_loss    = 0
        with torch.set_grad_enabled(training):
            for i, batch in enumerate(loader):

                self.optimizer.zero_grad()

                if self.mode == "spatiotemporal" or self.mode == "temporal" or self.mode == "fourier":

                    X = batch["X"].to(self.device)
                    y = batch["y"].to(self.device)

                    # Feed forward the data
                    prediction = self.model(X)

                elif self.mode == "all":
                    
                    X   = batch["X"].to(self.device)
                    rcs = batch["rcs"].to(self.device)
                    var = batch["var"].to(self.device)
                    y   = batch["y"].to(self.device)

                    # Feed forward the data
                    prediction = self.model(X, rcs, var)

                # Calculate the MSE loss
                loss = self.criterion(prediction, y)

                if training:
                    # Backpropagate the loss
                    loss.mean().backward()

                    if self.clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), options.clip)
                
                full_loss += torch.sum(loss.detach(), (0, 1))
                element_loss += torch.sum(loss.detach(), 0)


        result = {"loss": full_loss/(self.parameters*len(loader.dataset)), "param": element_loss/len(loader.dataset)}

        return result

    def _train_epoch(self, epoch):

        return self._run_epoch(epoch, True, self.train_loader)

    def _validate_epoch(self, epoch):

        return self._run_epoch(epoch, False, self.val_loader)

    def _train_implementation(self):
        raise NotImplementedError

    def train(self):
        self._train_implementation()

    def _configure_device(self, cuda):

        cuda_available = torch.cuda.is_available()

        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        #os.environ["CUDA_VISIBLE_DEVICES"]="1"

        # If we have a GPU available, we'll set our device to GPU. 
        # We'll use this device variable later in our code.
        if cuda_available and cuda:
            
            logging.info("GPU is available. Using GPU.")
            device = torch.device("cuda:"+self.gpu)

        else:
            logging.info("GPU not available. Using CPU.")
            device = torch.device("cpu")

        return device

    def _split_data(self, dataset, validation, train_size):

        if validation:

            if  len(dataset) == 2:# or (not isinstance(dataset, list)) or (not isinstance(dataset, tuple)):

                train_loader    = torch.utils.data.DataLoader(dataset[0], batch_size=self.batch_size, shuffle=True, num_workers=2)
                val_loader      = torch.utils.data.DataLoader(dataset[1], batch_size=self.batch_size, shuffle=True, num_workers=2)
                
            else:

                n_train = int(train_size * len(self.dataset))
                n_val   = len(self.dataset) - n_train

                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

                train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
                val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)


                

        else:

            train_loader    = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_loader      = None


        return train_loader, val_loader


class Approximator(BaseTrainer):

    def __init__(self, model, config, criterion, optimizer, dataset, model_dir):

        super().__init__(model, config, criterion, optimizer, dataset, model_dir)

        self.generator = self.dataset
        

    def _train_implementation(self):

        # Initialize generator for MATLAB data
        #self.generator.initialize_session()

        # Initialize parallel processing pool
        self.generator.initialize_pool()

        # Send model to cuda
        self.device = self._configure_device(self.cuda)
        self.model  = self.model.to(self.device)

        
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5, last_epoch=-1)

        # Start approximating
        for epoch in self.epochs:

            # Generate online data 
            self.generator.generate_batch()

            # Train for one batch
            training_result = self._train_epoch(epoch)

            # Change the learning rate
            scheduler.step()

            if self.tensorboard:
                self.writer.add_scalars("Loss", {"training": training_result["loss"]}, epoch)

                self.writer.add_scalars("Parameter-wise", {"D": training_result["param"][0], 
                                                           "C": training_result["param"][1], 
                                                           "α": training_result["param"][2]}, epoch)
                
                #self.writer.add_histogram("Gradients", np.asarray([param.grad for param in list(self.model.parameters())]), epoch)

            ############### MANUAL PRINTING #####################
            if self.verbose:
                logging.info('                                                      ')

                logging.info('Epoch:  %d | Loss: %.4f ' % (epoch, training_result["loss"]))

                logging.info('MSE for   | D: %.4f | C: %.4f | α: %.4f' % (training_result["param"][0], 
                                                                        training_result["param"][1], 
                                                                        training_result["param"][2]))
                logging.info('_____________________________________________________')

        self.loss.append(training_result["loss"] )
        
        if self.tensorboard:
            self.writer.close()
        
        self.generator.kill_pool()



class Trainer(BaseTrainer):

    def __init__(self, model, config, criterion, optimizer, dataset, model_dir):

        super().__init__(model, config, criterion, optimizer, dataset, model_dir)

    def _train_implementation(self):

        self.device = self._configure_device(self.cuda)

        #if torch.cuda.device_count() > 1:
        #    print("Using", torch.cuda.device_count(), "GPUs")
        #    self.model = nn.DataParallel(self.model, device_ids=[0, 1])

        self.model  = self.model.to(self.device)

        for epoch in self.epochs:

            training_result     = self._train_epoch(epoch)
            validation_result   = self._validate_epoch(epoch)


            if self.tensorboard:
                self.writer.add_scalars("Loss", {"training": training_result["loss"], 
                                                 "validation": validation_result["loss"]}, epoch)

                self.writer.add_scalars("Parameter-wise", {"D": validation_result["param"][0], 
                                                           "C": validation_result["param"][1], 
                                                           "α": validation_result["param"][2]}, epoch)

            ############### MANUAL PRINTING #####################
            if self.verbose:
                logging.info('                                                      ')

                logging.info('Epoch:  %d | Loss: %.4f | Validation: %.4f' % (epoch, 
                                                                        training_result["loss"], 
                                                                        validation_result["loss"]))

                logging.info('MSE for   | D: %.4f | C: %.4f | α: %.4f' % (validation_result["param"][0], 
                                                                        validation_result["param"][1], 
                                                                        validation_result["param"][2]))
                logging.info('_____________________________________________________')

            self.loss.append(np.array(validation_result))
        
        if self.tensorboard:
            self.writer.close()


class Mixed(BaseTrainer):

    def __init__(self, model, config, criterion, optimizer, dataset, model_dir):

        super().__init__(model, config, criterion, optimizer, dataset, model_dir)

        self.training_generator     = self.dataset[0]
        self.validation_generator   = self.dataset[1]

        self.n_superepochs  = self.config.params.n_datasets
        self.superepochs = range(1, self.n_superepochs+1)

    def _train_implementation(self):

        # Initialize generator for MATLAB data
        #self.generator.initialize_session()

        # Initialize parallel processing pool
        #self.training_generator.initialize_pool()
        #self.validation_generator.initialize_pool()


        # Generate validation data
        self.validation_generator.generate_batch()
        #self.validation_generator.kill_pool()


        # Send model to cuda
        self.device = self._configure_device(self.cuda)
        self.model  = self.model.to(self.device)
        #self.net = torch.nn.DataParallel(self.model, device_ids=[self.device])
        
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.n_epochs, gamma=0.5, last_epoch=-1)

        for superepoch in self.superepochs:

            self.train_loader, self.val_loader = self._split_data(self.dataset, self.validation, self.train_fraction)

            # Generate online data 
            self.training_generator.generate_batch()

            # Start approximating
            for epoch in self.epochs:

                # Train for one batch
                training_result     = self._train_epoch(epoch)
                validation_result   = self._validate_epoch(epoch)


                # Change the learning rate
                scheduler.step()

                iteration = (self.n_epochs)*(superepoch-1) + epoch

                if self.tensorboard:
                    self.writer.add_scalars("Loss", {"training": training_result["loss"], "validation": validation_result["loss"]}, iteration)

                    self.writer.add_scalars("Parameter-wise", {"D": validation_result["param"][0], 
                                                            "C": validation_result["param"][1], 
                                                            "α": validation_result["param"][2]}, iteration)
                    
                ############### MANUAL PRINTING #####################
                if self.verbose:
                    logging.info('                                                      ')

                    logging.info('Superepoch: %d | Epoch:  %d | Tot. Epoch: %d | Loss: %.4f | Val. Loss: %.4f' % (superepoch, 
                                                                                                                  epoch, 
                                                                                                                  iteration, 
                                                                                                                  training_result["loss"], 
                                                                                                                  validation_result["loss"]))

                    logging.info('MSE for   | D: %.4f | C: %.4f | α: %.4f' % (validation_result["param"][0], 
                                                                            validation_result["param"][1], 
                                                                            validation_result["param"][2]))
                    logging.info('_____________________________________________________')


            # Save state dictionary
            torch.save(self.model.state_dict(), self.save_path)

            # Record loss for plotting
            self.loss.append(np.array(validation_result))
        
        if self.tensorboard:
            self.writer.close()

        # Convert loss to numpy array and save
        self.loss = np.array(self.loss)
        np.save(self.loss, os.path.join(self.save_path, "loss.npy"))
        
        self.training_generator.kill_pool()


class Incrementer(BaseTrainer):

    def __init__(self, model, config, criterion, optimizer, dataset, model_dir):

        super().__init__(model, config, criterion, optimizer, dataset, model_dir)

        self.training_generator     = self.dataset[0]
        self.validation_generator   = self.dataset[1]

        #self.n_superepochs  = self.config.params.n_datasets
        #self.superepochs = range(1, self.n_superepochs+1)

    def _train_implementation(self):


        # Initialize pool
        self.training_generator.initialize_pool()

        # Generate validation data
        self.validation_generator.generate_batch()
        
        # Generate data
        self.training_generator.generate_batch()

        # Send model to cuda
        self.device = self._configure_device(self.cuda)
        self.model  = self.model.to(self.device)

        #self.net = torch.nn.DataParallel(self.model, device_ids=[self.device])
        
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2*self.n_epochs, gamma=0.5, last_epoch=-1)

        self.train_loader, self.val_loader = self._split_data(self.dataset, self.validation, self.train_fraction)

        # Start training
        for epoch in self.epochs:

            # Train for one batch
            training_result     = self._train_epoch(epoch)
            validation_result   = self._validate_epoch(epoch)


            # Change the learning rate
            scheduler.step()
            #iteration = (self.n_epochs)*(superepoch-1) + epoch

            if self.tensorboard:
                self.writer.add_scalars("Loss", {"training": training_result["loss"], "validation": validation_result["loss"]}, epoch)
                self.writer.add_scalars("Parameter-wise", {"D": validation_result["param"][0], 
                                                        "C": validation_result["param"][1], 
                                                        "α": validation_result["param"][2]}, epoch)
                
            ############### MANUAL PRINTING #####################
            if self.verbose:
                logging.info('                                                      ')
                logging.info('Epoch:  %d | Loss: %.4f | Val. Loss: %.4f' % (epoch, 
                                                                            training_result["loss"], 
                                                                            validation_result["loss"]))

                logging.info('MSE for   | D: %.4f | C: %.4f | α: %.4f' % (validation_result["param"][0], 
                                                                        validation_result["param"][1], 
                                                                        validation_result["param"][2]))
                logging.info('_____________________________________________________')

            # Augment the data with 10% new samples
            self.training_generator.augment_batch()

            # Save state dictionary
            if (epoch % 100) == 0:
                saved_dir = self.save_path
                if not os.path.exists(saved_dir):
                    os.makedirs(saved_dir)

                state = {'epoch': epoch+1, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}

                torch.save(state, os.path.join(saved_dir,str(epoch)+".pt"))

            # Record loss for plotting
            self.loss.append(np.array(validation_result))
        
        if self.tensorboard:
            self.writer.close()

        # Convert loss to numpy array and save
        self.loss = np.array(self.loss)
        np.save(os.path.join(self.save_path, "loss.npy"), self.loss)
        
        self.training_generator.kill_pool()
