# Deep Neural Networks for Estimation of FRAP Parameters
In the area of microscopy and bioscience, *fluorescence recovery after photobleaching* (FRAP) is a method for determining the kinetic diffusive properties of substances dissolved in a suspension or membrane. This is often applied to membranes such as protein-lipid layers in biological cells and pharmaceuticals. 

The diffusive attributes of the substance are integral to describing the properties of many chemical reactions, and its behaviour in the human body. As such, it is imperative that these attributes can be estimated well numerically.

Classical approaches include maximum likelihood and least squares estimations, but these are often based on flawed models. It is interesting to see if inference can be made quickly with model-agnostic methods, such as neural networks.

## Methods
This master thesis is based on the works of M. Röding et al. who provided a novel method for accurate simulations of FRAP microscopy experiments. This provides training data for a proof of concept deep neural network for regression.

## To do
- [x] A few basic architectures; fully-connected, LSTM
- [x] Create project file structure
- [x] Visualize the data in Tensorboard
- [ ] Rudimentary hyperparameter search
- [ ] Create architectures for perfect simulated image data
- [ ] Consider local phase with STFT
- [ ] Create architectures for noisy simulated image data