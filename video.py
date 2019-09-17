from torchsummary import summary
import torch
from models.onedee import Tratt
from models.voxnet import VoxNet


X = torch.rand(5, 1, 110, 256, 256)

model = Tratt(5)
#summary(VoxNet(5, 3).cuda(), input_size=(1, 110, 256, 256))
summary(model.cuda(), input_size=(1, 110, 256, 256))

#print(model(X).shape)
