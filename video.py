from models import CNN1D
from torchsummary import summary
import torch
from resnet import resnet18
from voxnet import VoxNet
X = torch.rand(5, 1, 110, 256, 256)

model = VoxNet(5, 3)

print(model(X).shape)
