from models import CNN1D
from torchsummary import summary
import torch
from resnet import resnet18

X = torch.rand(256, 1, 100)

model = resnet18(in_channels=1, dimension=1)
