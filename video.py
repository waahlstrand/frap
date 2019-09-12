from models import CNN3D
import torch

x = torch.randn(50, 1, 100, 256, 256)

n_filters = 2
n_hidden  = 16

model = CNN3D(n_filters, n_hidden)

print(model(x).shape)

