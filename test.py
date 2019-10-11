from torchsummary import summary
import torch
from models.spatiotemporal import ConvLSTM, Net, Carl, Fundo, FourierTratt
from models.resnet import resnet18, resnet183d
from models.temporal import *
#from models.voxnet import VoxNet


#model = TopHeavyTratt(5)
#summary(Tratt(5).cuda(), input_size=(1, 110, 256, 256))
#summary(ConvLSTM(n_input_channels = 1, n_hidden_channels = 5, kernel_size = 3, n_layers = 2, pool=False).cuda(), input_size=(1, 110, 256, 256))

#with torch.no_grad():
#    model = ConvLSTM(n_input_channels = 1, n_hidden_channels = 32, kernel_size = 3, pool=False, n_layers=1).cuda(device=0)
#    X = torch.rand((2, 1, 110, 256, 256)).cuda(device=0)
#    y = model(X)

#model2 = ConvLSTM(n_input_channels = 1, n_hidden_channels = 5, kernel_size = 3, pool=False, n_layers=2, shape=(1, 110, 100, 100)).cuda(device=0)
#X = torch.rand((10, 1, 110, 100, 100)).cuda(device=0)
#y = model2(X)

#model = Net(batch_size = 1).cuda()
#X = torch.rand((16, 1, 110, 256, 256)).cuda()
#y = model(X)
#model = Net(batch_size=64, input_shape=(1, 110, 20, 20)).cuda()
#model = resnet183d(in_channels=1, num_classes=3, dimension=3).cuda()
model = CNN1d(batch_size=2).cuda()
#X = torch.rand((64, 1, 110, 20, 20)).cuda()
#model(X)

summary(model, (1, 110))
