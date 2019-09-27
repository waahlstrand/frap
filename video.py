from torchsummary import summary
import torch
from models.spatiotemporal import Tratt, TopHeavyTratt, Filterer
from models.resnet import resnet18
#from models.voxnet import VoxNet


#model = TopHeavyTratt(5)
#summary(Tratt(5).cuda(), input_size=(1, 110, 256, 256))
summary(resnet18(num_classes=3, in_channels=110).cuda(), input_size=(1, 110, 256, 256))

#print(model(X).shape)
