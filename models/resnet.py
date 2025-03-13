import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F 

from torchsummary import summary

import numpy as np 


class ResNet(nn.Module):
    def __init__(self, input_dimension=3):
        super(ResNet, self).__init__()
        self.model = resnet50()
        self.model.conv1.in_channels = input_dimension
        # print(self.model)
    
    def forward(self, x):
        return self.model(x)


def main():
    resnet = ResNet(input_dimension=3)
    x = torch.tensor(np.random.rand(1,3,224,224).astype(np.float32))
    print(x.shape)

    y = resnet(x)
    print(y.shape,"\n", F.softmax(y, dim=1))
    # summary(resnet, (2, 256, 256))


if __name__ == "__main__":
    main()