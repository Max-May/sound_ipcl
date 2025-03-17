import torch
from torch import nn
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F 

from torchsummary import summary

import numpy as np 
# import sofa


# Input 
# 2 x 39 x Time

# Output
# 828 (HRTF locations)

class ResNet(nn.Module):
    def __init__(self, input_dimension: int = 3, num_classes: int = 1000):
        super(ResNet, self).__init__()
        self.model = resnet50()
        # self.model = resnet18()

        if input_dimension != 3:
            new_conv1 = nn.Conv2d(
                in_channels=input_dimension, 
                out_channels=self.model.conv1.out_channels, 
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias is not None
            )
            with torch.no_grad():
                new_conv1.weight[:, :input_dimension, :, :] = self.model.conv1.weight[:, :input_dimension, :, :]
                new_conv1.weight[:, input_dimension:, :, :] = self.model.conv1.weight[:, input_dimension:, :, :].mean(dim=1, keepdim=True)

            self.model.conv1 = new_conv1 

        if num_classes != 1000:
            new_fc = nn.Linear(
                in_features=self.model.fc.in_features, 
                out_features=num_classes
            )
            self.model.fc = new_fc

    
    def forward(self, x):
        return self.model(x)


def main():
    in_d = 2
    num_classes = 828
    resnet = ResNet(input_dimension=in_d, num_classes=num_classes)
    x = torch.rand([1,in_d,600,24])
    # print(x.shape)
    # print(resnet)
    # summary(resnet, x)
    resnet = resnet.to('cpu')
    y = resnet(x)
    print(y.shape)
    # print(y.shape,"\n", F.softmax(y, dim=1))


if __name__ == "__main__":
    main()