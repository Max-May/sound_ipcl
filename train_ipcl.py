import os

import numpy as np
import torch

from models.resnet import ResNet, Bottleneck
from ipcl.ipcl import IPCL

def main():
    input = torch.rand(5,2,39,8000)

    encoder = ResNet(
        block=Bottleneck,
        layers=[3,4,6,3],
        input_channels=2,
        num_classes=128,
        l2norm=True
    )

    learner = IPCL(
        base_encoder=encoder,
        numTrainFiles=18000,
        K=4096,
        T=0.07,
        out_dim=128,
        n_samples=5
    )

    loss, (x, y) = learner(input)



if __name__ == '__main__':
    main()