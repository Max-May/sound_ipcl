import os

import numpy as np
import torch

from models.resnet import ResNet, Bottleneck
from ipcl.ipcl import IPCL

def main():
    n_samples = 5
    input = torch.rand(n_samples,2,39,8000)
    targs = torch.randint(0, 828, (1,))
    targs = targs.repeat(n_samples)

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
        n_samples=n_samples
    )

    loss, (embeddings, prototype) = learner(input, targs)
    print(f'Loss: {loss}')
    print(f'X: {embeddings.shape}')
    print(f'Prototype: {prototype.shape}')
    feat = embeddings.chunk(n_samples)[0].detach().cpu()
    feat = feat.view(feat.shape[0],-1)
    print(f'Feat: {feat.shape}')


if __name__ == '__main__':
    main()