import numpy as np
import torch
import torchsummary

from models.resnet import ResNet


def main():
    nr_epochs = 1

    resnet = ResNet()
    # torchsummary.summary(resnet, (3, 128, 128))

    for epoch in range(nr_epochs):
        acc, loss = train(resnet)
        acc, loss = eval(resnet)


def train(model):
    acc = 0
    loss = 0
    return acc, loss


@torch.no_grad()
def eval(model):
    model.eval()
    acc = 0
    loss = 0
    return acc, loss


if __name__ == "__main__":
    main()