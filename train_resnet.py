import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
import torchsummary

from models.resnet import ResNet, Bottleneck
from dataloader.dataloader import AudioDataset

import sofa
from tqdm import tqdm


def train(model, dataloader, criterion, scheduler, optimizer, device):
    model.train()

    train_running_loss = 0.0
    train_running_acc = 0
    for idx, (data, labels) in enumerate(tqdm(dataloader, desc=f'Training')):
        data = data.to(device, dtype=torch.float)

        outputs = model(data)

        # Calculate loss
        labels = labels.to(device)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # Calculate accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_acc += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()

    scheduler.step()
    epoch_loss = train_running_loss / len(dataloader)
    epoch_acc = 100. * (train_running_acc / len(dataloader.dataset))
    return epoch_acc, epoch_loss


@torch.no_grad()
def eval(model):
    model.eval()
    acc = 0
    loss = 0
    return acc, loss


def main():
    nr_epochs = 5
    batch_size = 2

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f'Using device: "{device}"')

    # Model
    print('Building model...')
    model = ResNet(block=Bottleneck, layers=[3,4,6,3], input_channels=2, num_classes=828)
    model = model.to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40,60], gamma=.75) # 60 is added in case a higher number of epochs is used
    
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    train_data = AudioDataset(
                    dir='/home/maxmay/files_to_copy.txt', 
                    target_samplerate = 48000
                )
    # test_data = AudioDataset('test')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_data)

    # torchsummary.summary(resnet, (3, 128, 128))

    for epoch in range(nr_epochs):
        print(f'Epoch:[{epoch+1}/{nr_epochs}]')
        acc, loss = train(model, train_loader, criterion, scheduler, optimizer, device)
        print(f'Accuracy: {acc}\nLoss: {loss}\n')
        # acc, loss = test(resnet, test_loader)


if __name__ == "__main__":
    main()
