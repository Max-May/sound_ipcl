import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
import torchsummary

from models.resnet import ResNet, Bottleneck
from dataloader.dataloader import AudioDataset
from dataloader.WebAudioSet import WebAudioSet

import sofa
from tqdm import tqdm
import argparse
from utils.util import read_yaml, write_yaml, seed_all

import time 
from datetime import datetime


def train(model, dataloader, criterion, scheduler, optimizer, device):
    model.train()

    train_running_loss = 0.0
    train_running_acc = 0
    counter = 0
    total_guessed = 0

    time_total = 0
    time_start = time.time()
    # for idx, (data, labels) in enumerate(tqdm(dataloader, desc='Training')):
    for idx, (data, labels) in enumerate(dataloader):
        time_now = time.time()

        counter += 1

        avg_time = (time_now - time_start - time_total) / counter
        time_total += avg_time 


        total_guessed += data.shape[0]
        print(f'[Batch: {idx+1}, ({avg_time:.2f}s/it total:{time_total:.2f}s)]: {total_guessed:04d}', end="\r", flush=True)

    #     counter += 1
    #     total_guessed += data.shape[0]

    #     data = data.to(device, dtype=torch.float)
    #     labels = labels.to(device)

    #     # Forward pass
    #     outputs = model(data)
    #     # Calculate loss
    #     loss = criterion(outputs, labels)
    #     train_running_loss += loss.item()

    #     # Calculate accuracy
    #     _, preds = torch.max(outputs.data, 1)
    #     train_running_acc += (preds == labels).sum().item()

    #     loss.backward()
    #     optimizer.step()

    # scheduler.step()
    # epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_acc / total_guessed)
    # return epoch_acc, epoch_loss
    return None, None


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    valid_running_correct = 0
    valid_running_loss = 0
    counter = 0
    total_guessed = 0

    # for idx, (data, labels) in enumerate(tqdm(dataloader, desc='Validating')):
    #     counter += 1
    #     total_guessed += data.shape[0]

    #     data = data.to(device)
    #     labels = labels.to(device)
         
    #     # Forward pass.
    #     outputs = model(image)
    #     # Calculate the loss.
    #     loss = criterion(outputs, labels)
    #     valid_running_loss += loss.item()
    #     # Calculate the accuracy.
    #     _, preds = torch.max(outputs.data, 1)
    #     valid_running_correct += (preds == labels).sum().item()
    time_total = 0
    time_start = time.time()
    for idx, (data, labels) in enumerate(dataloader):
        time_now = time.time()

        counter += 1

        avg_time = (time_now - time_start - time_total) / counter
        time_total += avg_time 


        total_guessed += data.shape[0]
        print(f'[Batch: {idx+1}, ({avg_time:.2f}s/it total:{time_total:.2f}s)]: {total_guessed:04d}', end="\r", flush=True)

    # Loss and accuracy for the complete epoch.
    # epoch_loss = valid_running_loss / counter
    # epoch_acc = 100. * (valid_running_correct / total_guessed)
    # return epoch_acc, epoch_loss
    return None, None


def main(args):
    try:
        cfg = read_yaml(args.config)
    except Exception as e:
        print(f'Need config file, use "-c config.yaml"\n{e}')
        return None

    experiment = cfg['name']
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Seed everything for reproducibility
    seed = cfg['seed']
    seed_all(seed)

    print(f'=> Running experiment: {experiment} with id: {run_id}\nUsing seed: {seed}')

    # CUDA for PyTorch
    # use_cuda = torch.cuda.is_available()
    use_cuda = cfg['gpu']
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        curr_device = torch.cuda.current_device
    print(f'Using device: "{device}{": " + curr_device if use_cuda else ""}"')
    if use_cuda:
        print(f'[{torch.cuda.device(curr_device)}] name: "{torch.cuda.get_device_name(curr_device)}"')

    # Model
    arch = cfg['arch']
    print(f'=> Building model with arch: {arch["_component_"]}')
    if arch['block'].lower() == 'bottleneck':
        block = Bottleneck
    model = ResNet(
                block=block, 
                layers=arch['layers'], 
                input_channels=arch['in_channels'], 
                num_classes=arch['out_channels']
            )
    model = model.to(device)
    # torchsummary.summary(resnet, (3, 128, 128))

    # Optimizer
    optimizer = cfg['optimizer']
    if optimizer['_component_'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=optimizer['lr'])

    scheduler = cfg['scheduler']
    if scheduler['_component_'].lower() == 'multisteplr':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40,60], gamma=.75) # 60 is added in case a higher number of epochs is used

    # Loss function.
    criterion = cfg['criterion']
    if criterion['_component_'].lower() == 'crossentropyloss':
        criterion = nn.CrossEntropyLoss()

    # Old method --> not optimized for .tar files
    # train_data = AudioDataset(
    #                 dir='/home/maxmay/files_to_copy.txt', 
    #                 target_samplerate = 48000
    #             )
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # New method --> optimized for large tar file directories
    dataset = cfg['dataset']
    # WAS = WebAudioSet(
    #     base_data_dir = '/home/maxmay/Data/bal_train{00..07}.tar',
    #     val_data_dir = '/home/maxmay/Data/bal_train{08..09}.tar',
    #     hrtf_dir = '/home/maxmay/sound_ipcl/utils/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa',
    #     target_samplerate = 48000,
    #     batch_size = batch_size,
    #     resample=True
    #     )
    WAS = WebAudioSet(
    base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
    val_data_dir = dataset['base_data_dir']+dataset['val_split']+'.tar',
    hrtf_dir = dataset['sofa_dir'],
    target_samplerate = dataset['sample_rate'],
    batch_size = dataset['batch_size'],
    resample= dataset['resample']
    )
    WAS.setup('fit')
    train_loader = WAS.train_wds_loader()
    val_loader = WAS.val_wds_loader()

    # Since we resample the data, the dataloader is infinite and we need to set a limit per epoch
    # This is done inside the WebAudioSet class --> {train/val}_wds_loader()
    nr_epochs = cfg['trainer']['epochs']

    # for epoch in range(nr_epochs):
    #     print(f'Epoch:[{epoch+1}/{nr_epochs}]')
    #     train_acc, train_loss = train(model, train_loader, criterion, scheduler, optimizer, device)
    #     # train_acc, train_loss = train(model, val_loader, criterion, scheduler, optimizer, device)
    #     # print(f'Train Accuracy: {train_acc}\nLoss: {train_loss}\n')
    #     val_acc, val_loss = validate(resnet, val_loader, criterion, device)
        # print(f'Test Accuracy: {val_acc}\nLoss: {val_loss}\n')


def test_config(args):
    try:
        cfg = read_yaml(args.config)
    except Exception as e:
        print(f'Need config file, use "-c config.yaml"\n{e}')
        return None
    print(cfg['trainer']['epochs'])
    # print(cfg.name)
    # print(cfg.arch._component_)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Supervised training Resnet-50')
    args.add_argument('-c', '--config', 
                    default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-r', '--resume', 
                    default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    args = args.parse_args()
    # test_config(args)
    main(args)

    # write_yaml(Bottleneck, fn='./test.yaml')

