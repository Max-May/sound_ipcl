import os
import shutil
from pathlib import Path
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
from utils.util import read_yaml, write_yaml, seed_all, count_pattern_files

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
    for idx, (data, labels) in enumerate(dataloader):
        counter += 1
        total_guessed += data.shape[0]
        print(f'[Batch: {idx+1}/{dataloader.nsamples}]: {total_guessed}', end="\r", flush=True)

        data = data.to(device, dtype=torch.float)
        labels = labels.to(device)

        # Forward pass
        outputs = model(data)
        # Calculate loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # Calculate accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_acc += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()

    scheduler.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_acc / total_guessed)
    return epoch_acc, epoch_loss

        # time_now = time.time()

        # counter += 1

        # avg_time = (time_now - time_start - time_total) / counter
        # time_total += avg_time 


        # total_guessed += data.shape[0]
        # print(f'[Batch: {idx+1}, ({avg_time:.2f}s/it total:{time_total:.2f}s)]: {total_guessed:04d}', end="\r", flush=True)
    # return None, None


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    valid_running_correct = 0
    valid_running_loss = 0
    counter = 0
    total_guessed = 0

    for idx, (data, labels) in enumerate(dataloader):
        counter += 1
        total_guessed += data.shape[0]
        print(f'[Batch: {idx+1}/{dataloader.nsamples}]: {total_guessed}', end="\r", flush=True)

        data = data.to(device, dtype=torch.float)
        labels = labels.to(device)
         
        # Forward pass.
        outputs = model(data)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / total_guessed)
    return epoch_acc, epoch_loss

    # time_total = 0
    # time_start = time.time()
    # for idx, (data, labels) in enumerate(dataloader):
        # time_now = time.time()
        # counter += 1
        # avg_time = (time_now - time_start - time_total) / counter
        # time_total += avg_time 
        # total_guessed += data.shape[0]
        # print(f'[Batch: {idx+1}, ({avg_time:.2f}s/it total:{time_total:.2f}s)]: {total_guessed:04d}', end="\r", flush=True)
    # return None, None



def main(args):
    debug = args.debug
    if debug:
        print(f'=> Debugging mode is active!')
    try:
        cfg = read_yaml(args.config)
    except Exception as e:
        print(f'Need config file, use "-c config.yaml"\n{e}')
        return None

    # Seed everything for reproducibility
    seed = cfg['seed']
    seed_all(seed)

    experiment = cfg['name']
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # CUDA for PyTorch
    # use_cuda = torch.cuda.is_available()
    n_gpus = cfg['n_gpu']
    use_cuda = True if n_gpus > 0 else False
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        curr_device = torch.cuda.current_device()
    print(f'=> Using device: "{device}{": " + str(curr_device) if use_cuda else ""}"')
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
                num_classes=arch['out_channels'],
                norm_layer=nn.LayerNorm
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

    start_epoch = 0
    log = {}
    best_loss = float(1e9)
    current_loss = float(1e9)
    best_acc = 0.
    current_acc = 0.

    # If resuming, check if file exists and then load everything accordingly
    resume = args.resume
    if resume:
        assert len(resume.split(',')) == 3, "resume must in form: 'experiment,runID,suffix'"
        experiment,run_id,suffix = resume.split(',')
        ckpt = f'./results/{experiment}/{run_id}/checkpoint_{suffix}.pth'
        if not os.path.exists(ckpt):
            experiment = cfg['name']
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f'=> Checkpoint {ckpt} not found!\n=> Running experiment: {experiment} with id: {run_id}')
        else:
            print(f'=> Resuming from: {ckpt}')
            checkpoint = torch.load(ckpt, map_location=device)

            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']
            val_loss = checkpoint['current_loss']
            best_acc = checkpoint['best_acc']
            val_acc = checkpoint['current_acc']

            log = read_yaml(os.path.join('./logs', experiment, run_id, 'log.yaml'))
        
    # Setting up the save locations
    log_dir = cfg['log_dir']
    log_path = os.path.join(log_dir, experiment, run_id)
    save_dir = cfg['save_dir']
    save_path = os.path.join(save_dir, experiment, run_id)

    # Old method --> not optimized for .tar files
    # train_data = AudioDataset(
    #                 dir='/home/maxmay/files_to_copy.txt', 
    #                 target_samplerate = 48000
    #             )
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # New method --> optimized for large tar file directories
    dataset = cfg['dataset']
    WAS = WebAudioSet(
        base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
        val_data_dir = dataset['base_data_dir']+dataset['val_split']+'.tar',
        hrtf_dir = dataset['sofa_dir'],
        target_samplerate = dataset['sample_rate'],
        batch_size = dataset['batch_size'],
        resample= dataset['resample']
    )
    WAS.setup('fit')
    train_epoch_size = count_pattern_files(dataset['train_split'])
    val_epoch_size = count_pattern_files(dataset['val_split'])

    train_loader = WAS.train_wds_loader(epoch_size=train_epoch_size)
    val_loader = WAS.val_wds_loader(epoch_size=val_epoch_size)

    # Since we resample the data, the dataloader is infinite and we need to set a limit per epoch
    # This is done inside the WebAudioSet class --> {train/val}_wds_loader()
    trainer = cfg['trainer']
    nr_epochs = trainer['epochs']
    save_freq = trainer['save_freq']

    for epoch in range(start_epoch, nr_epochs):
        print(f'Epoch:[{epoch+1}/{nr_epochs}]')
        train_acc, train_loss = train(model, train_loader, criterion, scheduler, optimizer, device)
        print(f'Training accuracy: {train_acc:.3f} | Loss: {train_loss:.3f}')

        val_acc, val_loss = validate(model, val_loader, criterion, device)
        print(f'Validation Accuracy: {val_acc:.3f} | Loss: {val_loss:.3f}\n')

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'backbone': arch['_component_'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'current_loss': val_loss,
            'best_acc': best_acc,
            'current_acc': val_acc,
        }, is_best=is_best, save_path=save_path)

        if (epoch % save_freq) == 0:
            # create a logger here
            log = logger(log, epoch, 
                {'train_acc': train_acc, 
                'train_loss': train_loss, 
                'val_acc': val_acc, 
                'val_loss': val_loss},
                log_path=log_path)

    print(f'Done training!')


def save_checkpoint(state: dict, is_best: bool, save_path: str, fn: str='checkpoint_last.pth'):
    # Check if directory exists, else create all parent dirs
    Path(save_path).mkdir(parents=True, exist_ok=True) # results/sound_ipcl_base/runID/
    fn = os.path.join(save_path, fn)
    torch.save(state, fn)
    if is_best:
        fn_best = fn.replace('last', 'best')
        shutil.copyfile(fn, fn_best)


def logger(log: dict, epoch: int, parameters: dict, log_path:str, fn: str= 'log.yaml'):
    # Check if directory exists, else create all parent dirs
    Path(log_path).mkdir(parents=True, exist_ok=True)
    log[epoch] = parameters
    write_yaml(log, os.path.join(log_path, fn))
    return log


def test_dataloader(args):
    debug = args.debug
    if debug:
        print(f'Debugging mode is active')

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

    dataset = cfg['dataset']
    WAS = WebAudioSet(
        base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
        val_data_dir = dataset['base_data_dir']+dataset['val_split']+'.tar',
        hrtf_dir = dataset['sofa_dir'],
        target_samplerate = dataset['sample_rate'],
        batch_size = dataset['batch_size'],
        resample= dataset['resample'],
        debug=debug
    )
    WAS.setup('fit')
    train_epoch_size = count_pattern_files(dataset['train_split'])
    val_epoch_size = count_pattern_files(dataset['val_split'])

    train_loader = WAS.train_wds_loader(epoch_size=train_epoch_size)
    val_loader = WAS.val_wds_loader(epoch_size=val_epoch_size)

    # total_guessed = 0
    # for epoch in range(10):
    #     print(f'[Epoch: {epoch}/10]')
    #     for idx, (data, _) in enumerate(train_loader):
    #         total_guessed += data.shape[0]
    #         print(f'[Batch: {idx+1}]: {total_guessed}')
    #     print("")
    # print(f"Done")




if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Supervised training Resnet-50')
    args.add_argument('-c', '--config', 
                    default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-r', '--resume', 
                    default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--debug', 
                    default=False, action=argparse.BooleanOptionalAction,
                    help='turn on debuggin mode')  
    args = args.parse_args()

    main(args)
    # test_dataloader(args)


