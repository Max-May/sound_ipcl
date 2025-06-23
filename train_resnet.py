import os
import shutil
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim 
import torch.nn as nn
import torchsummary

from models.resnet import ResNet, Bottleneck, resnet50_groupnorm
from dataloader.dataloader import AudioDataset
from dataloader.WebAudioSet import WebAudioSet

import sofa
from tqdm import tqdm
import argparse
from utils.util import read_yaml, write_yaml, seed_all, count_pattern_files

import time 
from datetime import datetime


def check_for_nans(tensor, name="tensor"):
    if not torch.isfinite(tensor).all():
        print(f"[NaN/Inf DETECTED] in {name}")
        print(f"{tensor}")
        raise ValueError(f"Invalid value detected in {name}")


def train(model, dataloader, criterion, scheduler, optimizer, device, remap=None, steps=None, writer=None):
    model.train()

    train_running_loss = 0.
    train_running_acc = 0
    counter = 0
    total_guessed = 0

    # time_total = 0
    # time_start = time.time()

    for idx, (data, labels) in enumerate(dataloader):
        counter += 1
        total_guessed += data.shape[0]
        # print(f'[Batch: {idx+1}/{dataloader.nsamples}]: {total_guessed}', end="\r", flush=True)

        # Debugging purpose
        try:
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device)

            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(outputs.data, 1)
            accuracy = (preds == labels).sum().item()
            train_running_acc += accuracy 

            loss.backward()
            optimizer.step()
            if type(scheduler).__name__ == 'OneCycleLR':
                scheduler.step()

            if writer:
                writer.add_scalar(
                    'Loss/train/per_step',
                    loss.item(), global_step=writer.train_step
                )
                writer.add_scalar(
                    'Acc/train/per_step',
                    (accuracy/data.shape[0])*100., global_step=writer.train_step
                )

                writer.train_step += 1

            if steps is not None and counter == steps:
                break

        except Exception as e:
            print(f"Exception in training batch {idx+1}: {e}")
            break

    if type(scheduler).__name__ != 'OneCycleLR':
        scheduler.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_acc / total_guessed)
    return epoch_acc, epoch_loss



@torch.no_grad()
def validate(model, dataloader, criterion, device, remap=None, writer=None):
    model.eval()
    valid_running_correct = 0
    valid_running_loss = 0
    counter = 0
    total_guessed = 0

    nr_nans_batch = 0
    nr_nans_files = 0
    for idx, (data, labels) in enumerate(dataloader):
        counter += 1
        total_guessed += data.shape[0]
        # print(f'[Batch: {idx+1}/{dataloader.nsamples}]: {total_guessed}', end="\r", flush=True)

        try:
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device)
            
            # Forward pass.
            outputs = model(data)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            accuracy = (preds == labels).sum().item()
            valid_running_correct += accuracy

            if writer:
                writer.add_scalar(
                    'Loss/validate/per_step',
                    loss.item(), global_step=writer.val_step
                )
                writer.add_scalar(
                    'Acc/validate/per_step',
                    (accuracy/data.shape[0])*100., global_step=writer.val_step
                )

                writer.val_step += 1

        except Exception as e:
            print(f"Exception in validating batch {idx+1}")
            break

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / total_guessed)
    return epoch_acc, epoch_loss


def main(args):
    debug = args.debug
    store_all = args.store
    if debug:
        print(f'=> Debugging mode is active!')
    if not store_all:
        print(f"WARNING: 'store_all' = {store_all}, run won't be saved")
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
    gpu = cfg['gpu']
    n_gpus = gpu['n_gpu']
    use_cuda = True if n_gpus > 0 else False
    if use_cuda:
        cuda = gpu['cuda']
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" + f':{cuda}' if use_cuda else "cpu")
    print(f'=> Using device: "{device}"')
    if use_cuda:
        curr_device = torch.cuda.current_device()
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
            )
    # model = resnet50_groupnorm(
    #             input_channels=arch['in_channels'],
    #             num_classes=arch['out_channels'], 
    #             num_groups=1
    #             )
    model = model.to(device)
    # torchsummary.summary(resnet, (3, 128, 128))

    trainer = cfg['trainer']
    # stepwise = True if trainer['method'] == 'stepwise' else False
    steps = trainer['steps'] if ('steps' in trainer and trainer['steps'] > 0) else None
    nr_epochs = trainer['epochs']
    save_freq = trainer['save_freq']

    dataset = cfg['dataset']
    train_epoch_size = count_pattern_files(dataset['train_split'])
    val_epoch_size = count_pattern_files(dataset['val_split'])

    # Optimizer
    optimizer = cfg['optimizer']
    if optimizer['_component_'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=optimizer['lr'])
    elif optimizer['_component_'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=optimizer['lr'])

    scheduler = cfg['scheduler']
    if scheduler['_component_'].lower() == 'multisteplr':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3,6], gamma=.5) # Based on prelim investigation
    elif scheduler['_component_'].lower() == 'onecyclelr':
        if steps == None:
            steps = (train_epoch_size*2000)//dataset['batch_size']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=0.01, 
            epochs=nr_epochs, 
            steps_per_epoch=steps
        )
    print(f'=> Using "{type(scheduler).__name__}" as scheduler')
    
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

    train_step = 0
    val_step = 0
    writer_step = 0 # Global epoch SumaryWriter step in case of resume training

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
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_loss = checkpoint['best_loss']
            val_loss = checkpoint['current_loss']
            best_acc = checkpoint['best_acc']
            val_acc = checkpoint['current_acc']
            train_step = checkpoint['train_step']
            val_step = checkpoint['val_step']
            writer_step = checkpoint['writer_step']

            # log = read_yaml(os.path.join('./logs', experiment, run_id, 'log.yaml'))
        
    # Setting up the save locations
    log_dir = cfg['log_dir']
    log_path = os.path.join(log_dir, experiment, run_id)
    save_dir = cfg['save_dir']
    save_path = os.path.join(save_dir, experiment, run_id)

    # Labels to exclude, see documentation
    # removed = [34, 57, 80, 103, 126, 149, 172, 195, 218, 241, 264, 287, 310, 333, 356, 379, 402, 425, 
    # 448, 471, 494, 517, 540, 563, 586, 609, 632, 655, 678, 701, 724, 747, 770, 793, 816]
    # if removed:
    #     kept = [i for i in range(828) if i not in removed]
    #     remap = torch.tensor(kept, dtype=torch.int).to(device)
    # else:
    remap = None

    # New method --> optimized for large tar file directories
    WAS = WebAudioSet(
        base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
        val_data_dir = dataset['val_data_dir']+dataset['val_split']+'.tar',
        hrtf_dir = dataset['sofa_dir'],
        target_samplerate = dataset['sample_rate'],
        batch_size = dataset['batch_size'],
        resample= dataset['resample'],
        debug=debug
    )
    print(f"=> Train data path: {dataset['base_data_dir']+dataset['train_split']+'.tar'}")
    print(f"=> Validate data path: {dataset['val_data_dir']+dataset['val_split']+'.tar'}")
    WAS.setup('fit')

    # train_epoch_size = None
    # val_epoch_size = None

    train_loader = WAS.train_wds_loader(epoch_size=train_epoch_size)
    val_loader = WAS.val_wds_loader(epoch_size=val_epoch_size)

    # Since we resample the data, the dataloader is infinite and we need to set a limit per epoch
    # This is done inside the WebAudioSet class --> {train/val}_wds_loader()

    # Setup for Tensorboard SummaryWriter
    writer = None
    if store_all:
        writer = SummaryWriter(log_dir=log_path)
        if not hasattr(writer, 'train_step'):
            writer.train_step = train_step
        if not hasattr(writer, 'val_step'):
            writer.val_step = val_step
        if not hasattr(writer, 'step'):
            writer.step = writer_step

    for epoch in range(start_epoch, nr_epochs):
        print(f'Epoch:[{epoch+1}/{nr_epochs}]')
        train_acc, train_loss = train(model, train_loader, criterion, scheduler, optimizer, device, remap=remap, steps=steps, writer=writer)
        print(f'Training accuracy: {train_acc:.3f} | Loss: {train_loss:.3f}')

        val_acc, val_loss = validate(model, val_loader, criterion, device, remap=remap, writer=writer)
        print(f'Validation Accuracy: {val_acc:.3f} | Loss: {val_loss:.3f}\n')

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        best_acc = max(val_acc, best_acc)

        if store_all:
            save_checkpoint({
                'epoch': epoch + 1,
                'backbone': arch['_component_'],
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'current_loss': val_loss,
                'best_acc': best_acc,
                'current_acc': val_acc,
                'writer_step': writer_step,
                'train_step': train_step,
                'val_step': val_step
            }, is_best=is_best, save_path=save_path)

            writer.add_scalar(
                'Loss/train/avg',
                train_loss, global_step=writer.step
            )
            writer.add_scalar(
                'Acc/train/avg',
                train_acc, global_step=writer.step
            )
            writer.add_scalar(
                'Loss/validate/avg',
                val_loss, global_step=writer.step
            )
            writer.add_scalar(
                'Acc/validate/avg',
                val_acc, global_step=writer.step
            )

            writer.step += 1

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
    args.add_argument('-s', '--store',
                    default=True, action=argparse.BooleanOptionalAction,
                    help="Turn off if you don't want to store everything (--no-s or --no--store)")
    args.add_argument('-d', '--debug', 
                    default=False, action=argparse.BooleanOptionalAction,
                    help='turn on debuggin mode')  
    args = args.parse_args()

    main(args)
    # test_dataloader(args)


