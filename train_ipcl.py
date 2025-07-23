import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import argparse

from models.resnet import ResNet, Bottleneck
from ipcl.ipcl import IPCL
from dataloader.WebAudioSet import WebAudioSet
from dataloader.dataset_functions import Transform
from utils.knn import knn_monitor
from utils.util import read_yaml, write_yaml, seed_all, count_pattern_files


def train(learner, dataloader, n_samples, transform, scheduler, optimizer, device, steps=None, writer=None):
    learner.train()

    train_running_loss = 0.
    counter = 0

    features, targets = [], []

    for idx, (data, _) in enumerate(dataloader):
        counter += 1

        # Preprocessing: take 1 location and convolve 'n_samples' audio fragments
        data, targs = transform(data)
        data = data.to(device, dtype=torch.float)
        targs = targs.to(device)

        loss, (embeddings, prototypes) = learner(data, targs)

        if writer:
            writer.add_scalar(
                'Loss/train/per_step',
                loss.item(), global_step=writer.train_step
            )

            writer.train_step += 1

        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        feat = embeddings.chunk(n_samples)[0].detach().cpu()
        feat = feat.view(feat.shape[0],-1)
        features.append(feat)
        targets.append(targs.chunk(n_samples)[0].cpu())

        if scheduler is not None:
            scheduler.step()

        if steps is not None and counter == steps:
            break

    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)

    epoch_loss = train_running_loss / counter
    return epoch_loss, features, targets


@torch.no_grad()
def validate(learner, dataloader, device, writer=None):
    learner.eval()
    valid_running_loss = 0
    counter = 0

    for idx, (data, _) in enumerate(dataloader):
        counter += 1
        # print(f'Data before augmentation: {data.shape}')

        data, targets = transform(data)
        data = data.to(device, dtype=torch.float)
        targets = targets.to(device)
        # print(f'Data after augmentation: {data.shape}\nLabels: {targets.shape}\n{targets}')

        loss, _ = learner(data, targets)
        valid_running_loss += loss.item()
        # print(f'Embedding space: {embeddings.shape}')
        # print(f'Prototypes: {prototypes.shape}')
        # print(f'Loss: {loss.item()}')
        break

        if writer:
            writer.add_scalar(
                'Loss/validate/per_step',
                loss.item(), global_step=writer.val_step
            )

            writer.val_step += 1

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    return epoch_loss


@torch.no_grad()
def test_ipcl(learner, loader, n_samples, transform, device, steps=None):
    features, targets = [], []
    counter = 0

    for idx, (data, _) in enumerate(loader):
        counter += 1

        # Preprocessing: take 1 location and convolve 'n_samples' audio fragments
        data, targs = transform(data)
        data = data.to(device, dtype=torch.float)
        targs = targs.to(device)

        _, (embeddings, prototypes) = learner(data, targs)

        feat = embeddings.chunk(n_samples)[0].detach().cpu()
        feat = feat.view(feat.shape[0],-1)
        features.append(feat)
        targets.append(targs.chunk(n_samples)[0].cpu())

        if steps is not None and counter == steps:
            break

    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    return features, targets


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

    # Nr of samples for IPCL
    n_samples = cfg['n_samples']

    # Encoder
    _encoder = cfg['encoder']
    print(f'=> Using encoder with arch: {_encoder["_arch_"]}')
    if _encoder['block'].lower() == 'bottleneck':
        block = Bottleneck
    encoder = ResNet(
        block=block,
        layers=_encoder['layers'],
        input_channels=_encoder['in_channels'],
        num_classes=_encoder['out_channels'],
        l2norm=_encoder['l2_norm']
    ).float()

    # Dataset and loader
    dataset = cfg['dataset']
    hrtf = dataset['sofa_dir']
    batch_size = dataset['batch_size']
    train_epoch_size = count_pattern_files(dataset['train_split'])
    val_epoch_size = count_pattern_files(dataset['val_split'])

    was = WebAudioSet(
        base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
        val_data_dir = dataset['val_data_dir']+dataset['val_split']+'.tar',
        hrtf_dir = hrtf,
        target_samplerate = dataset['sample_rate'],
        batch_size = batch_size,  # This way you get [batch_size x n_samples] (128*5)
        resample= dataset['resample'],
        ipcl=True
    )
    was.setup('fit')

    train_loader = was.train_wds_loader(epoch_size=train_epoch_size)
    val_loader = was.val_wds_loader(epoch_size=val_epoch_size)

    # IPCL Model (called learner in this strategy)
    _learner = cfg['learner']
    if _learner['_arch_'] == 'ipcl':
        learner = IPCL(
            base_encoder=encoder,
            numTrainFiles=train_epoch_size*2000,
            K=_learner['queue_size'],
            T=_learner['temperature'],
            out_dim=_learner['embedding_space'],
            n_samples=n_samples
        ).float()
    learner = learner.to(device)

    trainer = cfg['trainer']
    # stepwise = True if trainer['method'] == 'stepwise' else False
    steps = trainer['steps']
    if steps == 0:
        steps = None
    else:
        print(f'=> Using {steps} steps per loop')

    nr_epochs = trainer['epochs']
    save_freq = trainer['save_freq']

    # Optimizer
    optimizer = cfg['optimizer']
    if optimizer['_component_'].lower() == 'sgd':
        optimizer = optim.SGD(learner.parameters(), lr=optimizer['lr'])
    elif optimizer['_component_'].lower() == 'adamw':
        optimizer = optim.AdamW(learner.parameters(), lr=optimizer['lr'])

    # Scheduler (option between batch-, or epoch-based scheduling)
    epoch_scheduler, batch_scheduler, tau_scheduler = None, None, None 
    scheduler = cfg['scheduler']['_component_'].lower()
    if scheduler == 'multisteplr':
        epoch_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5,10], gamma=.5)
    elif scheduler == 'onecyclelr':
        if steps is None:
            steps_per_epoch = (train_epoch_size*2000)//batch_size
        batch_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=0.01, 
            epochs=nr_epochs, 
            steps_per_epoch=steps_per_epoch
        )

    print(f'=> Using "{type(epoch_scheduler).__name__ if epoch_scheduler is not None else type(batch_scheduler).__name__}" as scheduler')

    # Augmentation
    transform = Transform(n_samples=n_samples, hrtf=hrtf, target_samplerate=48000)

    # Other parameters
    start_epoch = 0
    best_loss = float(1e9)
    best_top1 = 0.
    train_step = 0
    writer_step = 0 # Global epoch SumaryWriter step in case of resume training
    embeddings = {}
    writer = None

    # If resuming, check if file exists and then load everything accordingly
    resume = args.resume
    if resume:
        assert len(resume.split(',')) == 3, "resume must in form: 'experiment,runID,suffix'"
        experiment,run_id,suffix = resume.split(',')
        ckpt = f'./results/{experiment}/{run_id}/checkpoint_{suffix}.pth'
        ckpt_embeddings = f'./results/embeddings/{experiment}/{run_id}/embeddings.pth'
        if not os.path.exists(ckpt):
            experiment = cfg['name']
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f'=> Checkpoint {ckpt} not found!\n=> Running experiment: {experiment} with id: {run_id}')
        else:
            print(f'=> Resuming from: {ckpt}')
            checkpoint = torch.load(ckpt, map_location=device)

            start_epoch = checkpoint['epoch']
            learner.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if epoch_scheduler is not None:
                epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
            if batch_scheduler is not None:
                batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
            best_loss = checkpoint['best_loss']
            best_top1 = checkpoint['best_top1']
            train_step = checkpoint['train_step']
            writer_step = checkpoint['writer_step']
            # writer = checkpoint['writer']

            if os.path.exists(ckpt_embeddings):
                print(f'=> Fetching embeddings found at {ckpt_embeddings}')
                embeddings = torch.load(ckpt_embeddings)
            else:
                print(f'=> No saved embeddings found at {ckpt_embeddings}')


    # Setting up the save locations
    log_dir = cfg['log_dir']
    log_path = os.path.join(log_dir, experiment, run_id)
    save_dir = cfg['save_dir']
    save_path = os.path.join(save_dir, experiment, run_id)

    # Setup for Tensorboard SummaryWriter
    if store_all and writer is None:
        writer = SummaryWriter(log_dir=log_path)
        if not hasattr(writer, 'train_step'):
            writer.train_step = train_step
        if not hasattr(writer, 'step'):
            writer.step = writer_step

    # Main training loop
    for epoch in range(start_epoch, nr_epochs):
        if args.oneloop:
            print("=> Running for 1 epoch")

        # Save embeddings pre-trained for visualization purposes
        if epoch == 0 and args.embeddings:
            print('=> Obtaining embeddings pre-trained')
            test_embeddings,test_labels = test_ipcl(learner, train_loader, n_samples, transform, device=device)
            embeddings['-1'] = [test_embeddings, test_labels]
            print('=> Saving embeddings pre-trained')
            save_checkpoint(embeddings, is_best=False, save_path=os.path.join('./results/embeddings', experiment, run_id), fn='embeddings.pth')
            print(f'=> Saved embeddings to "{os.path.join("./results/embeddings", experiment, run_id, "embeddings.pth")}"')

        print(f'Epoch:[{epoch+1}/{nr_epochs}]')
        train_loss, trainX, trainY = train(learner, train_loader, n_samples, transform, batch_scheduler, optimizer, device, steps=steps, writer=writer)
        print(f'Training Loss: {train_loss:.3f}')
        
        # Validation with k-nearest neighbour
        if args.knn:
            top1, top5 = knn_monitor(
                learner.base_encoder, trainX, trainY, val_loader, sigma=learner.T, 
                K=200, num_chunks=200,
                n_samples=n_samples, hrtf=hrtf, device=device
            )
        else:
            top1 = 0
            top5 = 0

        if epoch_scheduler is not None:
            epoch_scheduler.step()

        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)
        best_top1 = max(top1, best_top1)

        if args.embeddings:
            embeddings[f'{epoch}'] = [trainX, trainY]
            save_checkpoint(embeddings, is_best=False, save_path=os.path.join('./results/embeddings', experiment, run_id), fn='embeddings.pth')
            print(f'=> Saved embeddings to "{os.path.join("./results/embeddings", experiment, run_id, "embeddings.pth")}"')
    
        # Save
        if store_all:
            writer.add_scalar(
                'Loss/train/avg',
                train_loss, global_step=writer.step
            )
            writer.add_scalar(
                'Acc/knn/top1',
                top1, global_step=writer.step
            )
            writer.add_scalar(
                'Acc/knn/top5',
                top5, global_step=writer.step
            )

            writer.step += 1

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': learner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_scheduler': epoch_scheduler.state_dict() if hasattr(epoch_scheduler, 'state_dict') else None,
                'batch_scheduler': batch_scheduler.state_dict() if hasattr(batch_scheduler, 'state_dict') else None,
                'best_top1': best_top1,
                'best_loss': best_loss,
                'writer_step': writer.step,
                'train_step': writer.train_step,
                # 'writer': writer
            }, is_best=is_best, save_path=save_path)
        if args.oneloop:
            print('=> Done 1 epoch, exiting now...')
            break

    print(f'Done training!')


def save_checkpoint(state: dict, is_best: bool, save_path: str, fn: str='checkpoint_last.pth'):
    # Check if directory exists, else create all parent dirs
    Path(save_path).mkdir(parents=True, exist_ok=True) # results/sound_ipcl_base/runID/
    fn = os.path.join(save_path, fn)
    torch.save(state, fn)
    if is_best:
        fn_best = fn.replace('last', 'best')
        shutil.copyfile(fn, fn_best)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Self supervised training Resnet-50')
    args.add_argument('-c', '--config', 
                    default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-r', '--resume', 
                    default=None, type=str,
                    help='path to latest checkpoint in form "experiment,runID,suffix" (default: None)')
    args.add_argument('-s', '--store',
                    default=True, action=argparse.BooleanOptionalAction,
                    help="Turn off if you don't want to store everything (--no-s or --no--store)")
    args.add_argument('-d', '--debug', 
                    default=False, action=argparse.BooleanOptionalAction,
                    help='turn on debuggin mode')  
    args.add_argument('-e', '--embeddings',
                    default=False, action=argparse.BooleanOptionalAction,
                    help="Turn on if you want to store the embeddings")
    args.add_argument('-k', '--knn',
                    default=False, action=argparse.BooleanOptionalAction,
                    help="Turn on if you want to validate ipcl using k-Nearest Neighbours")
    args.add_argument('-o', '--oneloop',
                    default=False, action=argparse.BooleanOptionalAction,
                    help="Turn on if you just want to run 1 epoch")
    args = args.parse_args()
    main(args)
