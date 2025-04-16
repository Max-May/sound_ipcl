import os
import time 
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

import sofa
from tqdm import tqdm

from models.resnet import ResNet, Bottleneck
from dataloader.dataloader import AudioDataset
from dataloader.WebAudioSet import WebAudioSet
from utils.util import read_yaml, write_yaml, seed_all, count_pattern_files


# TODO:
# Create dataloader to create cochleagrams to input into inference
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

    # experiment = cfg['name']
    # run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

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
                num_classes=arch['out_channels']
            )
    model = load_weights(model, args.weights, device=device)
    model = model.to(device)

    # Data
    train_data = AudioDataset(
                    mode='inf',
                    dir='/home/maxmay/files_to_copy.txt', 
                    target_samplerate = 48000
                )
    val_loader = DataLoader(train_data, batch_size=4, shuffle=False)
    # dataset = cfg['dataset']
    # WAS = WebAudioSet(
    #     base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
    #     val_data_dir = dataset['base_data_dir']+dataset['val_split']+'.tar',
    #     hrtf_dir = dataset['sofa_dir'],
    #     target_samplerate = dataset['sample_rate'],
    #     batch_size = dataset['batch_size'],
    #     resample= dataset['resample']
    # )
    # WAS.setup('inf')
    # val_epoch_size = count_pattern_files(dataset['val_split'])
    # val_loader = WAS.val_wds_loader(epoch_size=val_epoch_size, nr_workers=dataset['nr_workers'])

    print('=> Starting inference...')
    for epoch in range(2):
        total_guessed = 0
        for idx, (data, labels) in enumerate(val_loader):
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device)
            total_guessed += data.shape[0]
            print(f'[Batch: {idx+1}/{len(val_loader)}]: {total_guessed}')
        
            pred = infer(model, data)
            print(f'Currently predicted (labels, actual):')
            print(*zip(pred, labels))
            break
        break


def load_weights(model, fn: str, device='cpu'):
    assert os.path.exists(fn), f'"{fn}" must be a valid path'
    print(f'=> Loading weights from: {fn}')
    ckpt = torch.load(fn, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


@torch.no_grad()
def infer(model, input):

    outputs = model(input)
    _, preds = torch.max(outputs.data, 1)
    return preds


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Inference Resnet-50')
    args.add_argument('-c', '--config', 
                    default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-w', '--weights', 
                    default=None, type=str,
                    help='path to weights file (default: None)')
    args.add_argument('-d', '--debug', 
                    default=False, action=argparse.BooleanOptionalAction,
                    help='turn on debuggin mode')  
    args = args.parse_args()
    main(args)