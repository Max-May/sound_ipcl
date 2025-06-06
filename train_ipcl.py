import os
import time

import numpy as np
import torch

import argparse

from models.resnet import ResNet, Bottleneck
from ipcl.ipcl import IPCL
from dataloader.WebAudioSet import WebAudioSet
from dataloader.dataset_functions import augment
from utils.util import read_yaml, write_yaml, seed_all, count_pattern_files

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

    n_samples = 5

    encoder = ResNet(
        block=Bottleneck,
        layers=[3,4,6,3],
        input_channels=2,
        num_classes=128,
        l2norm=True
    ).float()

    learner = IPCL(
        base_encoder=encoder,
        numTrainFiles=18000,
        K=4096,
        T=0.07,
        out_dim=128,
        n_samples=n_samples
    ).float()

    dataset = cfg['dataset']
    hrtf = dataset['sofa_dir']
    train_epoch_size = count_pattern_files(dataset['train_split'])

    was = WebAudioSet(
        base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
        val_data_dir = dataset['val_data_dir']+dataset['val_split']+'.tar',
        hrtf_dir = hrtf,
        target_samplerate = dataset['sample_rate'],
        batch_size = dataset['batch_size'],
        resample= dataset['resample'],
        ipcl=True
    )
    was.setup('fit')

    train_loader = was.train_wds_loader(epoch_size=train_epoch_size)

    avg_time = 0.
    total = 0
    counter = 0

    device = 'cpu'
    for idx, (data, _) in enumerate(train_loader):
        counter += 1
        total += data.shape[0]
        print(f'Data before augmentation: {data.shape}')

        data, targets = augment(data, n_samples, hrtf=hrtf)
        data = data.to(device, dtype=torch.float)
        targets = targets.to(device)
        print(f'Data after augmentation: {data.shape}\nLabels: {targets.shape}\n{targets}')

        loss, (embeddings, prototypes) = learner(data, targets)
        print(loss)
        break
    # input = torch.rand(n_samples,2,39,8000)
    # targs = torch.randint(0, 828, (1,))
    # targs = targs.repeat(n_samples)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Self supervised training Resnet-50')
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