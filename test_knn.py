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
from utils.knn import knn_monitor, get_features
from utils.knn import run_kNN_chunky as run_knn
from utils.util import read_yaml, write_yaml, seed_all, count_pattern_files


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
    # train_epoch_size = count_pattern_files(dataset['train_split'])
    val_epoch_size = count_pattern_files(dataset['val_split'])
    test_epoch_size = count_pattern_files(dataset['test_split'])

    was = WebAudioSet(
        base_data_dir = dataset['base_data_dir'],
        train_data_dir = dataset['base_data_dir'] + dataset['train_split'] +'.tar',
        test_data_dir = dataset['base_data_dir']+dataset['test_split']+'.tar',
        val_data_dir = dataset['val_data_dir']+dataset['val_split']+'.tar',
        hrtf_dir = hrtf,
        target_samplerate = dataset['sample_rate'],
        batch_size = batch_size,  # This way you get [batch_size x n_samples] (128*5)
        resample= dataset['resample'],
        ipcl=True
    )
    was.setup('ipcl_inf')

    # train_loader = was.train_wds_loader(epoch_size=train_epoch_size)
    test_loader = was.test_wds_loader(epoch_size=test_epoch_size)
    val_loader = was.val_wds_loader(epoch_size=val_epoch_size)

    # IPCL Model (called learner in this strategy)
    _learner = cfg['learner']
    if _learner['_arch_'] == 'ipcl':
        learner = IPCL(
            base_encoder=encoder,
            numTrainFiles=val_epoch_size*2000,
            K=_learner['queue_size'],
            T=_learner['temperature'],
            out_dim=_learner['embedding_space'],
            n_samples=n_samples
        ).float()

    weights = args.weights
    ckpt_step = args.checkpoint
    assert len(weights.split(',')) == 3, "resume must in form: 'experiment,runID,suffix'"
    experiment,run_id,suffix = weights.split(',')
    if ckpt_step:
        suffix = ckpt_step
    weights_fn = f'./results/{experiment}/{run_id}/checkpoint_{suffix}.pth'
    print(weights_fn)
    if os.path.exists(weights_fn):
        learner = load_weights(learner, weights_fn, device=device)
    else:
        print(f"{weights_fn} does not exist")
    learner = learner.to(device)

    # Augmentation
    # transform = Transform(n_samples=n_samples, hrtf=hrtf, target_samplerate=48000)
    epoch_size = (val_epoch_size * 2000) // batch_size

    top1, top5 = run_knn(learner.base_encoder, test_loader, val_loader, n_samples, hrtf, epoch_size, device=device)
    # If embeddings exist, fetch them
    # if not args.checkpoint:
    #     step = ''
    # else:
    #     step = '_' + ckpt_step
    # ckpt_embeddings = f'./results/embeddings/{experiment}/{run_id}/embeddings{step}.pth'
    # print(ckpt_embeddings)
    # trainX, trainY = None, None
    # if os.path.exists(ckpt_embeddings):
    #     print(f'=> Fetching embeddings found at {ckpt_embeddings}')
    #     embeddings = torch.load(ckpt_embeddings)
    #     trainX = embeddings['embedding']
    #     trainY = embeddings['labels']

    # if not trainX and not trainY:
    #     print('=> Obtaining embeddings...')
    #     trainX, trainY = get_features(learner.base_encoder, test_loader, n_samples=n_samples, hrtf=hrtf, device=device)

    # print('=> Performing k-Nearest Neighbors...')
    # top1, top5 = knn_monitor(
    #     learner.base_encoder, trainX, trainY, val_loader, sigma=learner.T, 
    #     K=200, num_chunks=200,
    #     n_samples=n_samples, hrtf=hrtf, device=device
    # )


def load_weights(model, fn: str, device='cpu'):
    assert os.path.exists(fn), f'"{fn}" must be a valid path'
    print(f'=> Loading weights from: {fn}')
    ckpt = torch.load(fn, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Self supervised training Resnet-50')
    args.add_argument('-c', '--config', 
                    default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-w', '--weights', 
                    default=None, type=str,
                    help='path to weights file (default: None)')
    args.add_argument('--checkpoint',
                    default=None, type=int,
                    help='Checkpoint to fetch embeddings amd weights from')
    args.add_argument('-d', '--debug', 
                    default=False, action=argparse.BooleanOptionalAction,
                    help='turn on debuggin mode')  


    args = args.parse_args()
    main(args)