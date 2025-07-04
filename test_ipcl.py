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
    print('=> Testing IPCL mode')
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

    was = WebAudioSet(
        base_data_dir = dataset['base_data_dir']+dataset['train_split']+'.tar',
        val_data_dir = dataset['val_data_dir']+dataset['val_split']+'.tar',
        hrtf_dir = hrtf,
        target_samplerate = dataset['sample_rate'],
        batch_size = batch_size,  # This way you get [batch_size x n_samples] (128*5)
        resample= dataset['resample'],
        ipcl=True
    )
    was.setup('inf')
    val_epoch_size = count_pattern_files(dataset['val_split'])
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
    if args.weights:
        print(f'=> Using {args.weights} to initialize model')
        learner = load_weights(learner, args.weights, device=device)
    else:
        print('=> No weights file found, randomly initialize')
    learner = learner.to(device)

    # Augmentation
    transform = Transform(n_samples=n_samples, hrtf=hrtf, target_samplerate=48000)

    # Get embeddings
    ouptut = {}
    embeddings,labels = test_ipcl(learner, val_loader, n_samples, transform, device=device)

    ouptut['embedding'] = embeddings
    ouptut['labels'] = labels
    save_checkpoint(ouptut, is_best=False, save_path=os.path.join('./results/embeddings', experiment, run_id), fn='embeddings.pth')
    print(f'=> Saved embeddings to "{os.path.join("./results/embeddings", experiment, run_id, "embeddings.pth")}"')
    
    print(f'Done inferencing!')


def save_checkpoint(state: dict, is_best: bool, save_path: str, fn: str='checkpoint_last.pth'):
    # Check if directory exists, else create all parent dirs
    Path(save_path).mkdir(parents=True, exist_ok=True) # results/sound_ipcl_base/runID/
    fn = os.path.join(save_path, fn)
    torch.save(state, fn)
    if is_best:
        fn_best = fn.replace('last', 'best')
        shutil.copyfile(fn, fn_best)


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
    args.add_argument('-d', '--debug', 
                    default=False, action=argparse.BooleanOptionalAction,
                    help='turn on debuggin mode')  
    args.add_argument('-w', '--weights', 
                    default=None, type=str,
                    help='path to weights file (default: None)')
    args = args.parse_args()
    main(args)