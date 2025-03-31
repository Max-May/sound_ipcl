import os
from pathlib import Path

import sys
import time
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.functional import fftconvolve

# WebDataSet for handling tar files
import webdataset as wds
os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"

import sofa
from tqdm import tqdm

from dataset_functions import __getitem__
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.Generate_Cochleagram import generate_cochleagram


HOME_DIR = '/home/maxmay'
WORK_DIR = 'sound_ipcl'
TRAIN_DIR = 'Data/audio/bal_train'


def collate_fn(batch):
    print(batch)
    x = batch
    
    return x.flatten(start_dim=0, end_dim = 1)


class WebAudioSet(Dataset):
    def __init__(self):
        pass


def open_sofa(fn: str = 'KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa'):
    # Open .sofa file
    file_path = os.path.join(HOME_DIR, WORK_DIR, 'utils', fn)
    hrtf = sofa.Database.open(file_path)
    return hrtf


def main():
    shared_url = '/home/maxmay/Data/bal_train{00..09}.tar'
    hrtf = open_sofa()

    print('Creating dataset...')
    pre_process_function = partial(__getitem__,
                                    hrtf,
                                    target_samplerate=48000)

    dataset = (wds.WebDataset(shared_url, resampled=True, cache_dir='/tmp', shardshuffle=True)
            .shuffle(1000)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple('flac')
            .map(pre_process_function, handler=wds.warn_and_continue)
            .batched(16))

    print('Creating loader...')
    loader = DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=collate_fn)

    print('Trying to batch...')
    for idx, data in enumerate(loader):
        print(len(data))
        print(len(data[0]))
        print(len(data[0][0]))
        print(data[0][0][0][1])
        print(data[0][0][0][0].shape)

        break

if __name__ == '__main__':
    main()
