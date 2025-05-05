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

from .dataset_functions import __getitem__ #, __len__, __size__
from functools import partial


HOME_DIR = '/home/maxmay'
WORK_DIR = 'sound_ipcl'
TRAIN_DIR = 'Data/audio/bal_train'


def collate_fn(batch):
    x, y = batch
    # print("Collate X: ", x.shape)
    # print("Collate Y: ", y.shape) 
    return x, y.flatten(start_dim=0, end_dim = 1)


class WebAudioSet(Dataset):
    def __init__(self, 
                base_data_dir: str,
                val_data_dir : str,
                hrtf_dir: str,
                target_samplerate: int = 48000,
                batch_size: int = 32,
                resample: bool = True,
                debug=False
                ):
        super().__init__()
        self.base_data_dir = base_data_dir
        self.val_data_dir = val_data_dir
        self.hrtf = open_sofa(hrtf_dir)
        self.target_samplerate = target_samplerate
        self.batch_size = batch_size
        self.resample = resample
        self.debug = debug
    
    
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = self.make_web_dataset(self.base_data_dir, shuffle=1000)
            self.val_dataset = self.make_web_dataset(self.val_data_dir, shuffle=0)
        elif stage == 'inf':
            self.val_dataset = self.make_web_dataset(self.val_data_dir, shuffle=0)
    
    def make_web_dataset(self, path, shuffle):
        warning = wds.reraise_exception if self.debug else wds.warn_and_continue
        # warning = wds.reraise_exception
        pre_process_function = partial(__getitem__,
                                        hrtf=self.hrtf,
                                        target_samplerate=self.target_samplerate)

        # For IterableDataset objects, the batching needs to happen in the dataset.
        dataset = (wds.WebDataset(path, resampled=self.resample, cache_dir='/tmp', shardshuffle=True)
                .shuffle(shuffle)
                .decode(wds.torch_audio, handler=warning)
                .to_tuple('flac')
                .map(pre_process_function, handler=warning)
                .batched(self.batch_size))
        return dataset
    
    def train_wds_loader(self, epoch_size: int, nr_workers: int = 16, batch_size: int = None):
        if batch_size:
            sub_batch_size = batch_size
        else:
            sub_batch_size = self.batch_size

        print('=> Setting up the train data loader')
        loader = wds.WebLoader(
            self.train_dataset, 
            batch_size=None, 
            num_workers=nr_workers, 
            collate_fn=collate_fn
            )
        # Unbatch, shuffle between workers, then rebatch.
        loader = loader.unbatched().shuffle(1000).batched(sub_batch_size)
        loader = loader.with_epoch(epoch_size * 2000 // sub_batch_size)
        return loader

    def val_wds_loader(self, epoch_size: int, nr_workers: int = 16, batch_size: int = None):
        if batch_size:
            sub_batch_size = batch_size
        else:
            sub_batch_size = self.batch_size
        print('=> Setting up the validation data loader')
        loader = wds.WebLoader(
            self.val_dataset, 
            batch_size=None,
            shuffle=False, 
            pin_memory=True,
            num_workers=nr_workers, 
            collate_fn=collate_fn
            )
        
        # Unbatch, shuffle between workers, then rebatch.
        loader = loader.unbatched().shuffle(1000).batched(sub_batch_size)
        loader = loader.with_epoch(epoch_size * 2000 // sub_batch_size)
        return loader


def open_sofa(fn: str = None):
    # Open .sofa file
    if os.path.isfile(fn):
        file_path = fn
    else:
        file_path = os.path.join(HOME_DIR, WORK_DIR, 'utils', 'KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa')
    
    hrtf = sofa.Database.open(file_path)
    return hrtf


def main_old():
    shared_url = '/home/maxmay/Data/bal_train{00..01}.tar'
    hrtf = open_sofa()

    print('Creating dataset...')
    pre_process_function = partial(__getitem__,
                                    hrtf=hrtf,
                                    target_samplerate=48000)

    # For IterableDataset objects, the batching needs to happen in the dataset.
    dataset = (wds.WebDataset(shared_url, resampled=True, cache_dir='/tmp', shardshuffle=True)
            .shuffle(1000)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple('flac')
            .map(pre_process_function, handler=wds.warn_and_continue)
            .batched(64))

    print('Creating loader...')
    # Loader is iterable and due to size and repeatability it doesn't make sense to track its length
    # See: "https://github.com/webdataset/webdataset/issues/75" for more info on the topic
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=8, collate_fn=collate_fn)
    loader = loader.unbatched().shuffle(1000).batched(8)

    print('Trying to batch...')
    for idx, data in enumerate(loader):
        print(f'[{idx}]: {data[0].shape} | {data[1].shape}')
        if idx == 64 // 8:
            break


def main():
    # For IterableDataset objects, the batching needs to happen in the dataset.
    # Loader is iterable and due to size and repeatability it doesn't make sense to track its length
    # For more info see: "https://github.com/webdataset/webdataset/issues/75" on the topic
    WAS = WebAudioSet(
        base_data_dir = '/home/maxmay/Data/bal_train00.tar',
        val_data_dir = '/home/maxmay/Data/bal_train09.tar',
        hrtf_dir = '/home/maxmay/sound_ipcl/utils/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa',
        target_samplerate = 48000,
        batch_size = 32,
        resample=False
        )
    WAS.setup('fit')
    # train_loader = WAS.train_wds_loader()
    val_loader = WAS.val_wds_loader()

    # for idx, (audio, label) in enumerate(train_loader):
    #     print(f'[{idx}]: {audio.shape} | {label.shape}')
    #     break
    
    total_guessed = 0
    for idx, (audio, label) in enumerate(val_loader):
        # print(f'[{idx}]: {audio.shape} | {label.shape}')
        total_guessed += audio.shape[0]
        print(f'[Batch: {idx+1}]: {total_guessed:04d}', end="\r", flush=True)
    print()

if __name__ == '__main__':
    main()
