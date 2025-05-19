import os
import time

import numpy as np

from dataloader.dataloader import AudioDataset
from dataloader.WebAudioSet import WebAudioSet

from torch.utils.data import DataLoader

import sofa
from utils.util import seed_all, count_pattern_files



def main(setup='earth'):
    seed_all(42)

    if setup == 'snellius':
        train_data_path = '/gpfs/work3/2/managed_datasets/hf_cache_dir/datasets--agkphysics--AudioSet/snapshots/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/data/bal_train00.tar'
        sofa_path = '/gpfs/home4/mmay/sound_ipcl/utils/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa'

        dataset = WebAudioSet(
            base_data_dir=train_data_path,
            val_data_dir=train_data_path,
            hrtf_dir=sofa_path,
            target_samplerate=48000,
            batch_size=64,
            resample=True)
        dataset.setup('fit')

        train_epoch_size = count_pattern_files('00')
        
        loader = dataset.train_wds_loader(epoch_size=train_epoch_size)
        loader_length = loader.nsamples

    elif (setup == 'earth') or (setup == 'mars'):
        train_data_path = '/home/maxmay/Data/audio/bal_train'
        sofa_path = '/home/maxmay/sound_ipcl/utils/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa' 

        dataset = AudioDataset(
            mode='train',
            dir='/home/maxmay/files_to_copy.txt', 
            target_samplerate = 48000
        )

        loader = DataLoader(dataset)
        loader_length = len(loader)


    avg_time = []
    time0 = time.time()
    last_time = time0
    print(f'Starting at: {time0:.2f}s')
    for idx, (data,_) in enumerate(loader):
        print(f'[Batch: {idx+1}/{loader_length}]')
        time_now = time.time()
        time_diff = time_now - last_time
        last_time = time_now
        avg_time.append(time_diff)
        print(f'Took {time_diff:.2f}s [avg time: {np.mean(avg_time):.2f}s]')


if __name__ == '__main__':
    setup = 'earth'
    main(setup)
