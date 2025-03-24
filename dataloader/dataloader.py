import os
import sys
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.functional import fftconvolve
import sofa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.Generate_Cochleagram import generate_cochleagram

# hrtf = sofa.Database.open("./utils/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa")
# hrtf.Metadata.dump()
# source_positions = hrtf.Source.Position.get_values(system="cartesian")
# print(np.asarray(source_positions).shape)


HOME_DIR = '/home/maxmay'
WORK_DIR = 'sound_ipcl'
TRAIN_DIR = 'Data/audio/bal_train'


def read_txt(fn: str) -> [str]:
    train_loc = os.path.join(HOME_DIR, TRAIN_DIR)
    with open(fn, 'r') as f:
        files = [os.path.join(train_loc, line.strip('\n')) for line in f.readlines()]
    return files


def load_flac(fn: str, target_samplerate: int = 48000) -> (torch.Tensor, int):
    waveform, sample_rate = torchaudio.load(fn)
    if sample_rate < target_samplerate:
        waveform = torchaudio.functional.resample(waveform, orig_freq = sample_rate, new_freq = target_samplerate)
    return waveform


def slice_audio(audio: torch.Tensor, slice_seconds: int = 3, sample_rate: int = 48000) -> torch.Tensor:
    # Calculate the number of frames necessary based on the sample rate
    num_frames = slice_seconds * sample_rate

    # Pick a random off set, but make sure this won't exceed the total number of frames
    total = audio.shape[1]
    off_set = np.random.randint(0, (total - num_frames))

    # Slice the audio, accounting for multiple channels
    audio_slice = audio[:, off_set:off_set+num_frames]
    return audio_slice


# Dimensions:
# I: 1
# C: 3
# R: 2 -- Receivers (i.e. Left and Right)
# E: 1
# N: 256 -- Total time 
# M: 828 -- Measurement Locations
# S: 0
# Sampling rate: 48 kHz
def fetch_location(fn: str = 'KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa'):
    # Open .sofa file
    file_path = os.path.join(HOME_DIR, WORK_DIR, 'utils', fn)
    hrtf = sofa.Database.open(file_path)

    # Get the amount measurement locations
    locations = hrtf.Dimensions.M

    # Pick a random location
    measurement = np.random.randint(0, locations)

    # Get the sampling rate for that measurement
    sampling_rate = hrtf.Data.SamplingRate.get_values(indices={"M":measurement})
    t = np.arange(0,hrtf.Dimensions.N)*sampling_rate

    # Get the amount of receivers
    r = hrtf.Dimensions.R

    location_dist = np.zeros((r, hrtf.Dimensions.N))
    # location_dist = torch.zeros(r, hrtf.Dimensions.N)

    for receiver in range(r):
        location_dist[receiver] = hrtf.Data.IR.get_values(indices={'M': measurement, 'R': receiver, 'E': 0})

    location_dist = torch.from_numpy(location_dist)

    return location_dist, measurement


def transform(audio: torch.Tensor, location_dist: torch.Tensor) -> torch.Tensor:
    # Convolve sliced audio with the location cues
    convolved = fftconvolve(audio, location_dist)
    print(convolved.shape)

    # Remove the first and last second, due to noise
    second = 48000 # Sampling rate
    total = convolved.shape[1]

    convolved = convolved[:, second:total-second]
    print(f'Final shape: {convolved.shape}')
    return convolved


class AudioDataset(Dataset):
    # 1. load audio
    # 1.1. Check if Sampling_rate is 48 kHz, esle resample
    # 2. Create audio slice of 3 seconds
    # 3. Pick random location from .sofa
    # 3.1 Convolve audio -- fftconvolve
    # 4. Generate cochleagram
    # 4.1 remove first and last second due to noise

    def __init__(self, dir, target_samplerate: int = 48000):
        self.sampling_rate = target_samplerate
        self.audio = read_txt(dir)
        self.label = torch.rand(len(self.audio))
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.audio[idx], self.label[idx]


def main():
    target_samplerate = 48000
    dataset = AudioDataset(
                    dir='/home/maxmay/files_to_copy.txt', 
                    target_samplerate = target_samplerate
                    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    for idx, (audio, _) in enumerate(dataloader):
        wf = load_flac(audio[0], target_samplerate)
        print(f'Original shape: {wf.shape}')
        sliced = slice_audio(wf, 3)
        print(f'New shape: {sliced.shape}\n{sliced}')
        location_dist, label = fetch_location()
        print(f'Label: {label}')
        convolved = transform(sliced, location_dist)
        print(convolved)
        break

if __name__ == '__main__':
    main()
    # fetch_location()
    # wf = generate_cochleagram(torch.Tensor([0.]))
    # print(wf)
