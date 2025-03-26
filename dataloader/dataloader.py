import os
import sys
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.functional import fftconvolve

import sofa
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.Generate_Cochleagram import generate_cochleagram


HOME_DIR = '/home/maxmay'
WORK_DIR = 'sound_ipcl'
TRAIN_DIR = 'Data/audio/bal_train'
# Data storage Snellius: 
# '/projects/2/managed_datasets/hf_cache_dir/datasets--agkphysics--AudioSet/snapshots/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/'

def read_txt(fn: str) -> [str]:
    train_loc = os.path.join(HOME_DIR, TRAIN_DIR)
    with open(fn, 'r') as f:
        files = [os.path.join(train_loc, line.strip('\n')) for line in f.readlines()]
    return files


class AudioDataset(Dataset):
    # 1. load audio
    # 1.1. Check if Sampling_rate is 48 kHz, esle resample
    # 2. Create audio slice of 3 seconds
    # 3. Pick random location from .sofa
    # 3.1 Convolve audio -- fftconvolve
    # 4. Generate cochleagram
    # 4.1 remove first and last second due to noise

    def __init__(self, dir, target_samplerate: int = 48000):
        self.target_samplerate = target_samplerate
        self.audio_files = read_txt(dir)

        self.hrtf = self.open_sofa()
        self.locations = self.locations()
        self.labels = torch.randint(0, self.locations, (len(self.audio_files),))

        # currently store everything in memory
        # Future work could be to create tensor, only if data is called
        self.audio = self.create_tensor()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio[idx], self.labels[idx]

    def create_tensor(self):
        tensor = []
        for idx, fn in enumerate(tqdm(self.audio_files, desc='Generating Audio Dataset')):
            flac = self.load_flac(fn)
            sliced = self.slice_audio(flac, slice_seconds=3)
            localization = self.get_localization(self.labels[idx])
            # print("Sliced: ", sliced.shape)

            convolved = self.transform(sliced, localization, n=1)
            # print(f'Transformed: {convolved.shape}')

            cochleagram = generate_cochleagram(convolved, self.target_samplerate)
            # print(f'Coch shape: {cochleagram.shape}')
            tensor.append(torch.from_numpy(cochleagram))
        return torch.stack(tensor, dim=0)

    def load_flac(self, fn: str) -> (torch.Tensor, int):
        waveform, sample_rate = torchaudio.load(fn)
        # Due to Youtube's nature, audio files are irregular and can sometimes have multiple channels
        # To standardize the format, for every file the mean of the channels are taken if there are multiple.
        if waveform.shape[0] > 1:
            waveform  = torch.mean(waveform, dim=0, keepdim=True)
        # Youtube's sample rate can also differ between 44.1 kHz and 48 kHz, so upsample if need be.
        if sample_rate < self.target_samplerate:
            waveform = torchaudio.functional.resample(waveform, orig_freq = sample_rate, new_freq = self.target_samplerate)
        return waveform

    def slice_audio(self, audio: torch.Tensor, slice_seconds: int = 3) -> torch.Tensor:
        # Calculate the number of frames necessary based on the sample rate
        num_frames = slice_seconds * self.target_samplerate

        # Pick a random off set, but make sure this won't exceed the total number of frames
        total = audio.shape[1]
        off_set = np.random.randint(0, (total - num_frames))

        # Slice the audio, accounting for multiple channels
        audio_slice = audio[:, off_set:off_set+num_frames]
        return audio_slice


    def open_sofa(self, fn: str = 'KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa'):
        # Open .sofa file
        file_path = os.path.join(HOME_DIR, WORK_DIR, 'utils', fn)
        hrtf = sofa.Database.open(file_path)
        return hrtf

    # Dimensions:
    # I: 1
    # C: 3
    # R: 2 -- Receivers (i.e. Left and Right)
    # E: 1
    # N: 256 -- Total time 
    # M: 828 -- Measurement Locations
    # S: 0
    # Sampling rate: 48 kHz
    def locations(self):
        # Get the amount measurement locations
        return self.hrtf.Dimensions.M

    def get_localization(self, measurement: int):
         # Get the sampling rate for specific measurement
        sampling_rate = self.hrtf.Data.SamplingRate.get_values(indices={"M":measurement})
        # t = np.arange(0,hrtf.Dimensions.N)*sampling_rate

        # Get the amount of receivers
        r = self.hrtf.Dimensions.R

        location_dist = np.zeros((r, self.hrtf.Dimensions.N))
        # location_dist = torch.zeros(r, hrtf.Dimensions.N)

        for receiver in range(r):
            location_dist[receiver] = self.hrtf.Data.IR.get_values(indices={'M': measurement, 'R': receiver, 'E': 0})

        location_dist = torch.from_numpy(location_dist)
        return location_dist


    def transform(self, audio: torch.Tensor, location_dist: torch.Tensor, n: int = 1) -> torch.Tensor:
        # Convolve sliced audio with the location cues
        convolved = fftconvolve(audio, location_dist)
        # convolved = audio
        # print(convolved.shape)

        # Remove the first and last second (n=1), due to noise
        second = n * self.target_samplerate # Sampling rate
        total = convolved.shape[1]

        convolved = convolved[:, second:total-second-255] # 255 is hardcoded, is due to hrtf.Dimensions.N = 256; No solution yet
        # print(f'Final shape: {convolved.shape}')
        return convolved


def main():
    target_samplerate = 48000

    # hrtf = sofa.Database.open('/home/maxmay/sound_ipcl/utils/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa')
    # print(hrtf.Data.IR.get_values(indices={'M': 827, 'R': 0, 'E': 0}))
    dataset = AudioDataset(
                    dir='/home/maxmay/files_to_copy.txt', 
                    target_samplerate = target_samplerate
                    )

    # print(dataset.hrtf.Dimensions.N)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)


    for idx, (audio, label) in enumerate(dataloader):
        print(f'Audio: {audio.shape}\n{audio[0]}\nLabel(s): {label}')
        break

if __name__ == '__main__':
    main()
    # fetch_location()
    # wf = generate_cochleagram(torch.Tensor([0.]))
    # print(wf)
