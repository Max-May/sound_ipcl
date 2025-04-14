import os

import torch
import torchaudio
from torchaudio.functional import fftconvolve

import numpy as np
import sofa
import tarfile as tf

from .Generate_Cochleagram import generate_cochleagram


def change_name(fn: str) -> str:
    return fn.replace('{', '[').replace('..', '-').replace('}',']')


# Return number of files in .tarfile
def __len__(fn: str) -> int:
    fn = change_name(fn)
    # fn = fn.replace()
    length = 0
    with tf.open(fn) as archive:
        length = sum(1 for member in archive if member.isreg())
    return length


def slice_audio(audio: torch.Tensor, slice_seconds: float = 3., samplerate: int = 48000) -> torch.Tensor:
        # Calculate the number of frames necessary based on the sample rate
        num_frames = int(slice_seconds * samplerate)

        # Pick a random off set, but make sure this won't exceed the total number of frames
        total = audio.shape[1]
        # print(f'Audio: {audio.shape} | Total: {total} | Nr of frames: {num_frames}')
        assert num_frames <= total, f"Number of frames to pick cannot exceed the total number of frames ({num_frames} > {total})"
        off_set = np.random.randint(0, (total - num_frames))

        # Slice the audio, accounting for multiple channels
        audio_slice = audio[:, off_set:off_set+num_frames]
        return audio_slice


def locations(hrtf):
    # Get the amount measurement locations
    return hrtf.Dimensions.M


def get_localization(hrtf, measurement: int):
        # Get the sampling rate for specific measurement
    sampling_rate = hrtf.Data.SamplingRate.get_values(indices={"M":measurement})
    # t = np.arange(0,hrtf.Dimensions.N)*sampling_rate

    # Get the amount of receivers
    r = hrtf.Dimensions.R

    location_dist = np.zeros((r, hrtf.Dimensions.N))
    # location_dist = torch.zeros(r, hrtf.Dimensions.N)

    for receiver in range(r):
        location_dist[receiver] = hrtf.Data.IR.get_values(indices={'M': measurement, 'R': receiver, 'E': 0})

    location_dist = torch.from_numpy(location_dist)
    return location_dist


# --Function definitions
def rampsound(sndtemp, rampdur:float, target_samplerate: int = 48000):
    '''function to add an on and off ramp
    sndtemp = 2 channel waveform
    rampdur = ramp duration in seconds
    fs_snd = sampling rate'''
    fs_snd = target_samplerate

    rmpwin = int(np.floor(rampdur*fs_snd)) # define number of samples
    
    # define ramp
    if type(sndtemp) == np.ndarray:
        rampON = np.linspace(0,1,rmpwin)
        rampOFF = np.linspace(1,0,rmpwin)
    elif type(sndtemp) == torch.Tensor:
        rampON = torch.linspace(0,1,rmpwin)
        rampOFF = torch.linspace(1,0,rmpwin)
    else:
        raise TypeError(f'"{type(sndtemp)}" not recognized, expected "torch.Tenor" or "np.ndarray"')

    # ramp sound
    sndtemp[0,0:rmpwin] = rampON*sndtemp[0,0:rmpwin] # ON, left channel
    sndtemp[1,0:rmpwin] = rampON*sndtemp[1,0:rmpwin] # ON, right channel
    sndtemp[0,np.shape(sndtemp)[1]-rmpwin:] = rampOFF*sndtemp[0,np.shape(sndtemp)[1]-rmpwin:np.shape(sndtemp)[1]] # OFF, left channel
    sndtemp[1,np.shape(sndtemp)[1]-rmpwin:] = rampOFF*sndtemp[1,np.shape(sndtemp)[1]-rmpwin:np.shape(sndtemp)[1]] # OFF, right channel  

    # cast sndtemp to tensor
    if type(sndtemp) == np.ndarray:
        sndtemp = torch.from_numpy(sndtemp)

    return sndtemp


def transform(audio: torch.Tensor, location_dist: torch.Tensor, n: float = 1., target_samplerate: int = 48000) -> torch.Tensor:
    # Convolve sliced audio with the location cues
    convolved = fftconvolve(audio, location_dist)
    # convolved = audio
    # print(convolved.shape)
    ramped = rampsound(convolved, rampdur=0.01)

    # Remove the first and last second (n=1), due to noise
    second = int(n * target_samplerate) # Sampling rate
    total = convolved.shape[1]

    spatialized = ramped[:, second:total-second-255] # 255 is hardcoded, is due to hrtf.Dimensions.N = 256; No solution yet
    # print(f'Final shape: {convolved.shape}')
    return spatialized


def resample(waveform: torch.Tensor, sr: int, target_samplerate: int):
    # Youtube's sample rate can also differ between 44.1 kHz and 48 kHz, so upsample if need be.
    if sr < target_samplerate:
        waveform = torchaudio.functional.resample(waveform, orig_freq = sr, new_freq = target_samplerate)
    if waveform.shape[0] > 1:
            waveform  = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def __getitem__(sample,
                hrtf,
                target_samplerate:int = 48000):
    if isinstance(sample, dict):
        audio,sr = sample[".flac"]
    else:
        audio,sr = sample[0]
    audio = resample(audio, sr, target_samplerate)
    nr_locations = locations(hrtf)
    label = torch.randint(0, nr_locations, (1,))
    sliced = slice_audio(audio, slice_seconds=1.07) # This way you keep 1 second after last cut
    localization = get_localization(hrtf, label)
    # Add ramping -- currently inside transform
    transformed = transform(sliced, localization, n=0.035, target_samplerate=target_samplerate) # cut first and last 35ms from audio
    cochleagram = torch.from_numpy(generate_cochleagram(transformed, target_samplerate))
    return cochleagram, label
