import os
import sofa

import numpy as np

import torch
from torch.nn import RMSNorm
import torchaudio
from torchaudio.functional import fftconvolve

from dataloader.Generate_Cochleagram import generate_cochleagram

# LEFT = [45., 0., 1.5]
# RIGHT = [360-45., 0., 1.5]
# FRONT = [0., 0., 1.5]
locations = {'front': 4, 'right': 211, 'left': 639}


def open_sofa(fn: str = None):
    # Open .sofa file
    if os.path.isfile(fn):
        file_path = fn
    else:
        file_path = os.path.join(HOME_DIR, WORK_DIR, 'utils', 'KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa')
    
    hrtf = sofa.Database.open(file_path)
    return hrtf


def preprocess(fn, norm, cutoff=1.7, target_samplerate=48000):
    waveform, sr = torchaudio.load(fn)
    if waveform.shape[0] > 1:
        waveform = waveform[0]
    if sr < target_samplerate:
        waveform = torchaudio.functional.resample(waveform, orig_freq = sr, new_freq = target_samplerate)

    num_frames = int(cutoff * target_samplerate)

    waveform = waveform[0:num_frames]
    normalized = norm(waveform)
    normalized = torch.unsqueeze(normalized, 0)
    return normalized


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


def transform(audio: torch.Tensor, location_dist: torch.Tensor, n: float = 1., target_samplerate: int = 48000) -> torch.Tensor:
    # Convolve sliced audio with the location cues
    convolved = fftconvolve(audio, location_dist)
    # convolved = audio
    # print(convolved.shape)
    ramped = rampsound(convolved, rampdur=0.01)

    # Remove the first and last second (n=1), due to noise
    second = int(n * target_samplerate) # Sampling rate
    total = convolved.shape[1]

    spatialized = ramped[:, second:total-second-255] # 255 is hardcoded, is due to hrtf.Dimensions.N = 256
    # print(f'Final shape: {convolved.shape}')
    return spatialized


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
    # Currently testing only with ramp off
    # sndtemp[0,0:rmpwin] = rampON*sndtemp[0,0:rmpwin] # ON, left channel
    # sndtemp[1,0:rmpwin] = rampON*sndtemp[1,0:rmpwin] # ON, right channel
    sndtemp[0,np.shape(sndtemp)[1]-rmpwin:] = rampOFF*sndtemp[0,np.shape(sndtemp)[1]-rmpwin:np.shape(sndtemp)[1]] # OFF, left channel
    sndtemp[1,np.shape(sndtemp)[1]-rmpwin:] = rampOFF*sndtemp[1,np.shape(sndtemp)[1]-rmpwin:np.shape(sndtemp)[1]] # OFF, right channel  

    # cast sndtemp to tensor
    if type(sndtemp) == np.ndarray:
        sndtemp = torch.from_numpy(sndtemp)

    return sndtemp


def main():
    data_dir = '../Data/audio/bal_train'
    files = ['00mE-lhe_R8.flac', '00W1lcxW-WU.flac', '0150dZu3Na8.flac', '01B907_Gyys.flac']
    # files = ['00M9FhCet6s.flac']
    target_samplerate = 48000
    nr_frames = int(target_samplerate*1.7)
    print(f'Nr of frames: {nr_frames}')
    norm = RMSNorm([nr_frames])

    for file in files:
        print(file)
        filename = os.path.join(data_dir, file)
        fn = file.strip('.flac')
        fn += '_'
        save_path = os.path.join('./results/cochleagrams', fn)

        hrtf = open_sofa('./utils/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa')
        normalized = preprocess(filename, norm, cutoff=1.7)
        # print(normalized.shape)

        for direction in locations:
            m = locations[direction]
            print(direction, ':', m)
            dist = get_localization(hrtf, m)
            localized = transform(normalized, dist, n=0.35, target_samplerate=target_samplerate)
            cochleagram = generate_cochleagram(localized.detach().numpy(), target_samplerate)

            save_fn = save_path + direction
            print(save_fn)

            print(cochleagram.shape)
            print(cochleagram)
            print()

            np.save(save_fn + '.npy',
                    cochleagram)
            # torchaudio.save(save_fn+'.flac', 
            #                 localized, 
            #                 target_samplerate)

if __name__ == '__main__':
    main()