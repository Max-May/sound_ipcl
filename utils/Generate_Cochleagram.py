#Script to generate cochleagrams using the parameters described in Francl & McDermott 2022

import os
import numpy as np
from typing import Union

import torch
import torchaudio
from scipy.signal import welch, decimate, filtfilt, firwin, windows
import os 
from functools import partial
from scipy.fft import fft, ifft
from scipy.signal import firwin

# from pycochleagram
from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb
from pycochleagram import subband as sb

from tqdm import tqdm


def ownpow(a, b):
    '''
    Define power function because np.power in loop does not take power of negative (instead returns nan)
    This is the old code, now replaced by more efficient implementation
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            if a[i,j] > 0:
                a[i,j] = a[i,j]**b
            if a[i,j] < 0:
                temp = abs(a[i,j])**b
                a[i,j]= -1*temp
    return a'''
    return np.sign(a) * np.abs(a) ** b


# Function to perform FFT-based filtering for a 1D signal
def fft_filter(signal, filter_coefficients):
    """
    Apply FFT-based filtering to the input signal using the provided filter coefficients.
    
    Parameters:
        signal (numpy array): The input signal to filter.
        filter_coefficients (numpy array): The filter coefficients.
    
    Returns:
        numpy array: The filtered signal.
    """
    # Lengths of the signal and filter
    signal_len = len(signal)
    filter_len = len(filter_coefficients)
    
    # Pad the filter coefficients to match the signal length
    filter_padded = np.zeros(signal_len)
    filter_padded[:filter_len] = filter_coefficients
    
    # Compute the FFT of the signal and the filter
    signal_fft = fft(signal)
    filter_fft = fft(filter_padded)
    
    # Multiply in the frequency domain
    filtered_fft = signal_fft * filter_fft
    
    # Compute the inverse FFT to get the filtered signal
    filtered_signal = np.real(ifft(filtered_fft))
    
    return filtered_signal


# Function to perform zero-phase filtering (like filtfilt) using FFT for a 1D signal
def fft_filtfilt(signal, filter_coefficients):
    """
    Perform zero-phase filtering using FFT-based filtering (equivalent to filtfilt).
    
    Parameters:
        signal (numpy array): The input signal to filter.
        filter_coefficients (numpy array): The filter coefficients.
    
    Returns:
        numpy array: The zero-phase filtered signal.
    """
    # Forward filtering
    forward_filtered = fft_filter(signal, filter_coefficients)
    
    # Reverse the signal and filter again
    reversed_filtered = fft_filter(forward_filtered[::-1], filter_coefficients)
    
    # Reverse the result to restore the original order
    zero_phase_filtered = reversed_filtered[::-1]
    
    return zero_phase_filtered


# Function to apply FFT-based filtering to each row of a 2D array
def fft_filtfilt_2d(signal_2d, filter_coefficients):
    """
    Perform zero-phase filtering on each row of a 2D signal using FFT-based filtering.
    
    Parameters:
        signal_2d (numpy array): The 2D input signal to filter (shape: num_channels x num_samples).
        filter_coefficients (numpy array): The filter coefficients.
    
    Returns:
        numpy array: The zero-phase filtered 2D signal (shape: num_channels x num_samples).
    """
    num_channels, num_samples = signal_2d.shape
    filtered_signal_2d = np.zeros_like(signal_2d)
    
    # Loop over each channel (row) and apply zero-phase filtering
    for i in range(num_channels):
        filtered_signal_2d[i] = fft_filtfilt(signal_2d[i], filter_coefficients)
    
    return filtered_signal_2d


def generate_cochleagram(waveform: torch.Tensor, 
                        sample_rate: int, 
                        dtype: Union[torch.Tensor, np.ndarray] = np.ndarray,
                        debug: bool = False,
                        ) -> Union[torch.Tensor, np.ndarray]:
    assert dtype == np.ndarray, f"Function does not support {dtype} yet!"

    # Specifications
    target_samplerate = 48000   # sample rate of 48 kHz for this application
    n = 17 # the nr of filters and sampling factor interact,  it's n*2 + 2*sample_factor + 1 filters if sample factor = 1 and full_filter = True
            # here, input = 17 such that output = 39 filters, if then remove first and last filter, 39 filters are left
    sample_factor = 2 # reflects human hearing according to McDermott function 'human_cochleagram'
    numfilt = n*2 + 2*sample_factor + 1 # this results in 39 filters
    retmode = 'subband' #subband, envs
    low_lim = 45 # reflecting lower limit of human hearing
    hi_lim = 16975 # reflecting upper limit of human hearing 
        
    # upsample to 48 kHz if needed
    if sample_rate < target_samplerate:
        waveform = torchaudio.functional.resample(waveform, orig_freq = sample_rate, new_freq = target_samplerate)

    # generate filters
    erb_kwargs = {}
    filts, _, _ = erb.make_erb_cos_filters_nx(waveform.shape[1],
        target_samplerate, n, low_lim, hi_lim, sample_factor, padding_size=None,
        full_filter=True, strict=False, **erb_kwargs)

    # Deinfe low-pass filtering, used after half-wave rectification to simulate upper limit of phase locking in auditory nerve
    cutoff_freq = 4000   # Cut-off frequency
    num_taps = 4097      # Number of filter taps
    transition_width = 100  # Transition width in Hz
    filter_coefficients = firwin(num_taps, cutoff_freq, window=('kaiser', 14), fs=target_samplerate) # calculate filter coefficients using a Kaiser-windowed sinc function, beta = 14
    # Define downsampling
    ds_factor = int(np.floor(target_samplerate/8000)) # Downsample to 8 kHz
    custom_downsample_fx = partial(decimate, q = ds_factor, axis=1, ftype='fir', zero_phase=True)

    # Generate an empty array for the cochleagram
    coch_all = np.empty([np.shape(waveform)[0],numfilt,int(np.shape(waveform)[1]/ds_factor)]) # to accomodate multi-channel data, dimensions = [channels, filters, time]

    # for j in range(np.shape(waveform)[0]): # to accomodate multi-channel data
    for j in tqdm(range(np.shape(waveform)[0]), desc='Generating cochleagrams', disable=not debug):
        # generate subbands
        coch_temp = sb.generate_subbands(waveform[j], filts, padding_size=None, fft_mode='auto', debug_ret_all=False)

        # apply non-linearity (this has to be done here because the cochleagram function only does it for envelopes
        coch_temp = ownpow(coch_temp, 3.0 / 10.0)  # add epsilon to not take power of 0

        # half_wave rectification
        coch_temp = np.maximum(coch_temp,0) 

        # low-pass filtering
        for i in range(np.shape(coch_temp)[0]):
            coch_temp[i,:] = fft_filtfilt(coch_temp[i,:], filter_coefficients)

        # downsampling
        coch_temp = custom_downsample_fx(coch_temp)
        
        # add to array
        coch_all[j,:,:] = coch_temp
    return coch_all


def main():
    # Directories
    # dir_sounds = '/Users/kiki.vanderheijden/Documents/PythonScripts_Local/ESC-50-master/audio'
    # sound_name = '1-137-A-32.wav'
    dir_sounds = '../Data/audio/bal_train'
    sound_name = '01q8wKX4XEQ.flac'

    # Temporary example sound files of the bal_train audio_set:
    # sound_names = [
    # '00M9FhCet6s.flac',
    # '00mE-lhe_R8.flac',
    # '00W1lcxW-WU.flac',
    # '0150dZu3Na8.flac',
    # '01B907_Gyys.flac',
    # '01bTS8O2Xnc.flac',
    # '01hjVJN9xCg.flac',
    # '01k1v-NgjWs.flac',
    # '01PzcPKT3_E.flac',
    # '01q8wKX4XEQ.flac']


if __name__ == '__main__':
    main()
    # wf, sr = torchaudio.load('/home/maxmay/Data/audio/bal_train/00M9FhCet6s.flac')
    # coch = generate_cochleagram(wf, sr, dtype=np.ndarray, debug=True)
    # print(coch)

