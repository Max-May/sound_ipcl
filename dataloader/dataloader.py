import os
import torchaudio

DIR = '../Data/audio/bal_train'


def main():
    # Specifications
    target_samplerate = 480000   # sample rate  for this application
    n = 17 # the nr of filters and sampling factor interact,  it's n*2 + 2*sample_factor + 1 filters if sample factor = 1 and full_filter = True
            # here, input = 17 such that output = 39 filters, if then remove first and last filter, 39 filters are left
    sample_factor = 2 # reflects human hearing according to McDermott function 'human_cochleagram'
    numfilt = n*2 + 2*sample_factor + 1 # this results in 39 filters
    retmode = 'subband' #subband, envs
    low_lim = 45 # reflecting lower limit of human hearing
    hi_lim = 16975 # reflecting upper limit of human hearing     

    sound_names = os.listdir(DIR)[:10]

    for sound_name in sound_names:
        print(f'File: {sound_name}')
        waveform, sample_rate = torchaudio.load(os.path.join(DIR,sound_name)) 
        print(f'SR: {sample_rate}')
        print(f'Waveform: {waveform.shape}\n{waveform}')

        if sample_rate < target_samplerate:
            waveform = torchaudio.functional.resample(waveform, orig_freq = sample_rate, new_freq = target_samplerate)
        print(f'New Waveform: {waveform.shape}\n{waveform}')
        print()


if __name__ == '__main__':
    main()
