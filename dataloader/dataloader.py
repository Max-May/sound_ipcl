import os
import torchaudio

DIR = '../Data/audio/bal_train'


def main():
    sound_names = os.listdir(DIR)[:10]

    for sound_name in sound_names:
        print(f'File: {sound_name}')
        waveform, sample_rate = torchaudio.load(os.path.join(DIR,sound_name)) 
        print(f'SR: {sample_rate}')
        print(f'Waveform: {waveform.shape}\n{waveform}')
        print()



if __name__ == '__main__':
    main()
