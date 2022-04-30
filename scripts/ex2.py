from matplotlib import pyplot as plt
import numpy as np
import sounddevice
from scipy.io import wavfile

import fast_ica

# import youtube_dl

# def download_sources():

#     ydl_opts = {}
#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         ydl.download(['https://www.youtube.com/watch?v=wuvXCC9xy48'])
#         ydl.download(['https://www.youtube.com/watch?v=ea8rkQFXEr4'])

# def mix_sources(mixtures, apply_noise=False):
#     for i in range(len(mixtures)):
#         max_val = np.max(mixtures[i])
#         if max_val > 1 or np.min(mixtures[i]) < 1:
#             mixtures[i] = mixtures[i] / (max_val / 2) - 0.5
#     X = np.c_[[mix for mix in mixtures]]
#     if apply_noise:
#         X += 0.02 * np.random.normal(size=X.shape)
#     return X

# def example():
#     sampling_rate, mix1 = wavfile.read('mix1.wav')
#     sampling_rate, mix2 = wavfile.read('mix2.wav')
#     sampling_rate, source1 = wavfile.read('source1.wav')
#     sampling_rate, source2 = wavfile.read('source2.wav')
#     X = mix_sources([mix1, mix2])
#     S = ica(X)
#     plot_mixture_sources_predictions(X, [source1, source2], S)
#     wavfile.write('out1.wav', sampling_rate, S[0])
#     wavfile.write('out2.wav', sampling_rate, S[1])

def play(sound, fs=11025):
    for i in range(sound.shape[0]):
        print(f'playing mixed track {i}')
        sounddevice.play(sound[i, :], fs, blocking=True)

def load():
    X = np.loadtxt('mix.dat').T
    play(X)
    S_predicted, distances = fast_ica.ica(X, cycles=1000)
    play(S_predicted)

if __name__ == '__main__':
    load()
