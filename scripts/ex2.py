from matplotlib import pyplot as plt
import numpy as np
import sounddevice

import fast_ica

def play(sound, fs=11025):
    for i in range(sound.shape[0]):
        print(f'playing mixed track {i}')
        sounddevice.play(sound[i, :], fs, blocking=True)

def load():
    X = np.loadtxt('mix.dat').T
    # play(X)
    S_predicted, W, distances = fast_ica.ica(X, standardize=False)
    play(S_predicted)

if __name__ == '__main__':
    load()
