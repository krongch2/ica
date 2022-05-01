from matplotlib import pyplot as plt
import numpy as np
import sounddevice

import fast_ica

def play(sound, fs=11025):
    for i in range(sound.shape[0]):
        print(f'playing mixed track {i}')
        sounddevice.play(sound[i, :], fs, blocking=True)

def extract():
    '''
    Loads mixing sound samples from
    - Godfather
    - Southpark
    - Beethoven 5th
    - Austin Powers
    - The Matrix
    '''
    X = np.loadtxt('mix.dat').T
    # play(X)

    S_out, W, K, X_out, distances = fast_ica.ica(X)
    # play(S_out)
    play(fast_ica.center(S_out, divide_sd=True))
    # S_out, W, K, X_out, distances = fast_ica.ica_sk(X)
    # play(fast_ica.center(S_out, divide_sd=True))

if __name__ == '__main__':
    np.random.seed(0)
    extract()
