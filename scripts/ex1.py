from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.decomposition import FastICA, PCA

import fast_ica

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def load_data():
    N = 2000
    time = np.linspace(0, 8, N)
    s1 = np.sin(2*time) # sinusoidal
    s2 = np.sign(np.sin(3*time)) # square signal
    s3 = signal.sawtooth(2*np.pi*time) # saw tooth signal
    s4 = s1 + s2
    s5 = s1 - s2

    # S = np.array([s1, s2, s3])
    S = np.array([s3, s4, s5])
    A = np.array([
        [1, 1, 1],
        [0.5, 2, 1],
        [1.5, 1, 2]
        ])
    X = A @ S
    return X, S

def plot_sources(X, S, S_predicted, S_sklearn, S_pca):
    fig, ax = plt.subplots(nrows=2, ncols=3)

    for x in X:
        ax[0, 0].plot(x)
        ax[0, 0].set_title('Mixture signals ($X$)')

    for s in S:
        ax[0, 1].plot(s)
        ax[0, 1].set_title("Original sources ($S$)")

    for S_predicted in S_predicted:
        ax[1, 0].plot(-S_predicted)
        ax[1, 0].set_title("Predicted sources ($W X$) [Our FastICA]")

    for S_sklearn in S_sklearn:
        ax[1, 1].plot(S_sklearn)
        ax[1, 1].set_title("Predicted sources ($W X$) [Sklearn FastICA]")

    for S_pca in S_pca:
        ax[1, 2].plot(S_pca)
        ax[1, 2].set_title("Predicted sources ($W X$) [Sklearn PCA]")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    X, S = load_data()
    S_predicted, distances = fast_ica.ica(X, cycles=100)
    # print(distances)
    # plt.plot(distances)
    # plt.ylim((0, 0.1))
    # plt.show()
    # exit()
    S_sklearn = FastICA(n_components=3).fit_transform(X.T).T
    S_pca = PCA(n_components=3).fit_transform(X.T).T
    plot_sources(X, S, S_predicted, S_sklearn, S_pca)
