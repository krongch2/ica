from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import signal
from sklearn.decomposition import FastICA, PCA

import fast_ica

# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

def load_data():
    N = 2000
    time = np.linspace(0, 8, N)
    s1 = 5*np.sin(2*time) # sinusoidal
    s2 = 20*np.sign(np.sin(3*time)) # square signal
    s3 = 100*signal.sawtooth(2*np.pi*time) + 50# saw tooth signal
    s4 = s1 + s2
    s5 = s1 - s2
    s6 = np.cos(2*time)

    S = np.array([s1, s2, s3])
    A = np.array([
        [1, 1, 1],
        [0.5, 2, 1],
        [1.5, 1, 2]
        ])
    X = A @ S
    return X, S

def plot_sources(sources, titles, output=None):

    n = len(sources)
    fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(9, 7), sharex=True, sharey=False)
    for i in range(n):
        for source in sources[i]:
            ax[i].plot(source)
        ax[i].set_title(titles[i], fontsize=8)

    fig.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight')

def plot_distances(distances, output=None):
    n = len(distances)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(n):
        ax.plot(distances[i], 'o-', mec='white', color=colors[i], label=f'Component {i}')
    ax.set(xlabel='Cycle', ylabel='$||w_{n}^T w_{n-1}| - 1|$')
    ax.legend(fancybox=False, edgecolor='k')

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':
    np.random.seed(0)
    X, S = load_data()
    S_out, W, K, X_out, distances = fast_ica.ica(X)
    print(W)
    plot_distances(distances, output='ex1_dist.pdf')

    S_out_sk, W, K, X_out_sk, distances = fast_ica.ica_sk(X)
    S_pca = PCA().fit_transform(X.T).T

    source_title = [
        (S, 'Original sources ($S$)'),
        (X, 'Mixture signals ($X$)'),
        (fast_ica.whiten(X)[0], 'Whiten $X$'),
        (S_out, '$S_{\\mathrm{predicted}}$ [Our FastICA]'),
        (X_out, 'Retrieved $X = (W K)^{-1} S_{\\mathrm{predicted}}$ [Our FastICA]'),
        (S_out_sk, 'Predicted $S$ [Sklearn FastICA]'),
        (X_out_sk, 'Retrieved $X = (W K)^{-1} S_{\\mathrm{predicted}}$ [Sklearn FastICA]'),
        (S_pca, '$S_{\\mathrm{predicted}}$ [PCA]')
        ]
    sources, titles = zip(*source_title)
    plot_sources(sources, titles, output='ex1_sources.pdf')
    exit()


