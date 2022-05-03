from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import signal
from sklearn.decomposition import FastICA, PCA

import fast_ica

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def load_data():
    N = 2000
    time = np.linspace(0, 8, N)
    s1 = np.sin(2*time) # sinusoidal
    s2 = 2*np.sign(np.sin(3*time)) # square signal
    s3 = 4*signal.sawtooth(2*np.pi*time)  # saw tooth signal
    # s3 = 4*(1/(np.pi)*(2*np.pi*time%(2*np.pi)) - 1)
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

def plot_sources(sources, titles, output=None, colwrap=2, figsize=(7, 4)):

    n = len(sources)
    nrows = int(np.ceil(n/colwrap))
    ncols = colwrap
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey='row')
    for i in range(nrows):
        for j in range(ncols):
            idx = i*colwrap + j
            if idx >= n:
                break
            source = sources[idx]
            title = titles[idx]
            for component in source:
                ax[i, j].plot(component)
            ax[i, j].set_title(title, fontsize=8)
            if i == nrows - 1:
                ax[i, j].set_xlabel('Time')
            if j == 0:
                ax[i, j].set_ylabel('Value')

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
    fig.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight')

def run_ex1():
    np.random.seed(0)
    X, S = load_data()
    S_out, W, K, X_out, distances = fast_ica.ica(X)
    # print(W)
    plot_distances(distances, output='ex1_dist.pdf')
    X_whiten = fast_ica.retrieve_whiten(X, W, K, S_out)
    S_out_sk, W, K, X_out_sk, distances = fast_ica.ica_sk(X)
    print(la.inv(W @ K))
    S_pca = PCA().fit_transform(X.T).T

    source_title_list = [
        (S, '(a) Original sources ($S$)'),
        (X, '(b) Mixture signals ($X$)'),
        # (fast_ica.whiten(X)[0], 'Whiten $X$'),
        # (X_out, 'Retrieved $X = (W K)^{-1} S_{\\mathrm{predicted}}$'),
        # (X_whiten, '$X_{whiten}$ [Our FastICA]'),
        (X_out_sk, '(c) Retrieved signals ($X_{\\mathrm{retrieved}}$)'),
        (fast_ica.center(S_out, divide_sd=True), '(d) $S_{\\mathrm{predicted}}$ [Our FastICA]'),
        (fast_ica.center(S_out_sk, divide_sd=True), '(e) $S_{\\mathrm{predicted}}$ [Sklearn FastICA]'),
        (fast_ica.center(S_pca, divide_sd=True), '(f) $S_{\\mathrm{predicted}}$ [PCA]')
        ]
    sources, titles = zip(*source_title_list)
    plot_sources(sources, titles, output='ex1_sources.pdf', colwrap=3)

if __name__ == '__main__':
    run_ex1()