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
    fig, ax = plt.subplots(nrows=n, ncols=1, sharex=True, sharey=False)
    for i in range(n):
        for source in sources[i]:
            ax[i].plot(source)
        ax[i].set_title(titles[i], fontsize=8)

    fig.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':

    np.random.seed(0)
    X, S = load_data()
    S_predicted, W, K, _ = fast_ica.ica(X)
    tt= S_predicted.T @ la.inv(W @ K.T).T
    print(X.mean(axis=1))
    print(tt.shape)
    # mean = np.einsum('ij,i->ij', np.ones(X.shape), X.mean(axis=1, keepdims=True))
    X_new = ( S_predicted.T @ la.inv(W @ K.T).T + X.mean(axis=1)).T
    sklearn_ica = FastICA()
    S_sklearn = sklearn_ica.fit_transform(X.T).T
    # S_sklearn = fast_ica.whiten(S_sklearn)
    print(sklearn_ica.mean_)
    W_sklearn = sklearn_ica.mixing_
    X_new_sklearn = (S_sklearn.T @ W_sklearn.T + sklearn_ica.mean_).T

    source_title = [
        (S, 'Original sources ($S$)'),
        (X, 'Mixture signals ($X$)'),
        (fast_ica.whiten(X)[0], 'Whiten $X$'),
        (S_predicted, 'Predicted $S$ [Our FastICA]'),
        (X_new, 'Retrieved $X = W^{-1} S_{\\mathrm{predicted}}$ [Our FastICA]'),
        # (S_sklearn, 'Predicted $S$ [Sklearn FastICA]'),
        # (X_new_sklearn, 'Retrieved $X = W^{-1} S_{\\mathrm{predicted}}$ [Sklearn FastICA]'),
        ]
    sources, titles = zip(*source_title)
    plot_sources(sources, titles)
    exit()

    # S_sklearn = fast_ica.center(S_sklearn, standardize=True)
    # mean = np.einsum('ij,i->ij', np.ones(X.shape), sklearn_ica.mean_)
    # print(mean)
    # print(S_sklearn.shape)
    S_pca = PCA().fit_transform(X.T).T

