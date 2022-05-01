import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import seaborn as sns

import fast_ica

# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

def loadmat(fn):
    d = {}
    with h5py.File(fn) as f:
        for k, v in f.items():
            d[k] = np.array(v)
    return d

def plot_eeg(d, output='eeg.pdf'):
    ncomponents = d.shape[0]
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=False)
    for comp in range(ncomponents):
        i = int(comp/8)
        j = int(comp%8)
        ax[i, j].plot(d[comp])
        # ax[i, j].set_title(f'{comp}', fontsize=6)
        ax[i, j].set_yticks([])
        ax[i, j].set_xticks([])
    fig.tight_layout()
    plt.savefig(output)


def std(x):
    return (x - x.mean())/np.std(x)

if __name__ == '__main__':
    X = loadmat('eeg.mat')['Data']
    X = X[:2]
    plot_eeg(X, 'eeg_in.pdf')
    S_predicted, W, _ = fast_ica.ica(X, cycles=100)
    plot_eeg(S_predicted, 'eeg_out.pdf')
    S_predicted_noblinks = S_predicted.copy()
    S_predicted_noblinks[0, :] = np.zeros(X.shape[1])
    X_noblinks = la.inv(W) @ S_predicted_noblinks
    # plot_eeg(X_noblinks, 'egg_noblinks.pdf')

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].plot(std(X[0]))
    ax[1].plot(std(X_noblinks[0]))
    plt.show()
