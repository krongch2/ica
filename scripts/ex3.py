import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

import fast_ica

# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

def loadmat(fn):
    d = {}
    with h5py.File(fn) as f:
        for k, v in f.items():
            d[k] = np.array(v)
    return d

def plot_eeg(d, d2=None, output='eeg.pdf'):
    ncomponents = d.shape[0]
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for comp in range(ncomponents):
        i = int(comp/8)
        j = int(comp%8)
        ax[i, j].plot(d[comp], '-', color=colors[0])
        if d2 is not None:
            ax[i, j].plot(d2[comp], '-', color=colors[1])
        ax[i, j].text(0.5, 1, f'{comp}', transform=ax[i, j].transAxes, fontsize=6)
        # ax[i, j].set_yticks([])
        ax[i, j].set_xticks([])
    fig.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight')

def remove_blinks():
    np.random.seed(0)
    X = loadmat('ex2_eeg.mat')['Data']
    plot_eeg(X, output='ex2_raw.pdf')
    S_out, W, K, X_out, distances = fast_ica.ica(X)
    plot_eeg(S_out, output='ex2_ica.pdf')
    S_out_noblinks = S_out.copy()
    S_out_noblinks[3, :] = np.zeros(X.shape[1])
    X_noblinks = fast_ica.retrieve_X_out(X, W, K, S_out_noblinks)
    plot_eeg(X_noblinks, output='ex2_noblinks.pdf')
    plot_eeg(X, d2=X_noblinks, output='ex2_noblinks.pdf')



if __name__ == '__main__':
    remove_blinks()
