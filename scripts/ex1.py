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
    s2 = np.sign(np.sin(3*time)) # square signal
    s3 = signal.sawtooth(2*np.pi*time) # saw tooth signal
    s4 = s1 + s2
    s5 = s1 - s2
    s6 = np.cos(2*time)

    S = np.array([s1, s2, s3])
    S /= S.std(axis=1, keepdims=True)
    # S = np.array([s3, s4, s5])
    A = np.array([
        [1, 1, 1],
        [0.5, 2, 1],
        [1.5, 1, 2]
        ])
    X = A @ S
    return X, S

def plot_sources(X, S, S_predicted, S_predicted_std, S_sklearn, S_pca):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

    for x in X:
        ax[0, 0].plot(x)
        ax[0, 0].set_title('Mixture signals ($X$)')
        ax[0, 0].set(ylabel='Amplitude')

    for s in S:
        ax[0, 1].plot(s)
        ax[0, 1].set_title("Original sources ($S$)")

    for S_predicted in S_predicted:
        ax[1, 0].plot(S_predicted)
        ax[1, 0].set_title("Predicted sources ($W X$) [Our FastICA]")
        ax[1, 0].set(xlabel='Time', ylabel='Amplitude')

    for S_predicted_std in S_predicted_std:
        ax[1, 1].plot(S_predicted_std)
        ax[1, 1].set_title("Predicted sources ($W X$) [Our FastICA, std]")
        ax[1, 1].set(xlabel='Time', ylabel='Amplitude')

    # for S_sklearn in S_sklearn:
    #     ax[1, 1].plot(S_sklearn)
    #     ax[1, 1].set_title("Predicted sources ($W X$) [Sklearn FastICA]")
    #     ax[1, 1].set(xlabel='Time')

    # for S_pca in S_pca:
    #     ax[1, 2].plot(S_pca)
    #     ax[1, 2].set_title("Predicted sources ($W X$) [Sklearn PCA]")

    fig.tight_layout()
    # plt.show()
    plt.savefig('all.pdf', bbox_inches='tight')

if __name__ == '__main__':
    np.random.seed(0)
    X, S = load_data()
    # S_predicted, W, _ = fast_ica.ica(X, cycles=100, standardize=False)

    # S_predicted_std, W_std, _ = fast_ica.ica(X, cycles=100, standardize=True)

    # print(W)
    # print(W_std)
    # sd = X.std(axis=1, ddof=0)
    # print(sd)
    # print(W_std*sd)
    sklearn_ica = FastICA(n_components=3)
    S_sklearn = sklearn_ica.fit_transform(X.T).T
    # S_sklearn = fast_ica.center(S_sklearn, standardize=True)
    W = sklearn_ica.mixing_.T
    print(sklearn_ica.mean_)
    mean = np.einsum('ij,i->ij', np.ones(X.shape), sklearn_ica.mean_)
    # print(la.inv(W) @ S_sklearn + mean)
    X_new = la.inv(W) @ S_sklearn + mean
    X_new = fast_ica.center(X_new, standardize=True)
    S_pca = PCA(n_components=3).fit_transform(X.T).T
    plot_sources(X, S, S_sklearn, X_new, X_new, S_pca)
