import numpy as np
import numpy.linalg as la
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt

def data1():
    N = 2000
    time = np.linspace(0, 8, N)
    s1 = np.sin(2*time) # sinusoidal
    s2 = np.sign(np.sin(3*time)) # square signal
    s3 = signal.sawtooth(2*np.pi*time) # saw tooth signal

    S = np.array([s1, s2, s3])
    A = np.array([
        [1, 1, 1],
        [0.5, 2, 1],
        [1.5, 1, 2]
        ])
    X = A @ S
    return X, S

def data2():
    X = np.array([
        [1, 1, 2, 0, 5, 4, 5, 3],
        [3, 2, 3, 3, 4, 5, 5, 4]
        ])
    return X

def get_cov(X):
    return (X @ X.T)/(X.shape[1] - 1)

def center(X, standardize=False):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True, ddof=0) if standardize else 1
    D = (X - mu)/sd
    return D

def whiten(X):
    cov = np.cov(X)
    lamda, V = la.eigh(cov)
    lamda_inv = np.sqrt(la.inv(np.diag(lamda)))
    Z = lamda_inv @ V.T @ X
    print(np.cov(Z))
    return Z

def g(x, a1=1):
    return np.tanh(a1*x)

def dg(x):
    return 1 - g(x)**2

def test_gram_schmidt(w, W, i):
    w_new1 = w.copy()
    for j in range(i):
        w_new1 -= (w_new1 @ W[j].T) * W[j]
    w_new2 = w.copy()
    w_new2 -= W[:i] @ w_new2 @ W[:i]
    assert np.allclose(w_new1, w_new2)

def ica(X, iterations=1000, tol=1e-5):
    X = whiten(center(X))
    nrows = X.shape[0]
    W = np.zeros((nrows, nrows))
    for i in range(nrows):
        w = np.random.random((nrows))
        for j in range(iterations):
            w_new = (X * g(w.T @ X)).mean(axis=1) - dg(w.T @ X).mean() * w
            w_new -= W[:i] @ w_new @ W[:i]
            w_new /= (w_new ** 2).sum()**0.5
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            w = w_new
            print(distance)
            if distance < tol:
                break
        W[i, :] = w
    S = W @ X
    return S

def plot_sources(X, S, S_predicted):
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for x in X:
        ax[0, 0].plot(x)
        ax[0, 0].set_title('mixture signals')

    for s in S:
        ax[0, 1].plot(s)
        ax[0, 1].set_title("original sources")

    for S_predicted in S_predicted:
        ax[1, 0].plot(S_predicted)
        ax[1, 0].set_title("predicted sources")
    fig.tight_layout()
    plt.show()

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
#     S = ica(X, iterations=1000)
#     plot_mixture_sources_predictions(X, [source1, source2], S)
#     wavfile.write('out1.wav', sampling_rate, S[0])
#     wavfile.write('out2.wav', sampling_rate, S[1])

# from sklearn.preprocessing import StandardScaler
# X = StandardScaler().fit_transform(X)
# print(X)
# from sklearn.decomposition import PCA
# pca = PCA(whiten=True)
# pca.fit(D)
# D = pca.transform(D)
# print(D)

if __name__ == '__main__':
    np.random.seed(0)

    X = data2()
    S_predicted = ica(X, iterations=100)
    S = None
    plot_sources(X, S, S_predicted)
