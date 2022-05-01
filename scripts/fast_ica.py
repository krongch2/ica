import numpy as np
import numpy.linalg as la
from sklearn.decomposition import FastICA, PCA

import tests

def center(X, divide_sd=False):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True, ddof=0) if divide_sd else 1
    D = (X - mu)/sd
    return D

def whiten(X, test=True):
    X = center(X)
    cov = np.cov(X)
    lamda, V = la.eigh(cov)
    lamda_inv = la.inv(np.diag(lamda))**0.5
    K = lamda_inv @ V.T
    Z = K @ X

    if test:
        tests.test_identity(np.cov(Z))

    return Z, K

def whiten_svd(X):
    X = center(X)
    U, D, _ = la.svd(X, full_matrices=False)
    K = U / D
    Z = K.T @ X
    return Z, K.T

def g(x, a1=1):
    return np.tanh(a1*x)

def dg(x):
    return 1 - g(x)**2

def retrieve_X_out(X, W, K, S_out):
    X_out = la.inv(W @ K) @ S_out + X.mean(axis=1, keepdims=True)
    return X_out

def ica(X, cycles=200, tol=1e-5, test=False):
    X1, K = whiten(X)
    nrows, ncols = X.shape
    W = np.zeros((nrows, nrows))
    distances = []
    for i in range(nrows):
        w = np.random.random((nrows))
        dd = []
        for _ in range(cycles):
            w_new = (X1 * g(w.T @ X1)).mean(axis=1) - dg(w.T @ X1).mean() * w

            if test:
                tests.test_gram_schmidt(w_new, W, i)

            w_new -= w_new @ W[:i].T @ W[:i]
            w_new /= ((w_new**2).sum())**0.5
            distance = np.abs(np.abs(w @ w_new) - 1)
            dd.append(distance)
            w = w_new

            if distance < tol:
                distances.append(np.array(dd))
                break

        W[i, :] = w

    S_out = W @ K @ X
    X_out = retrieve_X_out(X, W, K, S_out)
    return S_out, W, K, X_out, distances

def ica_sk(X, cycles=200, tol=1e-5):
    ica = FastICA(max_iter=cycles, tol=tol)
    S_out = ica.fit_transform(X.T).T
    W = ica._unmixing
    K = ica.whitening_
    X_out = retrieve_X_out(X, W, K, S_out)
    return S_out, W, K, X_out, []
