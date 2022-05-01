import numpy as np
import numpy.linalg as la

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

    return Z, K.T

def whiten_svd(X):
    U, D, _ = la.svd(X, full_matrices=False)
    K = U / D
    Z = K.T @ X
    return Z, K

def g(x, a1=1):
    return np.tanh(a1*x)

def dg(x):
    return 1 - g(x)**2

def ica(X, cycles=1000, tol=1e-5, test=False):

    X = center(X)


    nrows, ncols = X.shape

    X1, K = whiten(X)


    W = np.zeros((nrows, nrows))
    distances = []
    for i in range(nrows):
        w = np.random.random((nrows))

        for _ in range(cycles):
            w_new = (X1 * g(w.T @ X1)).mean(axis=1) - dg(w.T @ X1).mean() * w

            if test:
                tests.test_gram_schmidt(w_new, W, i)

            w_new -= w_new @ W[:i].T @ W[:i]
            w_new /= ((w_new**2).sum())**0.5
            distance = np.abs(np.abs(w @ w_new) - 1)
            distances.append(distance)
            w = w_new

            if distance < tol:
                break
        W[i, :] = w

    print('my W', W)

    S_predicted = W @ K.T @ X
    return S_predicted, W, K, distances
