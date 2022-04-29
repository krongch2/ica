import numpy as np
import numpy.linalg as la

import tests

def center(X, standardize=False):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True, ddof=0) if standardize else 1
    D = (X - mu)/sd
    return D

def whiten(X, test=True):
    cov = np.cov(X)
    print(cov)
    lamda, V = la.eigh(cov)
    lamda_inv = np.sqrt(la.inv(np.diag(lamda)))
    Z = lamda_inv @ V.T @ X
    print(np.cov(Z))
    if test:
        tests.test_identity(np.cov(Z))

    return Z

def g(x, a1=1):
    return np.tanh(a1*x)

def dg(x):
    return 1 - g(x)**2

def ica(X, cycles=1000, tol=1e-5, test=False):
    X = whiten(center(X))
    nrows = X.shape[0]
    W = np.zeros((nrows, nrows))
    distances = []
    for i in range(nrows):
        w = np.random.random((nrows))
        for _ in range(cycles):
            w_new = (X * g(w.T @ X)).mean(axis=1) - dg(w.T @ X).mean() * w

            if test:
                tests.test_gram_schmidt(w_new, W, i)

            w_new -= W[:i] @ w_new @ W[:i]
            w_new /= (w_new**2).sum()**0.5
            distance = np.abs(np.abs(w @ w_new) - 1)
            distances.append(distance)
            w = w_new

            # check why it doesn't stop
            if distance < tol:
                break
        W[i, :] = w
    S_predicted = W @ X
    return S_predicted, distances
