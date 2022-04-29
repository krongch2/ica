import numpy as np

def test_gram_schmidt(w, W, i):
    w_new1 = w.copy()
    for j in range(i):
        w_new1 -= (w_new1 @ W[j].T) * W[j]
    w_new2 = w.copy()
    w_new2 -= W[:i] @ w_new2 @ W[:i]
    assert np.allclose(w_new1, w_new2)

def test_identity(A):
    assert np.allclose(A, np.eye(A.shape[0]))
