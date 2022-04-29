import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def exmaple_1():

    def get_s1(j):
        return 0.5*np.sin(j)

    def get_s2(N):
        return np.random.uniform(0, 1, N) - 0.5

    def get_x(s1, s2, a, b):
        return a*s1 + b*ss

    N = 500
    lin = np.linspace(1, 30, N)
    s1 = get_s1(lin)
    s2 = get_s2(N)
    x1 = get_x(s1, s2, 0.5, 0.5)
    x2 = get_x(s1, s2, 0.5, -0.5)
    # plt.plot(lin, x2)
    plt.plot(x1, x2, 'o')
    plt.show()

def example_2_1_2():
    s_1 = [1, 2, 1, 2]
    s_2 = [1, 1, 2, 2]
    s = np.array([s_1, s_2])
    A = np.array([
        [1, 2],
        [1, -1]
        ])
    x = A @ s
    W = la.inv(A)
    print(W @ x)

def get_cov(A):
    return (A.T @ A)/(A.shape[0] - 1)

def example_3_3():
    X = np.array([
        [1, 1, 2, 0, 5, 4, 5, 3],
        [3, 2, 3, 3, 4, 5, 5, 4]
        ]).T
    mu = X.mean(axis=0)
    D = X - mu
    print(D)
    sigma_D = get_cov(D)
    print(sigma_D)
    lamda, V = la.eigh(sigma_D)
    U = (V @ D.T).T
    sigma_U = get_cov(U)
    print(sigma_U)

    Z = np.einsum('i,ij -> ij', lamda**-0.5, U.T).T
    print(lamda.shape)
    print(U.shape)
    print(Z.T)
    sigma_Z = get_cov(Z)
    print(sigma_Z)

    # plt.plot(X[:, 0], X[:, 1], 'o')
    # plt.plot(U[:, 0], U[:, 1], 'o')
    plt.plot(Z[:, 0], Z[:, 1], 'o')
    plt.show()

example_3_3()

def pca():

    def get_s1(N):
        lin = np.linspace(1, 30, N)
        return 0.5*np.sin(lin)

    def get_s2(N):
        return np.random.uniform(0, 1, N) - 0.5

    N = 100
    s = np.array([get_s1(N), get_s2(N)])
    A = np.array([
        [1, 2],
        [1, -1]
        ])
    x = A @ s
    print(x)
    # plt.plot(x[0, :], x[1, :], 'o', mec='white')
    lin = np.linspace(1, 30, N)
    plt.plot(lin, x[1, :])
    plt.show()


# pca()

def data2():
    X = np.array([
        [1, 1, 2, 0, 5, 4, 5, 3],
        [3, 2, 3, 3, 4, 5, 5, 4]
        ])
    return X

def get_cov(X):
    return (X @ X.T)/(X.shape[1] - 1)

