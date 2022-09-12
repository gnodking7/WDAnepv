"""
=============================================
Wasserstein Discriminant Analysis (WDA)
Subfunctions shared by WDAgd, WDAeig, WDAnepv
=============================================
"""

# Author: Dong Min Roh <droh@ucdavis.edu>
#

import numpy as np
from scipy import linalg

###########################################################################

def split_classes(X, y):
    """
    Split data samples in data matrix X by classes using the labels y

    PARAMETERS
    ----------
    X :     ndarray, shape (n, d)
            Data matrix
    y :     ndarray, shape (n,)
            Labels for X

    RETURNS
    -------
    X_c :   list
            List of data class matrices
    """

    labels = np.unique(y)
    X_c = [X[y == i, :].astype(np.float32) for i in labels]
    return X_c

def dist(X, Y):
    """
    Compute squared Euclidean distance between
    all possible data points in data matrices X and Y

    PARAMETERS
    ----------
    X :     ndarray, shape (n, d)
            Data matrix
    Y :     ndarray, shape (m, d)
            Data matrix

    RETURNS
    -------
    M :     ndarray, shape (n, m)
            Metric matrix
    """

    X_inn_prod = np.sum(np.square(X), 1)
    Y_inn_prod = np.sum(np.square(Y), 1)
    M = X_inn_prod.reshape((-1, 1)) + Y_inn_prod.reshape((1, -1)) - 2 * np.matmul(X, Y.T)
    return M

def SK(K, tol = 1e-5, maxitr = 50):
    """
    Performs Sinkhorn-Knopp Iteration on a positive matrix K to obtain
    positive vectors u and v such that diag(u) * K * diag(v) is normalized doubly stochastic
    i.e., the sum of rows and the sum of columns of the resulting matrix are
    vectors 1/n and 1/m, respectively

    Article on Sinkhorn iteration:
    @article{sinkhorn1967diagonal,
        title={Diagonal equivalence to matrices with prescribed row and column sums},
        author={Sinkhorn, Richard},
        journal={The American Mathematical Monthly},
        volume={74},
        number={4},
        pages={402--405},
        year={1967},
        publisher={JSTOR}
    }

    PARAMETERS
    ----------
    K :         ndarray, shape (n, m)
                Positive matrix
    tol :       float, optional, default set to 1e-5
                Tolerance parameter for stopping criteria
    maxitr :    int, optional, default set to 50
                Number of maximum number of iterations

    RETURNS
    -------
    u :         ndarray, shape (n,)
                Optimal left vector
    v :         ndarray, shape (m,)
                Optimal right vector
    """

    n, m = K.shape[0], K.shape[1]
    vk = np.ones(m) / m # initial point
    # updates
    for i in range(maxitr):
        uk = np.ones(n) / (np.dot(K, vk)) / n
        new_vk = np.ones(m) / (np.dot(K.T, uk)) / m
        v_err = np.linalg.norm(new_vk - vk)   # error
        vk = new_vk
        if v_err < tol:
            break
    u = uk
    v = vk
    return u, v

def Acc_SK(K, tol = 1e-5, maxitr = 50):
    """
    Performs Accelerated Sinkhorn-Knopp Iteration on a positive matrix K to obtain
    positive vectors u and v such that diag(u) * K * diag(v) is normalized doubly stochastic
    i.e., the sum of rows and the sum of columns of the resulting matrix are
    vectors 1/n and 1/m, respectively

    Article on Acclerated SK iteration:
    @article{aristodemo2020accelerating,
        title={Accelerating the Sinkhorn--Knopp iteration by Arnoldi-type methods},
        author={Aristodemo, A and Gemignani, L},
        journal={Calcolo},
        volume={57},
        number={1},
        pages={1--17},
        year={2020},
        publisher={Springer}
    }

    PARAMETERS
    ----------
    K :         ndarray, shape (n, m)
                Positive matrix
    tol :       float, optional, default set to 1e-5
                Tolerance parameter for stopping criteria
    maxitr :    int, optional, default set to 50
                Number of maximum number of iterations

    RETURNS
    -------
    u :         ndarray, shape (n,)
                Optimal left vector
    v :         ndarray, shape (m,)
                Optimal right vector
    """

    n, m = K.shape[0], K.shape[1]
    vk = np.ones(m) / m # initial point
    # Functions
    U = lambda x : np.ones(len(x)) / x
    S = lambda x : U(np.dot(K, x))
    R = lambda x : (n / m) * U(np.dot(K.T, S(x)))
    J = lambda x : (m / n) * np.matmul((R(x) ** 2)[:, None] * K.T * (S(x) ** 2), K)
    # updates
    for i in range(maxitr):
        Jk = J(vk)
        D, V = linalg.eig(Jk)
        D = np.real(D)
        V = np.real(V)
        idx = np.argsort(-D)
        new_vk = V[:, idx[0]]
        v_err = np.linalg.norm(new_vk - vk)   # error
        if v_err < tol:
            break
        vk = new_vk
    u = np.ones(n) / (np.matmul(K, vk)) / n
    v = vk
    return u, v

def pair_tensor(T, X, Y):
    """
    Computes the sum of rank one matrices
        \sum_{ij}T(i,j) * [X(i,:) - Y(j,:)] * [X(i,:) - Y(j,:)]'
    efficiently by matrix-matrix multiplication

    PARAMETERS
    ----------
    T :     ndarray, shape (n, m)
            Cost matrix
    X :     ndarray, shape (n, d)
            Data matrix
    Y :     ndarray, shape (m, d)
            Data matrix

    RETURNS
    -------
    C_XY :  ndarray, shape (d, d)
            Matrix
    """

    d = X.shape[1]
    temp = X[:, None] - Y
    C = temp * np.sqrt(T)[:, :, None]
    C = C.reshape((-1, d))
    C_XY = np.matmul(C.T, C)
    return C_XY
