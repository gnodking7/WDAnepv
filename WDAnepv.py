"""
=============================================
Wasserstein Discriminant Analysis (WDA)
Nonlinear Eigenvalue based Algorithm: WDAnepv
=============================================
"""

# Author: Dong Min Roh <droh@ucdavis.edu>
#

import numpy as np
from scipy import linalg

###########################################################################

def sinkhorn(K, tol = 1e-5, maxitr = 10):
    """
    Performs Sinkhorn-Knopp Iteration on a positive matrix K to obtain
    positive vectors u and v such that diag(u) * K * diag(v) is normalized doubly stochastic

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
    maxitr :    int, optional, default set to 10
                Number of maximum number of iterations

    RETURNS
    -------
    u :         ndarray, shape (n,)
                Optimal left vector
    v :         ndarray, shape (m,)
                Optimal right vector
    v_err :     list
                List of 2-norm errors between consecutive v vectors
    """

    n, m = K.shape[0], K.shape[1]
    vk = np.ones(m) / m # initial point
    v_err = []
    # updates
    for i in range(maxitr):
        uk = np.ones(n) / (np.dot(K, vk)) / n
        new_vk = np.ones(m) / (np.dot(K.T, uk)) / m
        v_err.append(linalg.norm(new_vk - vk))   # error
        vk = new_vk
        if v_err[i] < tol:
            break
    u = uk
    v = vk
    return u, v

def Dink_TR(A, B, k, X, tol = 1e-5, maxitr = 100):
    """
    Computes the solution to Trace Ratio optimization (TRopt)
        max_{X^TX=I} q(X) := Tr(X'AX)/Tr(X'BX)
    by Dinkelbach's iteration
    Computes the k largest eigenvectors of A-q(X)B at each iteration

    Article of Dinkelbach's iteration:
    @article{dinkelbach1967nonlinear,
      title={On nonlinear fractional programming},
      author={Dinkelbach, Werner},
      journal={Management science},
      volume={13},
      number={7},
      pages={492--498},
      year={1967},
      publisher={INFORMS}
    }

    PARAMETERS
    ----------
    A :         ndarray, shape (n, n)
                Symmetric matrix
    B :         ndarray, shape (n, n)
                Symmetric positive definite matrix
    k :         int
                Size of subspace dimension
    X :         ndarray, shape (n, k)
                Initial projection
    tol :       float, optional, default set to 1e-5
                tolerance parameter for stopping criteria
    maxitr :    int, optional, default set to 100
                Number of maximum number of iterations

    RETURNS
    -------
    X :         ndarray, shape (n, k)
                Optimal orthogonal projection
    Q :         list
                List of TRopt values
    Err :       list
                List of TRopt errors between consecutive projections
    itr :       int
                Number of iterations
    """

    n = len(A)
    q = np.trace(np.matmul(X.T, np.matmul(A, X))) / np.trace(np.matmul(X.T, np.matmul(B, X)))
    Q, Err = [], []
    for i in range(maxitr):
        C = A - q * B
        C = (C + C.T) / 2   # ensure it's symmetric

        #D, V = linalg.eigh(C, subset_by_index = [n - k, n - 1]) # k largest eigs and eigenvectors, in ascending order
        #newX = np.fliplr(V) # eigenvectors now in descending order

        D, V = linalg.eigh(C)
        idx = np.argsort(-D)
        newX = V[:, idx[0:k]]

        newq = np.trace(np.matmul(newX.T, np.matmul(A, newX))) / np.trace(np.matmul(newX.T, np.matmul(B, newX)))

        err = np.abs(newq - q) / np.abs(newq)
        Err.append(err)
        # stopping criteria
        if (abs(linalg.subspace_angles(X, newX)[0]) < tol) or (err < 1e-5):
            X = newX
            Q.append(newq)
            break
        # Update the iterate
        Q.append(q)
        q = newq
        X = newX
    itr = i
    return X, Q, Err, itr

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


def pair_tensor(T, X, Y):
    """
    Computes the sum of rank one matrices
        \sum_{ij}T(i,j) * X(i,:)' * Y(j,:)

    PARAMETERS
    ----------
    T :     ndarray, shape (n, m)
            Cost matrix, obtained from Sinkhorn
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

def wda_nepv(DATA, LABEL, p, P, lamb=0.01, e=0, k=10, tol=1e-5, maxitr=50):
    """
    Performs Wasserstein Discriminant Analysis via Bi-Level Optimization method.
    Inner optimization involves solving Trace Ratio optimization (TRopt) using
    Self-Consistent-Field method (equivalent to Dinkelbach's algorithm).

    Mathematical formulation of WDA is:
        \max_{P^TP=I} Trace(P^TC_b(P)P)/Trace(P^TC_w(P)P)
    where C_b(P) and C_w(P) are sum of large number of rank one matrices of the form t*d*d'
    Rather than summing large number of times, it is far more efficient to
    form a (short and long) matrix with \sqrt{t}d columns then do matrix
    times matrix transpose to form C_b(P) and C_w(P).

    PARAMETERS
    ----------
    DATA :      ndarray, shape (n, d)
                Data matrix
    LABEL :     ndarray, shape (n,)
                Labels for DATA
    p :         int
                Size of subspace dimension
    P :         ndarray, shape (d, p)
                Initial projection
    lamb :      float, optional, default set to 0.01
                Wasserstein regularization parameter > 0
    e :         float, optional, default set to 0
                Perturbation parameter to regularize the denominator matrix
    k :         int, optional, default set to 10
                Number of Sinkhorn iterations
    tol :       float, optional, default set to 1e-5
                tolerance parameter for stopping criteria
    maxitr :    int, optional, default set to 50
                Number of maximum number of iterations

    RETURNS
    -------
    P :         ndarray, shape (d, p)
                Optimal orthogonal projection
    Sub_Err :   list
                List of subspace errors between consecutive projections
    Norm_Err :  list
                List of Frobenius errors between consecutive projections
    WDA_Val :   list
                List of WDA values
    proj :      callable
                Projection function
    """

    #mx = np.mean(DATA)
    #DATA -= mx.reshape((1, -1))

    d = DATA.shape[1]   # data point dimension
    Sub_Err, Norm_Err, WDA_Val = [], [], []

    X_c = split_classes(DATA, LABEL) # split data by classes

    for it in range(maxitr):
        # obtain between scatter A:=C_b(P) and within scatter B:=C_w(P)
        A = np.zeros((d, d))
        B = np.zeros((d, d))
        for i, xi in enumerate(X_c):
            pxi = np.matmul(xi, P)
            for j, xj in enumerate(X_c[i:]):
                pxj = np.matmul(xj, P)
                M = dist(pxi, pxj)
                K = np.exp(-lamb * M)
                u, v = sinkhorn(K, 1e-5, k)
                T = u.reshape((K.shape[0], 1)) * K * v.reshape((1, K.shape[1]))
                if j == 0:
                    B += pair_tensor(T, xi, xj)
                else:
                    A += pair_tensor(T, xi, xj)
        B += e * np.eye(d)  # perturb
        # store WDA value
        WDA_Val.append(np.trace(np.matmul(P.T, np.matmul(A, P))) / np.trace(np.matmul(P.T, np.matmul(B, P))))
        # solve TRopt \max_{P^TP=I} Trace(P^TAP)/Trace(P^TBP) by Dinkelbach's iteration
        NEW_P, Q, Err, itr = Dink_TR(A, B, p, P, tol=1e-3)
        # errors
        Sub_Err.append(abs(linalg.subspace_angles(P, NEW_P)[0]))
        Norm_Err.append(np.linalg.norm(P - NEW_P, 'fro') / np.linalg.norm(NEW_P, 'fro'))
        # update
        P = NEW_P
        # stopping criteria
        if it > 0:
            err = abs(WDA_Val[it] - WDA_Val[it - 1]) / abs(WDA_Val[it])
            if (Sub_Err[it] < tol) or (Norm_Err[it] < tol) or (err < 1e-5):
                break

    def proj(X):
        return np.matmul(X, P)
        #return (X - mx.reshape((1, -1))).dot(P)

    return P, Sub_Err, Norm_Err, WDA_Val, proj