"""
=============================================
Ratio Trace formulated
Wasserstein Discriminant Analysis (WDA)
Nonlinear Eigenvalue based Algorithm: WDAeig
=============================================
"""

# Author: Dong Min Roh <droh@ucdavis.edu>
#

import numpy as np
from scipy import linalg
import WDA_subfunc as sub

###########################################################################

def wda_eig(X, y, p, reg, P0, Breg=0, k=10, maxiter=100, verbose=0):
    """
    The function solves the ratio trace formulation of WDA.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Training samples.
    y : ndarray, shape (n,)
        Labels for training samples.
    p : int
        Size of dimensionality reduction.
    reg : float
        Wasserstein regularization term >0 (entropic regularization)
    P0 : ndarray, shape (d, p)
        Initial subspace for projection.
    Breg: float, optional, default set to 0
        Regularization for the B matrix in the denominator to make B positive definite
    k: int, optional, default set to 10
        Number of Sinkhorn iterations
    maxiter: int, optional, default set to 100
        Number of maximum number of iterations

    Returns
    -------
    Popt : ndarray, shape (d, p)
        Optimal transportation matrix for the given parameters
    proj : callable
        Projection function including mean centering
    obj  : list
        List of angles s_k to measure the distance between consecutive subspaces
    Q    : list
        List of WDA objective values
    PROJ : list
        List of ndarray projections
    Sub_Err : list
        List of subspace errors between consecutive subspaces
    """

    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = sub.split_classes(X, y)
    # compute uniform weighs
    wc = [np.ones((x.shape[0]), dtype=np.float32) / x.shape[0] for x in xc]
    P=P0

    obj = []

    Sub_Err = []
    PROJ = []
    Q = []

    for it in range(maxiter):
        loss_b = np.zeros((d,d))
        loss_w = np.zeros((d,d))
        for i, xi in enumerate(xc):
            pxi = np.dot(xi, P)
            for j, xj in enumerate(xc[i:]):
                pxj = np.dot(xj, P)
                M = sub.dist(pxi, pxj)
                K = np.exp(-reg * M)
                u, v = sub.Acc_SK(K, 1e-5, k)
                T = u.reshape((K.shape[0], 1)) * K * v.reshape((1, K.shape[1]))
                if j==0:
                    loss_w += sub.pair_tensor(T, xi, xj)
                else:
                    loss_b += sub.pair_tensor(T, xi, xj)
        if Breg==0:
            w, V = linalg.eig((loss_b+loss_b.T)/2, (loss_w+loss_w.T)/2)
        else:
            w, V = linalg.eig((loss_b+loss_b.T)/2, (loss_w+loss_w.T)/2+Breg*np.eye(d))
        w=np.real(w)
        V=np.real(V)
        idx = np.argsort(-w)
        Pnew = V[:, idx[0:p]]

        Pinv = np.linalg.inv(P.T.dot(P))
        Pninv = np.linalg.inv(Pnew.T.dot(Pnew))

        angle = np.linalg.norm((P.dot(Pinv.dot(P.T))-Pnew.dot(Pninv.dot(Pnew.T))), 2)

        obj.append(angle)
        if (verbose==1):
            print("Iter: % 2d, angle: % 2.8f" %(it, angle))

        err = linalg.subspace_angles(P, Pnew)[0]
        Sub_Err.append(err)
        PROJ.append(P)
        q = np.trace(np.matmul(P.T, np.matmul(loss_b, P))) / np.trace(np.matmul(P.T, np.matmul(loss_w, P)))
        Q.append(q)

        P=Pnew

        if (abs(angle)< 1e-3):
            break

    Popt = P

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt)

    return Popt, proj, obj, Q, PROJ, Sub_Err
