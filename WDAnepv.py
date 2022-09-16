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
import WDA_subfunc as sub

###########################################################################

def Dink_TR(A, B, k, X, tol = 1e-5, maxitr = 100):
    """
    Computes the solution to Trace Ratio optimization (TRopt)
        max_{X^TX=I_k} q(X) := Tr(X'AX)/Tr(X'BX)
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
    Sub_Err :   list
                List of subspace errors between consecutive projections
    PROJ:       list
                List of projections matrices
    itr :       int
                Number of iterations
    """

    n = len(A)
    q = np.trace(np.matmul(X.T, np.matmul(A, X))) / np.trace(np.matmul(X.T, np.matmul(B, X)))
    Q, Err, Sub_Err, PROJ = [], [], [], []
    
    for i in range(maxitr):
        C = A - q * B
        C = (C + C.T) / 2   # ensure it's symmetric

        D, V = linalg.eigh(C)
        idx = np.argsort(-D)
        newX = V[:, idx[0:k]]
        newq = np.trace(np.matmul(newX.T, np.matmul(A, newX))) / np.trace(np.matmul(newX.T, np.matmul(B, newX)))

        err = np.abs(newq - q) / np.abs(newq)
        Err.append(err)
        sub_err = linalg.subspace_angles(X, newX)[0]
        Sub_Err.append(sub_err)
        # Update the iterate
        Q.append(q)
        PROJ.append(X)
        q = newq
        X = newX
        # stopping criteria
        if (Sub_Err[i] < tol) or (err < 1e-5):
            Q.append(q)
            PROJ.append(X)
            break
    itr = i
    return X, Q, Err, Sub_Err, PROJ, itr + 1

def wda_nepv(X, y, p, reg, P0, Breg=0, k=10, maxitr=100, tol=1e-5):
    """
    Mathematical formulation of WDA is:
        \max_{P^TP=I} Trace(P^TC_b(P)P)/Trace(P^TC_w(P)P)
    where C_b(P) and C_w(P) are between and within covariance matrices, respectively

    Performs Wasserstein Discriminant Analysis via Bi-Level Optimization method:
        P_{k+1} = \argmax_{P^TP=I} Trace(P^TC_b(P_k)P)/Trace(P^TC_w(P_k)P)
    Above inner optimization involves solving Trace Ratio optimization (TRopt) using
    Self-Consistent-Field method (equivalent to Dinkelbach's algorithm).

    PARAMETERS
    ----------
    X :         ndarray, shape (n, d)
                Data matrix
    y :         ndarray, shape (n,)
                Labels for DATA
    p :         int
                Size of subspace dimension
    reg :       float
                Wasserstein regularization parameter > 0
    P0 :        ndarray, shape (d, p)
                Initial projection
    Breg :      float, optional, default set to 0
                Perturbation parameter to regularize the denominator matrix
    k :         int, optional, default set to 10
                Number of Acc_SK iterations
    maxitr :    int, optional, default set to 100
                Number of maximum number of iterations
    tol :       float, optional, default set to 1e-5
                tolerance parameter for stopping criteria

    RETURNS
    -------
    Popt :      ndarray, shape (d, p)
                Optimal orthogonal projection
    proj :      callable
                Projection function
    WDA_Val :   list
                List of WDA values
    PROJ :      list
                List of projections matrices
    Sub_Err :   list
                List of subspace errors between consecutive projections
    """

    mx = np.mean(X)
    X -= mx.reshape((1, -1))
    
    # data split between classe
    d = X.shape[1]   # data point dimension
    X_c = sub.split_classes(X, y) # split data by classes
    P = P0
    
    Sub_Err = []
    PROJ = []
    WDA_Val = []

    for it in range(maxitr):
        # obtain between scatter A:=C_b(P) and within scatter B:=C_w(P)
        A = np.zeros((d, d))
        B = np.zeros((d, d))
        for i, xi in enumerate(X_c):
            pxi = np.matmul(xi, P)
            for j, xj in enumerate(X_c[i:]):
                pxj = np.matmul(xj, P)
                M = sub.dist(pxi, pxj)
                K = np.exp(-reg * M)
                u, v, Err = sub.Acc_SK(K, 1e-5, k)
                T = u.reshape((K.shape[0], 1)) * K * v.reshape((1, K.shape[1]))
                if j == 0:
                    B += sub.pair_tensor(T, xi, xj)
                else:
                    A += sub.pair_tensor(T, xi, xj)
        B += e * np.eye(d)  # perturb
        # store WDA value
        A = (A + A.T) / 2
        B = (B + B.T) / 2
        WDA_Val.append(np.trace(np.matmul(P.T, np.matmul(A, P))) / np.trace(np.matmul(P.T, np.matmul(B, P))))
        # solve TRopt \max_{P^TP=I} Trace(P^TAP)/Trace(P^TBP) by Dinkelbach's iteration
        NEW_P, Q, Err, Dink_Sub_Err, Dink_PROJ, itr = Dink_TR(A, B, p, P)
        PROJ.append(P)
        # errors
        Sub_Err.append(abs(linalg.subspace_angles(P, NEW_P)[0]))
        # update
        P = NEW_P
        # stopping criteria
        if it > 0:
            err = abs(WDA_Val[it] - WDA_Val[it - 1]) / abs(WDA_Val[it])
            if (abs(Sub_Err[it]) < tol) or (err < 1e-5):
                PROJ.append(P)
                WDA_Val.append(np.trace(np.matmul(P.T, np.matmul(A, P))) / np.trace(np.matmul(P.T, np.matmul(B, P))))
                break
    
    Popt = P
    
    def proj(X):
        return (X - mx.reshape((1, -1))).dot(P)

    return Popt, proj, WDA_Val, PROJ, Sub_Err
