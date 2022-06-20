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
    Sub_Err :   list
                List of subspace errors between consecutive projections
    PROJ:       list
                List of projections matrices
    itr :       int
                Number of iterations
    """

    n = len(A)
    q = np.trace(np.matmul(X.T, np.matmul(A, X))) / np.trace(np.matmul(X.T, np.matmul(B, X)))
    Q, Err = [], []
    Sub_Err = []
    PROJ = []
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

def wda_nepv(DATA, LABEL, p, P, lamb=0.01, e=0, k=10, tol=1e-3, maxitr=50):
    """
    Mathematical formulation of WDA is:
        \max_{P^TP=I} Trace(P^TC_b(P)P)/Trace(P^TC_w(P)P)
    where C_b(P) and C_w(P) are sum of large number of rank one matrices of the form t*d*d'
    Rather than summing large number of times, it is far more efficient to
    form a (short and long) matrix with \sqrt{t}d columns then do matrix
    times matrix transpose to form C_b(P) and C_w(P).

    Performs Wasserstein Discriminant Analysis via Bi-Level Optimization method:
        P_{k+1} = \argmax_{P^TP=I} Trace(P^TC_b(P_k)P)/Trace(P^TC_w(P_k)P)
    Above inner optimization involves solving Trace Ratio optimization (TRopt) using
    Self-Consistent-Field method (equivalent to Dinkelbach's algorithm).

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
    Dink_Val :  list
                List of inner standard trace ratio (Dinkelbach) values
    proj :      callable
                Projection function
    itr :       int
                Number of iterations
    DINK_PROJ : list
                List of inner trace ratio (Dinkelbach) projections matrices
    PROJ :      list
                List of projections matrices
    """

    mx = np.mean(DATA)
    DATA -= mx.reshape((1, -1))

    d = DATA.shape[1]   # data point dimension
    Sub_Err, Norm_Err, WDA_Val = [], [], []
    ITR = []
    Dink_Val = []
    DINK_PROJ = []
    PROJ = []

    X_c = sub.split_classes(DATA, LABEL) # split data by classes

    for it in range(maxitr):
        # obtain between scatter A:=C_b(P) and within scatter B:=C_w(P)
        A = np.zeros((d, d))
        B = np.zeros((d, d))
        for i, xi in enumerate(X_c):
            pxi = np.matmul(xi, P)
            for j, xj in enumerate(X_c[i:]):
                pxj = np.matmul(xj, P)
                M = sub.dist(pxi, pxj)
                K = np.exp(-lamb * M)
                u, v = sub.Acc_SK(K, 1e-5, k)
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
        NEW_P, Q, Err, Dink_Sub_Err, Dink_PROJ, itr = Dink_TR(A, B, p, P, tol=1e-3)
        Dink_Val += Q
        DINK_PROJ += Dink_PROJ
        PROJ.append(P)
        # errors
        Sub_Err.append(abs(linalg.subspace_angles(P, NEW_P)[0]))
        Norm_Err.append(np.linalg.norm(P - NEW_P, 'fro') / np.linalg.norm(NEW_P, 'fro'))
        # update
        P = NEW_P
        ITR.append(itr)
        # stopping criteria
        if it > 0:
            err = abs(WDA_Val[it] - WDA_Val[it - 1]) / abs(WDA_Val[it])
            if (Sub_Err[it] < tol) or (Norm_Err[it] < tol) or (err < 1e-5):
                PROJ.append(P)
                WDA_Val.append(np.trace(np.matmul(P.T, np.matmul(A, P))) / np.trace(np.matmul(P.T, np.matmul(B, P))))
                break

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(P)

    return P, Sub_Err, Norm_Err, WDA_Val, Dink_Val, proj, ITR, DINK_PROJ, PROJ
