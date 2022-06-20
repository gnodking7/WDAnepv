"""
=============================================
Wasserstein Discriminant Analysis (WDA)
Gradient Descent based Algorithm: WDAgd
=============================================
"""

# Author: Dong Min Roh <droh@ucdavis.edu>
#

import autograd.numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import WDA_subfunc as sub

###########################################################################

def wda_gd(X, y, p=2, reg=1, k=10, maxiter=100, P0=None, verbosity=0):
    """
    Wasserstein Discriminant Analysis :ref:`[11] <references-wda>`

    The function solves the following optimization problem:

    .. math::
        \mathbf{P} = \mathop{\arg \min}_\mathbf{P} \quad
        \frac{\sum\limits_i W(P \mathbf{X}^i, P \mathbf{X}^i)}{\sum\limits_{i, j \neq i} W(P \mathbf{X}^i, P \mathbf{X}^j)}

    where :

    - :math:`P` is a linear projection operator in the Stiefel(`p`, `d`) manifold
    - :math:`W` is entropic regularized Wasserstein distances
    - :math:`\mathbf{X}^i` are samples in the dataset corresponding to class i

    PARAMETERS
    ----------
    X : ndarray, shape (n, d)
        Training samples.
    y : ndarray, shape (n,)
        Labels for training samples.
    p : int, optional
        Size of dimensionnality reduction.
    reg : float, optional
        Regularization term >0 (entropic regularization)
    k : int, optional, default set to 10
        Number of Sinkhorn iterations
    P0 : ndarray, shape (d, p)
        Initial starting point for projection.
    verbose : int, optional
        Print information along iterations.

    RETURNS
    -------
    P : ndarray, shape (d, p)
        Optimal transportation matrix for the given parameters
    proj : callable
        Projection function including mean centering.
    Itr : int
          Number of WDAgd iterations
    PROJ : list
           List of ndarray projections

    .. _references-wda:
    REFERENCES
    ----------
    .. [11] Flamary, R., Cuturi, M., Courty, N., & Rakotomamonjy, A. (2016).
            Wasserstein Discriminant Analysis. arXiv preprint arXiv:1608.08063.

    REMARKS
    -------
    The original codes provided by the authors of the reference is outdated.
    The following codes are written in accordance with the latest Pymanopt package.
        - Dong Min Roh -
        June, 2022
    """

    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = sub.split_classes(X, y)

    manifold = pymanopt.manifolds.Stiefel(d, p)

    @pymanopt.function.autograd(manifold)
    def cost(P):
        loss_b = 0
        loss_w = 0
        for i, xi in enumerate(xc):
            pxi = np.matmul(xi, P)
            for j, xj in enumerate(xc[i:]):
                pxj = np.matmul(xj, P)
                M = sub.dist(pxi, pxj)
                K = np.exp(-reg * M)
                u, v = sub.Acc_SK(K, 1e-5, k)
                T = u.reshape((K.shape[0], 1)) * K * v.reshape((1, K.shape[1]))
                if j == 0:
                    loss_w += np.sum(T * M)
                else:
                    loss_b += np.sum(T * M)
        # loss inversed because minimization
        return loss_w / loss_b

    problem = pymanopt.Problem(manifold, cost)

    # Steepest Descent
    optimizer = pymanopt.optimizers.SteepestDescent(verbosity=verbosity, log_verbosity=2)
    result = optimizer.run(problem, initial_point=P0)

    Popt = result.point
    Itr = result.iterations
    PROJ = result.log["iterations"]["point"]

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt)

    return Popt, proj, Itr, PROJ
