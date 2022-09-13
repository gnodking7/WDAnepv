"""
=============================================
Wasserstein Discriminant Analysis (WDA)
Gradient Descent based Algorithm: WDAgd
From:
@article{flamary2018wasserstein,
  title={Wasserstein discriminant analysis},
  author={Flamary, R{\'e}mi and Cuturi, Marco and Courty, Nicolas and Rakotomamonjy, Alain},
  journal={Machine Learning},
  volume={107},
  number={12},
  pages={1923--1945},
  year={2018},
  publisher={Springer}
}
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

def wda_gd(X, y, p, reg, P0, k=10, maxiter=100, verbosity=0):
    """
    Wasserstein Discriminant Analysis

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
    p : int
        Size of dimensionnality reduction.
    reg : float
        Regularization term >0 (entropic regularization)
    P0 : ndarray, shape (d, p)
        Initial starting point for projection.
    k : int, optional, default set to 10
        Number of Acc_SK iterations
    verbose : int, optional
        Print information along iterations.

    RETURNS
    -------
    Popt : ndarray, shape (d, p)
        Optimal transportation matrix for the given parameters
    proj : callable
        Projection function including mean centering.
    Itr : int
          Number of WDAgd iterations
    PROJ : list
           List of ndarray projections

    REMARKS
    -------
    Original code from:
    https://pythonot.github.io/_modules/ot/dr.html#wda
    is outdated with respect to the current version of Pymanopt (2.0.1)
    The following codes are written in accordance with the latest Pymanopt package.
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
                u, v, Err = sub.Acc_SK(K, 1e-5, k)
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
