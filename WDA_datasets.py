"""
=============================================
Wasserstein Discriminant Analysis (WDA)
Datasets shared by WDAgd, WDAeig, WDAnepv
=============================================
"""

# Author: Dong Min Roh <droh@ucdavis.edu>
#

import numpy as np
import os
import scipy.io

###########################################################################


def load_synth(d, n_tr, n_tst):
    """
    Generates data
      3 classes: X, Y, Z
      30% X, 40% Y, 30% Z
      Discriminative behavior for the first two components s.t. these components in
          X centers around (-5,0) and (5,0)
          Y centers around (-3,3) and (3,-3)
          Z centers around (-3,-3) and (3,3)
      using normal (Gaussian) distribution with standard deviation 0.5
      rest of d - 2 components are drawn from standard normal distribution

    PARAMETERS
    ----------
    n_tr :  int
            Number of training data points
    n_tst : int
            Number of testing data points
    d :     int
            Dimension of data points

    RETURNS
    -------
    TR :    ndarray, shape (n_tr, d)
            Training data
    TR_L :  ndarray, shape (n_tr,)
            Training label
    TST :   ndarray, shape (n_tst, d)
            Testing data
    TST_L : ndarray, shape (n_tst,)
            Testing label
    """

    # Training datasets
    n_X = int(np.floor(n_tr * 0.3))
    TR_X1 = np.hstack((np.random.normal(-5, 0.5, (int(n_X / 2), 1)), np.random.normal(0, 0.5, (int(n_X / 2), 1))))
    TR_X2 = np.hstack((np.random.normal(5, 0.5, (int(n_X / 2), 1)), np.random.normal(0, 0.5, (int(n_X / 2), 1))))
    TR_X = np.vstack((TR_X1, TR_X2))

    n_Y = int(np.floor(n_tr * 0.4))
    TR_Y1 = np.hstack((np.random.normal(-3, 0.5, (int(n_Y / 2), 1)), np.random.normal(3, 0.5, (int(n_Y / 2), 1))))
    TR_Y2 = np.hstack((np.random.normal(3, 0.5, (int(n_Y / 2), 1)), np.random.normal(-3, 0.5, (int(n_Y / 2), 1))))
    TR_Y = np.vstack((TR_Y1, TR_Y2))

    n_Z = int(np.floor(n_tr * 0.3))
    TR_Z1 = np.hstack((np.random.normal(-3, 0.5, (int(n_Z / 2), 1)), np.random.normal(-3, 0.5, (int(n_Z / 2), 1))))
    TR_Z2 = np.hstack((np.random.normal(3, 0.5, (int(n_Z / 2), 1)), np.random.normal(3, 0.5, (int(n_Z / 2), 1))))
    TR_Z = np.vstack((TR_Z1, TR_Z2))

    TR = np.vstack((TR_X, TR_Y, TR_Z))
    TR = np.hstack((TR, np.random.randn(n_tr, d - 2)))
    TR_L = np.concatenate((np.ones(n_X) * 1, np.ones(n_Y) * 2, np.ones(n_Z) * 3))

    TR = (TR - np.mean(TR, 0)) / np.std(TR, 0)  # standardize

    # Testing datasets
    n_X = int(np.floor(n_tst * 0.3))
    TST_X1 = np.hstack((np.random.normal(-5, 0.5, (int(n_X / 2), 1)), np.random.normal(0, 0.5, (int(n_X / 2), 1))))
    TST_X2 = np.hstack((np.random.normal(5, 0.5, (int(n_X / 2), 1)), np.random.normal(0, 0.5, (int(n_X / 2), 1))))
    TST_X = np.vstack((TST_X1, TST_X2))

    n_Y = int(np.floor(n_tst * 0.4))
    TST_Y1 = np.hstack((np.random.normal(-3, 0.5, (int(n_Y / 2), 1)), np.random.normal(3, 0.5, (int(n_Y / 2), 1))))
    TST_Y2 = np.hstack((np.random.normal(3, 0.5, (int(n_Y / 2), 1)), np.random.normal(-3, 0.5, (int(n_Y / 2), 1))))
    TST_Y = np.vstack((TST_Y1, TST_Y2))

    n_Z = int(np.floor(n_tst * 0.3))
    TST_Z1 = np.hstack((np.random.normal(-3, 0.5, (int(n_Z / 2), 1)), np.random.normal(-3, 0.5, (int(n_Z / 2), 1))))
    TST_Z2 = np.hstack((np.random.normal(3, 0.5, (int(n_Z / 2), 1)), np.random.normal(3, 0.5, (int(n_Z / 2), 1))))
    TST_Z = np.vstack((TST_Z1, TST_Z2))

    TST = np.vstack((TST_X, TST_Y, TST_Z))
    TST = np.hstack ((TST, np.random.randn(n_tst, d - 2)))
    TST_L = np.concatenate((np.ones(n_X) * 1, np.ones(n_Y) * 2, np.ones(n_Z) * 3))

    TST = (TST - np.mean(TST, 0)) / np.std(TST, 0)  # standardize

    return TR, TR_L, TST, TST_L

def load_shape(name):
    """
    Generates data
      2D shape datasets from http://cs.joensuu.fi/sipu/datasets/
      8 additional components drawn from standard normal distribution are appended
      Then, datasets are standardized
      50% training and 50% testing by random split

    PARAMETERS
    ----------
    name :  str
            Name of data set

    RETURNS
    -------
    TR :    ndarray, shape (n_tr, 10)
            Training data
    TR_L :  ndarray, shape (n_tr,)
            Training label
    TST :   ndarray, shape (n_tst, 10)
            Testing data
    TST_L : ndarray, shape (n_tst,)
            Testing label
    """

    # Load data
    os.chdir('/content/drive/My Drive/MAC/Matlab')
    txt = np.loadtxt(name + '.txt')
    data, label = txt[:, 0:2], txt[:, 2]
    n = data.shape[0]

    # Append additional components & standardize
    data = np.hstack((data, np.random.randn(n, 8)))
    data = (data - np.mean(data, 0)) / np.std(data, 0)  # standardize

    # Split data evenly between training and testing
    half = int(np.ceil(n / 2))
    rand_ind = np.arange(n)
    np.random.shuffle(rand_ind)
    TR_rand_ind = rand_ind[0:half]
    TST_rand_ind = rand_ind[half:]
    TR = data[TR_rand_ind, :]
    TR_L = label[TR_rand_ind]
    TR_L = TR_L.reshape(-1)
    TST = data[TST_rand_ind, :]
    TST_L = label[TST_rand_ind]
    TST_L = TST_L.reshape(-1)

    return TR, TR_L, TST, TST_L

def load_uci(name):
    """
    Generates data
      UCI datasets from https://archive.ics.uci.edu/ml/index.php
      Consider datasets are
                Wine, Iris, Vehicle, Credit, Ionosphere, LSVT
      Except for the LSVT dataset, 100 additional components drawn from Gaussian distribution N(1, 2) are appended
      50% training and 50% testing by random split

    PARAMETERS
    ----------
    name :  str
            Name of data set

    RETURNS
    -------
    TR :    ndarray, shape (n_tr, d)
            Training data
    TR_L :  ndarray, shape (n_tr,)
            Training label
    TST :   ndarray, shape (n_tst, d)
            Testing data
    TST_L : ndarray, shape (n_tst,)
            Testing label
    """

    # Load data
    os.chdir('/content/drive/My Drive/MAC/Matlab')
    mat = scipy.io.loadmat(name + '.mat')
    data, label = mat[name + '_data'], mat[name + '_label']
    n = data.shape[0]

    if name == 'ionosphere':
        data = data[:, 2:]

    # Append additional components & standardize
    data = np.hstack((data, np.sqrt(2) * np.random.randn(n, 100) + 1))
    data = (data - np.mean(data, 0)) / np.std(data, 0)  # standardize

    # Split data evenly between training and testing
    half = int(np.ceil(n / 2))
    rand_ind = np.arange(n)
    np.random.shuffle(rand_ind)
    TR_rand_ind = rand_ind[0:half]
    TST_rand_ind = rand_ind[half:]
    TR = data[TR_rand_ind, :]
    TR_L = label[TR_rand_ind]
    TR_L = TR_L.reshape(-1)
    TST = data[TST_rand_ind, :]
    TST_L = label[TST_rand_ind]
    TST_L = TST_L.reshape(-1)

    return TR, TR_L, TST, TST_L
