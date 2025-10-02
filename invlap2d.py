# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:29:54 2022

@author: 356493
"""

import numpy as np, pylab as plt
from scipy.optimize import nnls


def exp_decay(amp, t, T2):
    return amp*np.exp(-t/T2)


def inv_rec(amp, t, T1):
    return amp*(1-2*np.exp(-t/T1))


def twoD_ILT_tk(data, t2vals, t1vals, lamb, ncomp, lgT2=[-1,2], lgT1= [-1,2], npts=[50,50]):
    """two dimensional inverse Laplace transform for INV_CPMG data sets
    could be generalized by changing the input functions
    calcuates the l2-norm regularized data
    follows data compression scheme of Venkataramanan et al 2002 to speed up
    inverion.  Still very slow

    does not optimize lambda or number of singular values

    Parameters
    ---------
    data : ndarray
       2D matrix of the INV_CPMG data points defaults to T1 along rows
       and T2 along columns
    t2vals : ndarray
       vector containing the T2 time points (ms)
    t1vals : ndarray
       vector containing the T1 time points (s)
    lamb : float
       value for the regularization parameter lamda
    ncomp : float
       number of singular values to use for SVD compression
    lgT2 : list
       lower and upper levels for T2 inversion (log(ms))
       default : [-1, 2]
    lgT2 : list
       lower and upper levels for T1 inversion (log(s))
       default : [-1, 2]
    npts : list
       number of points to be inverted for T2 and T1
       default : [50,50]
    """

    # generate the
    t2_inv = np.logspace(lgT2[0],lgT2[1],num=npts[0]).reshape(1,-1)
    t1_inv = np.logspace(lgT1[0],lgT1[1],num=npts[1]).reshape(1,-1)

    # generate kernal matrices and compress using SVD
    K2 = exp_decay(1, t2vals.reshape(-1,1), t2_inv)
    U2,s2,V2 = np.linalg.svd(K2)
    K2_red = np.dot(np.diag(s2[:ncomp]), V2[:ncomp])

    K1 = inv_rec(1, t1vals.reshape(-1,1), t1_inv)
    U1,s1,V1 = np.linalg.svd(K1)
    K1_red = np.dot(np.diag(s1[:ncomp]), V1[:ncomp])

    # compress the initial data using the SVD
    M = np.dot(np.dot(U2[:,:ncomp].T, data), U1[:,:ncomp])

    # generate matrix Knroker product matrix K
    K = np.kron(K2_red, K1_red)

    X2 = np.concatenate([K, np.sqrt(lamb)*np.eye(K.shape[1])])
    b = np.concatenate([M.flatten(), np.zeros(K.shape[1])])

    #generate non-negative result using NNLS
    inv, res = nnls(X2,b)

    return t2_inv[0], t1_inv[0], inv.reshape(npts)


def main():
    np.random.seed(10)
    npts = 500
    t2 = np.linspace(0,100, num=npts)
    #t1 = np.array([0.001, 0.050, 0.100, 0.200, 0.300, 0.500, 0.800, 1, 1.25, 1.5, 2, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 8, 9, 10, 20])
    t1 = np.logspace(-3, np.log10(20), 22)
    sim_t21 = exp_decay(50, t2, 30)
    sim_t11 = inv_rec(50, t1, 2)
    sim_t22 = exp_decay(50, t2, 1)
    sim_t12 = inv_rec(20, t1, 1)
    t1t2 = np.dot(sim_t21.reshape(-1, 1), sim_t11.reshape(1, -1)) + np.dot(sim_t22.reshape(-1, 1), sim_t12.reshape(1, -1)) + 20*np.random.normal(size = (npts, len(t1)))

    t1t2_flat = t1t2.flatten()

    npts = 50
    ncomp = 8
    t2_inv = np.logspace(-1, 2, num=npts).reshape(1, -1)
    t1_inv = np.logspace(-1, 2, num=npts).reshape(1, -1)
    K2 = exp_decay(1, t2.reshape(-1, 1), t2_inv)
    U2,s2,V2 = np.linalg.svd(K2)
    K2_red = np.dot(np.diag(s2[:ncomp]), V2[:ncomp])

    K1 = inv_rec(1, t1.reshape(-1, 1), t1_inv)
    U1,s1,V1 = np.linalg.svd(K1)
    K1_red = np.dot(np.diag(s1[:ncomp]), V1[:ncomp])

    M = np.dot(np.dot(U2[:,:ncomp].T, t1t2), U1[:,:ncomp])

    K = np.kron(K2_red, K1_red)


    f = np.ones((50, 50))
    test = np.dot(np.dot(K2, f), K1.T)
    err = test - t1t2

    t2_inv, t1_inv, inv = twoD_ILT_tk(t1t2, t2, t1, 0.5, 8)

    #"#"#"#"#"#"
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.contour(inv, extent=[t2[0], t2[-1], t1[-1], t1[0]])
    plt.subplot(3, 1, 2)
    plt.plot(t2_inv)
    plt.subplot(3, 1, 3)
    plt.plot(t1_inv)
    plt.show()
    #"#"#"#"#"#"


if __name__ == '__main__':
    main()
