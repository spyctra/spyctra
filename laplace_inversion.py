"""
    determines regularization parameter using generalized cross validation
    Harris Mason 10/1/2025

"""

import matplotlib.pyplot as plt
import numpy as np
from plot_defaults import button
from scipy.optimize import nnls


def exp_dec(t, amp, T2):
    "Exponential decay for T2 and T1rho measurements"
    return amp*np.exp(-t/T2)


def inv_rec(t, amp, T1):
    "T1 measurement via inversion recovery"
    return amp*(1-2*np.exp(-t/T1))


def sat_rec(t, amp, T1):
    "T1 measurement via saturation recovery"
    return amp*(1-np.exp(-t/T1))


def compute_gcv(X, y):
    Gamma = np.diag(np.random.uniform(size=X.shape[1]))
    A = np.dot(X.T, X) + np.dot(Gamma.T, Gamma)
    A_inv = np.linalg.inv(A)
    b_hat = np.dot(A_inv, np.dot(X.T, y))
    y_pred = np.dot(X, b_hat)
    mse = np.mean((y - y_pred)**2)
    n = len(y)
    h_value = np.trace(np.identity(n) - np.dot(X, np.dot(A_inv, X.T))) / n

    return mse / h_value**2


def laplace_inversion(x, y, T_es, kernel=exp_dec):
    """Inverse Laplace Transform of 1D relaxation data with non-negative constraint
       can provide different kernals or user defined kernals.  Default is exponential decay"""

    # create regularization matrix
    X = kernel(x.reshape(-1, 1), 1.0, T_es.reshape(1, -1))

    lam = compute_gcv(X, y)
    ## perform the final inversion using the fit regularization parameter above
    X2 = np.concatenate([X, np.sqrt(lam)*np.eye(len(T_es))])
    b = np.concatenate([y, np.zeros(len(T_es))])
    t_fits, res = nnls(X2, b)  #final non-negative least squares fit

    return t_fits


def laplace_inversion_2D(data, t2_times, t1_times, lambda0, ncomp, T_1s, T_2s):
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
    t2_times : ndarray
       vector containing the sampled T2 time points
    t1_times : ndarray
       vector containing the sampled T1 time points
    lambda0 : float
       value for the regularization parameter lamda
    ncomp : float
       number of singular values to use for SVD compression
    T_1s : ndarray
       vector containing potential T1 values in data
    T_2s : ndarray
       vector containing potential T2 values in data
    """

    # generate kernal matrices and compress using SVD
    K2 = exp_dec(t2_times.reshape(-1,1), 1, T_2s)
    U2, s2, V2 = np.linalg.svd(K2)
    K2_red = np.dot(np.diag(s2[:ncomp]), V2[:ncomp])

    K1 = inv_rec(t1_times.reshape(-1,1), 1, T_1s)
    U1, s1, V1 = np.linalg.svd(K1)
    K1_red = np.dot(np.diag(s1[:ncomp]), V1[:ncomp])

    # compress the initial data using the SVD
    M = np.dot(np.dot(U2[:,:ncomp].T, data), U1[:,:ncomp])

    # generate matrix Knroker product matrix K
    K = np.kron(K2_red, K1_red)

    X2 = np.concatenate([K, np.sqrt(lambda0)*np.eye(K.shape[1])])
    b = np.concatenate([M.flatten(), np.zeros(K.shape[1])])

    #generate non-negative result using NNLS
    inv, res = nnls(X2, b)

    return inv.reshape([len(T_2s), len(T_1s)])


def test_suite_1D():
    from numpy.random import normal

    #create some fake data with known decay constants
    x = np.logspace(-2, 2, 32)
    y = np.zeros(len(x))
    y += exp_dec(x, 100, .5)
    y += exp_dec(x, 100, 12)
    y += normal(0, 2, len(y)) #add some noise since we fear not reality

    T_es = np.logspace(-2, 2, 1024) #the possible decay constants our data might contain

    t_fits = laplace_inversion(x, y, T_es)

    #"#"#"#"#"#"
    plt.figure(figsize=(16,9))

    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('data')

    plt.subplot(2, 1, 2)
    plt.plot(T_es, t_fits)
    plt.xlabel('t_e')
    plt.ylabel('amplitude')
    plt.xscale('log')
    plt.show()
    #"#"#"#"#"#"


def test_suite_2D():
    #create fake T1 and T2 data
    np.random.seed(10)
    npts = 500
    t2 = np.linspace(0,100, num=npts)
    t1 = np.logspace(-3, np.log10(20), 22)
    sim_t21 = exp_dec(t2, 50, 30)
    sim_t11 = inv_rec(t1, 50, 2)
    sim_t22 = exp_dec(t2, 50, 1)
    sim_t12 = inv_rec(t1, 20, 1)
    t1t2 = np.dot(sim_t21.reshape(-1, 1), sim_t11.reshape(1, -1)) + np.dot(sim_t22.reshape(-1, 1), sim_t12.reshape(1, -1)) + 20*np.random.normal(size = (npts, len(t1)))

    t1t2_flat = t1t2.flatten()

    npts = 50
    ncomp = 8
    t2_inv = np.logspace(-1, 2, num=npts).reshape(1, -1)
    t1_inv = np.logspace(-1, 2, num=npts).reshape(1, -1)
    K2 = exp_dec(t2.reshape(-1, 1), 1, t2_inv)
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

    #determine which decay constants to seach for
    T_1s = np.logspace(-1, 2, 25)
    T_2s = np.logspace(-1, 2, 50)

    #perform the 2d laplace inversion
    inv = laplace_inversion_2D(t1t2, t2, t1, 0.5, 8, T_1s, T_2s)

    #"#"#"#"#"#"
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.contour(inv)
    plt.xlabel('T_1s index')
    plt.ylabel('T_2s index')
    plt.subplot(3, 1, 2)
    plt.plot(T_2s)
    plt.xlabel('T_2s index')
    plt.ylabel('T_2 value')
    plt.subplot(3, 1, 3)
    plt.plot(T_1s)
    plt.xlabel('T_1s index')
    plt.ylabel('T_1 value')
    plt.tight_layout()
    plt.show()
    #"#"#"#"#"#"


def main():
    test_suite_1D()
    test_suite_2D()


if __name__ == '__main__':
    main()

