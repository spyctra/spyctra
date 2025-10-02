"""
    determines regularization parameter using generalized cross validation
    Harris Mason 10/1/2025

"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import nnls


def expDec(t, amp, T2):
    "Exponential decay for T2 and T1rho measurements"
    return amp*np.exp(-t/T2)


def invRec(t, amp, T1):
    "T1 measurement via inversion recovery"
    return amp*(1-2*np.exp(-t/T1))


def satRec(t, amp, T1):
    "T1 measurement via saturation recovery"
    return amp*(1-np.exp(-t/T1))


def compute_gcv(X,y):
    Gamma = np.diag(np.random.uniform(size=X.shape[1]))
    A = np.dot(X.T, X) + np.dot(Gamma.T, Gamma)
    A_inv = np.linalg.inv(A)
    b_hat = np.dot(A_inv, np.dot(X.T, y))
    y_pred = np.dot(X, b_hat)
    mse = np.mean((y - y_pred)**2)
    n = len(y)
    h_value = np.trace(np.identity(n) - np.dot(X, np.dot(A_inv, X.T))) / n

    return mse / h_value**2


def laplaceInversion(x, y, tlim=[-2,2], npts=200, kernal=expDec):
    """Inverse Laplace Transform of 1D relaxation data with non-negative constraint
       can provide different kernals or user defined kernals.  Default is exponential decay"""

    T = np.logspace(tlim[0], tlim[1], npts)
    # create regularization matrix
    X = kernal(x.reshape(-1,1), 1.0, T.reshape(1,-1))

    lam = compute_gcv(X, y)
    ## perform the final inversion using the fit regularization parameter above
    X2 = np.concatenate([X, np.sqrt(lam)*np.eye(len(T))])
    b = np.concatenate([y, np.zeros(len(T))])
    t, res = nnls(X2, b)  #final non-linear leastsquares fit

    return T, t


def invlap(x,y,T2):
    # create regularization matrix
    X = expDec(x.reshape(-1,1), 1.0, T2.reshape(1,-1))
    t = nnls(X, y)

    return t[0]


def main():
    from numpy.random import normal

    x = np.linspace(0, 100, 1024)
    y = np.zeros(len(x))
    y += expDec(x, 100, .5)
    y += expDec(x, 100, 12)
    #y += normal(0,10,len(y))

    T_2s, t = laplaceInversion(x, y)
    #t = invlap(x, y, T2s)


    #"#"#"#"#"#"
    plt.figure(figsize=(16,9))

    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('signal')

    plt.subplot(2, 1, 2)
    plt.plot(T_2s, t)
    plt.xlabel('t_2')
    plt.ylabel('inverse laplace')
    plt.xscale('log')
    plt.show()
    #"#"#"#"#"#"


if __name__ == '__main__':
    main()

