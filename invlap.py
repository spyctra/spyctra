"""
Code derived from Harris Mason's cpmg_proc.py
2017-04-26

From Mason's code:

    determines regularization parameter using the l-curve method
    Harris Mason 4/26/2017

"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import nnls

def expDec(t, amp, T2):
    return amp*np.exp(-t/T2)


def compute_gcv2(X,y):
    Gamma = np.diag(np.random.uniform(size=X.shape[1])) #np.identity(X.shape[1])*np.random.uniform()
    A = np.dot(X.T, X) + np.dot(Gamma.T, Gamma)
    A_inv = np.linalg.inv(A)
    b_hat = np.dot(A_inv, np.dot(X.T, y))
    y_pred = np.dot(X, b_hat)
    mse = np.mean((y - y_pred)**2)
    n = len(y)
    h_value = np.trace(np.identity(n) - np.dot(X, np.dot(A_inv, X.T))) / n

    return mse / h_value**2


def laplaceInversion(x, y, T2):
    X = np.zeros((len(x), len(T2)))

    # create regularization matrix
    for j in range(len(T2)):
        X[:,j] = expDec(x, 1.0, T2[j])

    lam = compute_gcv2(X, y)
    print(lam)
    ## perform the final inversion using the fit regularization parameter above
    X2 = np.concatenate([X, np.sqrt(lam)*np.eye(len(T2))])
    b = np.concatenate([y, np.zeros(len(T2))])
    t, res = nnls(X2,b)  #final non-linear leastsquares fit

    return t


def invlap(x,y,T2):
    X = np.zeros((len(x), len(T2)))

    # create regularization matrix
    for j in range(len(T2)):
        X[:,j] = expDec(x, 1.0, T2[j])

    t = nnls(X,y)
    return t[0]



def main():
    from numpy.random import normal

    x = np.linspace(0,100,1024)
    y = np.zeros(len(x))
    y += expDec(x,100,.5)
    y += expDec(x,100,12)
    #y += normal(0,10,len(y))

    T2s = np.logspace(-2, 2, 512)
    t = laplaceInversion(x, y, T2s)
    #t = invlap(x, y, T2s)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x,y)
    plt.subplot(2,1,2)
    plt.plot(T2s,t)
    #plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
