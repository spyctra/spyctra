from math import e, pi

from fitlib import fit
from function_library import comp_exp_dec
from spyctra import spyctra, fake_spyctra

import matplotlib.pyplot as plt
import numpy as np


def advanced_fitting():
    a = spyctra()
    from numpy.random import seed


    for i in range(10):
        a.add(fake_spyctra(amp=512, t_2=(i+1)*3e-3, points=128, df=10*i, phi=1+i*0.1, noise=800, delta=1e-5, seed=i))


    def processor(b):
        points = 16384
        lw = 120
        f0 = -1000
        f1 = 1000

        b.normalize(0.0001)
        b.exp_mult(lw)
        b.resize(points)
        b.fft()
        b.resize([f0,f1])

        return b


    b = a.copy()
    b = processor(b)

    dfs = b.find_df()
    phis = b.find_phi()
    lws = b.find_linewidth()
    x = a.delta*np.arange(a.points)


    def fitter(x0, amp, t_2, df, phi):
        y = comp_exp_dec(x, amp, t_2, df, phi)
        b = spyctra( data=[y]
                    ,delta=1e-5)

        b = processor(b)

        return b.data[0]


    p, d = fit(fitter, b.x, b.data,
              [ [512]*b.count
               ,1/2/lws
               ,dfs
               ,phis
              ]
              ,guess=0, check=0, result='a, t_2, df, phi')

    #"""
    from result import result
    d = result(d)

    plt.figure()
    plt.subplot(2,2,1)
    plt.errorbar(np.arange(b.count), d['a'], d['a_err'])
    plt.subplot(2,2,2)
    #plt.errorbar(np.arange(b.count), d['t_2'], d['t_2_err'])
    plt.plot(np.arange(b.count), d['t_2'])
    plt.subplot(2,2,3)
    plt.errorbar(np.arange(b.count), d['df'], d['df_err'])
    plt.plot([10*i for i in range(b.count)])
    plt.plot(dfs)
    plt.subplot(2,2,4)
    plt.errorbar(np.arange(b.count), np.array(d['phi'])%pi, np.array(d['phi_err']))
    plt.plot(phis)
    plt.show()
    #"""


def main():
    advanced_fitting()


if __name__ == '__main__':
    main()
