from spyctra import spyctra
import matplotlib.pyplot as plt
from TNT import read

from fitlib import fit
from function_library import comp_exp_dec, time_voigt

from math import e, pi


def main():
    """
    a = read('/home/mwmalone/code/spyctraRep/TNT/exp1_385/FID_', 85)

    a.decimate()
    a.save('./temp.p')
    #"""
    a = spyctra('./temp.p')
    a.shift(1)
    b = a.copy()
    #b.exp_mult(300)
    b.resize(2048)
    b.fft()
    b.resize([-1000,1000])
    b.phase()

    dfs = b.find_df()
    phis = b.phi
    peaks = b.find_peak()[1]
    lws = b.find_linewidth()

    p, d = fit(time_voigt, a.x, a.data,
              [ [4]*a.count
               ,1/lws/4
               ,1/lws/4
               ,dfs
               ,pi*3/2
               ,-1e-3
              ]
              ,guess=1, check=1, result='A, s, t_e, df, phi, x0', epsfcn=0.001)


    exit()


    p, d = fit(comp_exp_dec, a.x, a.data,
              [ [8]*a.count
               ,1/lws/2
               ,dfs
               ,phis
              ]
              ,guess=0, check=1, result='a, t_2, df, phi')

    print(d['df'])

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(dfs)
    plt.plot(d['df'])
    plt.subplot(2,1,2)
    plt.plot(a.freq + dfs)
    plt.plot(a.freq + d['df'])
    plt.show()


if __name__ == '__main__':
    main()
