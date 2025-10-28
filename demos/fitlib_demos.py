import sys
sys.path.append('../')

from fitlib import fit
import function_library as fl

from spyctra import spyctra, fake_spyctra
from numpy.random import uniform, seed, RandomState
import numpy as np

guess = 1
check = 1

def make_data(amps,t2s, dfs, phis, noise=100, points=1024):
    a = spyctra()

    for i in range(len(t2s)):
        a.add(fake_spyctra(t_2=t2s[i], points=points, delta=1e-5, df=dfs[i], amp=amps[i], phi=phis[i], seed=i, noise=noise))

    return a


def single_x_single_y_r():
    RandomState(0)
    traces = 1

    a = make_data([2000]*traces, [3e-3]*traces, [100]*traces, [1]*traces)

    a.resize(16384)
    a.fft()
    a.phase(a.get_phi())
    a.resize([-1000,1000])

    p,r = fit(fl.lorentzian, a.x, a.data[0].real,
              [ 10
               ,10
               ,1
               ],
               guess=guess, check=check, note='fitlibTest singleXsingleY_R')


def single_x_single_y_c():
    traces = 1

    a = make_data([2000]*traces, [3e-3]*traces, [100]*traces, [1]*traces)

    p,r = fit(fl.comp_exp_dec, a.x, a.data[0],
              [ 2500
               ,2e-2
               ,-100
               ,1
               ],
               guess=guess, check=check, note='fitlibTest singleXsingleY_C')


def single_x_multiple_y_r():
    traces = 3
    amps = uniform(1500, 2500, traces)
    phis = uniform(0,1, traces)

    a = make_data(amps, [3e-3]*traces, [100]*traces, phis)

    a.resize(16384)
    a.fft()
    a.phase(phis)
    a.resize([-1000,1000])

    p,r = fit(fl.lorentzian, a.x, [d.real for d in a.data],
              [ [10]*a.count
               ,[5]*a.count
               ,[0.5]*a.count
               ],
               guess=guess, check=check, note='fitlibTest singleXmultipleY_R')



def single_x_multiple_y_c():
    traces = 3
    amps = uniform(1500, 2500, traces)
    t2s = uniform(2e-2, 3e-4, traces)
    dfs = uniform(-500, 500, traces)
    phis = uniform(0,1, traces)

    a = make_data(amps, t2s, dfs, phis)

    p,r = fit(fl.comp_exp_dec, a.x, a.data,
              [ [200]*a.count
               ,[3e-3]*a.count
               ,[0]*a.count
               ,[0.5]*a.count
               ],
              guess=guess, check=check, note='fitlibTest singleXmultipleY_C')



def single_x_global_y_r():
    traces = 6
    amps = uniform(1500, 2500, traces)
    phis = uniform(0,1, traces)

    a = make_data(amps, [3e-3]*traces, [100]*traces, phis)

    a.resize(16384)
    a.fft()
    a.phase(phis)
    a.resize([-1000,1000])

    p,r = fit(fl.lorentzian, a.x, [d.real for d in a.data],
              [ [50]*a.count
               ,10
               ,50
               ],
               guess=guess, check=check, note='fitlibTest single_x_global_y_r')


def single_x_global_y_r2():
    traces = 60
    amps = [2000]*traces
    t2s = [3e-3]*traces
    dfs = [10]*traces
    phis = [0]*traces

    a = make_data(amps, t2s, dfs, phis,noise=0)

    a.resize(16384*4)
    a.fft()
    a.phase(phis)
    a.resize([-500,500])

    p,r = fit(fl.lorentzian, a.x, [d.real for d in a.data],
              [ 10
               ,6.5
               ,10
               ],
               guess=guess, check=check, note='fitlibTest single_x_global_y_r2')



def single_x_global_y_c():
    traces = 5
    amps = uniform(1500, 2500, traces)
    t2s = uniform(2e-3, 2e-3, traces)
    dfs = uniform(-500, 500, traces)
    phis = uniform(1,1, traces)

    a = make_data(amps, t2s, dfs, phis, noise=100, points= 512)
    b = a.copy()
    b.fft()

    p,r = fit(fl.comp_exp_dec, a.x, a.data,
              [ [2000]*traces
               ,3e-3
               ,[0]*traces
               ,0.5
               ],
               guess=guess, check=check, result='a,t2,df,phi', note='fitlibTest single_x_global_y_c')



def fit_error_worker():
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    ys = [x**2,1+x**2]
    ys[0][8]=0
    ys[1][9]=0
    errs = [[1,1,1,1,1,1,1,1,5,1],[1,1,1,1,1,1,1,1,1,5]]


    def parabola(x, a,b,c):
        return a*x**2 + b*x + c


    p,r = fit(parabola, x, ys,
              [ [1.0]*1
               ,[0.0]*len(ys)
               ,[0.0]*len(ys)
               ],
               guess=guess, check=check, result='a,b,c'
              ,note='fitlibTest fitErrorWorker_R')


    p,r = fit(parabola, x, ys,
              [ [1.0]*1
               ,[0.0]*len(ys)
               ,[0.0]*len(ys)
               ],
               guess=guess, check=check, result='a,b,c'
              ,sigma=errs,note='fitlibTest fitErrorWorker_with_sigmas_R')

    x = np.linspace(0, 0.5, 8)
    ys = fl.comp_exp_dec(x, 10, 1, 1, 2)

    ys[1] += 20-1j*20

    from numpy.random import normal

    errs = normal(0,4,len(x)) + 1j*normal(0,4,len(x))
    ys += errs
    errs[1] = 10 + 1j*10

    errs = np.abs(errs.real) + 1j*np.abs(errs.imag)

    p,r = fit(fl.comp_exp_dec, x, ys,
              [ 5
               ,1
               ,1
               ,1
               ],
               guess=guess, check=check, result='a, f, phi, te'
              ,note='fitlibTest fitErrorWorker_C')


    p,r = fit(fl.comp_exp_dec, x, ys,
              [ 5
               ,1
               ,1
               ,1
               ],
               guess=guess, check=check, result='a, f, phi, te'
              ,sigma=errs, note='fitlibTest fitErrorWorker_C')


def main():
    single_x_single_y_r()
    single_x_single_y_r()
    single_x_single_y_c()
    single_x_multiple_y_r()
    single_x_multiple_y_c()
    single_x_global_y_r()
    single_x_global_y_r2()
    single_x_global_y_c()
    fit_error_worker()


if __name__ == '__main__':
    main()
