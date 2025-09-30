"""
Test suite of spyctra operations. Used for debugging.
"""

import numpy as np
from time import time
from numpy.random import randint, seed, uniform
from result import result

from math import e, pi
import function_library as fl
import matplotlib.pyplot as plt

"""
CHANGE LOG

2025-09-29 added spyctra_test_suite_4() to cover negative shifts, array inputs for exp_mult, shift, phase
2025-09-27 the great get_ overhaul
2025-09-14 Initial release
"""

def spyctra_test_suite_0():
    from spyctra import spyctra, fake_spyctra

    seed(0)
    obs = 64
    pts = 2**14
    t2s = uniform(-6, 0.3, obs)
    t2s = 10**t2s
    dfs = randint(-250, 250, obs)

    a = spyctra()

    for i in range(len(t2s)):
        a.add(fake_spyctra( pts
                          ,t_2=t2s[i]
                          ,df=dfs[i]
                          ,phi=i
                          ,seed=i
                          ,noise=t2s[i]*100
                          ,meta={'i':[i]}
                          ,delta=100e-6))

    b = spyctra()

    for i in range(len(t2s)):
        b.add(fake_spyctra( pts
                          ,t_2=t2s[i]*3
                          ,df=dfs[i]*2
                          ,phi=i*2
                          ,seed=i*2
                          ,noise=t2s[i]*110
                          ,meta={'i':[i**2]}
                          ,delta=100e-6))

    a.add(b)
    a.add(a.copy(0,1,2,3))
    a.resize(a.points*4)
    a.fft()
    a.phase(2)
    a.resize([-1000,1000])

    res = result()
    res['freq'] = a.get_freq()
    res['LW'] = a.get_linewidth()
    res['LW_R'] = a.get_linewidth('R')
    res['LW_I'] = a.get_linewidth('I')
    res['LW_M'] = a.get_linewidth('M')
    res['noise'] = a.get_noise()
    res['noise8'] = a.get_noise(8)
    res['DF'] = a.get_df()
    res['offset'] = a.get_offset()
    res['offset4'] = a.get_offset(4)

    peaks, vals = a.get_peak()
    peaks_r, vals_r = a.get_peak('R')
    peaks_i, vals_i = a.get_peak('I')
    peaks_m, vals_m = a.get_peak('M')

    res['peak'] = peaks
    res['val'] = vals
    res['peak_r'] = peaks_r
    res['val_r'] = vals_r
    res['peak_i'] = peaks_i
    res['val_i'] = vals_i
    res['peak_m'] = peaks_m
    res['val_m'] = vals_m

    res['phase'] = a.get_phi()
    res['SNR'] = a.get_snr()
    res['SNR2'] = a.get_snr(a.points//2)
    res['peak_R'] = a.get_point(a.get_peak()[0], 'R')
    res['peak_I'] = a.get_point(a.get_peak()[0], 'I')
    res['peak_M'] = a.get_point(a.get_peak()[0], 'M')
    res['point_R'] = a.get_point(a.points//2, 'R')
    res['point_I'] = a.get_point(a.points//2, 'I')
    res['point_M'] = a.get_point(a.points//2, 'M')
    res['dataSumCheck'] = [np.sum(np.abs(d)) for d in a.data]

    res.pop(1)
    res.pop([1,2])
    res.print_ind()


def spyctra_test_suite_1():
    from spyctra import spyctra, fake_spyctra

    a = spyctra()

    for i in range(96):
        a.add(fake_spyctra( points=64*1800
                           ,t_2=(1+i)/2
                           ,df=i
                           ,phi=i
                           ,seed=i
                           ,noise=10+i*10
                           ,meta={'i':[i**2]}))

    a.decimate(4)
    a.fft()
    res = result()
    res['pos'] = a.get_peak()[0]
    res['peak'] = a.get_peak()[1]
    res.print_ind()


def spyctra_test_suite_2():
    from spyctra import spyctra, fake_spyctra

    a = spyctra()

    for i in range(9):
        a.add(fake_spyctra( points=64*1800
                           ,t_2=(1+i)/2
                           ,df=i
                           ,phi=i
                           ,seed=i
                           ,noise=10+i*10
                           ,meta={'i':[i**2]}))

    a.fft()
    res = result()
    res['R_0'] = a.integrate('R')
    ints = a.integrate('RI')
    res['R_1'] = ints[0]
    res['I_1'] = ints[1]
    ints = a.integrate('RIM')
    res['R_2'] = ints[0]
    res['I_2'] = ints[1]
    res['M_2'] = ints[2]

    phase_cors_0 = [3,10,2]
    a.phase_foc(phase_cors_0)
    res['R_3'] = a.integrate('R')
    ints = a.integrate('RI')
    res['R_4'] = ints[0]
    res['I_4'] = ints[1]
    ints = a.integrate('RIM')
    res['R_5'] = ints[0]
    res['I_5'] = ints[1]
    res['M_5'] = ints[2]


    phase_cors = [[phase_cors_0[0]+i,phase_cors_0[0]+i,phase_cors_0[0]+i] for i in range(a.count)]
    a.phase_foc(phase_cors)
    res['R_6'] = a.integrate('R')
    ints = a.integrate('RI')
    res['R_7'] = ints[0]
    res['I_7'] = ints[1]
    ints = a.integrate('RIM')
    res['R_8'] = ints[0]
    res['I_8'] = ints[1]
    res['M_8'] = ints[2]

    res.print_ind()


def spyctra_test_suite_3():
    from spyctra import spyctra, fake_spyctra

    amp = 512
    pts = 2048
    seed(0)

    a = spyctra()

    for i in range(96):
        t = fake_spyctra(meta={'i':[i**2]},seed=i)
        t.data = [randint(-amp, amp, pts)+1j*randint(-amp, amp, pts)]
        a.add(t)

    b = spyctra()

    for i in range(4):
        t = fake_spyctra(meta={'i':[i**2]},seed=i)
        t.data = [randint(-amp, amp, pts)+1j*randint(-amp, amp, pts)]
        b.add(t)

    a.add(b)
    a.sort(a.meta['i'])
    a.add(a.copy())
    a.sort(a.meta['i'])
    a.add(a.copy(0))
    a.sort(a.meta['i'])
    a.add(a.copy([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]))
    a.sort(a.meta['i'])
    a.decimate(4)
    a.report()
    b.decimate()
    a.exp_mult(3.14)
    a.exp_mult([i/100 for i in range(a.count)])
    a.resize(a.points*2)
    a.fft()
    a.fft()
    b.fft(divide=1)
    b.fft(rezero=0)
    a.shift(4)
    a.shift([i for i in range(a.count)])
    a.new_count(a.count*8)
    a.decimate(2)
    a.new_count(a.count//2)
    a.subtract(1.3)
    a.normalize()
    a.normalize(1/10)
    a.normalize([i+1 for i in range(a.count)])
    a.phase()
    a.phase(10)
    a.phase([i for i in range(a.count)])
    a.pop(0)
    a.pop([0,1,2])
    a.resize(a.points*2)
    a.resize(a.points//3)
    a.resize([a.start+a.delta*(a.points//10), a.start+a.delta*int(a.points*0.6)])
    a.smooth(2)
    a.subtract(1)
    a.subtract(a.get_offset())
    a.transpose()
    a.transpose()
    a.resize(a.points*64)
    a.fft()
    a.decimate(3)
    a.resize([a.start+a.delta*(a.points//10), a.start+a.delta*int(a.points*0.6)])
    a.report()

    res = result()
    res['freq'] = a.get_freq()
    res['LW'] = a.get_linewidth()
    res['LW_R'] = a.get_linewidth('R')
    res['LW_I'] = a.get_linewidth('I')
    res['LW_M'] = a.get_linewidth('M')
    res['noise'] = a.get_noise()
    res['noise8'] = a.get_noise(8)
    res['DF'] = a.get_df()
    res['offset'] = a.get_offset()
    res['offset4'] = a.get_offset(4)

    peaks, vals = a.get_peak()
    peaks_r, vals_r = a.get_peak('R')
    peaks_i, vals_i = a.get_peak('I')
    peaks_m, vals_m = a.get_peak('M')

    res['peak'] = peaks
    res['val'] = vals
    res['peak_r'] = peaks_r
    res['val_r'] = vals_r
    res['peak_i'] = peaks_i
    res['val_i'] = vals_i
    res['peak_m'] = peaks_m
    res['val_m'] = vals_m
    res['phase'] = a.get_phi()

    b.resize(a.count)
    res['phaseByTime'] = b.get_phi_by_time()
    res['phaseD'] = 180/3.14159*a.get_phi()

    res['SNR'] = a.get_snr()
    res['SNR2'] = a.get_snr(a.points//2)
    res['peak_R'] = a.get_point(a.get_peak()[0], 'R')
    res['peak_I'] = a.get_point(a.get_peak()[0], 'I')
    res['peak_M'] = a.get_point(a.get_peak()[0], 'M')
    res['point_R'] = a.get_point(a.points//2, 'R')
    res['point_I'] = a.get_point(a.points//2, 'I')
    res['point_M'] = a.get_point(a.points//2, 'M')
    res['dataSumCheck'] = [np.sum(np.abs(d)) for d in a.data]

    res.pop(1)
    res.pop([1,2])

    res.print_ind()


def spyctra_test_suite_4():
    from spyctra import spyctra, fake_spyctra

    a = spyctra()

    for i in range(8):
        a.add(fake_spyctra(t_2=3e-3*(i+1)))

    a.exp_mult(10*np.arange(a.count))
    a.shift(-30)
    a.shift(np.arange(a.count)*50)
    a.shift(-np.arange(a.count)*10)
    a.phase(np.arange(a.count))

    b = a[0:7:3]

    res = result()
    res['sums'] = b.integrate('R')

    res.print_ind()


def TNT_test():
    from spyctra import spyctra
    from TNT import read

    a = spyctra()
    a.add(read('../spyctraRep/TNT/test_sets/slse_', 4, ''))
    a.add(read('../spyctraRep/TNT/test_sets/slse_', 4, ''))

    for e in a.meta:
        if 'Read Time' not in e:
            res = result()
            res[e] = a.meta[e]

            res.print_ind()

    res = result()
    a.new_count(a.count*324)
    a.resize(4096)
    a.fft()
    a.phase()
    a.decimate(36)

    res['f0'] = a.freq
    res['df'] = a.get_df()
    res['Freq'] = a.get_freq()
    res['time'] = a.time
    res['phi'] = a.phi
    res['findPeak[0]'] = a.get_peak()[0]
    res['findPeak[1]'] = a.get_peak()[1]

    res.print_ind()


def SDF_test():
    from SDF import read
    from fitlib import fit

    raw = read('../spyctraRep/SDF/AN_sept2018c', '4')

    for a in raw:
        for m in a.meta:
            res0 = result()

            if m not in['Read Time (ms)']:
                res0[m] = a.meta[m]
                res0.print_ind()


    res = result()

    for i,a in enumerate(raw):
        b = a.copy()
        b.fft()

        amps = b.get_peak()[1]
        dfs = b.get_df()
        phis = b.get_phi()

        p,r = fit(fl.comp_exp_dec, a.x, a.data,
                  [amps*3,
                   6e-6,
                   dfs,
                   phis
                   ],
                  guess=0,check=0,result='a,t2,df,phi')

        As = r['a']
        taus = a.meta['TAUS']

        p,r = fit(fl.exp_dec_wo, taus, As,
                  [max(As), np.mean(taus), min(As)],
                  guess=0, check=0, result='A,T1,B')

        res.stack(r)

    res.print_ind()


def TREEV2_test():
    from TREEV2 import read

    a = read('../spyctraRep/TREEV2/CPMG_', 10)

    res = result()

    for m in a.meta:
        print(m, a.meta[m])
        if m not in['Read Time (ms)']:
            res[m] = a.meta[m]

    res.print_ind()

    a.new_count(a.count*a.points//int(a.meta['nSmpl'][0]))
    a.resize(4096)
    a.fft()
    a.decimate(50)

    res = result()
    res['phase'] = a.get_phi()
    res['Freq'] = a.get_freq()
    res['noise'] = a.get_noise()
    res['3xPhase'] = res['phase']*3

    res.print_ind()


def fitlib_test():
    from fitlib import fit
    from spyctra import spyctra, fake_spyctra
    from numpy.random import uniform, seed, RandomState

    guess = 0
    check = 0

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

        r = result(r)
        r.print_ind()
        r.print()


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

        res = result(r)
        res.print_ind()


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

        res = result(r)
        res.print_ind()


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

        res = result(r)
        res.print_ind()


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

        res = result(r)
        res.print_ind()


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

        res = result(r)
        res.print_ind()


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

        res = result(r)
        res.print_ind()


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
                  ,sigma=errs,note='fitlibTest fitErrorWorker_R')

        res = result(r)
        res.print_ind()

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

        res = result(r)
        res.print_ind()

    #"""
    single_x_single_y_r()
    single_x_single_y_c()
    single_x_multiple_y_r()
    single_x_multiple_y_c()
    single_x_global_y_r()
    single_x_global_y_r2()
    single_x_global_y_c()
    fit_error_worker()
    #"""


def main():
    c0 = time()
    #SDF_test()
    #exit()
    #"""
    spyctra_test_suite_0()
    spyctra_test_suite_1()
    spyctra_test_suite_2()
    spyctra_test_suite_3()
    spyctra_test_suite_4()
    TNT_test()
    SDF_test()
    TREEV2_test()
    fitlib_test()
    #"""
    print(f'TEST = {1000*(time()-c0):.1f} ms')


if __name__ == '__main__':
    main()
