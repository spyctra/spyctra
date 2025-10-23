"""
Spyctra calculation demonstrations
"""

import matplotlib.pyplot as plt
import numpy as np

from spyctra import spyctra, fake_spyctra
from fitlib import fit
from function_library import comp_exp_dec

from math import e, pi

"""
CHANGE LOG

2025-09-16 Initial release
"""

def spyctra_introduction():
    a = spyctra()

    for i in range(20):
        a.add(fake_spyctra(amp=0, t_2=(i+1)*3e-3, df=i*10, phi=i, noise=100))

    #"""
    a.plot()
    plt.show()
    #"""

    b = a.copy()
    b.resize(16384)
    b.fft()
    b.resize([-1000,1000])
    b.phase()

    b.plot()
    plt.show()

    dfs = b.get_df()
    phis = b.phi
    peaks = b.get_peak()[1]
    lws = b.get_linewidth()

    p, d = fit(comp_exp_dec, a.x, a.data,
              [ [512]*a.count
               ,1/lws/2
               ,dfs
               ,phis
              ]
              ,guess=0, check=0, result='a, t_2, df, phi')

    plt.figure()
    plt.subplot(2,1,1)
    plt.errorbar(np.arange(a.count), d['df'], d['df_err'])
    plt.subplot(2,1,2)
    plt.errorbar(np.arange(a.count), d['a'], d['a_err'])
    plt.show()


def advanced_fitting():
    a = spyctra()
    from numpy.random import seed


    for i in range(10):
        a.add(fake_spyctra(amp=512, t_2=(i+1)*3e-3, points=128, df=10*i, phi=i, noise=0*200, delta=1e-5, seed=i))


    def processor(b):
        points = 16384
        lw = 120
        f0 = -2000
        f1 = 2000

        b.normalize(0.0001)
        b.exp_mult(lw)
        b.resize(points)
        b.fft()
        b.resize([f0,f1])

        return b


    b = a.copy()
    b = processor(b)

    dfs = b.get_df()
    phis = b.get_phi()
    lws = b.get_linewidth()

    x = a.delta*np.arange(a.points)


    def fitter(x0, amp, t_2, df, phi):
        y = comp_exp_dec(x, amp, t_2, df, phi)
        b = spyctra( data = [y]
                    ,delta=1e-5)

        b = processor(b)

        return b.data[0]


    p, d = fit(fitter, b.x, b.data,
              [ [512]*b.count
               ,1/2/lws
               ,dfs
               ,phis
              ]
              ,guess=0, check=1, result='a, t_2, df, phi')

    #"""
    from result import result
    d = result(d)

    plt.figure()
    plt.subplot(2,2,1)
    plt.errorbar(np.arange(b.count), d['a'], d['a_err'])
    plt.subplot(2,2,2)
    plt.errorbar(np.arange(b.count), d['t_2'], d['t_2_err'])
    plt.subplot(2,2,3)
    plt.errorbar(np.arange(b.count), d['df'], d['df_err'])
    plt.subplot(2,2,4)
    plt.errorbar(np.arange(b.count), np.array(d['phi'])%pi, np.array(d['phi_err']))
    plt.show()
    #"""


def add_demo():
    a = spyctra()

    print(a.count)

    for i in range(4):
        a.add(fake_spyctra()) #add example

    b = spyctra()

    for i in range(8):
        b.add(fake_spyctra()) #add example

    print(a.count)

    a.add(b) #add example

    print(a.count)

    a.add(b[0:6:2]) #add example

    print(a.count)

    a.add(b.copy([i for i in range(2)])) #add example

    print(a.count)


def decimate_demo():
    a = spyctra()

    for i in range(256):
        a.add(fake_spyctra(t_2=3e-3, df=100, phi=1, noise=1000))

    a.plot(0)
    plt.show()

    a.decimate(64) #decimate example

    a.plot()
    plt.show()

    a.decimate() #decimate example

    a.plot()
    plt.show()


def exp_mult_demo():
    a = spyctra()

    for i in range(2):
        a.add(fake_spyctra(t_2=3e-3, df=100, phi=1, noise=1000))

    a.plot()
    plt.show()

    a.exp_mult(120) #exp_mult example

    a.plot()
    plt.show()

    a.exp_mult([120,1200]) #exp_mult example

    a.plot()
    plt.show()


def fft_demo():
    a = fake_spyctra(t_2=3e-3, df=100, phi=1, noise=64)

    a.plot()
    plt.show()

    a.fft() #fft example

    a.plot()
    plt.show()

    a.fft()
    a.fft(rezero=False) #fft example

    a.plot()
    plt.show()


def get_snr_demo():
    a = spyctra()

    trials = 16
    noises = [256*(i+1) for i in range(trials)]

    for i in range(trials):
        a.add(fake_spyctra(df=100, phi=i, noise=noises[i]))

    a.fft()
    a.phase()

    a.plot()
    plt.show()

    snrs = a.get_snr() #get_snr example
    snrs_at_x_0 = a.get_snr(0) #get_snr example
    snrs_at_x_i = a.get_snr(np.arange(a.count)) #get_snr example

    plt.figure()
    plt.title('get_snr() demo')
    plt.plot(noises, snrs, label='SNR')
    plt.plot(noises, snrs_at_x_0, label='SNR @x[0]')
    plt.plot(noises, snrs_at_x_i, label='SNR @x[i]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('noise in')
    plt.ylabel('SNR')
    plt.legend()
    plt.show()


def get_df_demo():
    a = spyctra()

    trials = 8
    dfs_in = [128*(i+1) for i in range(trials)]

    for i in range(trials):
        a.add(fake_spyctra(t_2=3e-3, df=dfs_in[i], phi=i, noise=4))

    a.resize(16384)
    a.fft()
    a.resize([-2000,2000])

    a.plot()
    plt.show()

    dfs = a.get_df() #get_df example

    plt.figure(figsize=(11,11))
    plt.suptitle('get_df() demo')
    plt.subplot(2,1,1)
    plt.plot(dfs_in, dfs)
    plt.ylabel('df (Hz)')
    plt.subplot(2,1,2)
    plt.plot(dfs_in, dfs_in-dfs)
    plt.axhline(0)
    plt.xlabel('signal off resonance (Hz)')
    plt.ylabel('act-obs off resonance (Hz)')
    plt.show()


def get_freq_demo():
    a = spyctra()

    trials = 8
    dfs_in = [128*(i+1) for i in range(trials)]

    for i in range(trials):
        a.add(fake_spyctra(t_2=3e-3, df=dfs_in[i], phi=i, noise=4))

    a.resize(16384)
    a.fft()
    a.resize([-2000,2000])

    a.plot()
    plt.show()

    freqs = a.get_freq() #get_freq example

    plt.figure()
    plt.title('get_freq() demo')
    plt.plot(dfs_in, freqs)
    plt.xlabel('signal off resonance (Hz)')
    plt.ylabel('frequency (Hz)')
    plt.show()


def get_linewidth_demo():
    a = spyctra()

    trials = 16
    t_2s = [(i+1)*5e-4 for i in range(trials)]

    for i in range(trials):
        a.add(fake_spyctra(t_2=t_2s[i], df=i*8, phi=i, noise=1, points=16384))

    a.fft()
    a.phase()
    a.resize([-1000,1000])

    a.plot()
    plt.show()

    lws = a.get_linewidth() #get_linewidth example
    lws_r = a.get_linewidth('R') #get_linewidth example
    lws_i = a.get_linewidth('I') #get_linewidth example
    lws_m = a.get_linewidth('M') #get_linewidth example

    plt.figure()
    plt.title('get_linewidth() demo')
    plt.plot(t_2s, lws, label='LW')
    plt.plot(t_2s, lws_r, label='LW_R')
    plt.plot(t_2s, lws_i, label='LW_I')
    plt.plot(t_2s, lws_m, label='LW_M')
    plt.xlabel('t_2 in (s)')
    plt.ylabel('linewidth (Hz)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


def get_noise_demo():
    a = spyctra()

    trials = 16
    noises_in = [8**(i/4) for i in range(trials)]

    for i in range(trials):
        a.add(fake_spyctra(noise=noises_in[i]))

    a.fft()

    a.plot()
    plt.show()

    noises = a.get_noise() #get_noise example
    noises_3 = a.get_noise(3) #get_noise example
    noises_8 = a.get_noise(8) #get_noise example

    plt.figure()
    plt.title('get_noise() demo')
    plt.plot(noises_in, noises,label='noise')
    plt.plot(noises_in, noises_3,label='noise_3')
    plt.plot(noises_in, noises_8,label='noise_8')
    plt.xlabel('noise in')
    plt.ylabel('noise')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


def get_offset_demo():
    a = fake_spyctra(t_2=1e-3, df=1000, noise=16)

    offset_in = 50*e**(1j*2)

    a.data[0] += 50*e**(1j*2)

    a.plot()
    plt.show()

    offset = a.get_offset() #get_offset example
    offset_2 = a.get_offset(2) #get_offset example
    offset_4 = a.get_offset(4) #get_offset example
    offset_8 = a.get_offset(8) #get_offset example

    print(f'{offset_in = }')
    print(f'{offset = }')
    print(f'{offset_2 = }')
    print(f'{offset_4 = }')
    print(f'{offset_8 = }')

    a.subtract(a.get_offset()) #subtract example

    a.plot()
    plt.show()


def get_peak_demo():
    a = spyctra()

    trials = 16
    amps_in = 16*np.arange(1, trials+1)
    dfs_in = 128*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps_in[i], t_2=3e-3, df=dfs_in[i], noise=16))

    a.fft()

    peaks = a.get_peak() #get_peak example
    peaks_R = a.get_peak('R') #get_peak example
    peaks_I = a.get_peak('I') #get_peak example
    peaks_M = a.get_peak('M') #get_peak example

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(dfs_in, peaks[0]  ,label='peak')
    plt.plot(dfs_in, peaks_R[0],label='peak_R')
    plt.plot(dfs_in, peaks_I[0],label='peak_I')
    plt.plot(dfs_in, peaks_M[0],label='peak_M')
    plt.ylabel('peak index')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(amps_in, np.abs(peaks[1]),label='peak')
    plt.plot(amps_in, np.abs(peaks_R[1]),label='peak_R')
    plt.plot(amps_in, np.abs(peaks_I[1]),label='peak_I')
    plt.plot(amps_in, np.abs(peaks_M[1]),label='peak_M')
    plt.xlabel('off resonance in (Hz)')
    plt.ylabel('peak value')
    plt.legend()
    plt.show()


def get_phi_by_time_demo():
    a = spyctra()

    for i in range(2):
        a.add(fake_spyctra(t_2=3e-3, df=(i+1)*10, noise=2))

    phase_by_time = a.get_phi_by_time() #get_phi_by_time example

    plt.figure()
    plt.title('get_phi_by_time() demo')
    plt.plot(a.x, phase_by_time[0], label='df=10')
    plt.plot(a.x, phase_by_time[1], label='df=20')
    plt.xlabel('time (s)')
    plt.ylabel('phase (radians)')
    plt.legend()
    plt.show()


def get_phi_demo():
    a = spyctra()

    trials = 16
    phis_in = np.linspace(0, 2*pi, trials)

    for i in range(trials):
        a.add(fake_spyctra(phi=phis_in[i], noise=4))

    a.fft()
    phis = a.get_phi() #get_phi example

    plt.figure()
    plt.title('get_phi() demo')
    plt.plot(phis_in, phis)
    plt.xlabel('time (s)')
    plt.ylabel('phase (radians)')
    plt.show()


def get_point_demo():
    a = spyctra()

    trials = 16
    amps_in = 256*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps_in[i], noise=4, points=4096, delta=1))

    a.fft()

    points = a.get_point(a.points//2 - 1) #get_point example
    points_x_i = a.get_point(np.arange(a.count)) #get_point example

    plt.figure()
    plt.title('get_point() demo')
    plt.plot(amps_in, np.abs(points), label='points')
    plt.plot(amps_in, np.abs(points_x_i), label='x_i')
    plt.legend()
    plt.xlabel('amps in')
    plt.ylabel('np.abs(point) @ a.points//2 - 1')
    plt.show()


def imshow_demo():
    a = spyctra()

    trials = 16
    dfs = 100*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(df=dfs[i], noise=4))

    a.resize(16384)
    a.fft()
    a.resize([-2000,2000])

    a.imshow() #imshow example
    plt.show()

    a.imshow([0,1,2,4,8], 'R') #imshow example
    plt.show()

    a.imshow('RI') #imshow example
    plt.show()

    a.imshow([0,3,6,9,12], 'RIM') #imshow example
    plt.show()



def integrate_demo():
    a = spyctra()

    trials = 16
    amps = np.logspace(1,4,trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps[i], noise=128))

    a.resize(16384)
    a.fft()
    a.resize([-1000, 1000])
    a.phase()

    integrals = a.integrate() #integrate example
    integrals_R = a.integrate('R') #integrate example
    integrals_I = a.integrate('I') #integrate example
    integrals_M = a.integrate('M') #integrate example

    plt.figure()
    plt.title('integrate() demo')
    plt.plot(amps, integrals, label='integrals_')
    plt.plot(amps, integrals_R, label='integrals_R')
    plt.plot(amps, integrals_I, label='integrals_I')
    plt.plot(amps, integrals_M, label='integrals_M')
    plt.legend()
    plt.xlabel('amps')
    plt.ylabel('integrals')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


def new_count_more_demo():
    a = fake_spyctra(t_2=3e-3, noise=64)

    a.new_count(4) #new_count example

    a.fft()

    a.plot()
    plt.show()


def new_count_less_demo():
    a = spyctra()

    trials = 16
    amps = np.logspace(3,1,trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps[i], noise=128, points=64))

    a.plot()
    plt.show()

    a.new_count(1) #new_count example

    a.plot()
    plt.show()


def normalize_demo():
    a = spyctra()

    trials = 8
    amps = np.logspace(3,1,trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps[i], noise=128, points=64))

    a.fft()

    a.plot()
    plt.show()

    a.normalize(1/amps) #normalize example

    a.plot()
    plt.show()

    a.normalize() #normalize example

    a.plot()
    plt.show()


def phase_demo():
    a = spyctra()

    trials = 8
    phis = np.linspace(0, 2*pi, trials)

    for i in range(trials):
        a.add(fake_spyctra(phi=phis[i], t_2=3e-3, noise=16))

    a.resize(16384)
    a.fft()
    a.resize([-1000,1000])

    a.plot()
    plt.show()

    a.phase(a.count) #phase demo

    a.plot()
    plt.show()

    a.phase() #phase demo

    a.plot()
    plt.show()


def plot_demo():
    a = spyctra()

    trials = 8
    amps = 16*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps[i], t_2=3e-3, noise=16))

    a.resize(16384)
    a.fft()
    a.resize([-1000,1000])

    a.plot(5) #plot example
    plt.show()

    a.plot() #plot example
    plt.show()

    a.plot([0,4,5]) #plot example
    plt.show()

    a.plot('R') #plot example
    plt.show()

    a.plot('RI') #plot example
    plt.show()

    a.plot([0,4,5], 'R') #plot example
    plt.show()

    a.plot([0,4,5], 'RI') #plot example
    plt.show()


def plot_over_demo():
    a = spyctra()

    trials = 8
    amps = 16*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps[i], t_2=3e-3, noise=16))

    a.resize(16384)
    a.fft()
    a.resize([-1000,1000])

    a.plot_over() #plot_over example
    plt.show()

    a.phase(np.arange(a.count))

    a.plot_over() #plot_over example
    plt.show()

    a.plot_over([0,4,5]) #plot_over example
    plt.show()

    a.plot_over('R') #plot_over example
    plt.show()

    a.plot_over('RI') #plot_over example
    plt.show()

    a.plot_over([0,4,5], 'R') #plot_over example
    plt.show()

    a.plot_over([0,4,5], 'RI') #plot_over example
    plt.show()


def plot_phase_cor_demo():
    a = spyctra()

    for i in range(2):
        a.add(fake_spyctra(points=16384*64, delta=1e-5, t_2=1e-2, df=20, amp=4096, phi=i+1, noise=2))

    a.subtract(a.get_offset())
    a.fft()
    a.resize([-100,100])

    phis = a.plot_phase_cor() #plot_phase_cor example
    a.phase_foc(phis)

    a.plot()
    plt.show()



def pop_demo():
    a = spyctra()

    trials = 8
    amps = 16*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(amp=amps[i], t_2=3e-3, noise=16))

    a.resize(16384)
    a.fft()
    a.resize([-1000,1000])

    a.plot()
    plt.show()

    a.pop([1,3,5,6]) #pop example

    a.plot()
    plt.show()


def resize_demo():
    a = fake_spyctra(t_2=3e-3, noise=16)

    a.plot()
    plt.show()

    a.resize(256) #resize example

    a.plot()
    plt.show()

    a.resize(2048) #resize example

    a.plot()
    plt.show()

    a.fft()

    a.plot()
    plt.show()

    a.resize([-1000,1000]) #resize example

    a.plot()
    plt.show()


def shift_demo():
    a = fake_spyctra(df=0.1, noise=16, points=64, delta=1)

    a.plot()
    plt.show()

    a.shift(10) #shift example

    a.plot()
    plt.show()

    a.shift(-10) #shift example

    a.plot()
    plt.show()


def smooth_demo():
    a = fake_spyctra(t_2=3e-3, df=100, noise=64)

    a.plot()
    plt.show()

    a.smooth(16) #smooth example

    a.plot()
    plt.show()



def sort_demo():
    a = spyctra()

    trials = 8
    dfs = 100*np.arange(trials)[::-1]

    for i in range(trials):
        a.add(fake_spyctra(df=dfs[i], t_2=3e-2, noise=4, points=16384))

    a.fft()
    a.resize([-1000,1000])

    a.plot()
    plt.show()

    a.sort(dfs) #sort example

    a.plot()
    plt.show()


def subtract_demo():
    get_offset_demo()


def transpose_demo():
    a = spyctra()

    trials = 16
    dfs = 128*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(df=dfs[i], noise=4))

    a.resize(16384)
    a.fft()
    a.resize([-2000,2000])

    a.imshow()
    plt.show()

    a.transpose() #transpose example

    a.imshow()
    plt.show()


def method_demos():
    plot_over_demo()

    add_demo()
    decimate_demo()
    exp_mult_demo()
    fft_demo()
    get_snr_demo()
    get_df_demo()
    get_freq_demo()
    get_linewidth_demo()
    get_noise_demo()
    get_offset_demo()
    get_peak_demo()
    get_phi_by_time_demo()
    get_phi_demo()
    get_point_demo()
    imshow_demo()
    integrate_demo()
    new_count_more_demo()
    new_count_less_demo()
    normalize_demo()
    phase_demo()
    plot_demo()
    plot_over_demo()
    plot_phase_cor_demo()
    pop_demo()
    resize_demo()
    shift_demo()
    smooth_demo()
    sort_demo()
    subtract_demo()
    transpose_demo()


def processing_demos():
    spyctra_introduction()
    advanced_fitting()


def main():
    #processing_demos()
    method_demos()

if __name__ == '__main__':
    main()
