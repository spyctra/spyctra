"""
Tool for understanding quadrature detection and spectrometers
"""
from math import e, pi
from numpy.random import normal
from scipy import signal
from time import perf_counter as time

import matplotlib.pyplot as plt
import numpy as np

from fitlib import fit
from function_library import comp_exp_dec
from spyctra import spyctra


"""
CHANGE LOG

2025-09-14 Added scipy.signal.butter to better match observed performance
2025-09-14 Initial release
"""

def cosine_exp_dec(x, a, t_e ,f_0 ,phi):
    return a*e**(-x/t_e)*np.cos(2*pi*f_0*x + phi)


def quad_detector(x0, y0, t_dwell, f_demod, bandwidth=100000, rec_phase=3):
    t_sample = x0[1] - x0[0]
    multiplier = int(t_dwell/t_sample)

    points = int((x0[-1] - x0[0])/t_dwell)
    x0 = x0[:multiplier*points]
    y0 = y0[:multiplier*points]

    y_real = y0*np.cos(2*pi*f_demod*x0 + rec_phase)
    y_imag = y0*np.sin(2*pi*f_demod*x0 + rec_phase)

    if bandwidth != 0:
        sos = signal.butter(10, bandwidth, 'lp', fs=1/t_sample, output='sos')
        y_real = signal.sosfilt(sos, y_real)
        y_imag = signal.sosfilt(sos, y_imag)

    y_real = np.floor(np.mean(y_real.reshape((points, multiplier)), 1))
    y_imag = np.floor(np.mean(y_imag.reshape((points, multiplier)), 1))

    return y_real + 1j*y_imag


def test_suite():
    t_sample = 1/10e6
    t_dwell = 1e-5
    f_demod = 4640000
    points = 4096

    multiplier = int(t_dwell/t_sample)

    x0 = np.array([i*t_sample for i in range(multiplier*points)])
    y0 = cosine_exp_dec(x0, a=2048, t_e=5e-2, f_0=f_demod + 100, phi=3.14159)
    y0 += normal(0, 8, len(y0))

    #"""
    a = spyctra()

    for bandwidth in [0, 100000]:
        y1 = quad_detector(x0,  y0, t_dwell, f_demod, bandwidth)

        a.add(spyctra(data=[y1], delta=t_dwell))

    a.plot()
    plt.show()

    a.shift(40)
    a.resize(a.points-40)

    p, d = fit(comp_exp_dec, a.x, a.data,
               [ [1024]*a.count
                ,[5e-3]*a.count
                ,[100]*a.count
                ,[0]*a.count
               ]
               ,guess=1, check=1, result='a,t_2,df,phi')


def main():
    test_suite()


if __name__ == '__main__':
    main()
