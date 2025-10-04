from numpy import e, pi
from scipy.special import wofz

import numpy as np

"""--- Complex Functions --- """
def comp_exp_dec(x, a, t_e, df, phi):
    return a*e**(-x/t_e)*e**(1j*(-2*pi*df*x + phi))


def comp_gaussian(x, a, s, x0, df, phi):
    return a*e**(-0.5*((x-x0)/s)**2)*e**(1j*(-(x-x0)*2*pi*df + phi))


def time_voigt(x, a, s, t_e, df, phi, x0):
    return ( a
            *e**(-0.5*((x-x0)/s)**2)
            *e**(-(x-x0)/t_e)
            *e**(1j*(-(x-x0)*2*pi*df + phi))
           )


"""--- Real functions ---"""
def exp_dec(x, a, t_e):
    return a*e**(-x/t_e)


def exp_dec_wo(x, a, t_e, b):
    return a*e**(-x/t_e) + b


def bessel32(x, a, peakX):
    alpha = x*2.0815759778/peakX
    return a*(np.sin(alpha) - alpha*np.cos(alpha))/alpha**2


def bi_exp_dec(x, a, t_1, b, t_2, C):
    return a*e**(-x/t_1) + b*e**(-x/t_2) + C


def exp_rec(x, a, t_e, b):
    return a*(1 - e**(-x/t_e)) + b


def gaussian_plus_exponential(x, a, p, t_g, t_e, c):
    return a*(p*e**(-0.5*(x/t_g)**2) + (1-p)*e**(-x/t_e)) + c


def gaussian(x, a, s, x0):
    return a*e**(-0.5*((x-x0)/s)**2)


def line(x, m, b):
    return m*x + b


def lorentzian(x, a, g, x0):
    return a*g**2/((x-x0)**2 + g**2)


def voigt(x, a, sigma, gamma, x0):
    fit = np.real(wofz((x-x0 + 1j*gamma)/sigma/np.sqrt(2)))/sigma
    return a*fit/max(fit) #ghastly hack..


def testSuite():
    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib import rcParams
    rcParams['font.family'] = 'Courier New'
    plt.rcParams["font.size"] = 14


    def plotter(x, func, params, is_complex, title):
        y = func(x, *params)

        plt.figure(figsize=(16,9))
        plt.title(title)

        if is_complex:
            plt.plot(y.real)
            plt.plot(y.imag)
            plt.plot(np.abs(y))
        else:
             plt.plot(y)
        plt.show()


    plotter(np.arange(100), comp_exp_dec, [512, 40, .02, 1], True, 'comp_exp_dec')
    plotter(np.arange(100), comp_gaussian, [512, 20, 10, 0.02, 1], True, 'comp_gaussian')
    plotter(np.arange(100), time_voigt, [512, 20, 20, 0.02, 1, 20], True, 'time_voigt')
    plotter(np.arange(100), exp_dec, [512, 20], False, 'exp_dec')
    plotter(np.arange(100), exp_dec_wo, [512, 20, 20], False, 'exp_dec_wo')
    plotter(1+ np.arange(100), bessel32, [512, 15], False, 'bessel32')
    plotter(np.arange(100), bi_exp_dec, [512, 10, 256, 100, 0], False, 'bi_exp_dec')
    plotter(np.arange(100), exp_rec, [512, 5, 0], False, 'exp_rec')
    plotter(np.arange(100), gaussian_plus_exponential, [512, 0.5, 30, 50, 0], False, 'gaussian_plus_exponential')
    plotter(np.arange(100), gaussian, [512, 10, 50,], False, 'gaussian')
    plotter(np.arange(100), line, [1.1,3], False, 'line')
    plotter(np.arange(100), lorentzian, [512, 5, 50], False, 'lorentzian')
    plotter(np.arange(100), voigt, [512, 5, 5, 50], False, 'voigt')


def main():
    testSuite()

if __name__ == '__main__':
    main()
