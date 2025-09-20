from numpy import e, pi
from scipy.special import wofz

import numpy as np

"""--- Complex Functions --- """
def comp_exp_dec(x, A, t_e, df, phi):
    return A*e**(-x/t_e)*e**(1j*(-2*pi*df*x + phi))


def comp_gaussian(x, A, s, x0, df, phi):
    return A*e**(-0.5*((x-x0)/s)**2)*e**(1j*(-(x-x0)*2*pi*df + phi))


def time_voigt(x, A, s, t_e, df, phi, x0):
    return ( A
            *e**(-0.5*((x-x0)/s)**2)
            *e**(-(x-x0)/t_e)
            *e**(1j*(-(x-x0)*2*pi*df + phi))
           )


"""--- Real functions ---"""
def exp_dec(x, A, t_e):
    return A*e**(-x/t_e)


def exp_dec_wo(x, A, t_e, B):
    return A*e**(-x/t_e) + B


def bessel32(x, A, peakX):
    alpha = x*2.0815759778/peakX
    return A*(np.sin(alpha) - alpha*np.cos(alpha))/alpha**2


def bi_exp_dec(x, A, t_1, B, t_2, C):
    return A*e**(-x/t_1) + B*e**(-x/t_2) + C


def exp_rec(x, A, t_e, B):
    return A*(1 - e**(-x/t_e)) + B


def gaussian_plus_exponential(x, A, p, t_g, t_e, C):
    return A*(p*e**(-0.5*(x/t_g)**2) + (1-p)*e**(-x/t_e)) + C


def gaussian(x, A, s, x0):
    return A*e**(-0.5*((x-x0)/s)**2)


def line(x, m, B):
    return m*x + B


def lorentzian(x, A, g, x0):
    return A*g**2/((x-x0)**2 + g**2)


def voigt(x, A, sigma, gamma, x0):
    fit = np.real(wofz((x-x0 + 1j*gamma)/sigma/np.sqrt(2)))/sigma
    return A*fit/max(fit) #ghastly hack..


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
    plotter(np.arange(100), bessel32, [512, 15], False, 'bessel32')
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
