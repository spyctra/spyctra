"""
A glorified wrapper for scipy.optimize.curve_fit that handles global fitting, complex
variable and has a handy visualization tool.

Written because LMFIT isn't that good.
"""

from copy import deepcopy
from math import e, isnan, pi
from multiprocessing import Pool
from scipy.optimize import curve_fit
from time import process_time as time

import matplotlib.pyplot as plt
import numpy as np

import sys

from plot_defaults import button


"""
CHANGE LOG

2025-10-04 summary keyword added to fit options. Plots all the fit parameters if multiple y.
2025-09-14 Initial release
"""

debug = 0

title_font = 14
xlabel_font = 11
ylabel_font = 11
xtick_font = 10
ytick_font = 10
legend_font = 10
summary_font_L = 12
summary_font_S = 12
button_font = 8


def fit(function, xs, ys, p0, **kwargs):
    t0 = time()

    xs, ys, sigmas, p0, meta, fit_kwargs = check_inputs(xs, ys, p0, **kwargs)

    if meta['global']:
        fit_inputs = [prep_global(xs, ys, sigmas, p0, meta, function, fit_kwargs)]
    else:
        fit_inputs =  [[function, xs[i], y, sigmas[i], [p[i] for p in p0], deepcopy(meta), deepcopy(fit_kwargs), f'{i}/{len(ys)}'] for i, y in enumerate(ys)]

    if meta['global'] or meta['multi'] < 2 or (meta['guess'] == 1 or meta['check'] == 1):
        raw_returns = [fitter(fit_input) for fit_input in fit_inputs]
    else:
        p = Pool(meta['multi'])
        raw_returns = p.map(fitter, fit_inputs)
        p.close()

    returns = parse_returns(meta, raw_returns)

    if meta['summary'] and len(ys) > 1:
        plot_summary(meta, returns[1], len(ys))

    get_time(t0, 'Fitting')

    return returns


def fitter(fit_input):
    t0 = time()

    if len(fit_input) == 10: #global fitting
        function, x, y, sigma, p0, meta, fit_kwargs, ys, p0_splits, label = fit_input
    else:
        function, x, y, sigma, p0, meta, fit_kwargs, label = fit_input
        p0_splits = None

    if meta['guess']:
        y1 = function(x, *p0)
        residuals = y - y1
        r_squared = calc_r_squared(y, residuals)
        summary = get_summary_text('GUESS', meta, p0, r_squared, residuals, label, function, p0_splits)

        xf, yf = None, None

        if meta['fine']>1:
            xf = np.linspace(x[0], x[-1], len(x)*meta['fine'])
            yf = function(xf, *p0)

        plotter(x, y, sigma, y1, residuals, r_squared, meta, summary, xf, yf)

    p1, cov, t_fit = calc_fit(function, x, y, sigma, p0, meta, fit_kwargs)
    y1 = function(x, *p1)
    residuals = y - y1
    r_squared = calc_r_squared(y, residuals)
    uncertainties = [cov[i][i]**0.5 for i in range(len(p1))]

    if not meta['quiet']:
        print_std_out(function, meta, label, p1, uncertainties, r_squared, p0_splits, t_fit)

    if meta['global']:
        ys_splits = np.cumsum([len(y) for y in ys])
        ind_residuals = np.split(residuals, ys_splits)
        ind_r_squareds = [calc_r_squared(y, ind_residuals[i]) for i,y in enumerate(ys)]
    else:
        ind_r_squareds = None

    if meta['check']:
        summary = get_summary_text('FIT', meta, p1, r_squared, residuals, label, function, p0_splits, uncertainties, ind_r_squareds, t_fit)

        xf, yf = None, None

        if meta['fine'] > 1:
            xf = np.linspace(x[0] ,x[-1], len(x)*meta['fine'])
            yf = function(xf, *p1)

        plotter(x, y, sigma, y1, residuals, r_squared, meta, summary, xf, yf)

    get_time(t0, f'Elapsed Fit {label}')

    if meta['global']:
        return [p1, uncertainties, r_squared, ind_r_squareds, p0_splits, residuals, ys_splits, t_fit]
    else:
        return [p1, uncertainties, r_squared, residuals, t_fit]


def print_std_out(function, meta, label, p1, uncertainties, r_squared, p0_splits, t_fit):
    t0 = time()

    func = repr(function).split(' ')[1]

    print(f'\n FIT {label} of {func} in {t_fit*1000:.1f} ms')

    if not meta['global']:
        for i, p in enumerate(p1):
            print(f'{meta['result'][i]:<{meta['max_result_label_length']}} {p: 8.5e} +/- {uncertainties[i]:8.5e}')
    else:
        p1s = np.split(p1, p0_splits)[:len(p0_splits)]
        uns = np.split(uncertainties, p0_splits)[:len(p0_splits)]

        for i, p in enumerate(p1s):
            print(f'{meta['result'][i]} {repr(p)[6:-1]} \n +/- {repr(uns[i])[6:-1]}')

    print(f' R^2 {r_squared:.3}\n')

    get_time(t0, 'print_std_out')


def parse_returns(meta, raw_returns):
    t0 = time()

    if meta['global']:
        p1, uncertainties, global_r_squared, ind_r_squareds, p0_splits, global_residuals, ys_splits, t_fit = raw_returns[0]

        p1s = np.split(p1, p0_splits)[:len(p0_splits)]
        uns = np.split(uncertainties, p0_splits)[:len(p0_splits)]
        p1s = [[p[0] if len(p) == 1 else p[i] for p in p1s] for i in range(len(ys_splits))]
        uns = [[p[0] if len(p) == 1 else p[i] for p in uns] for i in range(len(ys_splits))]

        ind_residuals = np.split(global_residuals, ys_splits)[:len(ys_splits)]

        if meta['complex']:
            mean_glb_res_real = np.mean(global_residuals.real)
            std_glb_res_real = np.std(global_residuals.real)
            mean_glb_res_imag = np.mean(global_residuals.imag)
            std_glb_res_imag = np.std(global_residuals.imag)
        else:
            mean_glb_res = np.mean(global_residuals.real)
            std_glb_res = np.std(global_residuals.real)

        raw_returns = [[p1s[i], uns[i], ind_r_squareds[i], ind_residuals[i], t_fit] for i in range(len(ys_splits))]
    else:
        p1s = [r[0] for r in raw_returns]

    res = {}

    for i, label in enumerate(meta['result']):
        res[label] = []
        res[label+'_err'] = []

    if meta['global']:
        res['R^2_ind'] = []

        if meta['complex']:
            res['<Res.real>_ind'] = []
            res['STD(Res.real)_ind'] = []
            res['<Res.imag>_ind'] = []
            res['STD(Res.imag)_ind'] = []
        else:
            res['<Res>_ind'] = []
            res['STD(Res)_ind'] = []

    res['R^2'] = []

    if meta['complex']:
        res['<Res.real>'] = []
        res['STD(Res.real)'] = []
        res['<Res.imag>'] = []
        res['STD(Res.imag)'] = []
    else:
        res['<Res>'] = []
        res['STD(Res)'] = []

    for i, raw_return in enumerate(raw_returns):
        p1, uncertainties, r_squared, residuals, t_fit = raw_return

        for j, label in enumerate(meta['result']):
            res[label].append(p1[j])
            res[label+'_err'].append(uncertainties[j])

        if meta['global']:
            res['R^2_ind'].append(r_squared)
        else:
            res['R^2'].append(r_squared)

        suffix = '_ind'*meta['global']

        if meta['complex']:
            res[f'<Res.real>{suffix}'].append(np.mean(residuals.real)),
            res[f'STD(Res.real){suffix}'].append(np.std(residuals.real))
            res[f'<Res.imag>{suffix}'].append(np.mean(residuals.imag))
            res[f'STD(Res.imag){suffix}'].append(np.std(residuals.imag))
        else:
            res[f'<Res>{suffix}'].append(np.mean(residuals))
            res[f'STD(Res){suffix}'].append(np.std(residuals))

    if meta['global']:
        if meta['complex']:
            res['R^2'] = [global_r_squared]*len(raw_returns)
            res['<Res.real>'] = [mean_glb_res_real]*len(raw_returns)
            res['STD(Res.real)'] = [std_glb_res_real]*len(raw_returns)
            res['<Res.imag>'] = [mean_glb_res_imag]*len(raw_returns)
            res['STD(Res.imag)'] = [std_glb_res_imag]*len(raw_returns)
        else:
            res['R^2'] = [global_r_squared]*len(raw_returns)
            res['<Res>'] = [mean_glb_res]*len(raw_returns)
            res['STD(Res)'] = [std_glb_res]*len(raw_returns)

    if not meta['quiet']:
        res_printer(res, len(raw_returns), meta['note'])

    get_time(t0, 'parse_returns')

    return [p1s, res]


def get_summary_text(status, meta, p0, r_squared, residuals, label, function, splits, uncertainties=None, ind_r_squareds=None, t_fit = None):
    t0 = time()

    label_length = min(max([len(label) for label in meta['result']]), 10)
    function = repr(function).split(' ')[1]
    function = (function).split('.')[-1]
    title_text=f' {status} {label} {function}'

    if t_fit != None:
        title_text += f' {t_fit*1000:.1f} ms'

    if meta['global']:
        p0 = np.split(p0, splits)

        if uncertainties != None:
            uncertainties = np.split(uncertainties, splits)

    param_text = ''

    for i, label in enumerate(meta['result']):
        param_text += '\n{} '.format(label.ljust(label_length)[:label_length])

        if meta['global'] and len(p0[i]) != 1:
            val = np.mean(np.mean(p0[i]))
            param_text += '{}<{:.3E}>'.format(' '*int(val>=0),val)

            if uncertainties != None:
                val = np.mean(uncertainties[i])
                param_text += fr' $\pm$ <{val:.3E}>'

            param_text += r' $\sigma$={:.3E}'.format(np.std(p0[i]))
        else:
            if meta['global']:
                val = float(p0[i][0])
            else:
                val = float(p0[i])

            param_text += ' {}{:.3E}'.format(' '*int(val>=0), val)

            if uncertainties != None:
                val = np.mean(uncertainties[i])
                param_text += fr'  $\pm$  {val:.3E}'

    r_squared_text = f'R^2 {r_squared:.3f}'

    if ind_r_squareds:
        r_squared_text += fr' <{np.mean(ind_r_squareds):.3f}> $\sigma$={np.std(ind_r_squareds):.3f}'

    residual_text = 'Residuals'

    if meta['complex']:
        r_val = np.mean(residuals.real)
        i_val = np.mean(residuals.imag)

        if r_val >= 0:
            r_pre = ' '
        else:
            r_pre = ''

        if i_val >= 0:
            i_pre = ' '
        else:
            i_pre = ''

        residual_text += '\n' + fr'Real {r_pre}<{r_val:.3E}>  $\sigma$={np.std(residuals.real):.3E}'
        residual_text += '\n' + fr'Imag {i_pre}<{i_val:.3E}>  $\sigma$={np.std(residuals.imag):.3E}'
    else:
        residual_text += '\n' + fr'<{np.mean(residuals):.3E}>  $\sigma$={np.std(residuals):.3E}'

    text = f'{title_text}\n{param_text}\n\n{r_squared_text}\n\n{residual_text}'
    get_time(t0, 'get_summary_text')

    return [status, text]


def calc_fit(function, x, y, sigma, p0, meta, fit_kwargs):
    t0 = time()

    if meta['complex']:
        xC = np.asarray(list(x.copy()) + list(x.copy())) #complex data requires this doubling
        yC = np.hstack([y.real, y.imag])

        def general_complex(xC, *params):
            vals = function(x, *params)

            return np.hstack([vals.real, vals.imag])

        if type(sigma) == np.ndarray:
            sigma = np.hstack([sigma.real, sigma.imag])

        popt, pcov = curve_fit(general_complex, xC, yC, p0=p0, sigma=sigma, **fit_kwargs)
    else:
        popt, pcov = curve_fit(function, x, y, p0=p0, sigma=sigma, **fit_kwargs)

    if np.isposinf(pcov[0][0]):
        print('WARNING: Covariance failed. Try epsfcn=0.001 or smaller')

    t_fit = time() - t0

    return [popt, pcov, t_fit]


###############################################################################

def check_inputs(xs, ys, p0, **kwargs):
    t0 = time()
    global debug

    if 'debug' in kwargs and debug != 1:
        debug = kwargs['debug']

    ys = check_vals(ys, 'ys')
    xs = check_vals(xs, 'xs')

    if len(xs) == 1 and len(ys) > len(xs):
        xs = [xs[0].copy() for _ in ys]

    if 'sigma' in kwargs:
        if kwargs['sigma'] is None:
            sigmas = [None]*len(ys)
            kwargs.pop('sigma')
        else:
            sigmas = check_vals(kwargs['sigma'], 'sigma')
            kwargs.pop('sigma')
    else:
        sigmas = [None]*len(ys)

    p0 = check_p0(p0)

    meta, fit_kwargs = parse_meta_kwargs(kwargs, xs, ys, sigmas, p0)
    check_consistency(xs, ys, sigmas, p0, meta)

    get_time(t0, 'check_inputs')

    return xs, ys, sigmas, p0, meta, fit_kwargs


def check_vals(vals, label):
    # Force x and y input arrays to be list of numpy arrays
    t0 = time()

    if debug:
        print(f'\n Checking {label}')

    #check for 2d data
    if type(vals[0]) in [list, np.array, np.ndarray]:
        if debug:
            print(f'{label} is 2d')
        try:
            if debug:
                for i, val in enumerate(vals):
                    print(f'ys[{i}] {type(val)}')

            newVals = [np.array(val) for val in vals]
        except:
            raise TypeError(f'Could not convert {label} of type {type(vals)} to numpy array')
    else: #must be 1d data
        if debug:
            print(f'{label} is 1d')

        if type(vals) != np.ndarray:
            try:
                print(f'WARNING: Expecting {label} array to be type np.ndarray, received {type(vals)}')

                newVals = [np.array(vals)]
            except:
                raise TypeError('Could not convert {label} of type {type(vals)} to numpy array')
        else:
            newVals = [vals]

    get_time(t0, 'check_vals '+label)

    return newVals


def check_p0(p0):
    # p0 should be list of np.arrays although single values are okay too
    t0 = time()

    if debug:
        print('\n Checking P0')

    if p0 == []:
        raise ValueError('p0 is empty')

    for i, p in enumerate(p0):
        if debug:
            test = len(p) if type(p) in [list, np.ndarray] else ''

            print(f'p{i} {type(p)} {test}')

        if type(p) == np.ndarray:
            pass
        elif type(p) == list:
            p0[i] = np.array(p)
        else: #presumably single integer
            p0[i] = np.array([p])

    get_time(t0, 'check_p0')

    return p0


def parse_meta_kwargs(kwargs, xs, ys, sigmas, p0):
    #Creates two dictionaries, meta for fitter() and
    #fit_kwargs for scipy.optimize.curve_fit()
    t0 = time()

    if debug:
        print( '\n Checking Meta and Kwargs')

    meta = {}     #Initialize to default options
    meta['check'] = False #plots final results of fit
    meta['fine'] = 0 #will return fitted curves
    #meta['fits'] = 0 #will return fitted curves
    meta['guess'] = False #plots initial guess results
    meta['multi'] = 0 #use multiprocessing
    meta['logx'] = 0 #plog x scale as log
    meta['logy'] = 0 #plot y scale as log
    meta['logy'] = 0 #plot y scale as log
    meta['note'] = '' #label for x axis
    meta['xlabel'] = '' #label for y axis
    meta['ylabel'] = '' #label for y axis
    meta['prefix'] = '' #prefix for result variables
    meta['quiet'] = False #suppresses result print to stdout
    #meta['residuals'] = 0 #will return residuals
    meta['result'] = '' #returns result with passed string used to label variables
    meta['suffix'] = '' #suffix for result variabless
    meta['summary'] = False #suffix for result variabless
    meta['savefig'] = '' #path with extension to save figure

    #If the kwarg is not option in meta, assumes it's for curve_fit
    fit_kwargs = {}

    for kwarg in kwargs:
        if kwarg in(meta):
            meta[kwarg] = kwargs[kwarg]
        elif kwarg != 'debug':
            fit_kwargs[kwarg] = kwargs[kwarg]

    #Additional meta parameters derived from x and data
    meta['global'] = len(ys) > 1 and 1 in [len(p) for p in p0]
    meta['complex'] = isinstance(ys[0][0], complex)

    if meta['global'] and meta['fine']:
        print('WARNING: Cannot fine plot global fits')

        meta['fine'] = 0

    if meta['result'] == '':
        meta['result'] = [f'p{i}' for i in range(len(p0))]
    else:
        meta['result'] = [s.strip() for s in meta['result'].split(',')]

    meta['result'] = [meta['prefix'] + label + meta['suffix'] for label in meta['result']]

    meta['max_result_label_length'] = max([len(r) for r in meta['result']])

    if meta['multi'] > 1 and meta['guess'] + meta['check'] > 0:
        print('WARNING: no multiprocessing when guessing or checking fits')

        meta['multi'] = 1

    if debug:
        for key in meta:
            print(f'{key} = {meta[key]}')

    get_time(t0, 'parse_meta_kwargs')

    return meta, fit_kwargs


def check_consistency(xs, ys, sigmas, p0, meta):
    t0 = time()

    if debug:
        print('\n Consistency check')

    def type_check(a, name, expected):
        if type(a) != expected:
            raise ValueError(f'Expected type of {name} is {expected} received {type(a)}')

    type_check(xs, 'xs', list)
    type_check(xs[0], 'xs[0]', np.ndarray)
    type_check(ys, 'ys', list)
    type_check(ys[0], 'ys[0]', np.ndarray)

    if len(xs) != len(ys): #Same number of xs and ys required
        raise ValueError(f'len(xs) != len(ys): {len(xs)}/{len(ys)}')

    for i, y in enumerate(ys): #Equal lengths of xs[i] and ys[i] required
        if len(xs[i]) != len(y):
            raise ValueError(f'len(xs[{i}]) != len(ys[{i}]): {len(xs[i])}/{len(y)}')

    for i, p in enumerate(p0):#Parameter lengths should be 1 or len(ys)
        if len(p) > 1 and len(p) != len(ys):
            raise ValueError(f'{len(ys)} ys but p0[{i}] has length {len(p)}')

    result_len = len(meta['result'])

    if len(p0) != result_len:
        raise ValueError(f'{len(p0)} variables in p0 but {result_len} variables in result')

    if debug:
        print('length check')

        for i, y in enumerate(ys):
            print(f'pair {i} {len(xs[0])}x{len(y)}')

    get_time(t0, 'check_consistency')


######################################################################

def prep_global(xs, ys, sigmas, p0_raw, meta, function, fit_kwargs):
    t0 = time()

    #Get location of repP0 parameters for each curve in self.data
    orders = [[0]*len(p0_raw) for _ in ys]
    num_ys = len(ys)

    for y in range(num_ys):
        offset = 0

        for p in range(len(p0_raw)):
            if len(p0_raw[p]) == num_ys:
                orders[y][p] = offset + y
                offset += num_ys
            else:
                orders[y][p] = offset
                offset += 1

    orders = np.array(orders)


    def globalFunc(x, *p):
        p = np.array(p)

        return np.hstack([function(xs[i], *p[order]) for i, order in enumerate(orders)])


    if type(sigmas[0]) == np.ndarray:
        sigma = np.hstack(sigmas)
    else:
        sigma = None

    y = np.hstack(ys)
    x = np.linspace(0, len(y)-1, len(y))
    p0 = np.array(np.hstack(p0_raw))
    p0_splits = np.cumsum([len(p) for p in p0_raw])
    get_time(t0, 'prep_global')

    return globalFunc, x, y, sigma, p0, deepcopy(meta), deepcopy(fit_kwargs), ys, p0_splits, '0/1'


def calc_r_squared(y0, residuals):
    t0 = time()

    if np.iscomplexobj(residuals):
        ss_res = np.sum(residuals.real**2 + residuals.imag**2)
        ss_tot = (  np.sum((y0.real - np.mean(y0.real))**2)
                  + np.sum((y0.imag - np.mean(y0.imag))**2))
    else:
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y0 - np.mean(y0))**2)

    r_squared = 1 - (ss_res / ss_tot)

    get_time(t0, 'calc_r_squared')

    if isnan(r_squared):
        return 0.0

    return r_squared


######################################################################

def plotter(x, y, sigma, y1, residuals, r_squared, meta, summary, xf=None, yf=None):
    t0 = time()

    fig = plt.figure(figsize=(13,11))
    gs = fig.add_gridspec(3, 2)
    gs.update(left=0.07, right=0.93, top=0.96, bottom=0.07, wspace=0.1, hspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[-1,: 1])
    ax3 = fig.add_subplot(gs[-1, 1])

    plot_data_vs_fit(ax0, x, y, sigma, y1, meta, summary, xf, yf)
    plot_residuals(ax1, x, residuals, meta)
    plot_residual_histogram(ax2, residuals, meta)
    plot_fit_summary(ax3, summary)

    if meta['savefig']:
        extension = meta['savefig'][-3:]
        plt.savefig(meta['savefig'], format=extension, dpi=fig.dpi)

    #Button to close program
    button()

    get_time(t0, 'plotter')

    if meta['check'] or meta['guess']:
        plt.show() #could add logic for hold in here....


def plot_data_vs_fit(ax, x, y, sigma, y1, meta, status, xf, y1f):
    t0 = time()

    if meta['complex']:
        if type(sigma) == np.ndarray:
            sigmaR = sigma.real
            sigmaI = sigma.imag
        else:
            sigmaR = None
            sigmaI = None

        ax.errorbar(x, y.real, sigmaR, marker='o', linestyle='None', c='r', alpha=0.35, markersize=4, label='Data.real', zorder=3)
        ax.errorbar(x, y.imag, sigmaI, marker='o', linestyle='None', c='g', alpha=0.35, markersize=4, label='Data.imag', zorder=3)

        if meta['fine']>1:
            x = xf
            y1 = y1f

        ax.plot(x, y1.real, c='r', label='Fit.real', zorder=4)
        ax.plot(x, y1.imag, c='g', label='Fit.imag', zorder=4)
    else:
        ax.errorbar(x, y, sigma, marker='o', linestyle='None', c='b', alpha=0.35, markersize=5, label='Data', zorder=3)

        if meta['fine']>1:
            x = xf
            y1 = y1f

        ax.plot(x, y1, label='Fit', zorder=4)

    if (ax.get_ylim()[0] < 0 < ax.get_ylim()[1]):
        ax.axhline(c='w', linewidth=1, zorder=2)
        ax.axhline(c='k', alpha=0.75, linestyle=':', linewidth=.9, zorder=5)

    if (ax.get_xlim()[0] < 0 < ax.get_xlim()[1]):
        ax.axvline(c='w', linewidth=1, zorder=2)
        ax.axvline(c='k', alpha=.75, linestyle=':', linewidth=.9, zorder=5)

    ax.tick_params(axis='x', labelsize=xtick_font)
    ax.tick_params(axis='y', labelsize=ytick_font)

    ax.set_title(f'Data vs {status[0]}', fontsize=title_font)

    if meta['logx']:
        ax.set_xscale('log')
    if meta['logy']:
        ax.set_yscale('log')
    if meta['xlabel'] != '':
        ax.set_xlabel(meta['xlabel'], fontsize=xlabel_font)
    if meta['ylabel'] != '':
        ax.set_ylabel(meta['ylabel'], fontsize=ylabel_font)

    ax.grid(linestyle=':',zorder=0)
    ax.legend(fontsize=legend_font)

    get_time(t0, 'plot_data_vs_fit')


def plot_residuals(ax, x, residuals, meta):
    t0 = time()

    if meta['complex']:
        ax.plot(x, residuals.real, c='r', linewidth=1, label='Res.real', zorder=3)
        ax.plot(x, residuals.imag, c='g', linewidth=1, label='Res.imag', zorder=3)
    else:
        ax.plot(x, residuals, label='Res', zorder=3)

    if (ax.get_ylim()[0] < 0 <ax.get_ylim()[1]):
        ax.axhline(c='w', linewidth=1, zorder=2)
        ax.axhline(c='k', alpha=0.75, linestyle=':', linewidth=0.9, zorder=4)

    if (ax.get_xlim()[0] < 0 < ax.get_xlim()[1]):
        ax.axvline(c='w', linewidth=1, zorder=2)
        ax.axvline(c='k', alpha=0.75, linestyle=':', linewidth=0.9, zorder=4)

    ax.tick_params(axis='x', labelsize=xtick_font)
    ax.tick_params(axis='y', labelsize=ytick_font)

    ax.set_title('Residuals', fontsize=title_font)
    ax.set_ylabel('(Data-Fit)',fontsize=ylabel_font)

    if meta['logx']:
        ax.set_xscale('log')
    if meta['xlabel'] != '':
        ax.set_xlabel(meta['xlabel'],fontsize=xlabel_font)

    ax.grid(linestyle=':', zorder=1)

    get_time(t0, 'plot_residuals')


def plot_residual_histogram(ax, residuals, meta):
    t0 = time()

    def hister(residual, suffix, c, a=1):
        try:
            hist = ax.hist(residual, bins=30, label='Hist' + suffix, alpha=a, color=c, zorder=4)
            centers = [(hist[1][i+1] + hist[1][i])*0.5 for i in range(len(hist[1])-1)]
            hopt, hcov = curve_fit(gaussian, centers, hist[0],
                                   p0=[max(hist[0]), np.std(residual), np.mean(residual)])
            ax.plot(centers, gaussian(centers, *hopt), c=c, zorder=5)
        except:
            pass

    if meta['complex']:
        hister(residuals.real, '.real', 'r', 0.5)
        hister(residuals.imag, '.imag', 'g', 0.5)
    else:
        hister(residuals, '', 'b', 0.5)

    ax.tick_params(axis='x', labelsize=xtick_font)
    ax.tick_params(axis='y', labelsize=ytick_font)
    ax.grid(linestyle=':', zorder=1)

    if (ax.get_xlim()[0]<0<ax.get_xlim()[1]):
        ax.axvline(c='w', linewidth=1, zorder=2)
        ax.axvline(c='k', alpha=0.75, linestyle=':', linewidth=0.9, zorder=6)

    ax.set_title('Residual Histogram', fontsize=title_font)
    ax.set_ylabel('Count',fontsize=ylabel_font)

    if meta['xlabel'] != '':
        ax.set_xlabel(meta['xlabel'], fontsize=xlabel_font)

    get_time(t0, 'plot_residual_histogram')


def plot_fit_summary(ax3, summary):
    t0 = time()

    status = summary[0]
    text = summary[1]
    lines = text.count('\n')

    if lines < 12:
        font = summary_font_L
    else:
        font = summary_font_S

    ax3.set_title(f'{status} parameter summary', fontsize=title_font)
    ax3.plot([0, 10], [10, 0], color='w')

    plt.tick_params( axis='both'
                    ,which='both'
                    ,bottom=False
                    ,top=False
                    ,left=False
                    ,right=False
                    ,labelbottom=False
                    ,labelleft=False)

    plt.text(-0.3, 9.99, text, verticalalignment='top', wrap=True, fontsize=font, family='Courier New')

    get_time(t0, 'plot_fit_summary')


def plot_summary(meta, result, count):
    num_plots = len(meta['result']) + 2

    if num_plots <= 8:
        fig, axs = plt.subplots(2, int(np.ceil(num_plots/2)))
    elif num_plots <= 12:
        fig, axs = plt.subplots(3, int(np.ceil(num_plots/3)))
    elif num_plots <= 16:
        fig, axs = plt.subplots(4, int(np.ceil(num_plots/4)))

    fig.suptitle('Fit Results Summary')

    indices = np.arange(count)

    for i, e in enumerate(meta['result']):
        ax = axs[i//len(axs[0])][i%len(axs[0])]
        ax.set_title(e)

        ax.errorbar(indices, result[e], result[f'{e}_err'])

    axs[-1][-2].set_title('Residuals')

    if meta['complex']:
        axs[-1][-2].errorbar(indices, result['<Res.real>'], result['STD(Res.real)'], label='R')
        axs[-1][-2].errorbar(indices, result['<Res.imag>'], result['STD(Res.imag)'], label='I')
        axs[-1][-2].legend()
    else:
        axs[-1][-2].errorbar(indices, result['<Res>'], result['STD(Res)'])


    axs[-1][-1].set_title('R^2')

    if meta['global']:
        axs[-1][-1].plot(indices, result['R^2_ind'])

    axs[-1][-1].plot(indices, result['R^2'])

    button()
    plt.tight_layout()
    plt.show()


def res_printer(res, length, note):
    print(f'\n{note}')
    print('\t'.join([e for e in res]))

    for i in range(length):
        print('\t'.join([f'{res[e][i]:.6}' for e in res]))

    print()
######################################################################

def gaussian(x, A, s, x0):
    return A*e**(-0.5*((x-x0)/s)**2)


def get_time(t0, label):
    t_delta = round(1000*(time()-t0))

    if debug:
        print(f'-{label}: {t_delta} ms\n')

    return t_delta


def fit_error_worker():
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    ys = [x**2,1+x**2]
    ys[0][8] = 0
    ys[1][9] = 0
    errs = [[1,1,1,1,1,1,1,1,5,1], [1,1,1,1,1,1,1,1,1,5]]


    def parabola(x, a,b,c):
        return a*x**2 + b*x + c


    for sigmas in [None, errs]:
        p,r = fit(parabola, x, ys,
                  [ [0.8]*1
                   ,[0.0]*len(ys)
                   ,[0.0]*len(ys)
                  ]
                  ,guess=0, check=1, result='a,b,c'
                  ,sigma=sigmas, note='CW')


def fine_worker():
    from numpy.random import normal,uniform,seed

    seed(0)
    """
    x = np.linspace(0,4*3.14159,50)

    def cos(x,a,f):
        return a*np.cos(2*3.14159*x*f)

    y = cos(x, 1, 1)
    y += 0.1*np.random.normal(0,1,len(x))

    p,r = fit(cos, x, y,
              [ 1
               ,1
              ]
              ,guess=1, check=1, result='a,f',fine=10)
    """

    xs = np.linspace(0, 20e-2, 1024)

    te = 1e-1
    phi0 = 2
    p0 = np.array([uniform(7,13), uniform(-20,20), te, phi0])
    ys = [compExpDec(xs, *p0) for _ in range(1)]
    ys = [y + 1*((normal(0,1,len(xs)) + 1j*normal(0,1,len(xs)))) for y in ys]

    p,r = fit(compExpDec, xs, ys,
              [ [10]*len(ys)
               ,[0]*len(ys)
               ,[5e-2]
               ,[1]
              ]
              ,guess=1, check=1, result='a,df,te,phi0',fine=10)


def intense_fit():
    from numpy.random import normal, uniform, seed
    seed(0)

    def compExpDec(x, a, df, te, phi0):
        return a*e**(-x/te)*e**(1j*(-x**df*2*pi + phi0))

    xs = np.linspace(0, 40e-2, 128)

    te = 1e-1
    phi0 = 1.1
    ys = [compExpDec(xs, uniform(7,13), uniform(-20,20), te, phi0) for _ in range(16)]
    ys = [y + 3*((normal(0,1,len(xs)) + 1j*normal(0,1,len(xs)))) for y in ys]

    p, d = fit(compExpDec, xs, ys,
               [ [10]*len(ys)
                ,[0]*len(ys)
                ,[5e-2]
                ,[1]
               ]
               ,guess=0, check=0, result='a,df,te,phi0', fine=10)


def worker():
    from spyctra import fake_spyctra, spyctra
    from function_library import comp_exp_dec

    a = spyctra()
    for i in range(25):
        a.add(fake_spyctra(amp=10000, points=256, t_2=(i+1)*3e-3, df=-i**2*.010, phi=i*0.01, seed=i, noise=8))

    b = a.copy()
    b.resize(b.points*8)
    b.fft()

    dfs = b.get_df()
    phis = b.get_phi()
    peaks = np.abs(b.get_peak()[1])
    lws = b.get_linewidth()

    """
    a.plot()
    plt.show()
    """

    p, d = fit(comp_exp_dec, a.x, a.data,
              [ peaks*20
               ,1/2/lws
               ,dfs
               ,phis
              ]
              ,guess=0, check=0, summary=1, result='a, t_2, df, phi')


def main():
    worker()


if __name__ == '__main__':
    main()
