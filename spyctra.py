from math import e, pi, ceil
from matplotlib.ticker import FuncFormatter
from numpy.random import normal, seed as random_seed
from os.path import isfile
from pickle import dump, load
from plot_phase_corrections import phase_correction_plotter
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.integrate import trapezoid
from time import perf_counter as time

import matplotlib.pyplot as plt
import numpy as np

from function_library import comp_exp_dec

#import plot_defaults
from plot_defaults import button

import sys

"""
CHANGE LOG

2025-09-27 find_ becomes get_ to make commands easier to remember
2025-09-16 Updated plot formatting
2025-09-15 Fixed negative shift behavior, listprint format
2025-09-05 Initial release
"""

debugger = 0 #prints sum of abs of all data after each method

class spyctra():
    def __init__(self, path=None, **kwargs):
        """
        The spyctra object

        Spyctras store a list of np.arrays as self.data and other useful data
        Assumes uniformly sampled data.

        Items in the dictionary self.meta should be lists to aid in deleting/appending

        get/calc naming convention
            get* returns a fact about the existing data
            calc* performs a calculation on the data using a subjective algorithm

        Raises:
            ValueError: No "data" kwarg passed
            ValueError: No "delta" kwarg passed

        2025-09-27
        """

        self.c0 = [] #list of times used to time methods
        self.data = []

        if path != None: #if path passed will attempt to load pickled spyctra
            self.open(path)
        elif kwargs: #If no kwargs then object is probably temporary and waiting for add/open
            if 'data' in kwargs:
                self.data = kwargs['data']
            else:
                raise ValueError('ERROR: No data passed to initialize spyctra')

            if 'delta' in kwargs:
                self.delta = kwargs['delta']
            else:
                raise ValueError('ERROR: No delta passed to initialize spyctra')

            if 'freq' in kwargs:
                self.freq = np.array(kwargs['freq'])
            else:
                self.freq = np.zeros(self.count)

            if 'meta' in kwargs:
                self.meta = kwargs['meta']
            else:
                self.meta = {}

            if 'phi' in kwargs:
                self.phi = np.array(kwargs['phi'])
            else:
                self.phi = np.zeros(self.count)

            if 'space' in kwargs:
                self.space = kwargs['space']
            else:
                self.space = 's'

            if 'start' in kwargs:
                self.start = kwargs['start']
            else:
                self.start = 0

            if 'time' in kwargs:
                self.time = np.array(kwargs['time'])
            else:
                self.time = np.zeros(self.count)


    def __getitem__(self, ind):
        """
        More pythonic way than a.copy() to copy spyctras

        Args:
            ind: can be a slice, integer, or list representing which spyctra to copy

        Returns:
            The list of spyctra to be copied by self.copy()

        Raises:
            TypeError if ind isn't an allowed type

        2025-09-07
        """

        if type(ind) == slice:
            return self.copy([i for i in range(self.count)][ind])
        elif type(ind) == list:
            return self.copy(ind)
        elif type(ind) == int:
            return self.copy([ind])
        else:
            raise TypeError(f'ERROR: __getitem__ cannot handle type {type(ind)}')


    #Use caution when using these properties in situations
    #where they are being modified and used at the same time
    #2025-09-06
    @property
    def count(self):
        #Returns the number of spyctra
        return len(self.data)


    @property
    def points(self):
        #Returns the length of the spyctra. User responsible to ensure all lengths equal.
        return len(self.data[0])


    @property
    def x(self):
        #Gets the x-values for the spyctra
        return self.start + self.delta*np.arange(self.points)


    @property
    def level(self):
        #For printing status to stdout
        return f'{" "*3*len(self.c0)}'


    def add(self, new_spyctra):
        """
        Add data from a second spyctra to self.

        User assumes space, start, delta, etc are the same.

        Args:
            new_spyctra: the new spyctra to be added to self.

        2025-09-06
        """

        a = new_spyctra.copy(quiet=1) #Need to copy here to continue working with original

        if self.data == []: #Check if current spyctra is empty
            self.__dict__.update(a.__dict__)
        else:
            self.data += a.data
            self.freq = np.append(self.freq, a.freq)
            self.phi = np.append(self.phi, a.phi)
            self.time = np.append(self.time, a.time)

            for item in self.meta:
                if item in a.meta:
                    if type(self.meta[item]) in [list, np.array, np.ndarray]:
                        self.meta[item] += a.meta[item]
                    else:
                        print(f'WARNING: deleting {item} from meta')

                        del self.meta[item]


    def copy(self, *user_copy, quiet=0):
        """
        Returns a duplicate of self.

        Args:
            user_copy: int or list of spyctra to copy
            quiet: Flag to suppress timing information

        Returns:
            Copy of original spyctra elements

        Raises:
            TypeError: if type(user_copy) isn't allowed

        2025-09-07
        """

        if user_copy:
            if type(user_copy[0]) in [list, np.array]:
                to_copy = np.array(user_copy[0])
            elif type(user_copy[0]) == int:
                to_copy = np.array([user_copy[0]])
            else:
                raise TypeError(f'ERROR: copy() expecting int or list received {type(user_copy[0])}')
        else:
            to_copy = np.arange(self.count)

        if quiet == 0:
            self.t0()

            print(f' Copying spyctra {list_print(to_copy)}:')

        data = [self.data[i].copy() for i in to_copy]
        freq = self.freq[to_copy].copy()
        phi = self.phi[to_copy].copy()
        time = self.time[to_copy].copy()

        meta = {}

        try:
            for m in self.meta:
                meta[m] = [self.meta[m][i] for i in to_copy]
        except Exception as e:
            print(e)
            print('WARNING: meta did not successfully copy')

        a = spyctra( data=data
                    ,delta=self.delta
                    ,freq=freq
                    ,meta=meta
                    ,phi=phi
                    ,space=self.space
                    ,start=self.start
                    ,time=time
                   )

        if quiet == 0:
            self.t1()

        return a


    def decimate(self, number=0):
        """
        Averages sequential spyctra and their metadata.

        Will drop remaining data if the number of datasets is
        not a multiple of number

        Args:
            number: The number of spyctra to be added together

        2025-09-27
        """

        if number == 0:
            number = self.count

        if number > 1: #Don't bother if number == 1
            self.t0()
            new_count = self.count//number

            print(f'{self.level} Decimating by {number}, creating {self.count}/{number}={new_count} spyctra of length {self.points}:')

            if self.count%number != 0:
                print(f'WARNING: self.count%number = {self.count}%{number} = {self.count%number}')
                print(f'WARNING: Will reduce number of spyctra to {new_count}')

            for i in range(new_count):
                for j in range(number-1):
                    self.data[i*number] += self.data[i*number + j + 1]

                self.data[i*number] /= number

            self.data = [self.data[i*number] for i in range(new_count)]

            self.freq = np.mean(np.reshape(self.freq[:new_count*number], (new_count,number)), axis=1)
            self.phi = np.mean(np.reshape(self.phi[:new_count*number], (new_count,number)), axis=1)
            self.time = np.mean(np.reshape(self.time[:new_count*number], (new_count,number)), axis=1)
            self.meta = {} #not dealing with this mess
            self.t1()


    def exp_mult(self, FWHM):
        """
        Weight each point using an exponential decay.

        Args:
            FWHM: The decay constant expressed as a FWHM in Hz.

        2025-09-27
        """

        self.t0()

        print(f'{self.level} Exponential multiplication by {list_print(FWHM)} Hz:')

        FWHM = ensure_iterable(FWHM, self.count)

        x = self.x

        for i, LW in enumerate(FWHM):
            self.data[i] *= e**(-pi*LW*x)

        self.t1()


    def fft(self, divide=0, rezero=1):
        """
        Fast Fourier Transform self.data updating x axis values.

        Accounts for domain of the data appropriately
        Normalizes by number of points to make y-axis units consistent

        Args:
            divide: If equal to 1 determines whether first point in x domain
                    should be normalized. Holdover from V0...
            rezero: fftshift the data. Default behavior.

        2025-09-06
        """

        self.t0()

        self.delta = 1/(self.delta*self.points)

        if self.space == 's':
            new_space = 'Hz'
        elif self.space == 'Hz':
            new_space = 's'

        print(f'{self.level} Fourier transforming into {new_space} domain with resolution {self.delta:.3f} {new_space}'
               + (divide == 1)*', dividing first point in half'
               + ':')

        if self.space == 's':
            self.space = 'Hz'

            if rezero == 1:
                self.start = (1 - self.points/2)*self.delta
            else:
                self.start = 0

            for i in np.arange(self.count):
                if divide:
                    self.data[i][0]/=2

                self.data[i] = fft(self.data[i])/self.points

                if rezero:
                    self.data[i] = self.data[i][::-1] #historical OriginLab artifact
                    self.data[i] = fftshift(self.data[i])

        elif self.space == 'Hz':
            self.space = 's'
            self.start = 0

            for i in np.arange(self.count):
                self.data[i] = ifftshift(self.data[i])
                self.data[i] = self.data[i][::-1]*self.points
                self.data[i] = ifft(self.data[i])

                if divide:
                    self.data[i][0]*=2

        self.t1()


    def get_df(self):
        """
        Quickly determine the off-resonance (delta-f) in frequency domain using the location
        of the peak magnitude

        Returns:
            np.array of off-resonances in Hz. Faster than np.fromiter

        Raises:
            ValueError when attempted in time domain

        2025-09-27
        """

        self.t0()

        print(f'{self.level} Calculating delta-f (F(act)-F(rf)) with precision {self.delta:.3f} Hz:')

        self.check_space('Hz')

        dfs = np.array([self.start + self.delta*peak for peak in self.get_peak('M')[0]])

        self.t1()

        return dfs


    def get_freq(self):
        """
        Roughly determine the signal frequency using self.freq and get_df()

        Returns:
            np.array of estimated signal frequencies in Hz

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Calculating frequency of signal with precision {self.delta:.3f} Hz:')

        freq = self.freq + self.get_df()

        self.t1()

        return freq


    def get_linewidth(self, comp='R'):
        """
        Quickly determine the linewidth (Full-Width-at-Half-Max) of the specified component of the signal

        Returns:
            np.array of linewidths in Hz

        Raises:
            ValueError: Errors if data is in time domain

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Calculating linewidths of {comp} with precision {self.delta:.3f} Hz:')

        check_component(comp)
        self.check_space('Hz')

        linewidths = np.zeros(self.count)
        peaks, vals = self.get_peak(comp)

        func = get_component_function(comp)

        half_vals = vals/2

        for i, peak in enumerate(peaks):
            data = func(self.data[i])

            lefts = data[:peak][::-1] < half_vals[i]
            rights = data[peak:] < half_vals[i]

            if len(lefts) != 0:
                left = np.argmax(lefts)
            else:
                left = peak-1

            if len(rights) != 0:
                right = np.argmax(rights)
            else:
                right = self.points-peak

            linewidths[i] = self.delta*(left + right) #can fidget with +1 if desired..

        self.t1()

        return np.array(linewidths)


    def get_noise(self, fraction=4):
        """
        Calculate the RMS of the magnitude of noise in the frequency domain using
        first and last 1/fraction of the data

        Returns:
            array of noise measurements

        Raises:
            ValueError: if performed in time domain

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Calculating noise using first and last 1/{fraction} of data:')

        self.check_space('Hz')

        sample = self.points//fraction
        #noise = np.array([0.5*( np.mean(np.abs(d[:sample])) +np.mean(np.abs(d[-sample:])))/(2**0.5) for d in self.data]) #noise calc from a less civilized time
        noise = np.array([np.mean(np.abs(np.hstack([d[:sample],d[-sample:]]))**2)**0.5 for d in self.data])

        self.t1()

        return noise


    def get_offset(self, fraction=8):
        """
        Calculate average value of the last 1/fraction of data to identify DC offsets

        Args:
            fraction: the last 1/fraction*self.points used to calculate the offset

        Returns:
            np.array of the complex offsets.

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Calculating offset in last 1/{fraction} of data:')

        points = self.points//fraction
        offset = np.array([np.mean(d[-points:]) for d in self.data])

        self.t1()

        return offset


    def get_peak(self, component='M'):
        """
        Returns the position and value of the largest value of the specified component in the data

        Args:
            component: which component of the data to check

        Returns:
            [np.array of positions, np.array of peak values]

        2025-09-27
        """

        self.t0()

        check_component(component)

        print(f'{self.level} Finding peak in {component}:')

        func = get_component_function(component)

        xVals = np.array([np.argmax(func(d)) for d in self.data])
        yVals = np.array([np.amax(func(d)) for d in self.data])

        self.t1()

        return [xVals, yVals]


    def get_phi(self):
        """
        Quickly find the signal phase at the peak in magnitude using arctan2

        Allows you to work in time domain, which is probably useless

        Returns:
            np.array of phases in radians.

        2025-09-13
        """

        self.t0()

        print(f'{self.level} Finding phase:')

        data = self.get_point(self.get_peak('M')[0], 'C')
        phis = np.arctan2(np.imag(data), np.real(data))

        self.t1()

        return np.array(phis)


    def get_phi_by_time(self):
        """
        Returns the data's phase at every point. Using for finding electromagnet instabilities.

        Returns:
            2D np.array of phases.

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Finding phase by time:')

        phases = np.array([np.arctan2(np.imag(data), np.real(data)) for data in self.data])

        self.t1()

        return phases


    def get_point(self, point, component='C'):
        """
        Return data at specified point

        Args:
            point: A list of integers specifying where to get the data, or a
                  single value applied to all spyctra
            component: the desired component of the data

        Returns:
            np.array of specified points

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Getting {component} data from point[s] {list_print(point)}:')

        check_component(component)

        point = ensure_iterable(point, self.count)
        func = get_component_function(component)
        data = func(np.array([d[int(point[i])] for i,d in enumerate(self.data)]))

        self.t1()

        return data


    def get_snr(self, peaks=None):
        """
        Calculate SNR of the abs(peak) to the RMS noise or user specified location.

        Returns:
            np.array of SNRs.

        Raises:
            ValueError: if not in frequency domain

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Finding SNR:')

        self.check_space('Hz')

        if peaks is None:
            snrs = self.get_peak()[1]/self.get_noise()
        else:
            peaks = ensure_iterable(peaks, self.count)
            snrs = self.get_point(peaks, 'M')/self.get_noise()

        self.t1()

        return np.array(snrs)


    def get_time(self, t0=None, scale=1):
        """
        Return the time of each spyctra since t0

        Args:
            t0: an initial reference time
            scale: a normalization factor expressed in seconds

        Returns:
            np.array of times since t0

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Determining spyctra elapsed times:')

        if t0 == None:
            t0 = self.time[0]

        times = (self.time - t0)/norm

        self.t1()

        return times


    def imshow(self, *args):
        """
        Plot the 2D data using imshow

        Args:
            component: which components (RIM) to plot

        2025-09-07
        """

        self.t0()

        if args:
            to_plot, components = plot_arg_parser(args, self.count)
        else:
            to_plot = np.arange(self.count);
            components = 'M'

        print(f'{self.level} Plotting imshow {components} for {list_print(to_plot)}:')

        fig, axs = plt.subplots(len(components))
        extent = [self.x[0], self.x[-1], (self.count-1), 0]

        for i, component in enumerate(components):
            func = get_component_function(component)

            if type(axs) == np.ndarray:
                ax = axs[i]
            else:
                ax = axs

            im = ax.imshow(func(self.data), interpolation='None', aspect='auto', extent=extent)
            ax.set_ylabel('spyctra')
            fig.colorbar(im, ax=ax, label=component)

        if self.space == 'Hz':
            plt.xlabel(r'$\Delta$f (Hz)')
            plt.suptitle('Frequency Domain')
        else:
            plt.xlabel('Time (s)')
            plt.suptitle('Time Domain')

        button()
        self.t1()



    def integrate(self, components='R'):
        """
        Trapezoidal intergration of selected component[s]

        Args:
            components: which component[s] to integrate

        Returns:
            np.array of the integrals

        2025-09-06
        """

        self.t0()
        comps = ensure_iterable(components, self.count)

        print(f'{self.level} Integrating {comps} component[s] in {self.space} domain:')

        integrals = np.empty((len(comps), self.count))

        for j, comp in enumerate(comps):
            func = get_component_function(comp)

            integrals[j] = np.array([trapezoid(func(self.data[i])) for i in range(self.count)])

        if len(comps)==1:
            integrals = integrals[0]

        self.t1()

        return integrals


    def new_count(self, N):
        """
        Refactor the data into N new spyctra
        Appends all the data and then breaks that into N new spyctra

        Args:
            N : the number of new spyctra to be created

        2025-09-06
        """

        if N > self.count:
            self.new_count_more(N)
        elif N < self.count:
            self.new_count_less(N)
        else:
            pass


    def new_count_less(self, N):
        #2025-09-06
        self.t0()

        if self.count%N != 0:
            raise ValueError(f'ERROR: {self.count%N = }')

        new_points = self.points*self.count//N

        print(f'{self.level} From {self.count} spyctra of length {self.points} creating {N} spyctra of length {new_points}:')

        all_points = np.concatenate(self.data)

        self.data = [all_points[i*new_points:(i + 1)*new_points] for i in range(N)]
        self.freq = np.zeros(N)
        self.phi = np.zeros(N)
        self.time = np.zeros(N)
        self.meta = {}

        self.t1()


    def new_count_more(self, N):
        #2025-09-06
        self.t0()

        if N%self.count != 0:
            raise ValueError(f'ERROR: {N%self.count = }')
        elif self.points%(N/self.count) != 0:
            raise ValueError(f'ERROR: {self.points%(N/self.count) = }')

        multiplier = N//self.count
        new_points = self.points*self.count//N

        print(f'{self.level} From {self.count} spyctra of length {self.points} creating {N} spyctra of length {new_points}:')

        newFreq = [None]*self.count
        newPhi = [None]*self.count
        newTime = [None]*self.count

        newData = np.concatenate(self.data)
        newData = np.reshape(newData, (N, new_points))

        for i in range(self.count):
            newFreq[i] = self.freq[i]*np.ones(multiplier)
            newPhi[i] = self.phi[i]*np.ones(multiplier)
            newTime[i] = self.time[i]*np.ones(multiplier)

        self.data = [d for d in newData]
        self.freq = np.concatenate(newFreq)
        self.phi = np.concatenate(newPhi)
        self.time = np.concatenate(newTime)
        self.meta = {}
        self.t1()


    def normalize(self, norm=None):
        """
        Divide data by specified normalization factor or peak of magnitude in each data

        Args:
            *norm: a single value or list used to normalize the data

        2025-09-06
        """

        self.t0()

        if norm is None:
            norm = self.get_peak()[1]
        else:
            norm = ensure_iterable(norm, self.count)

        print(f'{self.level} Normalizing by {list_print(norm)}')

        for i in range(self.count):
            self.data[i] /= norm[i]

        self.t1()


    def open(self, path):
        """
        Load a pickled spyctra from path

        2025-09-06
        """

        t0 = time()

        print(f'Opening {path}')

        with open(path, "rb") as input_file:
            a = load(input_file)

        a.report()
        self.__dict__.update(a.__dict__)

        print(f'  Done:{round((time()-t0)*1000, 3)} ms\n')


    def phase(self, phis=None):
        """
        Phase the data to the specified value[s]
        Default is to find the phase using peak of magnitude

        Args:
            phis: Single value or list/array corresponding to the phase
                  adjustment made to the data

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Phasing by {list_print(phis)}:')

        if phis is None:
            phis = self.get_phi()
        else:
            phis = np.array(ensure_iterable(phis, self.count))

        self.phi += phis
        adjs = e**(-1j*phis)

        for i in range(self.count):
            self.data[i] *= adjs[i]

        #self.data = [d[i]*adjs[i] for d in self.data] is this faster?

        self.t1()


    def phase_foc(self, phaseCorrs):
        """
        Applies first, and zeroth, order phase corrections to data
        [dPhidF,f0,phi0] is expected order of phase correction

        2025-09-06
        """

        self.t0()
        self.check_space('Hz')

        if type(phaseCorrs[0]) not in [list, np.array]:
            phaseCorrs = [phaseCorrs]*self.count

        print(f'{self.level} Phase using 1st & 0th order phase corrections dPhase=dPhi/dF*(f0-x) - phi0:')

        for i in range(self.count):
            print(phaseCorrs[i])

            if len(phaseCorrs[i]) == 2:
                phi0 = 0
                dPhidF, f0 = phaseCorrs[i]
            else:
                dPhidF, f0, phi0 = phaseCorrs[i]

            phiAdj = (f0 - self.x)*dPhidF - phi0
            self.data[i]*=e**(1j*phiAdj)

        self.t1()


    def plot(self, *args):
        """
        Plots selected spyctra as a single image of adjacent plots.
        Single image makes it much faster to look at dozens of spyctra.
        Default behavior is to append data.

        Args:
            *args: String used to specify components. Default is R,I,M
                   List of integers for which data to plot
                   Single integer isolates data to plot

        2025-09-27
        """

        self.t0()

        if args:
            to_plot, comps = plot_arg_parser(args, self.count)
        else:
            to_plot = np.arange(self.count);
            comps = 'RIM'

        print(f'{self.level} Plotting {comps} for {list_print(to_plot)}:')

        fig, ax = plt.subplots(figsize=(16,9))

        data = np.concatenate([self.data[i] for i in to_plot])

        if len(to_plot) > 1:
            x = np.arange(len(data))
        else:
            x = self.x

        if 'R' in comps:
            ax.plot(x, np.real(data), 'r', linewidth=2.0)
        if 'I' in comps:
            ax.plot(x, np.imag(data), 'g', linewidth=2.0)
        if 'M' in comps:
            ax.plot(x, np.abs(data), 'b', linewidth=2.0)

        ax.set_xlim([x[0],x[-1]])

        if self.space == 'Hz' and self.x[0] < 0 < self.x[-1]:
            has_x_equal_0 = True
            x0 = -self.x[0]*(self.points-1)/(self.x[-1]-self.x[0])
        else:
            has_x_equal_0 = False

        ys = plt.ylim()
        xs = plt.xlim()

        if len(to_plot)> 1:
            for i,_ in enumerate(to_plot[1:]): #vertical lines indicate different spyctra
                ax.axvline(self.points*(i+1), alpha=0.5, color='black')

            if self.space == 's':
                for i, s in enumerate(to_plot): #label spyctra index
                    ax.text(self.points/2*(2*i+1), 0.95*ys[1], s, alpha=0.9, fontsize=16)

                ticks0 = [self.points/2*i for i in range(2*len(to_plot))]
                ticks1 = [f'{self.start + self.delta*(v%self.points):.3}' for v in ticks0]
                ax.set_xticks(ticks0)
                ax.set_xticklabels(ticks1)

            if self.space == 'Hz':
                for i, s in enumerate(to_plot): #label spyctra index
                    ax.text(self.points/2*(2*i+1), 0.95*ys[1], s, alpha=0.5, fontsize=16)

                    if has_x_equal_0:
                        ax.axvline(x0 + self.points*i, alpha=0.5, color='black', linestyle=':')

                ticks0 = ( [self.points*i + self.points/4 for i in range(len(to_plot))]
                          +[self.points*(i+1) - self.points/4 for i in range(len(to_plot))])
                ticks1 = [f'{self.start + self.delta*(v%self.points):.3}' for v in ticks0]
                ax.set_xticks(ticks0)
                ax.set_xticklabels(ticks1)

            def format_coord(x0, y):
                rem = x0%self.points
                x1 = self.start + self.delta*rem

                return f'x={x1:.4e}, y={y:.4e}'

            ax.format_coord = format_coord

        self.plot_format(fig, ax, len(to_plot))

        self.t1()


    def plot_over(self, *args, increment=0):
        """
        Plots selected spyctra as a single image of overlayed plots.

        Args:
            *args: String used to specify components. Default is R,I,M
                   List of integers subsets the data to plot
                   Single integer isolates data to plot

        2025-09-06
        """
        self.t0()

        if args:
            to_plot, comps = plot_arg_parser(args, self.count)
        else:
            to_plot = np.arange(self.count);
            comps = 'RIM'

        print(f'{self.level} Plotting Over {comps} for {list_print(to_plot)}:')

        x = self.x

        fig, ax = plt.subplots(figsize=(16, 9))

        for i in to_plot:
            if 'R' in comps:
                ax.plot(x, increment*i + np.real(self.data[i]), linewidth=2.0, alpha=0.75, label=f'Real_{i}')
            if 'I' in comps:
                ax.plot(x, increment*i + np.imag(self.data[i]), linewidth=2.0, alpha=0.75, label=f'Imag_{i}')
            if 'M' in comps:
                ax.plot(x, increment*i + np.abs(self.data[i]), linewidth=2.0, alpha=0.75, label=f'Magn_{i}')

            if len(comps)*len(to_plot)<16:
                ax.legend()

        plt.xlim([self.x[0], self.x[-1]])

        self.plot_format(fig, ax, 1)
        self.t1()


    def plot_phase_cor(self, dPhidF=0, ddPhidF=0.001, f0=0, df0=0.01, phi0=0, dphi0=0.1):
        """
        Plots zero and first order phase corrections and returns values

        2025-09-06
        """

        phaseCorrs = [None]*self.count

        for j in range(self.count):
            b = phase_correction_plotter(self.x, self.data[j], dPhidF, ddPhidF, f0, df0, phi0, dphi0)
            phaseCorrs[j] = b.run()

        return phaseCorrs


    def pop(self, toRemove):
        """
        Removes/deletes specified spyctra

        Args:
            toRemove: the list of spyctra to remove

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Removing {list_print(toRemove)}:')

        if type(toRemove) not in [list, np.array]:
            toRemove = np.array([toRemove])

        toRemove.sort()

        for i in toRemove[::-1]:
            self.data.pop(int(i))
            self.freq = np.delete(self.freq, i)
            self.phi = np.delete(self.phi, i)
            self.time = np.delete(self.time, i)

            for item in self.meta:
                self.meta[item].pop(i)

        self.t1()


    def print(self, points=None):
        """
        Print data to standard output

        Args:
            points: the number of lines to print, default is all lines

        2025-09-06
        """

        if points is None:
            points = self.points

        print(f'Printing {points} points from {self.count} spyctra')

        labels = self.space + ''.join([f'\tR_{i}\tI_{i}\tM_{i}' for i in range(self.count)])

        print(labels)

        for j in range(points):
            print(self.x[j], end='')

            for i in range(self.count):
                val = self.data[i][j]

                print(f'\t{np.real(val)}\t{np.imag(val)}\t{np.abs(val)}', end='')

            print()


    def report(self):
        """
        Print basic facts about spyctra object
        """

        print(f'\nREPORT:\n'
              f' Count: {self.count}\n'
              f' Points: {self.points}\n'
              f' Delta: {self.delta}\n'
              f' Start: {self.start}\n'
              f' Space: {self.space}\n')

        for m in self.meta:
            if type (self.meta[m]) in [list, np.array, np.ndarray]:
                print(m, list_print(self.meta[m]))
            else:
                print(m, type(self.meta[m])) #this shouldn't happen...


    def resize(self, N):
        """
        Resize the data

        If N is integer, truncate or append zeros to make self.points == N.
        If N is negative, zeros are prepended

        If N is list, isolate data between the extremes in the list

        Args:
            N: The new length of each spyctra if integer; the x bounds if list

        2025-09-06
        """

        self.t0()

        if type(N) == int:
            if N > 0:
                if N<self.points:
                    print(f'{self.level} Truncating from {self.points} to {N} points:')

                    self.data = [d[:N] for d in self.data]
                else:
                    print(f'{self.level} Expanding from {self.points} to {N} points:')

                    self.data = [np.pad(d, (0, N-self.points)) for d in self.data]
            else:
                print(f'{self.level} Prepending zeros. Expanding from {self.points} to {abs(N)+self.points} points:')

                self.data = [np.pad(d, (N-self.points, 0)) for d in self.data]

            self.t1()
        #Keep only data between xmin and max
        elif type(N) == list:
            if (N[0] < self.start or N[1] > self.start + self.delta*self.points):
                raise ValueError(f'ERROR: Resize data range is unavailable {self.start}:{self.start + self.delta*self.points}')
            else:
                ind0 = int((N[0] - self.start)//self.delta)
                ind1 = int((N[1] - self.start)//self.delta)+1

                print(f'{self.level} Resizing from x={N[0]} to {N[1]} {self.space}, index {ind0}:{ind1}, {ind1-ind0+1} pts:')

                self.data = [self.data[i][ind0:ind1] for i in range(self.count)]
                self.start += ind0*self.delta

                self.t1()

                return [ind0,ind1]


    def save(self, path):
        """
        Save spyctra object using pickle

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Pickeling as {path}:')

        if isfile(path):
            print('WARNING: Overwriting existing file')
            #input('Proceed?')

        dump(self, open(path, 'wb'))

        self.t1()


    def shift(self, shifts):
        """
        Shift to the left (or right if negative shifts) by N points and replace with zeros

        Params:
            points: the number of points to be removed, int or list, negative values allowed

        2025-09-15
        """

        self.t0()
        print(f'{self.level} Shifting by {list_print(shifts)} point[s]:')

        if type(shifts) in([list, np.array]):
            print('WARNING: self.start may no longer be consistent between spyctra!')

            shifts = np.array(shifts)
        else:
            shifts = ensure_iterable(shifts, self.count)

        self.start += shifts[0]*self.delta

        for i in range(self.count):
            shift = int(shifts[i])
            if shift > 0:
                self.data[i] = np.concatenate((self.data[i][shift:], np.zeros(shift)))
            elif shift < 0:
                self.data[i] = np.concatenate((np.zeros(-shift), self.data[i][:shift]))

        self.t1()


    def smooth(self, smooth):
        """
        Average adjacent points.

        Args:
            smooth: The number of adjacent points to average.

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Smoothing data of length {self.points} by {smooth}:')

        if self.points%smooth != 0:
            print(f'WARNING: Will ignore {self.points%smooth = } points')

        self.start += 0.5*self.delta*smooth
        self.delta *= smooth
        new_points = self.points//smooth
        self.data = [np.mean(np.reshape(d[:new_points*smooth], (new_points,smooth)), axis=1) for d in self.data]

        self.t1()


    def sort(self, x):
        """
        Sorts spyctra by given key

        Args:
            x: Array whose sort indexes will be used to sort self.data

        2025-09-16
        """

        self.t0()

        print(f'{self.level} Sorting data, freq, phi, meta, and time:')

        sort_indexes = np.asarray(x).argsort(kind='stable')

        def sorter(list0):
            return [list0[i] for i in sort_indexes]

        self.data = sorter(self.data)
        self.freq = self.freq[sort_indexes]
        self.phi = self.phi[sort_indexes]
        self.time = self.time[sort_indexes]

        for m in self.meta:
            if type(self.meta[m]) == list:
                self.meta[m] = sorter(self.meta[m])
            elif type(self.meta[m]) == np.array:
                self.meta[m] = self.meta[key][sort_indexes]
            else:
                print(f'WARNING: Can\'t sort meta key {key} since type is {type(self.meta[key])}')

        self.t1()


    def subtract(self, offset):
        """
        Subtract the specified offset or offsets from the data.

        Args:
            offset: A list of offsets or a single offset to be removed
                subtracted from all spyctra.

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Subtracting by {list_print(offset)}:')

        offset = ensure_iterable(offset, self.count)
        self.data = [d - offset[i] for i, d in enumerate(self.data)]

        self.t1()


    def transpose(self):
        """
        Transpose spyctra data

        Note:
            User needs to redefine start, delta, space as appropriate

        2025-09-06
        """

        self.t0()

        print(f'{self.level} Transposing data from {self.count}x{self.points} to {self.points}x{self.count} (spyctra, points):')

        self.data = list(np.array(self.data).transpose())
        self.freq = np.zeros(self.count)
        self.meta = {}
        self.phi = np.zeros(self.count)
        self.time = np.zeros(self.count)

        self.t1()

    #Commonly used operations. Should probably have underscores preceeding their names
    def check_space(self, space):
        if self.space != space:
            raise ValueError(f'ERROR: Expecting self.space={space}, but self.space is {self.space}')


    def debug(self, last=0):
        sums = [np.sum(np.abs(d)) for d in self.data]

        if last == 0:
            for i, summ in enumerate(sums):
                print(i, summ)

        print(np.sum(sums))


    def plot_format(self, fig, ax, plots):
        """
        Ensures consistent formatting of plot and plot_over images

        2025-09-06
        """

        ys = plt.ylim()
        xs = plt.xlim()

        def setLabels(xlabel, ylabel, title):
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_title(title, fontsize=20)

        if self.space == 'Hz':
            setLabels(r'$\Delta$f (Hz)', 'FFT', 'Frequency domain')
        else:
            setLabels('Time (s)', 'Signal', 'Time domain')

        if np.log10(ys[1]) > 3:
            ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(x, ',')))
        if plots == 1 and np.log10(xs[1]) > 3:
            ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(x, ',')))

        ys = ax.get_ylim()
        xs = ax.get_xlim()

        if ys[0] < 0 <ys[1]:
            ax.axhline(0, alpha=0.5, color='black', linestyle=':')
        if xs[0] < 0 <xs[1]:
            ax.axvline(0, alpha=0.5, color='black', linestyle=':')

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        button()


    def t0(self):
        #Set internal timestamp
        return self.c0.append(time())


    def t1(self):
        #Print elapsed time of operation
        print(f'{self.level}   Done: {1000*(time()-self.c0.pop(-1)):.1f} ms')

        if debugger:
            self.debug(1)

        if len(self.c0) == 0:
            print()


def check_component(component):
    if component not in 'RIMC':
        raise ValueError(f'ERROR: Expecting comp in [R,I,M,C] received {component}')


def ensure_iterable(q, points):
    if not hasattr(q, '__iter__'):
        if type(q) != str:
            return q*np.ones(points)
        else:
            return [q]*points
    else:
        if type(q[0]) != str:
            return np.array(q)
        else:
            return q


def dull(x):
    return x


def get_component_function(component):
    if component == 'M':
        return np.abs
    elif component == 'R':
        return np.real
    elif component == 'I':
        return np.imag
    elif component == 'C':
        return dull
    else:
        raise ValueError(f'ERROR: Expecting comp in [R,I,M,C] received {component}')


def list_print(list_in):
    if not hasattr(list_in, '__iter__'):
        return list_in

    if len(list_in) <= 6:
        return f'[{", ".join([f"{a}" for a in list_in])}]'
    else:
        return f'[{", ".join([f"{a}" for a in list_in[:5]])} ... {list_in[-1]}]'


def plot_arg_parser(args, count):
    to_plot = []
    comps = []

    for arg in args:
        if type(arg) == int:
            to_plot = [arg]
        if type(arg) in [list, np.ndarray]:
            to_plot = list(arg)
        if type(arg) == str:
            comps = arg

    #Used to plot all spyctra if none specified
    if to_plot == []:
        to_plot = [i for i in range(count)]

    if comps == []:
        comps = 'RIM'

    return [to_plot, comps]


def fake_spyctra(points=1024, delta=10e-6, df=0, noise=0, t_2=np.inf, phi=0, seed=None, amp=512, scans=1, freq=0, meta={}):
    #2025-09-14
    t0 = time()

    if seed is None:
        seed = int(t0*10000)

    random_seed(seed)
    phi = float(phi)
    x = delta*np.arange(points)

    data = comp_exp_dec(x, amp, t_2, df, phi)
    noise /= scans**0.5
    data += normal(0, noise, size=points) + 1j*normal(0, noise, size=points)

    #simulates digitization of signal
    data = np.floor(data.real) + 1j*np.floor(data.imag)

    report =  f'FAKE Spyctra - Points: {points} DeltaF: {df:.3f} delta: {delta:.3e} '
    report += f'amp: {amp} noise: {noise} phase: {phi:.3f} t2: {t_2:.3f} seed: {seed} '
    report += f'-- {1000*(time()-t0):.1f} ms'

    print(report)

    return spyctra( data=[data]
                   ,freq=[freq]
                   ,delta=delta
                   ,space='s'
                   ,time=[time()]
                   ,phi=[0.0]
                   ,start=0
                   ,meta=meta
                  )


def work():
    a = spyctra()

    trials = 16
    dfs = 128*np.arange(trials)

    for i in range(trials):
        a.add(fake_spyctra(df=dfs[i], noise=4))

    a.resize(16384)
    a.fft()
    a.resize([-2000,2000])

    a.plot()
    plt.show()

    a.imshow('R') #imshow example
    a.imshow('RI') #imshow example
    a.imshow('RIM') #imshow example
    plt.show()


if __name__ == "__main__":
    work()



