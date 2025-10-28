spyctra
========================

The methods below are designed to be terse. One common convention is that methods will accept a single value and then apply it to all spyctra within *self*, or can accept an iterable, with length equal to *self.count*, with each value applied to the corresponding spyctra.


.. py:function:: spyctra.__init__(self, path=None, **kwargs)

   Initialize the python object.

   :param path: Optional path to a previously saved spyctra.
   :type path: str or None
   :param data: List of ndarrays() containing the individual spyctra data.
   :type data: list[ndarray]
   :param start: The initial x-axis value.
   :type start: float
   :param delta: The x-axis increment.
   :type delta: float
   :param space: A flag indicating if the data is in the time ('s') or frequency ('Hz') domain.
   :type space: str
   :param phi: The cumulative phase adjustments made to each spyctra. Default is 0.
   :type phi: ndarray or None
   :param meta: A dictionary of iterables that track additional metadata of the spyctra.
   :type meta: dict or None
   :param freq: The observation frequencies of the spyctra. Default is 0.
   :type freq: ndarray or None
   :param time: The datetime.timestamps of the spyctra.
   :type time: datetime.timestamp or None




.. py:function:: spyctra.__getitem__(self, ind)

   Copy individual spyctra into a new object in a pythonic way.

   :param ind: The spyctra to copy.
   :type ind: int or list[int] or slice
   :return: A new spyctra that is a subset of the original.
   :rtype: spyctra




.. py:function:: spyctra.count(self)

   Return the number of spyctra i.e. len(*self.data*).

   :return: len(*self.data*).
   :rtype: int





.. py:function:: spyctra.points(self)

   Return the length of the spyctra. Spyctra assumes, but does not require, *self.data[i]* all have the same length.

   :return: len(*self.data[0]*).
   :rtype: int





.. py:function:: spyctra.x(self)

   Return the x-axis of the data using *self.start* and *self.delta*. More efficient than always storing the whole thing since *self.start* and *self.delta* are generally more useful.

   :return: The x-axis of the data.
   :rtype: ndarray





.. py:function:: spyctra.add(self, new_spyctra)

   Add additional spyctra to *self*.

   :param new_spyctra: The spyctra to add to *self*.
   :type new_spyctra: spyctra





.. py:function:: spyctra.copy(self, *user_copy, quiet=0)

   Copy the selected spyctra.

   :param user_copy: The spyctra to copy. Default is all, i.e. *self.count*.
   :type user_copy: int or iterable[int] or None
   :param quiet: A flag that suppresses output during copy.
   :type quite: int[0, 1] or bool
   :return: A new spyctra object.
   :rtype: spyctra





.. py:function:: spyctra.decimate(self, number=0)

   Average *self.data* at each x value across number of spyctra.

   :param number: The number of spyctra to average. If *self.count%number* > 0, decimate will ignore the remainder. If *number* == 0, will decimate by *self.count*.
   :type number: int





.. py:function:: spyctra.exp_mult(self, FWHM)

   Weight the spyctra by an exponential decay with time constant 1/pi/*FWHM*.
   Doesn't check for time domain because spyctra lets you do you.

   :param FWHM: The *FWHM* used to define the time constants of the exponential weighting.
   :type FWHM: float or iterable[float]





.. py:function:: spyctra.fft(self, divide=0, rezero=1)

   FFT data from time to frequency domain or vice versa based on *self.space*.

   :param divide: If *divide* == 1 divides first point in half when going to frequency domain and corrects when going back to the time domain. Historical feature from v0 of spyctra and Originlab NMR Tools.
   :type divide: int[0, 1]
   :param rezero: If *rezero* == 1 performs fftshift on the FFTed data. Appropriate for demodulated data.
   :type rezero: int[0, 1]





.. py:function:: spyctra.get_df(self)

   Estimate the off-resonance (delta-f) of the peak, relative to the observation frequency, of the spyctra data in the frequency domain.

   :return: The array of off-resonances.
   :rtype: ndarray[float]





.. py:function:: spyctra.get_freq(self)

   Estimate the absolute frequency of the peak of the spyctra data in the frequency domain.

   :return: The array of frequencies for each element in *self.data*.
   :rtype: ndarray[float]





.. py:function:: spyctra.get_lw(self, component='R')

   Estimate the full-width at half-maximum (FWHM) of the signal in the frequency domain by 1) finding the index of the peak of the signal, 2) finding the left and right indices where the signal is less than half of the peak value.

   :param component: Which component (RIM) of the spyctra to use to calculate the linewidth.
   :type component: char or None
   :return: The array of FWHM, in Hz, for each spyctra.
   :rtype: ndarray[float].





.. py:function:: spyctra.get_noise(self, fraction=4)

   Estimate the RMS of the magnitude of noise in the frequency domain using the first and last 1/*fraction* of the data.

   :param fraction: The fraction of data, on both the left and right of the spyctra, that is sampled to calculate the RMS noise.
   :type fraction: int
   :return: The array of the RMS of the noise for each spyctra.
   :rtype: ndarray[float].





.. py:function:: spyctra.get_offset(self, fraction=8)

   Estimate the DC offset of each spyctra by sampling the last 1/*fraction* of the data.

   :param fraction: The *fraction* of data sampled to calculate offset.
   :type fraction: int
   :return: The array of the offsets for each spyctra.
   :rtype: ndarray[float, complex].





.. py:function:: spyctra.get_peak(self, component='M')

   Return the index and value of the peak of the signal for the specified component.

   :param component: Which component (RIM) of the spyctra to process.
   :type component: char
   :return: A list or two ndarrays containing the indices and values corresponding to the peak.
   :rtype: list[ndarray[int], ndarray[float, complex]]





.. py:function:: spyctra.get_phi(self)

   Calculate the phase adjustment needed to phase the data to the peak.

   :return: List of phases in radians.
   :rtype: ndarray[float]





.. py:function:: spyctra.get_phi_by_time(self)

   Calculate the phase of each point for each spyctra. Useful for identifying electromagnet instabilities.

   :return: ndarray of ndarrays of the phase for each spyctra.
   :rtype: ndarray[ndarray[float]]





.. py:function:: spyctra.get_point(self, x_indices, component='C')

   Return the specified component of the value at the specified point[s].

   :param x_indices: The specific index returned from each spyctra.
   :type kind: int or iterable[int]
   :param component: The component (RIMC) of the value to return.
   :type component: char
   :return: The desired values list.
   :rtype: ndarray[float, complex]





.. py:function:: spyctra.get_snr(self, peaks=None)

   Return the signal to noise ratio (SNR) at the peak signal location (default) or user specified location. Noise is always the default *self.noise* calculation.

   :param peaks: Where to calculate the signal. If *peaks* == None uses the absolute peak of the data.
   :type peaks: int or iterable[int] or None
   :return: The SNR[s].
   :rtype: ndarray[float]





.. py:function:: spyctra.get_time(self, t0=None, scale=1)

   Calculate the time since t0 for each spyctra and then divide by *scale* to put into desired units.

   :param t0: The initial time to compare to. If *t0* == None will use time of first spyctra.
   :type t0: datetime.timestamp
   :param scale: The number of seconds to divide the elapsed times by to convert to minutes (60), hours(60\*60), days(24\*60\*60).
   :type scale: float
   :return: the elapsed times.
   :rtype: ndarray[float]





.. py:function:: spyctra.imshow(self, *args)

   Plot a 2d representation of the desired subset of *self.data*.

   :param to_plot: Subset of spyctra to plot.
   :type to_plot: int or iterable[float]
   :param component: Which component[s] (RIM) of the data to plot.
   :type component: char or str





.. py:function:: spyctra.integrate(self, components='R')

   Return results of trapezoidal integration of the spyctra by the specified component[s].

   :param components: Which components (RIM) to integrate. Default is 'R'.
   :type components: char or str or None
   :return: the desired integrals.
   :rtype: ndarray[float] or ndarray[ndarray[float]]





.. py:function:: spyctra.new_count(self, N)

   Refactor the data into *N* new spyctra. If *N* > *self.count*, metadata such as *self.freq* is preserved. If *N* < *self.count*, metadata is lost.

   :param N: The number of new spyctra to create from the existing data.
   :type N: int





.. py:function:: spyctra.normalize(self, norm=None)

   Divide spyctra by the specified value[s]. Default is to normalize by the largest value in each spyctra.

   :param norm: The value used to divide each spyctra.
   :type norm: float or iterable[float]





.. py:function:: spyctra.open(self, path)

   Open a pickled spyctra object.

   :param path: The path to the pickled spyctra.
   :type path: str





.. py:function:: spyctra.phase(self, phis=None)

   Phase spyctra by the specified value[s].

   :param phis: The phase adjustment, in radians, to be made to each spyctra.
   :type phis: float or iterable[float]





.. py:function:: spyctra.phase_foc(self, phase_corrs)

   Apply first or phase correction[s].

   :param phase_corrs: The phase adjustment to be made to each spyctra.
   :type phase_corrs: iterable[iterable[dPhidF, f0, phi0]]





.. py:function:: spyctra.plot(self, *args)

   Plot the desired subset of spyctra as adjacent plots.

   :param to_plot: The subset of spyctra to plot.
   :type to_plot: int or iterable[int]
   :param component: Which components (RIM) of the data to plot.
   :type component: char or str





.. py:function:: spyctra.plot_over(self, *args)

   Plot the desired subset of spyctra in a single figure.

   :param to_plot: The subset of spyctra to plot.
   :type to_plot: int or iterable[int]
   :param component: Which components (RIM) of the data to plot.
   :type component: char or str





.. py:function:: spyctra.plot_phase_corr(self, dPhidF=0, dPhidF_inc=0.001, f0=0, f0_inc=0.01, phi0=0, phi0_inc=0.1)

   Open the interactive first order phase correction GUI and return the results.

   :param dPhidF: The rate of change of phase with respect to frequency in units of phase per Hz.
   :type dPhidF: float
   :param dPhidF_inc: The scale for the slider on the GUI for *dPhidF*.
   :type dPhidF_inc: float
   :param f0: The center frequency for the first order correction in units of Hz.
   :type f0: float
   :param f0_inc: The scale for the slider on the GUI for *f0*.
   :type f0_inc: float
   :param phi0: The zeroth order phase correction in units of radians.
   :type phi0: float
   :param dphi0_inc: The scale for the slider on the GUI for *dphi0*.
   :type dphi0_inc: float





.. py:function:: spyctra.pop(self, to_remove)

   Remove the specified spyctra from *self*.

   :param to_remove: The subset of spyctra to remove.
   :type to_remove: int or iterable[int]





.. py:function:: spyctra.print(self, points=None)

   Print spyctra *data*[:*points*] to standard output.

   :param points: The number of points to print.
   :type points: int





.. py:function:: spyctra.report(self)

   Print useful information about the spyctra.






.. py:function:: spyctra.resize(self, N)

   Resize spyctra by removing data or adding zeros.

 * If *N* is an integer, truncate or append zeros to make *self.points* == *N*.
 * If *N* is negative, zeros are prepended.
 * If *N* is list, isolate data between the extremes in the list.

   :param N: The new length of the spyctra.
   :type N: int or list
   :return: The indices used to subset the data if *N* is a list.
   :rtype: list





.. py:function:: spyctra.save(self, path)

   Pickle a spyctra and save to *path* for future processing.

   :param path: The path to save the pickled spyctra.
   :type path: str





.. py:function:: spyctra.shift(self, shifts)

   Remove the first *shift* points or add *shift* zeros to the spyctra data.

   :param shifts: If *shifts* > 0 then remove those points and append zeros. If *shifts* < 0 prepend shift zeros.
   :type shifts: int or iterable[int]





.. py:function:: spyctra.smooth(self, smooth)

   Successivle average points within a spyctra. Effectively a computationally cheap low-pass filter.

   :param smooth: the number of adjacent points to average. If *smooth%self.points* > 0 the remainder of points will be dropped.
   :type smooth: int





.. py:function:: spyctra.sort(self, x)

   Sort the spyctra by the keys of the iterable *x*. Sort all relevant parts of *self*: *data*, *freq*, *phi*, *meta*, *time*.

   :param x: The iterable used to identify how to sort the parts of *self*. len(*x*) must equal *self.count*.
   :type x: iterable





.. py:function:: spyctra.subtract(self, offset)

   Subtract offset[s] from the spyctra.

   :param offset: The offset[s] to subtract from each spyctra.
   :type offset: float(complex), iterable(float, complex)





.. py:function:: spyctra.transpose(self)

   Transpose the data in *self*. Resets *self*: *freq*, *time*, *meta*, *phi*.






.. py:function:: fake_spyctra(points=1024, delta=10e-6, df=0, noise=0, t_2=numpy.inf, phi=0, seed=None, amp=512, scans=1, freq=0, meta={}):

   A quick way to generate spyctra. Will even simulate digitization of the data. This is a powerful way to test methods and can even be used for improved spectral analysis of actual data in conjunction with fitlib. Assumes data will have the form of a complex exponential decay.

   :param points: The length of the spyctra data.
   :type points: int
   :param delta: The x increment of the spyctra x-axis.
   :type delta: int or list
   :param df: The off-resonance (delta-f) of the spyctra.
   :type df: float
   :param noise: The standard deviation of the noise in both the real and imaginary channels from one scan.
   :type noise: float
   :param t_2: The decay constant, in seconds, of the data.
   :type t_2: float
   :param phi: The phase offset, in radians, of the data.
   :type phi: float
   :param seed: A random seed to obtain distinct noise between spyctra. Default is to use the system time.
   :type seed: None or int
   :param amp: The initial amplitude of the data.
   :type amp: float
   :param scans: Simulates  averaged data by holding *amp* constant and reducing noise by *scans*\*\*0.5.
   :type scans: int
   :param freq: The observation frequency of the data. Default is 0.
   :type freq: float
   :param meta: A dictionary for other information about the spyctra.
   :type meta: dict
   :return: The fake spyctra.
   :rtype: spyctra

