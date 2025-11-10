fitlib
==================================

Because LMFIT is deficient. fitlib is a glorified wrapper for scipy.optimize.curve_fit. It handles global fitting (i.e. fitting multiple traces with one global parameter), complex variables, complex global fitting, and has a handy visualization tool so you can see how close your initial fit parameters were before you start fitting and wasting the time of all those poor electrons.

.. py:function:: fit(function, xs, ys, p0, **kwargs)

   The curve_fit wrapper that finds the parameters p1 that minimize the sum of the squared errors of *function*\(*xs*\[i], \*p1) - *ys*\[i]

   :param function: The name of the function to be fit.
   :type function: object
   :param xs: The x values of the y data. Can be 2D.
   :type xs: iterable[float] or iterable[iterable[float]]
   :param ys: The y values of the data. Can be 2D.
   :type ys: iterable[float, complex] or iterable[iterable[float, complex]]
   :param p0: The initial guess of the fit parameters. If there are multiple y iterables (i.e. len(*ys*) > 1), but one *p0* parameter is not an iterable with len() > 1, a global fit will be performed for that parameter.

   Examples:

   * individual fits:
      * len(*ys*) == 1: *p0*\=[1.0, 1.1, 1.2, 1.3], one value for each parameter

      * len(*ys*) > 1: *p0*\= [[1.0]\*len(*ys*), [1.1]\*len(*ys*), [1.2]\*len(*ys*), [1.3]\*len(*ys*)] each parameter has same length as *ys*

   * global fits:
      * len(*ys*) > 1: *p0*\=[1.0, [1.1]\*len(*ys*), [1.2]\*len(*ys*), [1.3]\*len(*ys*)], at least one parameter be a single value (or an iterable of length(1))

   :type p0: iterable[float, complex, iterable[float, complex]]
   :param guess: A flag indicating whether to plot the fit comparison using the initial *p0*. Default is 0.
   :type guess: int[0, 1] or bool
   :param check: A flag indicating whether to plot the fit comparison using the final fit parameters *p1*. Default is 0.
   :type check: int[0, 1] or bool
   :param result: A single string of comma seperate values serving as the label for the displayed fit parameters and result dictionary. Default is p_0, p_1, etc. Example: result='amplitude, frequency, phase, linewidth'
   :type result: str or None
   :param fine: A flag indicating whether to plot guess or check plots with a finer array of x values than the default of the *xs* values. Default is 0.
   :type fine: int[0, 1] or bool
   :param multi: The number of threads to use when fitting a large number of *ys* in a non-global fit. Beware Amdahl's Law: multithreading overhead limits performance gains. Default is 1.
   :type multi: int
   :param logx: A flag indicating if a log scale should be used for the x axis in the guess and check plots. Default is 0.
   :type logx: int[0, 1] or bool
   :param logy: A flag indicating if a log scale should be used for the y axis in the guess and check plots. Default is 0.
   :type logy: int[0, 1] or bool
   :param note: Text to display next to printout of the fit results summary. Default is ''.
   :type note: str
   :param xlabel: Text for the xlabel of the guess and check plots. Default is ''.
   :type xlabel: str
   :param ylabel: Text for the ylabel of the guess and check plots. Default is ''.
   :type ylabel: str
   :param prefix: Text to prepend to the labels of the parameter variable names based on result. Default is ''.
   :type prefix: str
   :param quiet: A flag to suppress printing results to standard out. Default is 0.
   :type quiet: int[0, 1] or bool
   :param suffix: Text to append to the labels of the parameter variable names based on *result*. Default is ''.
   :type suffix: str
   :param summary: A flag indicating whether to plot a summary of the fit parameters for calls to fit with len(*ys*)>1. Default is 0.
   :type summary: int[0, 1] or bool
   :param savefig: If specified, a path, with file extension, of where to save the guess and check figures. Default is ''.
   :type savefig: str
   :param \*\*kwargs: Other \*\*kwargs recognized by scipy.optimize.curve_fit.

   :return: The fit parameters and a summary dictionary as *p1* and *r*
   :rtype: [p1=ndarray], r=dict]
