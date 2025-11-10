Laplace Inversion
==================================

Calculate 1 and 2D Laplace inversions to recover the distribution of time constants in data.

.. py:function:: laplace_inversion(x, y, T_es, kernel=exp_dec)

   Perform a 1D Laplace inversion of the data defined by *x* and *y* using the decay constants *T_es* and the specific kernel. The kernel can be and 'exp_dec' for an exponential decay, 'inv_rec' for an inversion recovery, or 'sat_rec' for a saturation recovery.

   :param x: The x values of the data with same length as y.
   :type x: iterable(float)
   :param y: The y values of the data with same length as x.
   :type y: iterable(float)
   :param T_es: The possible decay constants of the data.
   :type T_es: iterable[float]
   :param kernel: The functional form the signal is being decomposed into. Default is 'exp_dec'.
   :type kernel: function reference
   :return: The y values of the Laplace inversion corresponding to T_es.
   :rtype: ndarray



.. py:function:: laplace_inversion_2D(data, t2_times, t1_times, lambda0, ncomp, T_1s, T_2s)

   Perform a 2D Laplace inversion of the data taken along T2 (direct) and T1 (indirect) dimensions. Assumes exponential decay along T2 dimension and inversion recovery along T1 dimension but can be altered.

   :param data: The magnitude of the 2D data with T2 along rows and T1 along columns.
   :type data: 2D ndarray
   :param t2_times: The time of the 2D observations.
   :type t2_times: ndarray
   :param t1_times: The time of the 1D observations.
   :type t1_times: ndarray
   :param lambda0: The value for the regularization parameter lamda.
   :type lambda0: float
   :param ncomp: The number of singular values to use for SVD compression.
   :type ncomp: float
   :param T_1s: The possible T1 values in the data.
   :type T_1s: float
   :param T_2s: The possible T2 values in the data.
   :type T_2s: float

   :return: The 2D Laplace inversion of the data.
   :rtype: 2D ndarray
