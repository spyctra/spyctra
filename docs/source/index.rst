spyctra_v6 Documentation
========================
Spyctra is a python library to quickly process and visualize maganetic resonance data, although any discretely sampled data is suitable. Spyctra exists to allow the experimentalist to audit data as it comes in from the spectrometer to concurrently identify problems and redesign experiments. This feedback loop allows for rapid convergence on the *right* experiment and the *right* processing.

Spyctra Overview
========================

Spyctra objects contain

* *self.data*: a list constaining np.arrays (for the individual spyctra) that make up the data.
* *self.delta*: the increment of the x axis for the data
* *self.start*: the initial x axis value
* *self.space*: indicates if data is in the time ('s') or frequency ('Hz') domain
* *self.phi*: an ndarray tracking the net phase adjustment made to the data
* *self.meta*: a dictionary containing any other metadata about the spyctra object.
* *self.freq*: an ndarray with the observation frequency of the data
* *self.time*: an ndarray of datetime.timestamps representing the time of the data

Design choices
========================

* spyctra assumes you know what you're doing and has minimal error conditions. For example spyctra assumes, but does not require, all spyctra within an object to have the same length.
* Why lists and not 2D numpy arrays to store data? Becasuse self.data is a list it is much easier to add or remove spyctra from self. Especially when data is streaming in, it may not be clear how much data you have to process and the performance hit from forcing 2D numpy arrays can be obnoxious.
* Why no multithreading inside spyctra? It's been attempted but no clear performance benefit has been found on the datasets commonly acquired by the authors. But using a multithreaded process where each subprocess works on a spyctra object might be useful for some users with really large spyctra.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   spyctra_man.rst
   fitlib_man.rst
   TNT_man.rst
   SDF_man.rst
   quad_detector_simulator_man.rst






