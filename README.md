# spyctra_v6

Spyctra is a set of tools for quickly manipulating, visualizing, and processing discretely sampled data with a focus on magnetic resonance problems.

Spyctra arranges data as a list of numpy arrays, instead of a 2d-numpy array, to simplify the memory requirements of creating spyctra from many files.

./demos contains .py files with introduction to basic processing and fitting approaches.
./tests contains test_suite.py and compare_test_suite_outputs.py. By running test_suite.py with different spyctra versions you can then compare the outputs with compare_test_suite_outputs.py to see if anything was broken, or fixed.

spyctra can read:\
TNT files from Tecmag Inc., including pulse sequence variables and their values\
SDF files from Stelar, including all metadata\
examples can be found in the test_suites() of the file readers although you must provide your own test files

The most important version of spyctra is the next version. Generally, backwards compatability is not preserved between different versions (e.g. v5 to v6) but there are no guarantees the intermediate versions will be consistent.

Find a bug? Please send a minimal working example to mwmalone@gmail.com
