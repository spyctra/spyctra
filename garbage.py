import numpy as np
from math import pi, e

xs = pi*e*np.logspace(-34,34,100)

for x in xs:
    print(f'{x:.6}')


