"""
Moidification to python dictionary object to simplify test_suite analysis
"""

from copy import deepcopy
from os import remove
from os.path import exists

import numpy as np
import sys

"""
CHANGE LOG

2025-09-07 Initial release
"""

if exists(sys.argv[0][:-3]+'.res'):
    remove(sys.argv[0][:-3]+'.res')
if exists(sys.argv[0][:-3]+'.res2'):
    remove(sys.argv[0][:-3]+'.res2')

class result(dict):
    def __init__(self, d0=None):
        if d0 is not None:
            for e in d0:
                self[e] = d0[e]


    def __repr__(self):
        length = len(self[next(iter(self))])

        t = 'ROW\t' + '\t'.join([e for e in self]) + '\n'

        for i in range(length):
            t += f'{i}\t' + '\t'.join([f'{self[e][i]}' for e in self]) + '\n'

        return t


    def pop(self, to_remove):
        to_remove = ensure_iterable(to_remove)
        to_remove.sort()

        for e in self:
            if type(self[e]) == list:
                for i in to_remove[::-1]:
                    self[e].pop(int(i))
            elif type(self[e]) in [np.array, np.ndarray]:
                #print(to_remove)
                self[e] = np.delete(self[e], to_remove)
            else:
                raise TypeError(f'ERROR: element {e} is type {type(self[e])} should be iterable')


    def print(self, quiet=0):
        path = f'{sys.argv[0][:-3]}.res'
        write_mode = 'a'

        f = open(path, write_mode)
        f.write(f'{self}')
        f.close()

        if not quiet:
            print(f'{self}')


    def print_ind(self):
        path = f'{sys.argv[0][:-3]}.res2'
        write_mode = 'a'

        t = ''

        for i, tag in enumerate(self):
            t += tag + '\n'

            for val in self[tag]:
                t += f' {val}\n'

            t += '\n'

        f = open(path, write_mode)
        f.write(t)
        f.close()

        print(t)


    def stack(self, d1):
        if not bool(self):
            for e in d1:
                self[e] = deepcopy(d1[e])

            return

        for item in self:
            if type(self[item]) in [list]:
                self[item] += d1[item]
            elif type(self[item]) in [np.array, np.ndarray]:
                self[item] = np.append(self[item], d1[item])
            else:
                raise TypeError(f'ERROR: element {item} is not list or numpy array. Type {type(item)}')


def ensure_iterable(q):
    if not hasattr(q, '__iter__'):
        return [q]
    else:
        return q


def test_suite():
    import numpy as np

    r0 = result()

    r0['a'] = 1.6*np.ones(3, dtype=np.float64)
    r0['b'] = [1,2,3]

    r1 = result()

    r1['a'] = np.arange(5)
    r1['b'] = [i for i in range(len(r1['a']))]


    r0.pop(1)
    r0.print()
    r0.print_ind()

    r1.pop([2,3])

    r2 = result()
    r2.stack(r1)
    r2.print()


    r0.stack(r1)
    r0.print()



def work():
    import numpy as np

    r = result()

    r['a'] = 1.6*np.ones(3, dtype=np.float64)
    r['b'] = [1,2,3]

    r.pop(1)
    r.print()
    r.print_ind()

    print(r)


if __name__ == "__main__":
    test_suite()
    work()
