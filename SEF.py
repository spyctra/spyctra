from os.path import getmtime
from spyctra import spyctra
from time import ctime, perf_counter as time

import numpy as np

"""
CHANGE LOG

2025-10-03 _ overhaul
"""


debug = 0
timer = 0
quiet = 0


def t1(t0):
    return round(1000*(time()-t0),1)


def parse_path_and_options(path_data, *options):
    t0 = time()

    global debug
    global quiet

    if type(path_data) == list:
        directory, filename = path_data[0], path_data[1]
    else:
        path = path_data.replace('\\','/')
        file_start = path.rfind('/') + 1
        filename = path[file_start:]
        directory = path[:file_start]

    if filename[-4:] != '.sef':
        filename += '.sef'

    if directory == '':
        directory = './'

    if options:
        options = options[0]
    else:
        options = []

    if 'quiet' in options:
        quiet = 1

    if 'debug' in options:
        debug = 1

    if 'quiet' not in options:
        print(f' {filename}', end='')

    if debug:
        print()

    a = open(directory + filename, 'r')

    meta = { 'directory': directory
            ,'filename': filename
            ,'fileTime': ctime(getmtime(directory + filename)) #OS time
           }

    if debug:
        print('\n initialization')
        for e in meta:
            print(e, meta[e])

    if timer:
        print(f'\n  Opened file, parsed path and options in {t1(t0)} ms')

    return a, meta


def getData(a):
    t0 = time()

    for i in range(7):
        a.readline()

    data = []
    current_block=[]
    start = 0
    delta = 0

    for line in a:
        if line[:5] != 'Block':
            line = line.strip('\n')
            line = line.replace('\t',' ')
            words = line.split(' ')
            vals = [float(word) for word in words if word != '']

            if start == 0:
                start = float(vals[0])

            if delta == 0:
                delta = float(vals[0])-start

            current_block.append(vals[1] +1j*vals[2])
        else:
            if debug:
                print('New Block')

            data.append(np.array(current_block))
            current_block = []

    data.append(np.array(current_block))

    lengths = [len(block) for block in data]

    if len(set(lengths)) > 1:
        print(f'ERROR: Unequal block lengths: {lengths}')
        exit()

    if timer:
        print(f'\n  Read {len(data)} blocks in {t1(t0)} ms')

    return start, delta, data


def make_spyctra(start, delta, data, meta, t0):
    t00 = time()

    meta['Read Time (ms)'] = [t1(t0)]

    a = spyctra( data=data
                ,freq=[23e6]*len(data)
                ,delta=delta
                ,start=start
               )

    if timer:
        print(f'Made {len(data)} spyctra in {t1(t00)} ms')

    return a


def SEF_reader(path_data, *options):
    t0 = time()
    a, meta = parse_path_and_options(path_data, *options)
    start, delta, data = getData(a)

    return make_spyctra(start, delta, data, meta, t0)


def read(path, *rawOptions):
    from fileReader import masterReader

    return masterReader(path, '.sef', *rawOptions)


def test_suite():
    import matplotlib.pyplot as plt
    from fitlib import fit
    from math import e, pi

    a = SEF_reader('../spyctraRep/Stelar/PDMS_mj_2Aug2018_4MHz_T1.sef')
    #a.plot()
    #plt.show()

    def time_voigt(x, A, s, te, offRes, phase):
        return ( A
                *e**(-0.5*((x)/s)**2)
                *e**(-(x)/te)
                *e**(1j*(-(x)*2*pi*offRes + phase))
               )

    b = a.copy()
    b.resize(4096)
    b.fft()
    amps = [np.mean(np.abs(data[:10])) for data in a.data]
    dfs = b.get_df()
    phases = b.get_phi()


    p, r = fit(time_voigt, a.x, a.data,
              [ amps
               ,1e-4
               ,3e-4
               ,dfs
               ,phases
              ]
             ,guess=1, check=1, result='a,s,t_e,df,phi')

    plt.figure()
    plt.errorbar(np.arange(a.count), r['a'], r['a_err'])
    plt.show()



def main():
    test_suite()


if __name__ == '__main__':
    main()
