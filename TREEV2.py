from datetime import datetime, timedelta
from nptdms import TdmsFile
from os.path import getmtime
from spyctra import spyctra
from time import ctime, perf_counter as time

import numpy as np

"""
CHANGE LOG

2023-06-03 Initial release
"""

debug = 0
timer = 0
quiet = 0

def suffix(char):
    return {'u': 1e-6,
            'm': 1e-3,
            's': 1}[char]


def t1(t0):
    return round(1000*(time()-t0), 1)


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

    if filename[-5:] != '.tdms':
        filename += '.tdms'

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

    a = TdmsFile(directory + filename, 'rb')

    if debug:
        print('\n TdmsFile descriptions')

        for p in dir(a):
            print(p)

    meta = { 'directory': directory
            ,'filename': filename
            ,'fileTime': ctime(getmtime(directory+filename)) #OS time
           }

    if debug:
        print('\n initialization')
        for e in meta:
            print(e, meta[e])

    if timer:
        print(f'\n  Opened file, parsed path and options in {t1(t0)} ms')

    return a, meta


def get_data(a):
    t0 = time()

    all_channels = [repr(c) for c in a['Running AVG'].channels()]
    real_channels = [c for c in all_channels if 'Real' in c]

    data = np.empty((len(real_channels), len(a['Running AVG']['Real 0'])), dtype=complex)

    for i in range(len(real_channels)):
        data[i] = a['Running AVG'][f'Real {i}'].data

        if len(real_channels) == len(all_channels)//2:
            data[i] += 1j*a['Running AVG'][f'Imag {i}'].data

    if debug:
        print('\n data')
        print(f'{data.shape[0]}x{data.shape[1]}')

    if timer:
        print(f'  Read data in {t1(t0)} ms')

    return data


def get_metadata(properties):
    t0 = time()
    meta = {}
    lines = properties['Run Variables'] + properties['Run Settings']
    lines = lines.split('\n')

    for line in lines:
        line = line.strip('\r')
        words = line.split('=')

        if words != ['']:
            if words[1][-1] in ['u','m','s']:
                try:
                    words[1] = float(words[1][:-1])*suffix(words[1][-1])
                except:
                    print(f'WARNING: Could not parse {line}')
            else:
                try:
                    words[1] = float(words[1])
                except:
                    pass

            meta[words[0]] = words[1]

    if debug:
        print('\n meta')
        for e in meta:
            print(e, meta[e])

    if timer:
        print(f'  Read metadata in {t1(t0)} ms')

    return meta


def make_spyctra(a, data, meta, t0):
    t00 = time()
    reps = int(a.properties['Repetitions'])
    time_since = float(a.properties['Sec from Jan 1st 1904'])

    time_stamp = (datetime(1903, 12, 31, 18, 0) + timedelta(seconds=time_since)).timestamp()
    time_check = (datetime(1903, 12, 31, 18, 0) + timedelta(seconds=time_since))
    time_check = (repr(time_check.year) + '-' +
                  repr(time_check.month) + '-' +
                  repr(time_check.day) + ' ' +
                  repr(time_check.hour) + ':' +
                  repr(time_check.minute) + ':' +
                  repr(time_check.second))

    if not quiet:
        print(  f'\tFreq: {meta["Fops"]}'
              + f' Dwell Time: {meta["dInt"]:1.2e}'
              + f' Data: {reps} scans {data.shape[1]} pts {data.shape[0]} channels'
              + f' {time_check}  {meta["fileTime"]}'
              + f' Read in {t1(t0)} ms'
             )

    for e in meta:
        meta[e] = [meta[e]]

    meta['Read Time (ms)'] = [t1(t0)]

    a = [spyctra( data=[d]
                 ,freq=meta['Fops']
                 ,delta=meta['dInt'][0]
                 ,start=0
                 ,phase=np.array([0.0])
                 ,space='s'
                 ,time=np.array([time_stamp])
                 ,meta=meta
                ) for d in data]

    if len(a) == 1:
        if timer:
            print(f'Made 1 spyctra in {t1(t00)} ms')

        return a[0]
    else:
        if timer:
            print(f'Made {len(data)} spyctra in {t1(t00)} ms')

    return a


def TREEV2_reader(path_data, *options):
    t0 = time()
    a, meta = parse_path_and_options(path_data, *options)
    data = get_data(a)
    meta.update(get_metadata(a.properties))

    return make_spyctra(a, data, meta, t0)


def read(path=None, *rawOptions):
    from file_reader import master_reader

    return master_reader(path, '.tdms', *rawOptions)


def test_suite():
    from result import result

    TREEV2_reader('../spyctraRep/TREEV2/FID_0', 'quiet')
    TREEV2_reader('../spyctraRep/TREEV2/FID_0.tdms')
    TREEV2_reader('../spyctraRep/TREEV2/FID_0', 'quiet,debug')
    TREEV2_reader('../spyctraRep/TreeV2/Multi_channel/test_013')


def work():
    from file_reader import master_reader
    import matplotlib.pyplot as plt

    a = TREEV2_reader('../spyctraRep/TreeV2/Multi_channel/test_013')

    b = spyctra()
    b.add(a)
    b = b[2:]
    b.fft()

    b.plot_over('M')
    plt.show()


if __name__ == "__main__":
    #test_suite()
    work()

