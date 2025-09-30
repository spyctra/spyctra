"""
File reader for Stelar's FFC data format
"""

from datetime import datetime
from math import log10
from os.path import exists, getmtime
from spyctra import spyctra
from time import ctime, perf_counter as time

import numpy as np

"""
CHANGE LOG

2025-09-14 Initial Release
"""

debug = 0
timer = 0
quiet = 0

def t1(t0):
    return round(1000*(time()-t0), 1)


def get_taus(meta):
    try:
        BINI = meta['BINI'].strip("()")
        words = BINI.split('*')
        start = float(words[0])*meta['T1MX']

        BEND = meta['BEND'].strip("()")
        words = BEND.split('*')
        stop = float(words[0])*meta['T1MX']

        if meta['BGRD'] == 'LIN':
            taus = np.linspace(start, stop,
                               num=meta['NBLK'])
        elif meta['BGRD'] == 'LOG':
            taus = np.logspace(log10(start), log10(stop),
                               num=meta['NBLK'])
    except Exception as e:
        taus = None

        print('WARNING: Could not get tau values')
        print(e)

    return taus


def get_delta(meta):
    try:
        delta = 1/meta['SW']
    except:
        print('WARNING: could not determine tDwell')
        print(' This is common in multipulse sequences')

        delta = 1

    return delta


def get_time_stamp(meta):
    data_time_line = meta['TIME']
    date_and_time = data_time_line.split('\t')
    words = date_and_time[1].split(' ')
    mon, day, year = list(map(int, words[0].split('/')))
    hour, mn, sec = list(map(int, words[1].split(':')))

    if words[2] == 'AM' and hour == 12:
        hour = 0
    if words[2] == 'PM' and hour < 12:
        hour += 12

    return datetime(year, mon, day, hour, mn, sec).timestamp()


def parse_path_and_options(pathData, *options):
    t0 = time()

    global debug
    global timer
    global quiet

    if type(pathData) == list:
        directory, filename = pathData[0], pathData[1]
    else:
        path = pathData.replace('\\','/')
        file_start = path.rfind('/') + 1
        filename = path[file_start:]
        directory = path[:file_start]

    if filename[-4:] != '.sdf':
        filename += '.sdf'

    if directory == '':
        directory = './'

    if options:
        options = options[0]
    else:
        options = []

    find = 0

    for option in options:
        if option.isdigit():
            find = int(option)

    if 'debug' in options:
        debug = 1

    if 'quiet' in options:
        quiet = 1

    if 'timer' in options:
        timer = 1

    a = open(directory + filename, 'r')

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

    return a, meta, find


def parse_meta(words):
    words[0] = words[0].replace(' ', '')

    if words[0] != 'TIME':
        words[1] = words[1].replace('\t', '')

    #see if value is string, float or integer
    try:
        value = float(words[1]) #check if number

        if words[1].find('.') == -1: #no decimal place assume integer
            value = int(words[1])

        label = words[0]
    except:
        label = words[0]
        value = words[1] #give up and take string

    if debug:
        print(label, value)

    return {label: value}


def get_data(a, meta, options):
    if meta['NBLK'] > 0:
        pts = meta['NBLK']*meta['BS']
    else:
        pts = meta['BS']

    data = np.empty(pts, dtype=complex)

    for i in range(pts):
        line = a.readline()
        line = line.strip('\n')
        words = line.split('\t')

        if 'stellarmag' in options:
            data[i] = int(words[2])
        else:
            data[i] = int(words[0]) + int(words[1])*1j

    return data


def make_spyctra(data, meta):
    taus = meta['taus']
    del meta['taus']

    for key in meta:
        meta[key] = [meta[key]]

    a = spyctra( data=[data]
                ,freq=np.array([meta['SYF1']])
                ,delta=meta['delta'][0]
                ,space='s'
                ,time=np.array([meta['timeStamp']])
                ,phase=np.array([0.0])
                ,start=0
                ,meta=meta
               )

    a.normalize(meta['NS'])

    if a.meta['NBLK'][0] > 0:
        new_count = a.meta['NBLK'][0]
        a.new_count(new_count)

        for e in meta:
            meta[e] = meta[e]*new_count

            if debug:
                print(e, meta[e])

        meta['TAUS'] = taus
        a.meta = meta

    return a


def SDF_reader(pathData, *options):
    t0 = time()

    spyctras = []
    read_times = []
    found = 0

    a, meta0, find = parse_path_and_options(pathData, *options)

    for line in a:
        if len(line)>3 and line[:4] == 'ZONE':
            t00 = time()
            meta = meta0.copy()
            found += 1

        line = line.strip('\n')
        words = line.split('=')

        if len(words) == 2:
            meta.update(parse_meta(words))

            if line[:4] == 'DATA':
                data = get_data(a, meta, options)
                meta['taus'] = get_taus(meta)
                meta['delta'] = get_delta(meta)
                meta['timeStamp'] = get_time_stamp(meta)

                if meta['taus'] is not None:
                    spyctras.append(make_spyctra(data, meta))
                    read_times.append(t1(t00))

                    print(f'Zone {len(spyctras)}\n')
                else:
                    found -= 1
                    pass

                if find > 0 and find == found:
                    print(f'pulled {len(spyctras)} zones from {meta["directory"] + meta["filename"]}')

                    return spyctras

    if find < 1:
        print(f'{meta0["directory"] + meta0["filename"]} contained {len(spyctras)} zones')

    if not quiet:
        for i in range(len(spyctras)):
            print(f'  {i} Zone: {i+1}  Count: {spyctras[i].count}  Points: {spyctras[i].points}  Delta: {spyctras[i].delta} Read in {read_times[i]} ms')

    print(f'Read in {t1(t0)} ms')

    return spyctras


def read(path=None, *options):
    from file_reader import master_reader

    return master_reader(path, '.sdf', *options)


def test_suite():
    import matplotlib.pyplot as plt

    sdfs = SDF_reader('../spyctraRep/Stelar/AN_sept2018c')

    for a in sdfs:
        a.plot()
        plt.show()


def work():
    from file_reader import master_reader
    import matplotlib.pyplot as plt

    a = master_reader('../spyctraRep/Stelar/AN_sept2018c','.sdf')
    exit()

    b = a[1]
    b.report()
    b.shift(12)
    b.resize(4*4096)
    b.fft()
    b.resize([-50000,50000])
    b.plot()
    plt.show()


def main():
     test_suite()


if __name__ == "__main__":
    main()

