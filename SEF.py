from datetime import datetime, timedelta
from os.path import getmtime
from spyctraV6 import spyctra
from time import ctime, perf_counter as time

import numpy as np

debug = 0
timer = 0
quiet = 0


def t1(t0):
    return round(1000*(time()-t0),1)


def parsePathAndOptions(pathData, *options):
    t0 = time()

    global debug
    global quiet

    if type(pathData) == list:
        directory, filename = pathData[0], pathData[1]
    else:
        path = pathData.replace('\\','/')
        fileStart = path.rfind('/') + 1
        filename = path[fileStart:]
        directory = path[:fileStart]

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
            ,'fileTime': ctime(getmtime(directory+filename)) #OS time
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
    currentBlock=[]
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
            
            currentBlock.append(vals[1] +1j*vals[2])
        else:
            if debug:
                print('New Block')
            
            data.append(np.array(currentBlock))
            currentBlock = []
    
    data.append(np.array(currentBlock))

    lengths = [len(block) for block in data]
    if len(set(lengths)) > 1:
        print(f'ERROR: Unequal block lengths: {lengths}')
        exit()
        
    if timer:
        print(f'\n  Read {len(data)} blocks in {t1(t0)} ms')

    return start, delta, data


def makeSpyctra(start, delta, data, meta, t0):
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
        

def SEFreader(pathData, *options):
    t0 = time()
    a, meta = parsePathAndOptions(pathData, *options)
    start, delta, data = getData(a)

    return makeSpyctra(start, delta, data, meta, t0)


def read(path, *rawOptions):
    from fileReader import masterReader
    return masterReader(path, '.sef', *rawOptions)


def testSuite():
    import matplotlib.pyplot as plt
    from fitlib import fit
    from math import e, pi
    
    a = SEFreader('../spyctraRep/Stelar/PDMS_mj_2Aug2018_4MHz_T1.sef')
    #a.plot()
    #plt.show()

    def timeVoigt(x, A, s, te, offRes, phase):
        return ( A
                *e**(-0.5*((x)/s)**2)
                *e**(-(x)/te)
                *e**(1j*(-(x)*2*pi*offRes + phase))
               )

    b = a.copy()
    b.resize(4096)
    b.fft()
    amps = [np.mean(np.abs(data[:10])) for data in a.data]
    dfs = b.findDf()
    phases = b.findPhase()
    
    
    p, r = fit(timeVoigt, a.x, a.data,
              [ amps
               ,1e-4
               ,3e-4
               ,dfs
               ,phases
              ]
             ,guess=1, check=1,result='A,s,te,offRes,phase')


def main():
    testSuite()


if __name__ == '__main__':
    main()
      