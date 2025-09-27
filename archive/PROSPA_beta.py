from datetime import datetime
from os.path import exists, getmtime, isdir
from spyctra import spyctra
from struct import unpack
from time import clock, ctime

import numpy as np

"""CHANGE LOG
2018-11-30 V5 development
"""

def monthToNum(shortMonth):
    return {'Jan' : 1,
            'Feb' : 2,
            'Mar' : 3,
            'Apr' : 4,
            'May' : 5,
            'Jun' : 6,
            'Jul' : 7,
            'Aug' : 8,
            'Sep' : 9,
            'Oct' : 10,
            'Nov' : 11,
            'Dec' : 12}[shortMonth]


def parseAcquPar(path):
    parDictionary = {}
    acqParams = open(path + '/' + 'acqu.par')

    for line in acqParams:
        line = line.strip('\n')
        words = line.split(' ')
        parDictionary[words[0]] = words[2]

    return parDictionary


def reader(path):
    f = open(path, "rb")
    f.seek(12)
    f.seek(16)
    sizeTD2 = unpack('<i', f.read(4))[0]
    f.seek(32)
    data1 = np.fromiter((unpack('<f', f.read(4))[0] for i in range(sizeTD2)), dtype=complex, count=sizeTD2)

    return data1


def dirread(path):
    parDictionary = parseAcquPar(path)
    cdata = reader(path + '/' + 'data.1d')
    fileTime = (ctime(getmtime(path + '/' + 'data.1d')))
    dwellTime = float(parDictionary['acqTime'])/float(parDictionary['nrPnts'])

    timeWords = fileTime.split(' ')
    try:
        year = int(timeWords[-1])
        mon = int(monthToNum(timeWords[1]))
        day = int(timeWords[2])

        hour, mn, sec = [int(v) for v in timeWords[3].split(':')]

        timeStamp = (datetime(year, mon, day, hour, mn, sec)-spyctraEpoch).total_seconds()
    except:
        print('WARNING: could not parse fileTime', fileTime)
        timeStamp = 0

    print('B1Freq:', int(parDictionary['b1freq']),
          ' Points:', int(parDictionary['nrPnts']),
          ' Acq. Time:', float(parDictionary['acqTime']),
          ' Dwell Time:', dwellTime,
          ' Scans:', int(parDictionary['nrScans']),
          ' RxGain:', int(parDictionary['rxGain']),
          path,
          fileTime)

    plainData = spyctra(data=[cdata],
                        freq=[0],
                        delta=dwellTime,
                        space='s',
                        time=[timeStamp],
                        phase=[0],
                        start=0)

    return plainData


def read(path, *num):
    start = clock()

    if num:
        if type(num[0]) == int:
            suffixes = [i for i in range(num[0])]
        elif type(num[0]) == list:
            suffixes = num[0]
        else:
            raise ValueError(('ERROR: Expecting "Int" or "List"'+
                              'when reading directories but received'), type(num[0]))

        count = len(suffixes)
        found = count

        print('    Trying to read', count, 'directories named', path)
        b = spyctra()

        for i in range(len(suffixes)):
            suffx = repr(suffixes[i])
            filename = path + suffx
            filename0 = path + '0' + suffx
            filename00 = path + '00' + suffx

            if exists(filename):
                b.add(dirread(filename))
            elif exists(filename0):
                b.add(dirread(filename0))
            elif exists(filename00):
                b.add(dirread(filename00))
            else:
                print('WARNING: Couldn\'t find directory', filename);
                found -= 1
    else:
        if exists(path):
            b = dirread(path)
            found = 1
            count = 1
        else:
            print('ERROR: Couldn\'t find directory', path);
            exit()

    print('Found:', found, 'of', count, 'in', round((clock()-start)*1000, 3), 'ms','\n')
    return b


def readFFT():
    a = open('../Magritek_RepFiles/asciifft.txt', 'r')
    x = []
    data = []
    for line in a:
        line = line.strip('\n')
        words = line.split(',')
        x.append(float(words[0]))
        data.append(complex(float(words[1]),float(words[2])))

    data = np.array(data)

    b = spyctra(data = [data],
                start = x[0],
                delta = x[1]-x[0],
                space = 'Hz')
    return b


def test():
    t0 = clock()
    a=read('../Magritek_RepFiles')
    a.fft(divide=1)

    a.normalize(0.5)
    a.start*=-1
    a.delta*=-1

    b = readFFT()
    b.data[0] = np.append(b.data[0][1:], b.data[0][:1])

    a.add(b)
    a.plotOver('I')
    from result import result
    res = result()
    res.add(a)
    res.print(quiet=1)
    print(clock()*1000 - t0*1000)
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(np.real(a.data[1])-np.real(a.data[0]))
    plt.subplot(1,3,2)
    plt.plot(np.imag(a.data[1])-np.imag(a.data[0]))
    plt.subplot(1,3,3)
    plt.plot(np.abs(a.data[1])-np.abs(a.data[0]))

    plt.show()


def main():
    from result import result
    import matplotlib.pyplot as plt
    a=read('../spyctra_repFiles_Magritek/MeOH_180710_',13)

    print(a.time)

    a.fft()
    a.plotOver('M')
    res = result()
    res.add(a.getTime(60), 'time')
    res.add(a.findPhaseD(), 'Phase')
    res.plot('1', 'time', 'Phase','sortx')
    res.show()
    plt.show()


if __name__ == "__main__":
     #test()
     main()


