from datetime import datetime
from os.path import exists, getmtime
from spyctra import spyctra, spyctraEpoch
from struct import unpack
from time import clock, ctime

import numpy as np

import matplotlib.pyplot as plt

"""CHANGE LOG
2018-11-07 In development
"""

""" Notes
https://github.com/jjhelmus/nmrglue/blob/master/nmrglue/fileio/bruker.py
http://www.chem.wilkes.edu/~trujillo/NMR/How_To..._/Parameter_Reference.pdf
"""

def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch

    return delta.total_seconds()

def acqusDic(file):
    acqus = open(file+'/acqus', 'r')
    acqDic = {}
    
    for line in acqus:
        if line[:3] == '##$':
            line = line[3:]
            line = line.strip('\n')
            words = line.split('= ')
            acqDic[words[0]] = words[1]
            
    return acqDic


def reader(file, quiet=0):

    acqDic = acqusDic(file)
    
    print(acqDic['TD']) #points complex counts double...
    print(acqDic['DTYPA']) #0 for int, 2 for float
    print(acqDic['BYTORDA']) #1 big endian
    print(acqDic['SW']) #Spectral Width
    print(acqDic['SFO1']) #1 big endian
    
    tDwell = 10e6/(2*float(acqDic['SW'])*float(acqDic['SFO1']))
    tDwell *= 1e-6
    print(tDwell) #dwell time
    
    tDwell_2 = float(acqDic['AQ'])/int(acqDic['TD'])
    print(tDwell_2)


    data = open(file+'/fid', 'rb')
    cdata = np.fromiter((complex(unpack('>i', data.read(4))[0], 
                                 unpack('>i', data.read(4))[0])
                         for i in range(int(acqDic['TD'])//2)
                         ), dtype=complex)
    
    plt.figure()
    plt.plot(np.real(cdata))
    plt.plot(np.imag(cdata))
    plt.show()

    plainData = spyctra(data=[cdata],
                        freq=[freq],
                        delta=delta,
                        space='s',
                        time=[timeStamp],
                        phase=[0.0],
                        start=0)

    if dims[1] > 1:
        plainData.newCount(dims[1])
        
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
                              'when reading files but received'), type(num[0]))

        count = len(suffixes)
        found = count

        print('    Trying to read', count, 'files named', path)
        b = spyctra()

        for i in range(len(suffixes)):
            suffx = repr(suffixes[i]) + '.tnt'
            filename = path +suffx
            filename0 = path + '0' + suffx
            filename00 = path + '00' + suffx

            if exists(filename):
                b.add(reader(filename, len(num)-1))
            elif exists(filename0):
                b.add(reader(filename0, len(num)-1))
            elif exists(filename00):
                b.add(reader(filename00, len(num)-1))
            else:
                print('WARNING: Couldn\'t find file', filename);
                found -= 1
    else:
        if '.tnt' not in path:
            path += '.tnt'
        print('    Trying to read', path)

        if exists(path):
            b = reader(path, 0)
            found = 1
            count = 1
        else:
            print('ERROR: Couldn\'t find file', path);
            exit()

    print('Found:', found, 'of', count, 'in', round((clock()-start)*1000, 3), 'ms','\n')
    return b


def main():
    import matplotlib.pyplot as plt
    t0 = clock()
    a=reader('../spyctra_repFiles_Bruker/1H_4mm_CaO_050715/1')



if __name__ == "__main__":
    main()


