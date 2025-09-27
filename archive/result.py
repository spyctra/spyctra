import matplotlib.pyplot as plt

from copy import deepcopy
from math import ceil
from matplotlib.widgets import Button
from os import remove
from os.path import exists
from pickle import dump, load
from time import time

import numpy as np
import sys

"""CHANGE LOG
2021-11-29 markers now valid option for plot
2021-07-06 copy() added
2021-06-23 options abs, raw added to show; huge error bars no longer plot 
           by default
2021-03-13 only imports spyctra if necessary so other people can use this
2020-12-13 cleaned show()
2020-12-08 __repr__ defines string representation of result object
2020-09-22 pull now pulls selected rows and returns new result
2020-07-16 note for result added to provide descriptive text to print
2020-05-08 result() now accepts path to pickled result
2020-04-08 __getitem__ allows result[tag] as a way to call pull(tag)
2020-01-20 v5 Overhaul, reformatted
"""


debug = 0

if exists(sys.argv[0][:-3]+'.res'):
    remove(sys.argv[0][:-3]+'.res')
if exists(sys.argv[0][:-3]+'.res2'):
    remove(sys.argv[0][:-3]+'.res2')

class result():
    def __init__(self, *args):
        """
        The result object is a simple database for rapid storage
        and analysis of data

        probably just a cheap version of pandas...

        add() horizontally add other results to self
        open() opens a picked result file
        plot() plot data with various options
        print() write result data to screen and/or file
        pull() pull a row from self
        save() saves a results as picked object
        show() shows the plotted data
        space() adds blank column to result
        sort() sort result by a column
        stack() vertically add results to self
        ['tag'] pulls variable 'tag' from self
        """

        self.tags = [] #Labels for each variable
        self.data = [] #Values for each variable
        self.subplots = [] #Tracks data to plot
        self.note = ''

        if args:
            if len(args) == 1 and type(args[0]) == str: #assume it's the pickled object
                self.open(args[0])
            else:
                self.data, self.tags = self.formatInput(*args)


    def __getitem__(self, tag):
        """
        returns data associated with specified tag
        """

        if tag[-1] == '*':
            matches = [i for i, val in enumerate(self.tags) if val[:-1] == tag]
        else:
            matches = [i for i, val in enumerate(self.tags) if val == tag]

        if len(matches) == 0:
            raise ValueError('ERROR: tag {} not in {}'.format(tag, self.tags))
        elif len(matches) == 1:
            index = matches[0]
        else:
            raise ValueError('ERROR: multiple tags {} in {}'.format(tag, self.tags))

        try:
            return np.array(self.data[index].copy())
        except:
            return self.data[index].copy()


    def __repr__(self):
        if len(self.data) == 0:
            return 'WARNING: NO DATA FOUND IN RESULT OBJECT'

        text = ''

        if self.note != '':
            text += '{}\n\n'.format(self.note)

        text += '\t'.join(self.tags) + '\n\n\n'

        for r in range(self.rows):
            for c in range(self.cols):
                if r < len(self.data[c]):
                    text += '{}'.format(self.data[c][r]) + '\t'*(c<self.cols-1)
                else:
                    text += '\t'

            text += '\n'

        text += '-\n\n\n'

        return text


    @property
    def cols(self):
        return len(self.data)


    @property
    def rows(self):
        return max([len(d) for d in self.data])


    def add(self, *args):
        """
        Horizontally add data to the result object
        """

        #If self is empty, create new result using newData and newTags
        if self.tags == []:
            self.data, self.tags = self.formatInput(*args)
        else:
            newData, newTags = self.formatInput(*args)
            self.data += newData[1:]
            self.tags += newTags[1:]

        #The ROW count needs to be updated if longer data comes in
        self.data[0] = [i for i in range(self.rows)]


    def copy(self):
        a = result()
        a.data = self.data.copy()
        a.tags = self.tags.copy()


    def formatInput(self, *args):
        argType = type(args[0])


        if argType in [list, np.array, np.ndarray]:
            return formatListAndArray(args[0], args[1])
        elif argType in [int, float, complex, np.float64, np.complex128, np.int32, str]:
            return formatSingleValue(*args)
        elif argType == result:
            return formatResult(args[0])
        elif argType.__name__ == 'spyctra':
            return formatSpyctra(*args)
        else:
            raise TypeError('ERROR: Received unknown type {}'.format(argType))


    def open(self, path):
        """
        Opens pickled result file
        """

        print('Opening result', path, '\n')

        with open(path, "rb") as input_file:
            a = load(input_file)

        self.__dict__.update(a.__dict__)


    def plot(self, title, x, ys, options='none'):
        """
        Call to define plots which are only made with show()
        """

        self.subplots.append([title, x, ys, options])


    def pop(self, toRemove):
        """
        Remove selected rows from result
        """

        if type(toRemove) == int:
            toRemove = [toRemove]

        toRemove.sort(reverse=True)

        for i, d in enumerate(self.data):
            if len(d) > toRemove[-1]:
                for j in toRemove:
                    if type(d) == list:
                        self.data[i].pop(j)
                    else:
                        self.data[i] = np.delete(self.data[i], j)

        self.data[0] = [i for i in range(self.rows)]


    def print(self, *path, quiet=0):
        """
        Print result object to screen/file
        """

        if path:
            path = path[0]
            writeMode = 'w'
        else:
            path = sys.argv[0][:-3]+'.res'
            writeMode = 'a'

        print('\nSAVING result in', path, '\n')

        f = open(path, writeMode)
        f.write(repr(self))
        f.close()

        if not quiet:
            print(repr(self))


    def printInd(self):
        text = ''
        for i, tag in enumerate(self.tags):
            if i == 0:
                continue
            
            text += tag + '\n'
            for val in self.data[i]:
                text += repr(val)  + '\n'
            text += '\n'

        path = sys.argv[0][:-3]+'.res2'
        writeMode = 'a'

        print('\nSAVING result in', path, '\n')

        f = open(path, writeMode)
        f.write(text)
        f.close()

        print(text)




    def pull(self, ind):
        """
        returns new result from rows ind
        """

        if type(ind) not in [list, np.ndarray]:
            ind = [ind]

        r = result()

        for i in ind:
            line = result()

            for j, c in enumerate(self.data[1:]):
                line.add(c[i],self.tags[j+1])

            r.stack(line)

        return r


    def save(self, path):
        """
        Save result object using pickle
        """

        print('Pickeling:')
        dump(self, open(path, 'wb'))
        print('    Done')


    def show(self):
        """
        Create actual images for each plot
        Useful since plt.show() stops processing until closed
        """

        c0 = time()
        print('Showing')

        titles = list(set([subplot[0] for subplot in self.subplots]))

        for title in titles:
            plt.figure(title)
            plt.suptitle(title, fontsize=16)

            subplots = [i for i in range(len(self.subplots))
                        if self.subplots[i][0] == title]
            numSubplots = len(subplots)

            if numSubplots > 6:
                raise ValueError('ERROR: {}/6 subplots requested in \'{}\':'
                                 .format(numSubplots, title))
            else:
                rows = ceil(numSubplots/2)
                if numSubplots<2:
                    numCols = 1
                else:
                    numCols = 2

                axs = [None]*len(subplots)

                for i, index in enumerate(subplots):
                    axs[i] = plt.subplot(rows, numCols, i+1)
                    self.makeSubplot(axs[i], index)

            #Button to close program
            axprev = plt.axes([0.01, 0.96, 0.03, 0.03])
            bnext = Button(axprev, 'STOP')
            bnext.on_clicked(exit)
            axprev._button = bnext

            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')


    def makeSubplot(self, ax, index):
        #self.subplots are [0] title, [1] xs, [2] ys, [3] options
        x0 = self.subplots[index][1]
        ys = self.subplots[index][2]
        options = self.subplots[index][3].lower()
        options = options.strip().split(",")


        markers = '-'
        for option in options:
            if option in '.,ov^<>12348spP*hH+xXDd|_':
                markers = option


        #parse x
        x = self.__getitem__(x0)
        if 'sortx' in options:
            sortIndexes = np.asarray(x).argsort()
            x = [x[i] for i in sortIndexes]

        #parse ys
        ys = ys.split(',')

        newYs = []
        for y in ys:
            if y[-1] != '*':
                newYs.append(y)
            else:
                for tag in self.tags:
                    if tag[:len(y)-1] == y[:-1] and tag[-3:] != 'err':
                        newYs.append(tag)
        ys = newYs

        minData =  float(np.inf)
        maxData = -float(np.inf)
        
        #do plotting
        for y0 in ys:
            yData = self.__getitem__(y0)
            
            if optionCheck(['abs'], options):
                yData = np.abs(yData)

            if optionCheck(['sort', 'sortx'], options):
                yData = [yData[i] for i in sortIndexes]

            minData = np.min([np.min(yData), minData])
            maxData = np.max([np.max(yData), maxData])

            if optionCheck(['err'], options):
                yErr = self.data[self.tags.index(y0)+1]
                plt.errorbar(x[:len(yData)], yData, yerr=yErr, linewidth=2.0, label=y0)
            else:
                plt.plot(x[:len(yData)], yData, markers, linewidth=2.0, label=y0)

        ax.legend()

        if not optionCheck(['raw'], options):
            ymin, ymax = plt.ylim()
            if (ymax-ymin)>10*(maxData-minData):
                plt.ylim(ymax=maxData*1.1)
                plt.ylim(ymin=minData*1.1)

        if optionCheck(['x0','org'], options):
            xmin, xmax = plt.xlim()

            if xmin > 0:
                plt.xlim(xmin=0)
            if xmax < 0:
                plt.xlim(xmax=0)

        if optionCheck(['y0','org'], options):
            ymin, ymax = plt.ylim()

            if ymin > 0:
                plt.ylim(ymin=0)
            if ymax < 0:
                plt.ylim(ymax=0)

        ax.set_xlabel(x0, fontsize=16)

        if optionCheck(['logx','loglog'], options):
            ax.set_xscale("log", nonpositive='clip', base=10)

        if optionCheck(['logy','loglog'], options):
            ax.set_yscale("log", nonpositive='clip', base=10)

        if not optionCheck(['nogrid'], options):
            ax.grid()


    def sort(self, tag):
        c0 = time()
        print('\nSorting', end=' ')

        try:
            ind = self.tags.index(tag)
            print('by', tag)
        except:
            raise ValueError('ERROR: no tag', tag, 'in', self.tags)

        sortIndexes = np.asarray(self.data[ind]).argsort()

        for j, col in enumerate(self.data[1:]):
            if self.tags[j+1] != '':
                self.data[j+1] = [col[i] for i in sortIndexes]


    def space(self):
        self.add([''],'')


    def stack(self, result):
        if self.data == []:
            self.data = deepcopy(result.data)
            self.tags = result.tags
        else:
            for i in range(len(self.tags)):
                if self.tags[i] != result.tags[i]:
                    raise ValueError('ERROR: Tags don\'t match:',
                                     self.tags, '\n', result.tags)

                if type(result.data[i]) == list:
                    #There will ALWAYS be at least 2 lists in data!!!
                    self.data[i] += deepcopy(result.data[i])
                elif type(result.data[i]) == np.ndarray:
                    self.data[i] = np.append(self.data[i], deepcopy(deepcopy(result.data[i])))

        for i in range(len(self.data[0])):
            self.data[0][i] = i


def formatSingleValue(val, tag):
    if type(val) == str:
        newData = [[0], [val]]
    else:
        newData = [[0], np.array([val])]
    newTags = ['ROW', tag]
    return newData, newTags


def formatSpyctra(a, comps='RIM'):
    from spyctra import spyctra
    newTags = ['ROW']
    newData =[[i for i in range(a.points)]]

    if a.space == 's':
        newTags.append('Time (s)')
    else:
        newTags.append('Freq (Hz)')
    newData.append(a.x)

    for i in range(a.count):
        if 'R' in comps:
            newData.append(a.data[i].real)
            newTags.append('REAL_' + repr(i))
        if 'I' in comps:
            newData.append(a.data[i].imag)
            newTags.append('IMAG_' + repr(i))
        if 'M' in comps:
            newData.append(abs(a.data[i]))
            newTags.append('MAGN_' + repr(i))

    return newData, newTags


def formatResult(a):
    if debug:
        print('Formating result')
    return a.data, a.tags


def formatListAndArray(a, tags):
    if debug:
        print('Formating list or array')

    newTags = ['ROW']

    if ',' in tags:
        tags = tags.split(',')
    #check if 2d data passed
    if type(a[0]) in [list, np.ndarray, np.array]:
        if debug:
            print('2d data')
        newData = [[i for i in range(len(a[0]))]] #For ROW column

        for data in a:
            newData.append(data)

        #used supplied tags or add suffix automatically
        if type(tags) == list:
            newTags += tags
        else:
            newTags += ['{}_{}'.format(tags,i) for i in range(len(a))]

    else:
        newData = [[i for i in range(len(a))]]
        newData.append(a)
        newTags.append(tags)

    if debug:
        print(' ',newTags)

    return newData, newTags


def optionCheck(valid, options):
    return any(x in valid for x in options)


def testSuite():
    from spyctra import fakeSpyctra, spyctra
    a = spyctra()
    for i in range(2):
        a.add(fakeSpyctra(points=8,
                          t2=2.3e-3,
                          delta=1e-5,
                          noise=200,
                          offRes=100,
                          phase=i,
                          amp=10**(5-i),
                          meta={'a':[i],'b':[10-i]},
                          seed=i))
    b = spyctra()
    for i in range(2):
        b.add(fakeSpyctra(points=8,
                          t2=2.3e-3,
                          delta=1e-5,
                          noise=200,
                          offRes=100,
                          phase=i,
                          amp=10**(5-i),
                          meta={'a':[i],'b':[10-i]},
                          seed=10+i*2))

    r = result()
    r.add(a)
    r.space()
    #r.add(result(b))
    r.sort('Time (s)')
    r.add([0,1,2,3], 'atime')
    r.add(a.data[0][0],'cw')
    r.add([0,1,2,3], 'atime')
    r.add(a.findPhase(), 'phi')

    r.plot('1','Time (s)','MAGN_0')
    r.plot('1','Time (s)','REAL_0,IMAG_0')
    r.plot('1','Time (s)','REAL_0,IMAG_0')
    r.plot('1','Time (s)','REAL_0,IMAG_0')
    r.plot('1','Time (s)','REAL_0,IMAG_0')
    r.plot('1','ROW','phi')
    r.show()
    plt.show()
    r.print()


def work():
    #"""
    r = result()
    r.add([i*2.1 for i in range(10)],'a')
    r.add([i**2 for i in range(10)],'a_err')
    
    r.plot('1','ROW','a','*,logx,lo')
    r.show()
    plt.show()

if __name__ == "__main__":
    #testSuite()
    work()