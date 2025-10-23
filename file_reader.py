"""
The function master_reader() is called from individual file readers: TNT, SDF, TREEV2...

It does sanity checks on the path sent to master_reader() to help
debug issues with using the correct folder and identifies issues
in the options sent to the individual file reader

final options are a single string

The code will create a list of suffixes to search for valid files
 pass a list and it will use that list
 pass an int and it will look for files with suffixes '_#' up to #=int-1
 pass a * and it will return all files matching the filename with option to limit
 pass text and it will send that to the individual file readers as options
and then return a spyctra obsect [or list of spyctras] of the found files
"""

from os import listdir, scandir
from os.path import exists, getmtime, isdir, split
from spyctra import spyctra, list_print
from time import perf_counter as time

import sys
import numpy as np

"""
CHANGE LOG

2025-10-22 forced params numbered_files and text_options and now allow kwargs and send to reader
2025-09-18 GUI option use_GUI added
2025-09-07 Initial release
"""

debug = 0


def t1(t0):
    return round(1000*(time()-t0), 1)


def master_reader(path, extension, numbered_files=None, text_options='', **kwargs):
    t0 = time()

    if path is not None:
        directory, filename = parse_path(path)
        suffixes, text_options = parse_options(numbered_files, text_options)

        filenames = match_files(directory, filename, suffixes, extension)
    else:
        directory, filenames, text_options = use_GUI(extension)


    return reader(directory, filenames, extension, text_options, t0, **kwargs)


def parse_path(path):
    """
    Parses path to get the directory and filename/fragment

    If no valid directory is found searches for a valid directory
    and prints the subfolders in that directory
    """

    path = path.replace('\\','/')
    directory, filename = split(path)

    if directory == '':
        directory = './'
    if directory[-1] != '/':
        directory += '/'

    if debug:
        print('\nparse_path()')
        print(f' filename: {filename}')
        print(f' directory: {directory}')

    if isdir(directory):
        return directory, filename
    else:
        print(f'\nERROR: Cannot find directory {directory}')

        while directory.find('/') >= 0:
            directory = directory[:directory[:-1].rfind('/') + 1]

            print(f'  Searching for directory: {directory}', end = '')

            if isdir(directory):
                print(' FOUND')
                print('    Valid subdirectories')

                subdirs = [f.path[len(directory):] for f in scandir(directory) if f.is_dir()]

                for subdir in subdirs:
                    print(f'      {subdir}')

                break

            print(' FAILED')

        sys.exit()


def parse_options(numbered_files, text_options):
    if debug:
        print(f'{numbered_files = }')
        print(f'{text_options = }')

    suffixes = []

    if numbered_files is not None:
        if type(numbered_files) == int:
            suffixes = [repr(i) for i in range(numbered_files)]
        elif type(numbered_files) in([list, np.array]):
            suffixes = [repr(i) for i in numbered_files]
        elif type(numbered_files) == str: #nasty hack in case you don't specify numbered files
            text_options = numbered_files
        else:
            raise TypeError(f'ERROR: numbered_files is unsupported type {type(numbered_files)}')

    text_options = text_options.lower()

    if debug:
        print('\nparse_options()')
        print(f' suffixes: {suffixes}')
        print(f' options: {text_options}')

    return suffixes, text_options


def match_files(directory, filename, suffixes, extension):
    if '*' == filename[-1]:
        return match_starred_files(directory, filename, suffixes, extension)
    elif len(suffixes) == 0:
        return match_single_file(directory, filename, extension)
    elif len(suffixes) > 0:
        return match_suffixed_files(directory, filename, suffixes, extension)


def match_single_file(directory, filename, extension):
    if exists(directory + filename + extension):
        if debug:
            print('\nmatch_single_file()')
            print(f' found {directory + filename + extension}')

        return [filename + extension]
    else:
        print(f'\nERROR: No file {filename + extension} found in {directory}')

        print_possible_files(directory, extension)

        sys.exit


def match_starred_files(directory, filename, suffixes, extension):
    t0 = time()

    filename = filename[:-1] #remove the *

    matched_files = [f for f in listdir(directory)
                     if (f[:len(filename)] == filename
                     and f[-len(extension):] == extension)] #check name and extension are correct

    if len(matched_files) > 0:
        try:
            #if format is file_num.ext sort by num
            matched_files = sorted(matched_files, key = lambda x: int(x.split('_')[-1].split('.')[0]))
        except:
            matched_files.sort()

        print(f'\nFound {len(matched_files)} {extension} files matching {filename}*')

        if len(suffixes) > 0: #if user asked for a subset of file
            print(f'Pulling {len(suffixes)} files')

            matched_files = matched_files[:min(len(matched_files), len(suffixes))]

        if debug:
            print('\nmatch_starred_files()')

            for f in matched_files:
                print(' {f}')

        return matched_files
    else:
        print(f'\nERROR: No file {filename}*{extension} files found in \n {directory}')

        print_possible_files(directory, extension)

        sys.exit()


def match_suffixed_files(directory, filename, suffixes, extension):
    print(f'\nTrying to read {len(suffixes)} {extension} files matching {directory + filename}')

    missing_files = []
    matched_files = [None]*len(suffixes)
    found = 0

    for suffix in suffixes:
        suf = suffix
        match = False

        for i in range(3):
            if exists(directory + filename + suf + extension):
                matched_files[found] = filename + suf + extension
                found += 1
                match = True
                break

            suf = '0' + suf #historical artifact from when suffixes looked like _001, _002, etc.

        if match == False:
            missing_files.append(suffix)

    if len(missing_files) > 0:
        print(f'WARNING: Did not find {len(missing_files)} file[s]')

        if found == 0:
            print(f'\nERROR: No {filename} files found in \n {directory}')

            print_possible_files(directory, extension)

            sys.exit()

    return matched_files[:found]


def reader(directory, filenames, extension, text_options, t0, **kwargs):
    if extension == '.tnt':
        from TNT import TNT_reader

        func = TNT_reader
    elif extension == '.sdf':
        from SDF import SDF_reader

        func =  SDF_reader
    elif extension == '.tdms':
        from TREEV2 import TREEV2_reader

        func =  TREEV2_reader

    if extension != '.sdf':
        a = spyctra()

        for filename in filenames:
            a.add(func(f'{directory}{filename}', text_options, **kwargs))
    else:
        a = []

        for filename in filenames:
            a += func(f'{directory}{filename}', text_options, **kwargs)

    if len(filenames) != 1:
        print(f'Read {a.count} in {t1(t0)} ms\n')

    return a


def print_possible_files(directory, extension):
    from numpy.random import choice

    possible = [f for f in listdir(directory) if (f[-len(extension):] == extension)]

    print(f'\n  Found {len(possible)} {extension} files')

    for file in choice(possible, min(10, len(possible))):
        print(f'   {file}')

    sys.exit()


def use_GUI(extension):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    if '.tnt' in extension:
        label = 'TNT'
    if '.tdms' in extension:
        label = 'TREEV2'
    if '.sdf' in extension:
        label = 'SDF'

    # Open the file dialog
    file_paths = filedialog.askopenfilenames(title='Select File[s]',
                                             initialdir='./',  # Start directory
                                             filetypes=((f'{label} files', '*' + extension), ('All files', '*.*'))
                                            )
    root.destroy()

    if file_paths:
        directory = split(file_paths[0])[0] + '/'
        filenames = [split(file_path)[1] for file_path in file_paths]

        return directory, filenames, ''
    else:
        print("No file selected.")
        sys.exit()


def test_suite():
    options = ['', 'debug', 'nometa', 'quiet', 'nometa,quiet']
    #options = ['']
    #"""

    master_reader('../spyctra_rep/TNT/exp1_385/slse_', '.tnt', 4, joy='loveless')
    exit()
    master_reader('FID_0', '.tnt')
    master_reader('./FID_0', '.tnt')
    master_reader('../spyctra_rep/TNT/exp1_385/FID_0', '.tnt')
    master_reader(None, '.tnt')
    exit()

    for option in options:
        master_reader('../spyctra_rep/TNT/exp1_385/slse_*', '.tnt', option)
        master_reader('../spyctra_rep/TNT/exp1_385/slse_*', '.tnt', 10, option)
        master_reader('../spyctra_rep/TNT/exp1_385/slse_*', '.tnt', [i for i in range(5)], option)

        master_reader('../spyctra_rep/TNT/exp1_385/slse_0', '.tnt', option)
        master_reader('../spyctra_rep/TNT/exp1_385/slse_', '.tnt', 10, option)
        master_reader('../spyctra_rep/TNT/exp1_385/slse_', '.tnt', [i for i in range(5)], option)

        master_reader('../spyctra_rep/TREEV2/CPMG_*', '.tdms', option)
        master_reader('../spyctra_rep/TREEV2/CPMG_*', '.tdms', 10, option)
        master_reader('../spyctra_rep/TREEV2/CPMG_*', '.tdms', [i for i in range(5)], option)

        master_reader('../spyctra_rep/TREEV2/CPMG_00', '.tdms', option)
        master_reader('../spyctra_rep/TREEV2/CPMG_', '.tdms', 10, option)
        master_reader('../spyctra_rep/TREEV2/CPMG_', '.tdms', [i for i in range(5)], option)
    #"""
    #error cases
    #master_reader('../spyctraRep/TNTno/a/a/a/a/a/a/slse_*', '.tnt') #no folder
    #master_reader('../spyctraRep/TNT/slsb_*', '.tnt') #no starred files
    #master_reader('../spyctraRep/TNT/slsb_*', '.tnt', 10) #no starred files
    #master_reader('../spyctraRep/TNT/slse_', '.tnt', [100+i for i in range(10)]) #no suffics files
    master_reader('../spyctraRep/TNT/slse_999', '.tnt') #no file


def main():
    test_suite()


if __name__ == '__main__':
    main()
