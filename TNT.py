"""
Spyctra's TNT file reader.
"""

from datetime import datetime
from os.path import getmtime
from spyctra import spyctra
from time import ctime, perf_counter as time

import numpy as np
import TNT_reader_lame as TNT

"""
CHANGE LOG

2025-09-14 Initial release
"""

debug = 0 #debugging option
quiet = 0 #halts output, useful when reading many files
timer = 0 #prints times of individual functions


def t1(t0):
    return round(1000*(time()-t0), 1)


def parse_path_and_options(path_data, options=''):
    t0 = time()

    global quiet

    if type(path_data) == list:
        directory, filename = path_data[0], path_data[1]
    else:
        path = path_data.replace('\\','/')
        file_start = path.rfind('/') + 1
        filename = path[file_start:]
        directory = path[:file_start]

    if filename[-4:] != '.tnt':
        filename += '.tnt'

    if directory == '':
        directory = './'

    meta = { 'directory': directory
            ,'filename': filename
            ,'file_time': ctime(getmtime(directory + filename)) #OS time
           }

    if 'quiet' in options:
        quiet = 1

    if debug:
        print('\n initialization')

        for e in meta:
            print(e, meta[e])

    if timer:
        print(f'\n  Opened file, parsed path and options in {t1(t0)} ms')

    return meta


def make_spyctra(data, meta, t0):
    t00 = time()

    def remove_flag(key):
        if key in meta:
            del meta[key]

    remove_flag('pseq_pass')
    remove_flag('table_pass')
    remove_flag('variable_pass')

    date, raw_time = meta['date'].split()
    d = [int(d) for d in date.split('/')]
    t = [int(t) for t in raw_time.replace('\x00','').split(':')]
    time_stamp = datetime(d[0], d[1], d[2], t[0], t[1], t[2]).timestamp()

    freq = round(meta["ob_freq"][meta["obs_channel"]-1]*1e6, 1)

    if not quiet:
        print(  f'\tFreq: {freq}  '
              + f'Dwell Time: {meta["dwell"][0]}  '
              + f'Rec. Gain: {meta["receiver_gain"]}  '
              + f'Data: {meta["actual_scans"]}x{meta["actual_npts"]}  '
              + f'Tecmag Date: {meta["date"]}  '
              + f'Start Time: {raw_time}  {meta["file_time"]} '
              + f'Read in {t1(t0)} ms'
             )

    #normalize by scans because duh
    data /= meta['actual_scans']

    #create single object with all data
    plain_data = spyctra( data=[data]
                         ,freq=np.array([freq])
                         ,delta=meta['dwell'][0]
                         ,space='s'
                         ,time=np.array([time_stamp])
                         ,phase=np.array([0.0])
                         ,start=0
                         )

    count = meta['npts'][1]
    plain_data.new_count(count)

    meta['Read Time (ms)'] = t1(t0)

    for m in meta:
        meta[m] = [meta[m]]*count

    plain_data.meta = meta

    if timer:
        print(f'\nMade Spyctra in {t1(t00)} ms')

    return plain_data


def TNT_reader(path_data, options=''):
    t0 = time()

    meta = parse_path_and_options(path_data, options)
    data, meta1 = TNT.TNT_reader(meta['directory'] + meta['filename'], options)
    meta.update(meta1)

    return make_spyctra(data, meta, t0)


def read(path=None, *options):
    from file_reader import master_reader

    return master_reader(path, '.tnt', *options)


def test_suite():
    path = '../spyctraRep/TNT/test_files/'

    """
    TNT_reader(path + 'EXP_00') #1.005
    exit()
    #"""

    TNT_reader(path + 'GJL001_pPAPs_f1=15_02272019_0cm_open_vary_excite') #1.005
    TNT_reader(path + 'Sensor2314_InterferenceRejection_1p31Mhz_F1=10_11292018_v5') #1.003
    TNT_reader(path + 'SpinLock_SPinEcho_02212019_0cm_Variable_Refocusing_weekend') #1.005
    TNT_reader(path + 'AFPLockSweep_1m') #TNT1.003
    TNT_reader(path + 'FindWY_1m') #1.003
    TNT_reader(path + 'GD460_VaryTauY27us_Exp_Pu19_F1_99_32by20_Pu28.263_794.994nm_Pr27.6CnmGolY0SadX47mATs157Vt43mV.tnt') #1.003
    TNT_reader(path + 'Cu2O_spinEcho_v3') #good 1.005
    TNT_reader(path + '31P_1pulse') #1.005
    TNT_reader(path + 'FID_423_v10') # 1.008
    TNT_reader(path + 'EXP_00') #1.005
    TNT_reader(path + 'slse_0') #1.001
    TNT_reader(path + 'LW_0') #1.001
    TNT_reader(path + 'NQRI_SLSE_5') #1.008
    TNT_reader(path + 'slse_0_e90180') #1.005
    TNT_reader(path + '1H_H2O_CPMG_blank_2048_5u_32') #1.005
    TNT_reader(path + 'N68_012816_AN_QFS_150cc_flush__001') #1.008
    TNT_reader(path + 'SLSE_00') #1.003
    TNT_reader(path + 'A0_1008_2024_F-HCl_coil12_150K_cpmg') #1.008
    TNT_reader(path + 'A1_1008_2024_F-HCl_coil12_150K_cpmg_shorter_tau') #1.003

    #"""
    exit()

    import matplotlib.pyplot as plt

    a = TNT_reader(path + 'Cu2O_spinEcho_v3', 'debug') #1.005

    a.plot()
    plt.show()

if __name__ == '__main__':
    test_suite()
