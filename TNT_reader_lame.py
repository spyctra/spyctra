"""
Generic TNT file reader

Send me files that this can't read and I'll try to take a look
Michael Malone: mwmalone@gmail.com

There is a non-lame version...
"""

from numpy import fromiter, array
from struct import unpack
from time import perf_counter as time
from _io import BufferedReader
import traceback
import traceback

"""
CHANGE LOG

2025-09-14 Initial release
"""

debug = 0 #prints metadata as it is processed
timer = 0 #prints times of individual functions
telling = 0 #debugging tool stating location of new metadata params
skip_meta = 0 #stops loading metadata
quiet = 0 #halts output, useful when reading many files
fine = 0 #painfully describes what's going on

class TNT_BufferedReader(BufferedReader):
    def chomp(self, chomps):
        #useful for debugging
        a0 = self.tell()

        for i in range(chomps):
            print(i, self.read(1))

        self.seek(a0, 0)


    def reads(self, length, name=''):
        if fine:
            print('\n')
            print(f'reads {name}: length {length}')

            self.chomp(length)

        return self.read(length)


    def read_str(self, length, name=''):
        if fine:
            print(f'\nread_str {name}: length {length}')

            self.chomp(length)

        string = self.read(length)

        value = ''

        for i in string:
            if 1 < i < 128:
                c = str(chr(i))
                value += c

        value = value.replace('\n', ' ')
        value = value.replace('\r', ' ')

        return value


    def read_double(self, name=''):
        if fine:
            print(f'\nread_double {name}')

            self.chomp(8)

        return unpack('<d', self.read(8))[0]


    def read_int(self, name=''):
        if fine:
            print(f'\nread_int {name}')

            self.chomp(4)

        return unpack('<i', self.read(4))[0]


    def read_long(self, name=''):
        if fine:
            print(f'\nread_long {name}')

            self.chomp(4)

        return unpack('<l', self.read(4))[0]


    def read_short(self, name=''):
        if fine:
            print(f'\nread_short {name}')

            self.chomp(2)

        return unpack('<h', self.read(2))[0]


def t1(t0):
    return round(1000*(time()-t0), 1)


def read_structure(a):
    t0 = time()
    struct_meta = {}
    struct_meta['versionID'] = a.read_str(8)
    struct_meta['TMAGtag'] = a.read_str(4)
    boolean = a.read_int()
    structure_length = a.read_int()

    if structure_length != 1024:
        #never observed
        print('\nERROR: Expecting structure length of 1024, received {structure_length}')
        exit()
    else:
        #Number of points and scans in all dimensions
        struct_meta['npts'] = [a.read_long() for _ in range(4)]
        struct_meta['actual_npts']  = [a.read_long() for _ in range(4)]
        struct_meta['acq_points'] = a.read_long()
        struct_meta['npts_start'] = [a.read_long() for _ in range(4)]
        struct_meta['scans'] = a.read_long()
        struct_meta['actual_scans'] = a.read_long()
        struct_meta['dummy_scans'] = a.read_long()

        struct_meta['repeat_times'] = a.read_long()
        struct_meta['sa_dimension'] = a.read_long()
        struct_meta['sa_mode'] = a.read_long()

        #space

        #Field and Frequencies
        struct_meta['magnet_field'] = a.read_double()
        struct_meta['ob_freq'] = [a.read_double() for _ in range(4)]
        struct_meta['base_freq'] = [a.read_double() for _ in range(4)]
        struct_meta['offset_freq'] = [a.read_double() for _ in range(4)]
        struct_meta['ref_freq'] = a.read_double()

        struct_meta['NMR_frequency'] = a.read_double()
        struct_meta['obs_channel'] = a.read_short()
        a.read(42)

        #Spectral width, dwell and filter
        struct_meta['sw'] = [a.read_double() for _ in range(4)]
        struct_meta['dwell'] = [a.read_double() for _ in range(4)]
        struct_meta['filter'] = a.read_double()
        struct_meta['experiment_time'] = a.read_double()
        struct_meta['acq_time'] = a.read_double()
        struct_meta['last_delay'] = a.read_double()

        struct_meta['spectrum_direction'] = a.read_short()
        struct_meta['hardware_sideband'] = a.read_short()
        struct_meta['Taps'] = a.read_short()
        struct_meta['Type'] = a.read_short()

        struct_meta['bDigRec'] = a.read_int()
        struct_meta['nDigitialCenter'] = a.read_long()
        a.read(16)

        #Hardware settings
        struct_meta['transmitter_gain'] = a.read_short()
        struct_meta['receiver_gain'] = a.read_short()
        struct_meta['number_of_receivers'] = a.read_short()
        struct_meta['RG2'] = a.read_short()
        struct_meta['receiver_phase'] = a.read_double()
        a.read(4)


        #Spinning speed information:
        struct_meta['set_spin_rate'] = a.read_short()
        struct_meta['actual_spin_rate'] = a.read_short()

        #Lock information:
        struct_meta['lock_field'] = a.read_short()
        struct_meta['lock_power'] = a.read_short()
        struct_meta['lock_gain'] = a.read_short()
        struct_meta['lock_phase'] = a.read_short()
        struct_meta['lock_freq_mhz'] = a.read_double()
        struct_meta['lock_ppm'] = a.read_double()
        struct_meta['H2O_freq_ref'] = a.read_double()
        a.read(16)

        #VT Information
        struct_meta['set_temperature'] = a.read_double()
        struct_meta['actual_temperature'] = a.read_double()

        #Shim Information
        struct_meta['shim_units'] = a.read_double()
        struct_meta['shims'] = [a.read_short() for _ in range(36)]
        struct_meta['shim_FWHM'] = a.read_double()

        #Bruker
        struct_meta['HH_dcpl_attn'] = a.read_short()
        struct_meta['DF_DN'] = a.read_short()
        struct_meta['F1_tran_mode'] = [a.read_short() for _ in range(7)]
        struct_meta['dec_BW'] = a.read_short()

        struct_meta['grd_orientation'] = [a.read_str(1) for _ in range(4)]
        struct_meta['Latch_LP'] = a.read_long()
        struct_meta['grd_Theta'] = a.read_double()
        struct_meta['grd_Phi'] = a.read_double()
        a.read(264)

        #Time variables
        struct_meta['start_time'] = a.read_long()
        struct_meta['finish_time'] = a.read_long()
        struct_meta['elapsed_time'] = a.read_long()

        #Text variables
        struct_meta['date'] = a.read_str(19)
        struct_meta['data2'] = a.read_str(13)

        struct_meta['nucleus'] = a.read_str(16)
        struct_meta['nucleus_2D'] = a.read_str(16)
        struct_meta['nucleus_3D'] = a.read_str(16)
        struct_meta['nucleus_4D'] = a.read_str(16)
        struct_meta['sequence'] = a.read_str(32)
        struct_meta['lock_solvent'] = a.read_str(16)
        struct_meta['lock_nucleus'] = a.read_str(16)

        if debug:
            print('\n  TNT Structure:')

            for elem in struct_meta:
                print(f'{elem}: {struct_meta[elem]}')

    if timer:
        print(f'  Read structure in {t1(t0)} ms')

    return struct_meta


def read_data(a):
    t0 = time()
    a.read(8)
    data_length = a.read_int()//8

    data = fromiter((complex(unpack('<f', a.read(4))[0],
                             unpack('<f', a.read(4))[0])
                             for i in range(data_length)),
                             dtype=complex)

    if timer:
        print(f'  Read {len(data)} complex points in {t1(t0)} ms')

    return data


def read_variable(a, meta, c):
    def read_string_careful(a, name=''):
        l = a.read_int()

        if 0 <= l <= 40:
            val = a.read_str(l, name)
        else:
            return 0

        if val in['Every pass', '+ Add']:
            return 0

        return val


    if meta['versionID'] != 'TNT1.001':
        name = read_string_careful(a, 'Variable Name')

        if type(name) == int:
            return 0
        elif name in['Every pass','+ Add']:
            return 0

        a.reads(4)

        data = read_string_careful(a, 'Variable Value')

        if type(data) == int:
            return 0

        a.reads(4)
        read_string_careful(a)
        read_string_careful(a)
        a.reads(8)
        read_string_careful(a)
        name2 = read_string_careful(a, 'Variable Name 2')
        read_string_careful(a)
        read_string_careful(a)
        a.reads(12)
    else:
        name = read_string_careful(a)

        if type(name) == int:
            return 0

        data = read_string_careful(a)

        if type(data) == int:
            return 0

        a.reads(4)
        read_string_careful(a)
        read_string_careful(a)
        a.reads(4)

    if debug:
        print(f'{name} = {data}')

    return {name: data}


def find_and_read_variables(a, meta):
    #import io
    #global fine
    #fine = 1
    #a.seek(0, io.SEEK_END)
    #a.seek(144509)
    #a.chomp(10)

    trials = 0
    a0 = a.tell()

    done = False

    while not done:
        var_meta = {}

        a.seek(a0 + trials, 0)
        trials += 1

        if debug:
            a.chomp(10)
            input('advance? ')

        try:
            num_vars = a.read_int()

            if debug:
                print(f' {a.tell()}')
                print(f' {num_vars = }')

            if num_vars < 0 or num_vars > 100:
                if debug:
                    print('bad num_vars', num_vars)

                continue

            for _ in range(num_vars):
                temp = read_variable(a, meta, _)

                if type(temp) is dict:
                    var_meta.update(temp)
                else:
                    if debug:
                        print('bad var read')

                    continue

            if num_vars > 1 and len(var_meta) == num_vars:
                done = True

        except Exception as e:
            traceback.print_exc()

    if debug:
        for e in var_meta:
            print(f'{e}:{ var_meta[e]}')

        input('advance? ')

    return var_meta


def TNT_reader(path, options=''):
    t0 = time()
    options = options.lower()

    global debug
    global timer
    global telling
    global skip_meta
    global quiet

    if 'debug' in options:
        debug = 1

    if 'skip_meta' in options:
        skip_meta = 1

    if 'quiet' in options:
        quiet = 1

    if 'timer' in options:
        timer = 1

    if 'telling' in options:
        telling = 1

    if path[-4:] != '.tnt':
        path += '.tnt'

    if not quiet:
        print(path)

    a = TNT_BufferedReader(open(path, 'rb'))

    meta = {}
    meta.update(read_structure(a))
    data = read_data(a)

    """
    meta = { 'table_pass':True
            ,'versionID':'TNT1.008'}
    a.read(144390)

    read_variables(a, meta)
    exit()
    """

    if not skip_meta:
        meta.update(find_and_read_variables(a, meta))

    return data, meta


def main():
    path = '../spyctraRep/TNT/test_files/'
    #data, meta = TNT_reader(path + 'slse_0') #1.003
    #exit()
    data, meta = TNT_reader(path + 'GJL001_pPAPs_f1=15_02272019_0cm_open_vary_excite') #1.005
    data, meta = TNT_reader(path + 'Sensor2314_InterferenceRejection_1p31Mhz_F1=10_11292018_v5') #1.003 huge
    data, meta = TNT_reader(path + 'SpinLock_SPinEcho_02212019_0cm_Variable_Refocusing_weekend') #1.005
    data, meta = TNT_reader(path + 'AFPLockSweep_1m') #TNT1.003
    data, meta = TNT_reader(path + 'FindWY_1m') #1.003
    data, meta = TNT_reader(path + 'GD460_VaryTauY27us_Exp_Pu19_F1_99_32by20_Pu28.263_794.994nm_Pr27.6CnmGolY0SadX47mATs157Vt43mV.tnt') #1.003
    data, meta = TNT_reader(path + 'Cu2O_spinEcho_v3') #good 1.005
    data, meta = TNT_reader(path + '31P_1pulse') #1.005
    data, meta = TNT_reader(path + 'FID_423_v10') # 1.008
    data, meta = TNT_reader(path + 'EXP_00') #1.005
    data, meta = TNT_reader(path + 'slse_0') #1.001
    data, meta = TNT_reader(path + 'LW_0') #1.001
    data, meta = TNT_reader(path + 'NQRI_SLSE_5') #1.008
    data, meta = TNT_reader(path + 'slse_0_e90180') #1.005
    data, meta = TNT_reader(path + '1H_H2O_CPMG_blank_2048_5u_32') #1.005
    data, meta = TNT_reader(path + 'N68_012816_AN_QFS_150cc_flush__001') #1.008
    data, meta = TNT_reader(path + 'SLSE_00') #1.003
    data, meta = TNT_reader(path + 'A0_1008_2024_F-HCl_coil12_150K_cpmg') #1.008
    data, meta = TNT_reader(path + 'A1_1008_2024_F-HCl_coil12_150K_cpmg_shorter_tau') #1.003

    for e in meta:
        print(f'{e}: { meta[e]}')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(data.real)
    plt.plot(data.imag)
    plt.show()


if __name__ == '__main__':
    main()
