"""
V6
Michael Malone: mwmalone@gmail.com
Send me files that this can't read and I'll try to take a look

Code based heavily on Cris LaPierre's MATLAB code developed
at Dr. Matthew Rosen's lab (rosen@cfa.harvard.edu, rosenlab.org)
at the Athinoula A. Martinos Center for Biomedical Imaging
at Massachusetts General Hospital with Harvard Medical School

Used with permission

Citation of original code:
Title: Read_TNT.m or Read_TNT_pseq_delays.m
Author: Cris LaPierre
Date: 2012-08-03
Code version: 2.1.1
"""

from numpy import fromiter, array
from struct import unpack
from time import perf_counter as time
from _io import BufferedReader

import traceback

"""CHANGE LOG
2023-06-03 V6 cleanup, fstrings, etc.
2022-03-11 Sequence printout cleaner; table reader hacked for Dave P.
2021-12-01 Code breakout for Tecmag
2020-12-07 Code refactored for improved readability and consistency
2020-07-14 read_variables modified to handle variable references
2019-07-26 reader object created, tables and variables validated on entire test suite
2019-07-17 v5 tested and released.
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
        #if chomps > 400:
        #    exit()
        for i in range(chomps):
            print(i, self.read(1))

        self.seek(a0,0)


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


def read_pulse_sequence(a):
    t0 = time()
    pseq_meta = {}

    if telling:
        print(f'\nPulse sequence starts @ {a.tell()}')

    if True:
    #try:
        TMG2_tag = a.read_str(4)
        boolean = a.read_int()
        TMG2_data = a.read(a.read_int()) #typically 4 bytes
        PSEQ_tag = a.read_str(4)

        if PSEQ_tag != 'PSEQ':
            raise ValueError('WARNING: Could not find "PSEQ" tag')

        if a.read_int():
            pseq_meta['rev'] = a.read_str(8)
            a.read_str(a.read_int())

        if pseq_meta['rev'] == '1.18 BIN':
            a.read(8)
            a.read(a.read_int())
            a.read(a.read_int())

        pseq_meta['num_fields'] = a.read_int()
        pseq_meta['num_events'] = a.read_int()

        if debug:
            print(f'\n  Pulse Sequence: num_fields {pseq_meta["num_fields"]}, num_events {pseq_meta["num_events"]}')

        for field in range(pseq_meta['num_fields']):
            if telling:
                print(f'New seq field @ {a.tell()}')

            _ , field_pass = read_pseq_field(a, 60) #currently not working with returned field

        pseq_meta['pseq_pass'] = True

        if timer:
            print(f'  Read pulse sequence in {t1(t0)} ms')
    else:
    #except:
        pseq_meta['pseq_pass'] = False

        print('Warning: Could not read pulse sequence')

    return pseq_meta


def read_pseq_field(a, entry_bytes):
    num_local_events = a.read_int()
    field = ['']*num_local_events
    field_pass = True
    a.read(24)

    for event in range(num_local_events):
        if event == 0:
            a.read_str(a.read_int())
            label_str = a.read_str(a.read_int())
            a.read_str(a.read_int()) #identical to label_str?
            a.seek(entry_bytes - 4, 1)
            field[0] = label_str
        else:
            raw = ''
            val = ''
            for bytez in range(entry_bytes//4):
                nm_size = a.read_int()

                if nm_size > 0:
                    if bytez in [0, 1, 3, 5, 7, 9, 11]:
                        val = a.read_str(nm_size)
                    elif bytez == 14:
                        if nm_size == 1:
                            a.read_str(a.read_int())
                            a.read(a.read_int())
                            a.read(a.read_int())
                            a.read(a.read_int())
                            a.read_str(a.read_int())
                            val = a.read_int()
                            a.seek(2, 1)
                    else:
                        pass #might be bytez should be read

                    field[event] = repr(val)
    if debug:
        print(','.join(field))

    return field, field_pass


def read_tables(a, meta):
    t0 = time()
    table_meta = {}

    if not meta['pseq_pass']:
        return {'table_pass': False}

    if telling:
        print(f'\nTables start @ {a.tell()}')

    try:
        a0 = a.tell()
        num_0 = a.read_int()
        a.seek(a0)
        if meta['versionID'] in(['TNT1.001', 'TNT1.003']) and num_0*4 != 128:
            num_tables = a.read_int()
        else:
            a.read(a.read_int()*4)
            num_tables = a.read_int()

        if debug:
            print(f'\n  Tables: {num_tables}')

        for _ in range(num_tables):
            if telling:
                print(f'Table @ {a.tell()}')

            table, table_pass = read_any_table(a, meta)

            if table_pass:
                table_meta.update(table)
            else:
                print('WARNING: Unable to read tables')
                return

        a0 = a.tell()
        if meta['versionID'] in ['TNT1.003']: #might have hidden tables
            check = True
            found = 0

            while check:
                flag = a.read(8)
                seq = a.read(8)
                a.seek(-16, 1)

                if seq == b'Sequence':
                    break

                table, table_pass = read_any_table(a, meta)

                if table_pass:
                    #table_meta.update(table) #currently not storing table
                    a0 = a.tell()
                    found += 1
                else:
                    check = False
                    a.seek(a0)
            if found > 0 and not quiet:
                print(f'WARNING: Found {found} extra tables')

        table_meta['table_pass'] = True

        if timer:
            print(f'  Read {len(table_meta)} tables in {t1(t0)} ms')
    except:
        table_meta['table_pass'] = False

        print(traceback.format_exc())
        print('Warning: Could not read tables')
        exit()

    return {'table_pass': table_meta['table_pass']}


def read_any_table(a, meta):
    table_name = a.read_str(a.read_int('table_nameL'), 'table_name')
    table_entry_len = a.read_int('table_entry_len')

    if table_entry_len > 0 or meta['versionID'] in ['TNT1.005','TNT1.008']:
        table_entry = a.read_str(table_entry_len, 'table_entry')
        table_entry  = ' '.join(table_entry.split())
        table_entry = table_entry.split(' ')
        inc_Op = a.read_str(a.read_int('inc_OpL'), 'inc_Op')
        inc_val = a.read_str(a.read_int('inc_valL'), 'inc_val')
        inc_sch = a.reads(a.read_int('inc_schL'), 'inc_sch')

        a.reads(4, 'null')
        st = a.read_str(4, 'st')
        flag_0 = a.read_int('flag_0')

        if meta['versionID'] in(['TNT1.001', 'TNT1.003']):
            a.reads(8, 'null')
            flag_1 = a.read_int('flag_1')

            if (flag_1==1 and flag_0==0):
                pass
            elif (flag_0 in([1, 2]) and meta['versionID'] == 'TNT1.001'):
                a.reads(4, 'null0')
            else:
                a.reads(12,'null1')
                a.read_str(a.read_int(), 'null2')
                a.read_str(a.read_int(), 'null3')
                a.read_str(a.read_int(), 'null4')

            #hack
            #some files have an extra 4 bytes
            #some files have empty tables
            if meta['versionID'] in ['TNT1.003']:
                a0 = a.tell()

                if a.reads(4) == b'\x00\x00\x00\x00':
                    a1 = a.tell()

                    if a.reads(4) == b'\x00\x00\x00\x00': #only true if extra table
                        a.seek(a0)
                    else:
                        a.seek(a1)
                else:
                    a.seek(a0)
        else:
            a.reads(24,'null5')
            a.reads(a.read_int(), 'null6')
            a.reads(a.read_int(), 'null7')
            a.reads(a.read_int(), 'null8')
            a.reads(8,'null9')
            unknown = a.read_int('unknown')

            if unknown != 0:
                a.seek(-8, 1)

                if debug:
                    print(f'WARNING: unknown flag {unknown}')
    else:
        table_entry =''
        inc_Op = ''
        inc_val = ''
        inc_sch = ''
        st = a.read_str(4,'st')
        a.read_str(a.read_int('a'))
        a.read_str(a.read_int('b'))
        a.read_str(a.read_int('c'))

    if debug:
        print(f'{table_name} {st} {table_entry}, {inc_Op}, {inc_val}, {inc_sch}')

    if table_entry == '':
        return {table_name: table_entry}, -1

    return {table_name: table_entry}, 1


def read_variables(a, meta):
    t0 = time()
    var_meta = {}

    if not meta['table_pass']:
        return {'var_pass': False}

    if telling:
        print(f'\nVariables start @ {a.tell()}')

    try:
        flag = a.read_int()

        if flag:
            if debug:
                print('\nvariable flag')

            section = a.read_str(a.read_int())

            if section != 'Sequence':
                raise ValueError(f'ERROR: Expecting "Sequence" received {section}')

            for _ in range(a.read_int()):
                a.read_str(a.read_int())

        num_vars = a.read_int()

        if debug:
            print(f'\n  Variables: {num_vars}')

        for _ in range(num_vars):
            if telling:
                print(f'New variable starts @ {a.tell()}')

            var_meta.update(read_variable(a, meta, _))

        var_meta['variablePass'] = True

        if timer:
            print(f'  Read {len(var_meta)} variables in {t1(t0)} ms\n')

    except:
        var_meta['variablePass'] = False

        print(traceback.format_exc())
        print('Warning: Could not read variables')

    if debug:
        for e in var_meta:
            print(f'{e}:{ var_meta[e]}')

        input('advance? ')

    return var_meta


def read_variable(a, meta, c):
    if meta['versionID'] != 'TNT1.001':
        name = a.read_str(a.read_int())
        a.reads(4)
        data = a.read_str(a.read_int())
        a.reads(4)
        a.read_str(a.read_int()) #
        a.read_str(a.read_int())
        a.reads(8)
        a.read_str(a.read_int())
        name2 = a.read_str(a.read_int())
        a.read_str(a.read_int())
        a.read_str(a.read_int())
        a.reads(12)
    else:
        name = a.read_str(a.read_int())
        data = a.read_str(a.read_int())
        a.reads(4)
        a.read_str(a.read_int())
        a.read_str(a.read_int())
        a.reads(4)

    if debug:
        print(f'{name} = {data}')

    return {name: data}


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
        meta.update(read_pulse_sequence(a))
        meta.update(read_tables(a, meta))
        meta.update(read_variables(a, meta))

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
