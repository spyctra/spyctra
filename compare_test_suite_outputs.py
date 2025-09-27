"""
Tool for comparing spyctra calculations
"""

import numpy as np


def main():
    new_file_path = './test_suite.res2'
    old_file_path = '../spyctra_V6/test_suite.res2'

    with open(new_file_path, 'r') as file:
        new_file = [line.rstrip() for line in file]

    with open(old_file_path, 'r') as file:
        old_file = [line.rstrip() for line in file]

    print(f'{new_file_path = }')
    print(f'{old_file_path = }')
    print('variable, index, new, old, ratio')

    for i in range(len(new_file)):
        #identify titles
        if len(new_file[i]) > 0 and new_file[i][0] != ' ':
            title = new_file[i]

            print(f'\n{title}')

            continue

        #they're unlikley variables titles at this point
        #now comparing data
        if new_file[i] == old_file[i]:
            pass
        else:
            new_text = new_file[i][1:]
            old_text = old_file[i][1:]
            is_float = True

            try:
                new_val = float(new_text)
                old_val = float(old_text)
            except:
                if title == 'directory':
                    continue

                try:
                    new_val = complex(new_text)
                    old_val = complex(old_text)
                    is_float = False
                except:
                    print("what's this data type?", i, new_file[i], old_file[i])
                    print(title)

                    exit()
            if np.abs(old_val) == 0:
                print(title, new_val, old_val)
            else:
                try:
                    if is_float:
                        ratio = new_val/old_val
                    else:
                        ratio_r = np.real(new_val)/np.real(old_val)
                        ratio_i = np.imag(new_val)/np.imag(old_val)
                except:
                    print('better be div by zero', i, title, new_val, old_val)

                    continue

                if is_float:
                    if ratio < 0.999999 or ratio > 1.000001:
                        print(title, i, new_file[i], old_file[i], ratio)
                elif ratio_r < 0.999999 or ratio_r > 1.000001 or ratio_i < 0.999999 or ratio_i > 1.000001:
                    print(title, i, new_file[i], old_file[i], ratio_r, ratio_i)


if __name__ == '__main__':
    main()
