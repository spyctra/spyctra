import numpy as np

def main():
    new = './testSuite_worker.res2'
    old = '../spyctraV5/testSuite_worker.res2'

    with open(new, 'r') as file:
        a = [line.rstrip() for line in file]

    with open(old, 'r') as file:
        b = [line.rstrip() for line in file]

    for i in range(len(a)):
        #identify titles
        try:
            float(a[i])
        except:
            try:
                complex(a[i])
            except:
                title = a[i]
                print(title)
                continue

        #they're unlikley variables titles at this point
        #now comparing data
        if a[i] == b[i]:
            pass
        else:
            isFloat = True
            try:
                vala = float(a[i])
                valb = float(b[i])
            except:
                try:
                    vala = complex(a[i])
                    valb = complex(a[i])
                    isFloat = False
                except:
                    print("what's this data type?", i, a[i], b[i])
                    print(title)
                    exit()
            if np.abs(valb) == 0:
                print(title, vala, valb)
            else:
                try:
                    if isFloat:
                        ratio = vala/valb
                    else:
                        ratioR = np.real(vala)/np.real(valb)
                        ratioI = np.imag(vala)/np.imag(valb)
                except:
                    print('better be div by zero', i, title, vala, valb)
                    continue
                if isFloat:
                    if ratio < 0.999999 or ratio > 1.000001:
                        print(title, i, a[i], b[i], ratio)
                elif ratioR < 0.999999 or ratioR > 1.000001 or ratioI < 0.999999 or ratioI > 1.000001:
                    print(title, i, a[i], b[i], ratioR, ratioI)


if __name__ == '__main__':
    main()
