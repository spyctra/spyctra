import matplotlib.pyplot as plt

def tnt_read():
    from TNT import read

    a = read()
    a.plot()
    plt.show()

    a = read('../spyctraRep/TNT/exp1_385/FID_', 10)
    a.plot()
    plt.show()


def treev2_read():
    from TREEV2 import read

    a = read()
    a.plot()
    plt.show()

    a = read('../spyctraRep/TREEV2/CPMG_', 10)
    a.plot()
    plt.show()


def sdf_read():
    from SDF import read

    a = read()
    a[0].plot()
    plt.show()

    a = read('../spyctraRep/SDF/AN_sept2018c')
    a[0].plot()
    plt.show()


def main():
    sdf_read()
    exit()
    tnt_read()
    treev2_read()


if __name__ == '__main__':
    main()
