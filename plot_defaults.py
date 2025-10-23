import sys
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from matplotlib import rcParams

rcParams['font.family'] = 'Courier New'
rcParams['font.size'] = 14
rcParams['figure.figsize'] = [9, 9]
rcParams['axes.labelsize'] = 16  # font size of the x and y labels
rcParams['xtick.labelsize'] = 14  # font size of the tick labels
rcParams['ytick.labelsize'] = 12  # font size of the tick labels

#https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample


def button():
    #Button to close program
    axprev = plt.axes([0.01, 0.96, 0.03, 0.03])
    bnext = Button(axprev, 'STOP', hovercolor='red')
    bnext.on_clicked(sys.exit)
    axprev._button = bnext
    bnext.label.set_fontsize(9)


def defaults():
    import numpy as np

    #"#"#"#"#"#"#"#
    plt.figure()
    plt.suptitle('plt.suptitle')

    plt.subplot(2,1,1)
    plt.title('plt.title()')
    plt.plot(np.arange(10), np.arange(10)**2, label='a')
    plt.xlabel('plt.xlabel()')
    plt.ylabel('plt.ylabel()')
    plt.legend()

    plt.subplot(2,1,2)
    plt.title('plt.title()')
    plt.plot(np.arange(10), np.arange(10))
    plt.xlabel('plt.xlabel()')
    plt.ylabel('plt.ylabel()')
    plt.xticks([])
    plt.yticks([])

    button()
    #"#"#"#"#"#"#"#


    #"#"#"#"#"#"#"#
    fig, axs = plt.subplots(2,1)
    fig.suptitle('fig.suptitle()')

    axs[0].set_title('axs[0].set_title()')
    axs[0].plot(np.arange(10), np.arange(10)**2)
    axs[0].set_xlabel('axs[0].set_xlabel()')
    axs[0].set_ylabel('axs[0].set_ylabel()')

    axs[1].set_title('axs[1].set_title()')
    axs[1].plot(np.arange(10), np.arange(10))
    axs[1].set_xlabel('axs[1].set_xlabel()')
    axs[1].set_ylabel('axs[1].set_ylabel()')

    button()
    plt.show()
    #"#"#"#"#"#"#"#


def specialized():
    import numpy as np

    #"#"#"#"#"#"#"#
    plt.figure()
    plt.suptitle('plt.title')

    plt.subplot(2,1,1)
    plt.plot(np.arange(10), np.arange(10)**2)
    plt.xlabel('plt.xlabel', fontsize=16)
    plt.ylabel('plt.ylabel', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)

    plt.subplot(2,1,2)
    plt.plot(np.arange(10), np.arange(10))
    plt.xlabel('plt.xlabel', fontsize=16)
    plt.ylabel('plt.ylabel', fontsize=20)
    plt.xticks([], fontsize=12)
    plt.yticks([], fontsize=10)
    button()
    #"#"#"#"#"#"#"#


    #"#"#"#"#"#"#"#
    fig, axs = plt.subplots(2,1)
    fig.suptitle('fig.suptitle()')

    axs[0].set_title('axs[0].set_title()', fontsize=16)
    axs[0].plot([0,1,2,3,4,5,6,7,8], [0,1,4,9,16,25,36,49,64])
    axs[0].set_xlabel('axs[0].set_xlabel()', fontsize=18)
    axs[0].set_ylabel('axs[0].set_ylabel()', fontsize=18)
    axs[0].tick_params(axis='x', labelsize=12)

    axs[1].set_title('axs[1].set_title()', fontsize=20)
    axs[1].plot([0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8])
    axs[1].set_xlabel('axs[1].set_xlabel()')
    axs[1].set_ylabel('axs[1].set_ylabel()')
    axs[1].tick_params(axis='both', labelsize=24)

    button()
    plt.show()
    #"#"#"#"#"#"#"#


def main():
    #defaults()
    specialized()


if __name__ == '__main__':
    main()
