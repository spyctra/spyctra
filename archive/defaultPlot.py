from matplotlib.widgets import Button

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 25

#mpl.rcParams['figure.figsize'] = [8,8]
#mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['figure.subplot.hspace'] = 0.6

plt.rcParams["font.family"] = "monospace"
mpl.rcParams['font.monospace'] ='consolas'
mpl.rcParams['font.size'] = 16

mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.format'] = 'png'

mpl.rcParams['xtick.labelsize'] = 12 
mpl.rcParams['ytick.labelsize'] = 12 

mpl.rcParams['grid.linewidth'] = 0.2

mpl.rcParams['axes.facecolor'] = 'black'
mpl.rcParams['figure.facecolor'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['axes.titlecolor'] = 'white'
mpl.rcParams['figure.edgecolor'] = 'white'
mpl.rcParams['grid.color'] = 'white'
mpl.rcParams['legend.labelcolor'] = 'white'
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['xtick.labelcolor'] = 'inherit'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['ytick.labelcolor'] = 'inherit'
mpl.rcParams['axes.edgecolor'] = 'white'



def button():
    axprev = plt.axes([0.01, 0.96, 0.03, 0.03])
    bnext = Button(axprev, 'STOP')
    bnext.label.set_fontsize(8)
    bnext.on_clicked(exit)
    axprev._button = bnext



def single():
    f, (ax2) = plt.subplots(1,1)
    f.suptitle('f.suptitle')
    ax2.plot([0,1,4,9,16,25], [0,1,2,3,4,5], label='legend')
    ax2.plot([0,1,4,9,16,25], [1,2,3,4,5,6], label='other')
    ax2.title.set_text('ax2.title.set_text')
    ax2.set_xlabel('ax2.set_xlabel')
    ax2.set_ylabel('ax2.set_ylabel')
    ax2.legend()
    plt.savefig('./defaultPlot.png')
    plt.grid()
    button()
    plt.show()


def twoVert():
    f, (ax1,ax2) = plt.subplots(1,2)
    f.suptitle('f.suptitle')
    ax1.plot([0,1,2,3,4,5],[0,1,4,9,16,25], label='label')
    ax1.title.set_text('ax1.title.set_text')
    ax1.set_xlabel('ax1.set_xlabel')
    ax1.set_ylabel('ax1.set_ylabel')
    ax1.legend()

    ax2.plot([0,1,4,9,16,25], [0,1,2,3,4,5], label='legend')
    ax2.title.set_text('ax2.title.set_text')
    ax2.set_xlabel('ax2.set_xlabel')
    ax2.set_ylabel('ax2.set_ylabel')
    ax2.legend()
    plt.savefig('./defaultPlot.png')
    plt.show()
    
    
def twoHor():
    f, (ax1,ax2) = plt.subplots(2,1)
    f.suptitle('f.suptitle')
    ax1.plot([0,1,2,3,4,5],[0,1,4,9,16,25], label='label')
    ax1.title.set_text('ax1.title.set_text')
    ax1.set_xlabel('ax1.set_xlabel')
    ax1.set_ylabel('ax1.set_ylabel')
    ax1.legend()

    ax2.plot([0,1,4,9,16,25], [0,1,2,3,4,5], label='legend')
    ax2.title.set_text('ax2.title.set_text')
    ax2.set_xlabel('ax2.set_xlabel')
    ax2.set_ylabel('ax2.set_ylabel')
    ax2.legend()

    plt.savefig('./defaultPlot.png')
    plt.show()


def main():
    #print(mpl.rcParams.keys())
    #exit()
    single()
    twoVert()
    twoHor()

    
if __name__ == '__main__':
    main()