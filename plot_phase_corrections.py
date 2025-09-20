"""
Visualizer for determining phase corrections
"""

import matplotlib

from matplotlib.pyplot import draw, pause
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from math import e, pi
import matplotlib.animation as animation

from plot_defaults import button

"""
CHANGE LOG

2025-09-14 Initial release
"""

class phase_correction_plotter():
    def __init__(self, x, data, dPhidF=0, dPhidF_inc=0.001, f0=0, f0_inc=0.01, phi0_inc=0, dphi0_inc=0.1):
        self.x = x
        self.data = data
        self.dPhidF = dPhidF
        self.dPhidF_inc = dPhidF_inc
        self.f0 = f0
        self.f0_inc = f0_inc
        self.phi0_inc = phi0_inc
        self.dphi0_inc = dphi0_inc


    def run(self):
        self.fig, self.ax0 = plt.subplots(figsize=(16,9))
        plt.subplots_adjust(bottom=0.25)

        self.yM, = self.ax0.plot(self.x, np.abs(self.data), c='b',linewidth=3, alpha=0.5)
        self.yR, = self.ax0.plot(self.x, self.data.real, label='real', c='r')
        self.yI, = self.ax0.plot(self.x, self.data.imag, c='g')

        self.ax0.set_xlim([min(self.x), max(self.x)])
        self.ax0.set_ylim([-max(self.ax0.get_ylim()), max(self.ax0.get_ylim())])
        self.ax0.set_xlabel('Frequency (Hz)')
        self.ax0.set_ylabel('Signal')
        self.ax0.set_title('First Order Phase Correction Visualizer')
        self.ax0.grid()
        plt.legend()
        self.fig.text(0.825, 0.19, 'value')
        self.fig.text(0.875, 0.19, 'increment')



        #1 dPhidF slider
        self.ax_dPhidF = plt.axes([0.125, 0.15, 0.65, 0.03])
        self.s_dPhidF = Slider(self.ax_dPhidF, 'dPhidF', self.dPhidF-self.dPhidF_inc*100, self.dPhidF+self.dPhidF_inc*100, valinit=self.dPhidF, valstep=self.dPhidF_inc)
        self.s_dPhidF.on_changed(self.phase_update)


        #1 dPhidF box
        self.b_ax_dPhidF = plt.axes([0.825, 0.15, 0.045, 0.03])
        self.b_dPhidF = TextBox(self.b_ax_dPhidF, '', initial=self.dPhidF)
        self.b_dPhidF.on_submit(self.b_dPhidF_update)


        #1 dPhidF_inc box
        self.b_ax_dPhidF_inc = plt.axes([0.875, 0.15, 0.045, 0.03])
        self.b_dPhidF_inc = TextBox(self.b_ax_dPhidF_inc, '', initial=self.dPhidF_inc)
        self.b_dPhidF_inc.on_submit(self.b_dPhidF_inc_update)


        #"""
        #2 f0 slider
        self.ax_f0 = plt.axes([0.125, 0.10, 0.65, 0.03])
        self.s_f0 = Slider(self.ax_f0, 'f0', self.f0-self.f0_inc*100, self.f0+self.f0_inc*100, valinit=self.f0, valstep=self.f0_inc)
        self.s_f0.on_changed(self.phase_update)

        #2 f0 box
        self.b_ax_f0 = plt.axes([0.825, 0.10, 0.045, 0.03])
        self.b_f0 = TextBox(self.b_ax_f0, '', initial=self.f0)
        self.b_f0.on_submit(self.b_f0_update)

        #2 f0_inc box
        self.b_ax_f0_inc = plt.axes([0.875, 0.10, 0.045, 0.03])
        self.b_f0_inc = TextBox(self.b_ax_f0_inc, '', initial=self.f0_inc)
        self.b_f0_inc.on_submit(self.b_f0_inc_update)
        #"""


        #"""
        #3 phi0_inc slider
        self.ax_phi0_inc = plt.axes([0.125, 0.05, 0.65, 0.03])
        self.s_phi0_inc = Slider(self.ax_phi0_inc, 'phi0_inc', self.phi0_inc-self.dphi0_inc*100, self.phi0_inc+self.dphi0_inc*100, valinit=self.phi0_inc, valstep=self.dphi0_inc)
        self.s_phi0_inc.on_changed(self.phase_update)

        #3 f0 box
        self.b_ax_phi0_inc = plt.axes([0.825, 0.05, 0.045, 0.03])
        self.b_phi0_inc = TextBox(self.b_ax_phi0_inc, '', initial=self.phi0_inc)
        self.b_phi0_inc.on_submit(self.b_phi0_inc_update)

        #3 f0_inc box
        self.b_ax_dphi0_inc = plt.axes([0.875, 0.05, 0.045, 0.03])
        self.b_dphi0_inc = TextBox(self.b_ax_dphi0_inc, '', initial=self.dphi0_inc)
        self.b_dphi0_inc.on_submit(self.b_dphi0_inc_update)
        #"""

        #Button to close program
        button()

        plt.show()

        return [self.dPhidF, self.f0, self.phi0_inc]


    #plot updater
    def phase_update(self, sliderVal):
        self.dPhidF = self.s_dPhidF.val
        self.f0 = self.s_f0.val
        self.phi0_inc = self.s_phi0_inc.val

        phiAdj = -self.phi0_inc + (self.f0-self.x)*self.dPhidF
        data = self.data*e**(1j*phiAdj)

        self.yR.set_ydata(data.real)
        self.yI.set_ydata(data.imag)


    #1 dPhidF button updaters
    def b_dPhidF_update(self, dPhidF):
        self.dPhidF = np.float64(dPhidF)
        self.s_dPhidP_update()

    def b_dPhidF_inc_update(self, dPhidF_inc):
        self.dPhidF_inc = np.float64(dPhidF_inc)
        self.s_dPhidP_update()

    def s_dPhidP_update(self):
        self.ax_dPhidF.clear()
        self.s_dPhidF.__init__(self.ax_dPhidF, 'dPhidF', self.dPhidF-self.dPhidF_inc*100, self.dPhidF+self.dPhidF_inc*100, valinit=self.dPhidF, valstep=self.dPhidF_inc)
        self.s_dPhidF.on_changed(self.phase_update)
        self.phase_update(0)


    #2 f0 button updaters
    def b_f0_update(self, f0):
        self.f0 = np.float64(f0)
        self.s_f0_update()

    def b_f0_inc_update(self, f0_inc):
        self.f0_inc = np.float64(f0_inc)
        self.s_f0_update()

    def s_f0_update(self):
        self.ax_f0.clear()
        self.s_f0.__init__(self.ax_f0, 'f0', self.f0-self.f0_inc*100, self.f0+self.f0_inc*100, valinit=self.f0, valstep=self.f0_inc)
        self.s_f0.on_changed(self.phase_update)
        self.phase_update(0)


    #3 phi0_inc button updaters
    def b_phi0_inc_update(self, phi0_inc):
        self.phi0_inc = np.float64(phi0_inc)
        self.s_phi0_inc_update()

    def b_dphi0_inc_update(self, dphi0_inc):
        self.dphi0_inc = np.float64(dphi0_inc)
        self.s_phi0_inc_update()

    def s_phi0_inc_update(self):
        self.ax_phi0_inc.clear()
        self.s_phi0_inc.__init__(self.ax_phi0_inc, 'phi0_inc', self.phi0_inc-self.dphi0_inc*100, self.phi0_inc+self.dphi0_inc*100, valinit=self.phi0_inc, valstep=self.dphi0_inc)
        self.s_phi0_inc.on_changed(self.phase_update)
        self.phase_update(0)


def test_suite():
    from spyctra import spyctra, fake_spyctra

    a = spyctra()

    for i in range(2):
        a.add(fake_spyctra(points=16384*64, delta=1e-5, t_2=1e-1, df=20, amp=4096, phi=i+1, noise=0))

    a.subtract(a.find_offset())
    a.fft()
    a.resize([-100,100])

    phis = a.plot_phase_cor()
    a.phase_FOC(phis)

    a.plot()
    plt.show()


def main():
    test_suite()


if __name__ == '__main__':
    main()
