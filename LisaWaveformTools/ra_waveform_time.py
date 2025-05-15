"""subroutines for running lisa code"""

from collections import namedtuple

import numpy as np
from numba import njit

from LisaWaveformTools.algebra_tools import gradient_homog_2d_inplace
from LisaWaveformTools.ra_waveform_freq import (RAantenna_inplace,
                                                get_tensor_basis,
                                                get_xis_inplace,
                                                spacecraft_vec)

StationaryWaveformTime = namedtuple('StationaryWaveformTime', ['T', 'PT', 'FT', 'FTd', 'AT'])
SpacecraftChannels = namedtuple('SpacecraftChannels', ['T', 'RR', 'II', 'dRR', 'dII'])


@njit(fastmath=True)
def ExtractAmpPhase_inplace(spacecraft_channels, AET_waveform, waveform, NT, lc, wc):
    """get the amplitude and phase for LISA"""
    AA = waveform.AT
    PP = waveform.PT
    FT = waveform.FT
    FTd = waveform.FTd

    AET_Amps = AET_waveform.AT
    AET_Phases = AET_waveform.PT
    AET_FTs = AET_waveform.FT
    AET_FTds = AET_waveform.FTd

    RRs = spacecraft_channels.RR
    IIs = spacecraft_channels.II
    dRRs = spacecraft_channels.dRR
    dIIs = spacecraft_channels.dII

    # TODO check absolute phase aligns with Extrinsic_inplace
    polds = np.zeros(wc.NC)
    js = np.zeros(wc.NC)

    gradient_homog_2d_inplace(RRs, wc.DT, NT, 3, dRRs)
    gradient_homog_2d_inplace(IIs, wc.DT, NT, 3, dIIs)

    n = 0
    for itrc in range(0, wc.NC):
        polds[itrc] = np.arctan2(IIs[itrc, n], RRs[itrc, n])
        if polds[itrc] < 0.:
            polds[itrc] += 2*np.pi
    for n in range(0, NT):
        fonfs = FT[n]/lc.fstr

        # including TDI + fractional frequency modifiers
        Ampx = AA[n]*(8*fonfs*np.sin(fonfs))
        Phase = PP[n]
        for itrc in range(0, wc.NC):
            RR = RRs[itrc, n]
            II = IIs[itrc, n]

            if RR == 0. and II == 0.:
                # TODO is this the correct way to handle both ps being 0?
                p = 0.
                AET_FTs[itrc, n] = FT[n]
            else:
                p = np.arctan2(II, RR)
                AET_FTs[itrc, n] = FT[n]-(II*dRRs[itrc, n]-RR*dIIs[itrc, n])/(RR**2+II**2)/(2*np.pi)

            if p < 0.:
                p += 2*np.pi

            # TODO implement integral tracking of js
            if p-polds[itrc] > 6.:
                js[itrc] -= 2*np.pi
            if polds[itrc]-p > 6.:
                js[itrc] += 2*np.pi
            polds[itrc] = p

            AET_Amps[itrc, n] = Ampx*np.sqrt(RR**2+II**2)
            AET_Phases[itrc, n] = Phase+p+js[itrc]  # +2*np.pi*kdotx[n]*FT[n]

    for itrc in range(0, wc.NC):
        AET_FTds[itrc, 0] = (AET_FTs[itrc, 1]-AET_FTs[itrc, 0]-FT[1]+FT[0])/wc.DT+FTd[0]
        AET_FTds[itrc, NT-1] = (AET_FTs[itrc, NT-1]-AET_FTs[itrc, NT-2]-FT[NT-1]+FT[NT-2])/wc.DT+FTd[NT-1]

    for n in range(1, NT-1):
        FT_shift = -FT[n+1]+FT[n-1]
        FTd_shift = FTd[n]
        for itrc in range(0, wc.NC):
            AET_FTds[itrc, n] = (AET_FTs[itrc, n+1]-AET_FTs[itrc, n-1]+FT_shift)/(2*wc.DT)+FTd_shift


# TODO check factor of 2pi
@njit()
def AmpFreqDeriv_inplace(waveform, Amp, phi0, FI, FD0, TS):
    """Get time domain waveform to lowest order, simple constant fdot"""
    AS = waveform.AT
    PS = waveform.PT
    FS = waveform.FT
    FDS = waveform.FTd

    NT = TS.size
    #  compute the intrinsic frequency, phase and amplitude
    for n in range(0, NT):
        t = TS[n]
        FS[n] = FI+FD0*t
        FDS[n] = FD0
        PS[n] = -phi0+2*np.pi*FI*t+np.pi*FD0*t**2
        AS[n] = Amp


# TODO do consistency checks
class BinaryTimeWaveformAmpFreqD():
    """class to store a binary waveform in time domain and update for search
        assuming input binary format based on amplitude, frequency, and frequency derivative"""
    def __init__(self, params, NT_min, NT_max, lc, wc, n_pad_T, freeze_limits=False):
        """initalize the object"""
        self.params = params
        self.n_pad_T = n_pad_T
        self.NT_min = NT_min-self.n_pad_T
        self.NT_max = NT_max+self.n_pad_T
        self.NT = self.NT_max-self.NT_min
        self.freeze_limits = freeze_limits
        self.lc = lc
        self.wc = wc

        # TODO ensure this handles padding self consistently
        self.nt_low = self.NT_min
        self.nt_high = self.NT_max

        self.nt_range = self.nt_high-self.nt_low

        self.TTs = self.wc.DT*np.arange(self.nt_low, self.nt_high)

        AmpTs = np.zeros(self.NT)
        PPTs = np.zeros(self.NT)
        FTs = np.zeros(self.NT)
        FTds = np.zeros(self.NT)

        self.waveform = StationaryWaveformTime(self.TTs, PPTs, FTs, FTds, AmpTs)

        RRs = np.zeros((self.wc.NC, self.NT))
        IIs = np.zeros((self.wc.NC, self.NT))
        dRRs = np.zeros((self.wc.NC, self.NT))
        dIIs = np.zeros((self.wc.NC, self.NT))

        self.spacecraft_channels = SpacecraftChannels(self.TTs, RRs, IIs, dRRs, dIIs)

        self.xas = np.zeros(self.NT)
        self.yas = np.zeros(self.NT)
        self.zas = np.zeros(self.NT)
        self.xis = np.zeros(self.NT)
        self.kdotx = np.zeros(self.NT)

        _, _, _, self.xas[:], self.yas[:], self.zas[:] = spacecraft_vec(self.TTs, self.lc)

        AET_AmpTs = np.zeros((self.wc.NC, self.NT))
        AET_PPTs = np.zeros((self.wc.NC, self.NT))
        AET_FTs = np.zeros((self.wc.NC, self.NT))
        AET_FTds = np.zeros((self.wc.NC, self.NT))

        self.AET_waveform = StationaryWaveformTime(self.TTs, AET_PPTs, AET_FTs, AET_FTds, AET_AmpTs)

        self.update_params(params)

    def update_params(self, params):
        self.params = params
        self.update_intrinsic()
        self.update_extrinsic()

    def update_intrinsic(self):
        """get amplitude and phase for taylorT3"""
        amp = self.params[0]
        costh = np.cos(np.pi/2 - self.params[1])  # TODO check
        phi = self.params[2]
        freq0 = self.params[3]
        freqD = self.params[4]
        phi0 = self.params[6] + np.pi

        kv, _, _ = get_tensor_basis(phi, costh)  # TODO check intrinsic extrinsic separation here
        get_xis_inplace(kv, self.TTs, self.xas, self.yas, self.zas, self.xis, self.lc)

        AmpFreqDeriv_inplace(self.waveform, amp, phi0, freq0, freqD, self.xis)

    def update_extrinsic(self):
        # Calculate cos and sin of sky position, inclination, polarization
        costh = np.cos(np.pi/2-self.params[1])
        phi = self.params[2]
        cosi = np.cos(self.params[5])
        psi = self.params[7]

        # TODO fix F_min and nf_range
        RAantenna_inplace(self.spacecraft_channels, cosi, psi, phi, costh, self.TTs, self.waveform.FT, 0, self.NT, self.kdotx, self.lc)
        ExtractAmpPhase_inplace(self.spacecraft_channels, self.AET_waveform, self.waveform, self.NT, self.lc, self.wc)
