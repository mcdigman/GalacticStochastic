"""subroutines for running lisa code"""

from collections import namedtuple

import numpy as np
from numba import njit

from LisaWaveformTools.algebra_tools import gradient_homog_2d_inplace
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_freq import RAantenna_inplace, get_tensor_basis, get_xis_inplace, spacecraft_vec
from WaveletWaveforms.wdm_config import WDMWaveletConstants

StationaryWaveformTime = namedtuple('StationaryWaveformTime', ['T', 'PT', 'FT', 'FTd', 'AT'])
SpacecraftChannels = namedtuple('SpacecraftChannels', ['T', 'RR', 'II', 'dRR', 'dII'])


@njit()
def ExtractAmpPhase_inplace(
    spacecraft_channels: SpacecraftChannels,
    AET_waveform: StationaryWaveformTime,
    waveform: StationaryWaveformTime,
    NT,
    lc: LISAConstants,
    dt: float,
) -> None:
    """Get the amplitude and phase for LISA"""
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

    nc_channel = RRs.shape[0]

    polds = np.zeros(nc_channel)
    js = np.zeros(nc_channel)

    gradient_homog_2d_inplace(RRs, dRRs, dt)
    gradient_homog_2d_inplace(IIs, dIIs, dt)

    n = 0
    for itrc in range(nc_channel):
        polds[itrc] = np.arctan2(IIs[itrc, n], RRs[itrc, n])
        if polds[itrc] < 0.0:
            polds[itrc] += 2 * np.pi
    for n in range(NT):
        fonfs = FT[n] / lc.fstr

        # including TDI + fractional frequency modifiers
        Ampx = AA[n] * (8 * fonfs * np.sin(fonfs))
        Phase = PP[n]
        for itrc in range(nc_channel):
            RR = RRs[itrc, n]
            II = IIs[itrc, n]

            if RR == 0.0 and II == 0.0:
                p = 0.0
                AET_FTs[itrc, n] = FT[n]
            else:
                p = np.arctan2(II, RR)
                AET_FTs[itrc, n] = FT[n] - (II * dRRs[itrc, n] - RR * dIIs[itrc, n]) / (RR**2 + II**2) / (2 * np.pi)

            if p < 0.0:
                p += 2 * np.pi

            if p - polds[itrc] > 6.0:
                js[itrc] -= 2 * np.pi
            if polds[itrc] - p > 6.0:
                js[itrc] += 2 * np.pi
            polds[itrc] = p

            AET_Amps[itrc, n] = Ampx * np.sqrt(RR**2 + II**2)
            AET_Phases[itrc, n] = Phase + p + js[itrc]

    for itrc in range(nc_channel):
        AET_FTds[itrc, 0] = (AET_FTs[itrc, 1] - AET_FTs[itrc, 0] - FT[1] + FT[0]) / dt + FTd[0]
        AET_FTds[itrc, NT - 1] = (
            AET_FTs[itrc, NT - 1] - AET_FTs[itrc, NT - 2] - FT[NT - 1] + FT[NT - 2]
        ) / dt + FTd[NT - 1]

    for n in range(1, NT - 1):
        FT_shift = -FT[n + 1] + FT[n - 1]
        FTd_shift = FTd[n]
        for itrc in range(nc_channel):
            AET_FTds[itrc, n] = (AET_FTs[itrc, n + 1] - AET_FTs[itrc, n - 1] + FT_shift) / (2 * dt) + FTd_shift


# TODO check factor of 2pi
@njit()
def AmpFreqDeriv_inplace(waveform: StationaryWaveformTime, Amp, phi0, FI, FD0, TS) -> None:
    """Get time domain waveform to lowest order, simple constant fdot"""
    AS = waveform.AT
    PS = waveform.PT
    FS = waveform.FT
    FDS = waveform.FTd

    NT = TS.size
    #  compute the intrinsic frequency, phase and amplitude
    for n in range(NT):
        t = TS[n]
        FS[n] = FI + FD0 * t
        FDS[n] = FD0
        PS[n] = -phi0 + 2 * np.pi * FI * t + np.pi * FD0 * t**2
        AS[n] = Amp


# TODO do consistency checks
class BinaryTimeWaveformAmpFreqD:
    """class to store a binary waveform in time domain and update for search
    assuming input binary format based on amplitude, frequency, and frequency derivative
    """

    def __init__(self, params, nt_min, nt_max, lc: LISAConstants, wc: WDMWaveletConstants, nc_waveform) -> None:
        """Initalize the object"""
        self.params = params
        self.nt_min = nt_min
        self.nt_max = nt_max
        self.nt_range = self.nt_max - self.nt_min
        self.lc = lc
        self.wc = wc
        self.nc_waveform = nc_waveform

        self.TTs = self.wc.DT * np.arange(self.nt_min, self.nt_max)

        AmpTs = np.zeros(self.nt_range)
        PPTs = np.zeros(self.nt_range)
        FTs = np.zeros(self.nt_range)
        FTds = np.zeros(self.nt_range)

        self.waveform = StationaryWaveformTime(self.TTs, PPTs, FTs, FTds, AmpTs)

        RRs = np.zeros((self.nc_waveform, self.nt_range))
        IIs = np.zeros((self.nc_waveform, self.nt_range))
        dRRs = np.zeros((self.nc_waveform, self.nt_range))
        dIIs = np.zeros((self.nc_waveform, self.nt_range))

        self.spacecraft_channels = SpacecraftChannels(self.TTs, RRs, IIs, dRRs, dIIs)

        self.xas = np.zeros(self.nt_range)
        self.yas = np.zeros(self.nt_range)
        self.zas = np.zeros(self.nt_range)
        self.xis = np.zeros(self.nt_range)
        self.kdotx = np.zeros(self.nt_range)

        _, _, _, self.xas[:], self.yas[:], self.zas[:] = spacecraft_vec(self.TTs, self.lc)

        AET_AmpTs = np.zeros((self.nc_waveform, self.nt_range))
        AET_PPTs = np.zeros((self.nc_waveform, self.nt_range))
        AET_FTs = np.zeros((self.nc_waveform, self.nt_range))
        AET_FTds = np.zeros((self.nc_waveform, self.nt_range))

        self.AET_waveform = StationaryWaveformTime(self.TTs, AET_PPTs, AET_FTs, AET_FTds, AET_AmpTs)

        self.update_params(params)

    def update_params(self, params) -> None:
        self.params = params
        self.update_intrinsic()
        self.update_extrinsic()

    def update_intrinsic(self) -> None:
        """Get amplitude and phase for taylorT3"""
        amp = self.params[0]
        costh = np.cos(np.pi / 2 - self.params[1])  # TODO check
        phi = self.params[2]
        freq0 = self.params[3]
        freqD = self.params[4]
        phi0 = self.params[6] + np.pi

        kv, _, _ = get_tensor_basis(phi, costh)  # TODO check intrinsic extrinsic separation here
        get_xis_inplace(kv, self.TTs, self.xas, self.yas, self.zas, self.xis, self.lc)

        AmpFreqDeriv_inplace(self.waveform, amp, phi0, freq0, freqD, self.xis)

    def update_extrinsic(self) -> None:
        # Calculate cos and sin of sky position, inclination, polarization
        costh = np.cos(np.pi / 2 - self.params[1])
        phi = self.params[2]
        cosi = np.cos(self.params[5])
        psi = self.params[7]

        # TODO fix F_min and nf_range
        RAantenna_inplace(
            self.spacecraft_channels,
            cosi,
            psi,
            phi,
            costh,
            self.TTs,
            self.waveform.FT,
            0,
            self.nt_range,
            self.kdotx,
            self.lc,
        )
        ExtractAmpPhase_inplace(
            self.spacecraft_channels, self.AET_waveform, self.waveform, self.nt_range, self.lc, self.wc.DT
        )
