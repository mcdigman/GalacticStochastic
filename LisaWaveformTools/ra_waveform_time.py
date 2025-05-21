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


#@njit(fastmath=True, parallel=True)
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
    # Note that if RR and II are both very near zero there can be a step function
    # in the phase that would correspond to a dirac delta function in the frequency,
    # that currently is not represented in the frequency or derivative at all,
    # because the formula used for the frequency derivative does not include a delta function.
    # It appears naturally in the phase due to the two argument arctangent.
    # (the limit of the two argument arctangent as one argument approaches zero
    # is one possible representation of a step function)
    # I think this may be the best behavior given the current use of passing to
    # Wavelet domain taylor expansion methods given typically sparse sampling for wavelets
    # but I am not sure if including such delta functions would be better in other use cases.
    # The code implicitly assumes postivie frequencies.
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

    # Get the starting local phase for wrapping
    for itrc in range(nc_channel):
        polds[itrc] = np.arctan2(IIs[itrc, 0], RRs[itrc, 0]) % (2*np.pi)

    for n in range(NT):
        fonfs = FT[n] / lc.fstr

        # including TDI + fractional frequency modifiers
        Ampx = AA[n] * (8 * fonfs * np.sin(fonfs))
        for itrc in range(nc_channel):
            RR = RRs[itrc, n]
            II = IIs[itrc, n]

            if RR == 0.0 and II == 0.0:
                # Handle zero denominator case without a delta function.
                # Note that the second derivative could be more complicated here,
                # But we ignore that for now and just take a numerical derivative.
                p = 0.0
                AET_FTs[itrc, n] = FT[n]
            else:
                # Handle general case, with the analytic derivative of the phase.
                # dRR and dII are currently computed through a numerical derivative,
                # But could be constructed analytically.
                # Ignores delta functions in FT that can happen when RR or II pass through 0.
                # Keep the local phase postive for self-consistent wrapping.
                p = np.arctan2(II, RR) % (2 * np.pi)
                AET_FTs[itrc, n] = FT[n] - (II * dRRs[itrc, n] - RR * dIIs[itrc, n]) / (RR**2 + II**2) / (2 * np.pi)

            # If the phase has increased or decreased more than 6 (<~2 pi)
            # try absorbing that change into the reported phase permanently,
            # as it is likely represents wrapping.
            # In testing 6 is a decent choice of the cutoff.
            # It might be possible to detect wrapping by multiple factors of 2 pi
            # using the analytic part of the perturbation in AET_Fts.
            # Note that wrapping due to the linear part of the frequency is assumed
            # to have already been done in the original waveform generation.
            if p - polds[itrc] > 6.0:
                js[itrc] -= 2 * np.pi
            elif polds[itrc] - p > 6.0:
                js[itrc] += 2 * np.pi

            # Store the current phase for wrapping
            polds[itrc] = p

            # Set the amplitude
            AET_Amps[itrc, n] = Ampx * np.sqrt(RR**2 + II**2)
            # Set the phase, including the input base phase, the perturbation from this iteration,
            # and any previous wrapping we have applied.
            AET_Phases[itrc, n] = PP[n] + p + js[itrc]

    # Get the frequency derivative using a numerical derivative, with offsets for
    # improved numerical accuracy. Because of the behavior near RR or II=0,
    # the numerical derivative may have better practical accuracy than
    # inserting an analytic result, at least without some additional numerical stabilizers.
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
