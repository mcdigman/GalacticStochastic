"""helper functions for Chirp_WDM"""
from collections import namedtuple

import numpy as np

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.wdm_config import WDMWaveletConstants

LinearChirpletIntrinsicParams = namedtuple('LinearChirpletIntrinsicParams', ['amp0_t', 'phi0', 'F0', 'FTd0'])


# @njit(fastmath=True)
def amp_phase_t(ts, params):
    """Get amplitude and phase for chirp_time"""
    xs = (ts - params[8]) / params[0]
    fs = params[7] * xs + params[9]
    fac = np.sqrt(params[7] / params[0])
    phase = params[6] + 2. * np.pi * (params[8] - ts) * fs + np.pi * params[7] * params[0] * xs**2 + np.pi / 4.
    amp = fac * params[3] * np.exp(-xs**2 / 2.0)
    fds = np.full(xs.shape, params[7] / params[0])
    return phase, amp, fs, fds


def chirp_time(params, wc: WDMWaveletConstants):
    """Directly compute time domain intrinsic_waveform"""
    ts = wc.dt * np.arange(0, wc.Nf * wc.Nt)

    Phases, Amps, fas, fdas = amp_phase_t(ts, params)
    Phases = - Phases

    hs = Amps * np.cos(Phases)

    waveform = StationaryWaveformTime(ts, Phases, fas, fdas, Amps)
    return hs, waveform


# @njit(fastmath=True)
def amp_phase_f(fs, params):
    """Get amplitude and phase for frequency chirp"""
    xs = (fs - params[9]) / params[7]
    Phase = params[6] + 2.0 * np.pi * params[8] * fs + np.pi * params[0] * params[7] * xs**2
    Amp = params[3] * np.exp(-xs**2 / 2.0)
    return Phase, Amp


def ChirpWaveletT(params, TFs, wc):
    """Get the wavelet in time domain"""
    TFs = np.arange(0, wc.Nt) * wc.DT
    Phases, Amps, fas, fdas = amp_phase_t(TFs, params)
    Phases = - Phases

    # get the intrinsic intrinsic_waveform
    return StationaryWaveformTime(TFs, np.array([Phases]), np.array([fas]), np.array([fdas]), np.array([Amps]))


# @njit(fastmath=True)
def toff(f, params):
    """Get times for sparse freq method"""
    return params[0] * (f - params[9]) / params[7] + params[8]


# @njit(fastmath=True)
def foft(t, params):
    """Get frequencies for sparse time method"""
    return params[7] * (t - params[8]) / params[0] + params[9]
