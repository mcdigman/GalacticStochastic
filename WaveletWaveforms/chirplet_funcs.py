"""helper functions for Chirp_WDM"""
from collections import namedtuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.wdm_config import WDMWaveletConstants

LinearChirpletIntrinsicParams = namedtuple('LinearChirpletIntrinsicParams', ['amp_center_f', 'phi0', 'f_center', 't_center', 'tau', 'gamma'])


@njit()
def chirplet_time_intrinsic(waveform: StationaryWaveformTime, intrinsic_params: LinearChirpletIntrinsicParams, t_in: NDArray[np.float64]) -> None:
    """Get amplitude and phase for chirp_time"""
    AT = waveform.AT
    PT = waveform.PT
    FT = waveform.FT
    FTd = waveform.FTd

    nt_loc = t_in.size

    # amplitude multiplier to convert from frequency domain amplitude to time domain amplitude
    amp_center_t = np.sqrt(intrinsic_params.gamma / intrinsic_params.tau) * intrinsic_params.amp_center_f

    phase_center_t = - np.pi / 4. - intrinsic_params.phi0
    #  compute the intrinsic frequency, phase and amplitude
    for n in range(nt_loc):
        t = t_in[n]
        delta_t = t - intrinsic_params.t_center
        x = delta_t / intrinsic_params.tau
        FT[n] = intrinsic_params.gamma * x + intrinsic_params.f_center

        # TODO check sign convention on phase
        PT[n] = phase_center_t + 2. * np.pi * delta_t * FT[n] - np.pi * intrinsic_params.gamma * intrinsic_params.tau * x ** 2
        AT[n] = amp_center_t * np.exp(-x**2 / 2.0)
        FTd[n] = intrinsic_params.gamma / intrinsic_params.tau


# @njit(fastmath=True)
def amp_phase_t(ts, params: LinearChirpletIntrinsicParams):
    """Get amplitude and phase for chirp_time"""
    xs = (ts - params.t_center) / params.tau
    fs = params.gamma * xs + params.f_center
    amp_center_t = np.sqrt(params.gamma / params.tau) * params.amp_center_f
    # TODO check sign convention on phase
    phase = - np.pi / 4. - params.phi0 + 2. * np.pi * (ts - params.t_center) * fs - np.pi * params.gamma * params.tau * xs**2
    amp = amp_center_t * np.exp(-xs**2 / 2.0)
    fds = np.full(xs.shape, params.gamma / params.tau)
    return phase, amp, fs, fds


def chirp_time(params, wc: WDMWaveletConstants):
    """Directly compute time domain intrinsic_waveform"""
    ts = wc.dt * np.arange(0, wc.Nf * wc.Nt)

    Phases, Amps, fas, fdas = amp_phase_t(ts, params)

    hs = Amps * np.cos(Phases)

    waveform = StationaryWaveformTime(ts, Phases, fas, fdas, Amps)
    return hs, waveform


# @njit(fastmath=True)
def amp_phase_f(fs, params: LinearChirpletIntrinsicParams):
    """Get amplitude and phase for frequency chirp"""
    xs = (fs - params.f_center) / params.gamma
    # TODO check sign convention on phase
    Phase = params.phi0 + 2.0 * np.pi * params.t_center * fs + np.pi * params.tau * params.gamma * xs**2
    Amp = params.amp_center_f * np.exp(-xs**2 / 2.0)
    return Phase, Amp


def ChirpWaveletT(params, TFs, wc):
    """Get the wavelet in time domain"""
    TFs = np.arange(0, wc.Nt) * wc.DT
    Phases, Amps, fas, fdas = amp_phase_t(TFs, params)

    # get the intrinsic intrinsic_waveform
    return StationaryWaveformTime(TFs, np.array([Phases]), np.array([fas]), np.array([fdas]), np.array([Amps]))


# @njit(fastmath=True)
def toff(f, params: LinearChirpletIntrinsicParams):
    """Get times for sparse freq method"""
    return params.tau * (f - params.f_center) / params.gamma + params.t_center


# @njit(fastmath=True)
def foft(t, params: LinearChirpletIntrinsicParams):
    """Get frequencies for sparse time method"""
    return params.gamma * (t - params.t_center) / params.tau + params.f_center
