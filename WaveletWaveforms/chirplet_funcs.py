"""helper functions for Chirp_WDM"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numba import njit

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq, StationaryWaveformTime

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
    from WaveletWaveforms.wdm_config import WDMWaveletConstants


class LinearChirpletIntrinsicParams(NamedTuple):
    amp_center_f: float
    phi0: float
    f_center: float
    t_center: float
    tau: float
    gamma: float


@njit()
def chirplet_time_intrinsic(waveform: StationaryWaveformTime, intrinsic_params: LinearChirpletIntrinsicParams, t_in: NDArray[np.floating], nt_lim: PixelGenericRange) -> None:
    """Get amplitude, phase, frequency, and frequency derivative for the waveform.
    Uses a separate t_in array, which may be different from the time array in the waveform object.
    The separate t_in allows for conversion between different time coordinates, e.g., SSB to guiding center.
    """
    AT = waveform.AT
    PT = waveform.PT
    FT = waveform.FT
    FTd = waveform.FTd

    assert 0 <= nt_lim.nx_min < nt_lim.nx_max <= t_in.size
    assert AT.shape == PT.shape == FT.shape == FTd.shape
    assert len(AT.shape) == 1
    assert AT.shape[-1] <= nt_lim.nx_max

    # amplitude multiplier to convert from frequency domain amplitude to time domain amplitude
    phase_center_t = - intrinsic_params.phi0
    ftd = intrinsic_params.gamma / intrinsic_params.tau
    amp_center_t = np.sqrt(ftd) * intrinsic_params.amp_center_f

    #  compute the intrinsic frequency, phase and amplitude
    for n in range(nt_lim.nx_min, nt_lim.nx_max):
        t = t_in[n]
        delta_t = t - intrinsic_params.t_center
        x = delta_t / intrinsic_params.tau

        FT[n] = ftd * delta_t + intrinsic_params.f_center

        # TODO check sign convention on phase
        PT[n] = phase_center_t + 2.0 * np.pi * delta_t * FT[n] - np.pi * ftd * delta_t ** 2
        AT[n] = amp_center_t * np.exp(-x**2 / 2.0)
        FTd[n] = ftd


@njit()
def chirplet_freq_intrinsic(waveform: StationaryWaveformFreq, intrinsic_params: LinearChirpletIntrinsicParams, f_in: NDArray[np.floating]) -> None:
    AF = waveform.AF
    PF = waveform.PF
    TF = waveform.TF
    TFp = waveform.TFp

    nf_loc = f_in.size
    #  compute the intrinsic frequency, phase and amplitude
    phase_0_f = - np.pi / 4. + intrinsic_params.phi0
    ftd = intrinsic_params.gamma / intrinsic_params.tau
    tfp = 1. / ftd

    for n in range(nf_loc):
        f = f_in[n]
        delta_f = f - intrinsic_params.f_center
        x = delta_f / intrinsic_params.gamma
        PF[n] = phase_0_f + 2.0 * np.pi * intrinsic_params.t_center * f + np.pi * tfp * delta_f ** 2
        TF[n] = tfp * delta_f + intrinsic_params.t_center
        TFp[n] = tfp
        AF[n] = intrinsic_params.amp_center_f * np.exp(-x ** 2 / 2.0)


@njit()
def amp_phase_t(t:  NDArray[np.floating], params: LinearChirpletIntrinsicParams) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Get amplitude and phase for chirp_time"""
    x = (t - params.t_center) / params.tau
    f = params.gamma * x + params.f_center
    amp_center_t = np.sqrt(params.gamma / params.tau) * params.amp_center_f
    phase = - params.phi0 + 2. * np.pi * (t - params.t_center) * f - np.pi * params.gamma * params.tau * x ** 2
    amp = amp_center_t * np.exp(-x**2 / 2.0)
    fds = np.full(x.shape, params.gamma / params.tau)
    return phase, amp, f, fds


@njit()
def amp_phase_f(f:  NDArray[np.floating], params: LinearChirpletIntrinsicParams) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get amplitude and phase for frequency chirp"""
    x = (f - params.f_center) / params.gamma
    Phase = - np.pi / 4.0 + params.phi0 + 2.0 * np.pi * params.t_center * f + np.pi * params.tau * params.gamma * x**2
    Amp = params.amp_center_f * np.exp(-x**2 / 2.0)
    return Phase, Amp


def ChirpWaveletT(params: LinearChirpletIntrinsicParams, wc: WDMWaveletConstants) -> StationaryWaveformTime:
    """Get the wavelet in time domain"""
    TFs = np.arange(0, wc.Nt) * wc.DT
    Phases, Amps, fas, fdas = amp_phase_t(TFs, params)

    # get the intrinsic intrinsic_waveform
    return StationaryWaveformTime(TFs, np.array([Phases]), np.array([fas]), np.array([fdas]), np.array([Amps]))


# TODO toff and foft have unclear names and are redundant with functions above
@njit(fastmath=True)
def toff(f:  NDArray[np.floating], params: LinearChirpletIntrinsicParams) ->  NDArray[np.floating]:
    """Get times for sparse freq method"""
    return params.tau * (f - params.f_center) / params.gamma + params.t_center


@njit(fastmath=True)
def foft(t:  NDArray[np.floating], params: LinearChirpletIntrinsicParams) ->  NDArray[np.floating]:
    """Get frequencies for sparse time method"""
    return params.gamma * (t - params.t_center) / params.tau + params.f_center
