"""A source with linearly increasing frequency and constant amplitude."""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.linear_frequency_source import LinearFrequencyIntrinsicParams
from LisaWaveformTools.stationary_freq_source import StationarySourceWaveformFreq
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq


# TODO check factor of 2pi
@njit()
def linear_frequency_intrinsic_freq(waveform: StationaryWaveformFreq, intrinsic_params: LinearFrequencyIntrinsicParams, f_in: NDArray[np.floating]) -> None:
    """
    Get time domain intrinsic_waveform for a linearly increasing frequency source with constant amplitude.

    For use in the stationary phase approximation. The actual intrinsic_waveform would be given:
    h = amp_t*np.cos(-phi0 + 2*pi*F0*t + pi*FTd0*t**2)

    pt = params[6] + 2.0 * np.pi * (params[8] - ts) * fs + np.pi * params[7] * params[0] * xs**2 + np.pi / 4.
    pf = params[6] + 2.0 * np.pi * params[8] * fs + np.pi * params[0] * params[7] * xs**2
    Note that t_in is not the same as the time coordinate in the input intrinsic_waveform object;
    it may be a different time coordinate, due to the need to convert from the solar system barycenter
    frame to the constellation guiding center frame (or the frames of the individual spacecraft).

    Parameters
    ----------
    waveform : StationaryWaveformTime
        The intrinsic_waveform object to be updated in place with the time domain intrinsic_waveform.
    intrinsic_params : LinearFrequencyIntrinsicParams
        Dictionary or namespace containing intrinsic parameters of the source.
    t_in : np.ndarray
        Array of arrival times (or evaluation times) for the signal.

    Returns
    -------
    None
        The function updates the input `intrinsic_waveform` object in place.
    """
    AF = waveform.AF
    PF = waveform.PF
    TF = waveform.TF
    TFp = waveform.TFp

    nf_loc = f_in.size
    #  compute the intrinsic frequency, phase and amplitude
    for n in range(nf_loc):
        f = f_in[n]
        TF[n] = -intrinsic_params.F0 / intrinsic_params.FTd0 + f / intrinsic_params.FTd0
        TFp[n] = 1. / intrinsic_params.FTd0
        PF[n] = intrinsic_params.phi0 - np.pi / 4. + intrinsic_params.F0**2 * np.pi / intrinsic_params.FTd0 - 2 * np.pi * intrinsic_params.F0 / intrinsic_params.FTd0 * f + np.pi / intrinsic_params.FTd0 * f**2
        AF[n] = intrinsic_params.amp0_t / np.sqrt(intrinsic_params.FTd0)


# TODO do consistency checks
class LinearFrequencySourceWaveformFreq(StationarySourceWaveformFreq):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant amplitude in the time domain.
    """

    @override
    def _update_intrinsic(self) -> None:
        """Update the intrinsic_waveform with respect to the intrinsic parameters."""
        if not isinstance(self.params.intrinsic, LinearFrequencyIntrinsicParams):
            msg = 'Intrinsic parameters must be of type LinearFrequencyIntrinsicParams.'
            raise TypeError(msg)

        linear_frequency_intrinsic_freq(self._intrinsic_waveform, self.params.intrinsic, self._intrinsic_waveform.F)
        self._consistent_intrinsic: bool = True
