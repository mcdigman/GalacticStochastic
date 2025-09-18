"""A source with linearly increasing frequency and constant amplitude."""

from collections import namedtuple
from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.stationary_source_waveform import SourceParams, StationaryWaveformTime
from LisaWaveformTools.stationary_time_source import StationarySourceWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletTaylorTime
from WaveletWaveforms.wdm_config import WDMWaveletConstants

LinearFrequencyIntrinsicParams = namedtuple('LinearFrequencyIntrinsicParams', ['amp0_t', 'phi0', 'F0', 'FTd0'])

LinearFrequencyIntrinsicParams.__doc__ = """
Store the intrinsic parameters for a source with linearly increasing frequency.

Galactic binaries are the prototypical example of such a source.
The amplitude is assumed constant in the time domain.
The time domain intrinsic_waveform is given
h(t) = amp_t*cos(-phi0 + 2*pi*F0*t + pi*FTd0*t^2)

Parameters
----------
amp0_t: float
    The constant time domain intrinsic_waveform amplitude
phi0: float
    The time domain phase at t=0
F0: float
    The frequency at t=0
FTd0: float
    The frequency derivative with respect to time dF/dt at t=0
"""


# TODO check factor of 2pi
@njit()
def linear_frequency_intrinsic(waveform: StationaryWaveformTime, intrinsic_params: LinearFrequencyIntrinsicParams, t_in: NDArray[np.float64]) -> None:
    """
    Get time domain intrinsic_waveform for a linearly increasing frequency source with constant amplitude.

    For use in the stationary phase approximation. The actual intrinsic_waveform would be given:
    h = amp_t*np.cos(-phi0 + 2*pi*F0*t + pi*FTd0*t**2)
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
    AT = waveform.AT
    PT = waveform.PT
    FT = waveform.FT
    FTd = waveform.FTd

    nt_loc = t_in.size
    #  compute the intrinsic frequency, phase and amplitude
    for n in range(nt_loc):
        t = t_in[n]
        FT[n] = intrinsic_params.F0 + intrinsic_params.FTd0 * t
        FTd[n] = intrinsic_params.FTd0
        PT[n] = -intrinsic_params.phi0 + 2 * np.pi * intrinsic_params.F0 * t + np.pi * intrinsic_params.FTd0 * t ** 2
        AT[n] = intrinsic_params.amp0_t


# TODO do consistency checks
class LinearFrequencySourceWaveformTime(StationarySourceWaveformTime):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant amplitude in the time domain.
    """

    @override
    def _update_intrinsic(self) -> None:
        """Update the intrinsic_waveform with respect to the intrinsic parameters."""
        self._consistent_intrinsic = False
        if not isinstance(self.params.intrinsic, LinearFrequencyIntrinsicParams):
            msg = 'Intrinsic parameters must be of type LinearFrequencyIntrinsicParams.'
            raise TypeError(msg)

        linear_frequency_intrinsic(self._intrinsic_waveform, self.params.intrinsic, self.wavefront_time)
        self._consistent_intrinsic = True


class LinearFrequencyWaveletWaveformTime(BinaryWaveletTaylorTime):
    """Store a sparse wavelet intrinsic_waveform for a source with linearly increasing frequency and constant amplitude,
    using the Taylor time method.
    """

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelTimeRange, *, wavelet_mode: int = 1, response_mode=0) -> None:
        """Construct a binary wavelet object."""
        # get the intrinsic_waveform
        source_waveform = LinearFrequencySourceWaveformTime(params, nt_lim_waveform, lc, response_mode=response_mode)

        super().__init__(params, wc, lc, nt_lim_waveform, source_waveform, wavelet_mode=wavelet_mode)
