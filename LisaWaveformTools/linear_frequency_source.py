"""A source with linearly increasing frequency and constant amplitude."""

from typing import NamedTuple, override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.source_params import AbstractIntrinsicParamsManager, ExtrinsicParams, SourceParams
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from LisaWaveformTools.stationary_time_source import StationarySourceWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletTaylorTime
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class LinearFrequencyIntrinsicParams(NamedTuple):
    """
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

    amp0_t: float
    phi0: float
    F0: float
    FTd0: float


N_LINEAR_FREQUENCY_PACKED = 4


def _load_intrinsic_linear_frequency_from_packed_helper(
    params_packed: NDArray[np.floating],
) -> LinearFrequencyIntrinsicParams:
    assert len(params_packed.shape) == 1
    assert params_packed.size == N_LINEAR_FREQUENCY_PACKED
    amp0_t = params_packed[0]
    phi0 = params_packed[1]
    F0 = params_packed[2]
    FTd0 = params_packed[3]
    return LinearFrequencyIntrinsicParams(amp0_t, phi0, F0, FTd0)


def _packed_from_intrinsic_linear_frequency_helper(params: LinearFrequencyIntrinsicParams) -> NDArray[np.floating]:
    return np.array([params.amp0_t, params.phi0, params.F0, params.FTd0])


def _validate_intrinsic_linear_frequency_helper(params: LinearFrequencyIntrinsicParams) -> bool:
    del params
    return True


class LinearFrequencyParamsManager(AbstractIntrinsicParamsManager[LinearFrequencyIntrinsicParams]):
    """Manage creation, translation, and handling of ExtrinsicParams objects."""

    def __init__(self, params_load: LinearFrequencyIntrinsicParams) -> None:
        self._n_packed: int = N_LINEAR_FREQUENCY_PACKED
        super().__init__(params_load)

    @property
    @override
    def n_packed(self) -> int:
        return self._n_packed

    @property
    @override
    def params_packed(self) -> NDArray[np.floating]:
        return _packed_from_intrinsic_linear_frequency_helper(self._params)

    @params_packed.setter
    @override
    def params_packed(self, params_in: NDArray[np.floating]) -> None:
        assert params_in.size == self.n_packed
        self.params: LinearFrequencyIntrinsicParams = _load_intrinsic_linear_frequency_from_packed_helper(params_in)

    def is_valid(self) -> bool:
        return _validate_intrinsic_linear_frequency_helper(self.params)


# TODO check factor of 2pi
@njit()
def linear_frequency_intrinsic(
    waveform: StationaryWaveformTime, intrinsic_params: LinearFrequencyIntrinsicParams, t_in: NDArray[np.float64]
) -> None:
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
        PT[n] = -intrinsic_params.phi0 + 2 * np.pi * intrinsic_params.F0 * t + np.pi * intrinsic_params.FTd0 * t**2
        AT[n] = intrinsic_params.amp0_t


# TODO do consistency checks
class LinearFrequencySourceWaveformTime(StationarySourceWaveformTime[LinearFrequencyIntrinsicParams, ExtrinsicParams]):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant time domain amplitude."""

    @override
    def _update_intrinsic(self) -> None:
        """Update the intrinsic_waveform with respect to the intrinsic parameters."""
        self._consistent_intrinsic: bool = False
        if not isinstance(self.params.intrinsic, LinearFrequencyIntrinsicParams):
            msg = 'Intrinsic parameters must be of type LinearFrequencyIntrinsicParams.'
            raise TypeError(msg)

        linear_frequency_intrinsic(self._intrinsic_waveform, self.params.intrinsic, self.wavefront_time)
        self._consistent_intrinsic = True

    @override
    def _create_intrinsic_params_manager(
        self, params_intrinsic: LinearFrequencyIntrinsicParams
    ) -> LinearFrequencyParamsManager:
        return LinearFrequencyParamsManager(params_intrinsic)


class LinearFrequencyWaveletWaveformTime(BinaryWaveletTaylorTime[LinearFrequencyIntrinsicParams, ExtrinsicParams]):
    """Store a wavelet waveform for a source with linearly increasing frequency and constant amplitude.

    Uses the Taylor time method.
    """

    def __init__(
        self,
        params: SourceParams,
        wc: WDMWaveletConstants,
        lc: LISAConstants,
        nt_lim_waveform: PixelGenericRange,
        *,
        wavelet_mode: int = 1,
        response_mode: int = 0,
        table_cache_mode: str = 'check',
        table_output_mode: str = 'skip',
    ) -> None:
        """Construct a binary wavelet object."""
        # get the intrinsic_waveform
        source_waveform = LinearFrequencySourceWaveformTime(params, nt_lim_waveform, lc, response_mode=response_mode)

        super().__init__(
            params,
            wc,
            lc,
            nt_lim_waveform,
            source_waveform,
            wavelet_mode=wavelet_mode,
            table_cache_mode=table_cache_mode,
            table_output_mode=table_output_mode,
        )
