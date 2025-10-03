"""A source with linearly increasing frequency and constant amplitude."""

from typing import override

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.source_params import ExtrinsicParams, SourceParams
from LisaWaveformTools.stationary_time_source import StationarySourceWaveformTime
from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams, LinearChirpletParamsManager, chirplet_time_intrinsic
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange
from WaveletWaveforms.sparse_wavelet_time import get_sparse_source_t_grid
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletSparseTime, BinaryWaveletTaylorTime
from WaveletWaveforms.wdm_config import WDMWaveletConstants


# TODO do consistency checks
class LinearChirpletSourceWaveformTime(StationarySourceWaveformTime[LinearChirpletIntrinsicParams, ExtrinsicParams]):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant amplitude in the time domain."""

    @override
    def _create_intrinsic_params_manager(self, params_intrinsic: LinearChirpletIntrinsicParams) -> LinearChirpletParamsManager:
        return LinearChirpletParamsManager(params_intrinsic)

    @override
    def _update_intrinsic(self) -> None:
        """Update the intrinsic_waveform with respect to the intrinsic parameters."""
        self._consistent_intrinsic: bool = False
        if not isinstance(self.params.intrinsic, LinearChirpletIntrinsicParams):
            msg = 'Intrinsic parameters must be of type LinearChirpletIntrinsicParams.'
            raise TypeError(msg)

        chirplet_time_intrinsic(self._intrinsic_waveform, self.params.intrinsic, self.wavefront_time, self._nt_lim_waveform)
        self._consistent_intrinsic = True


class LinearChirpletWaveletTaylorTime(BinaryWaveletTaylorTime[LinearChirpletIntrinsicParams, ExtrinsicParams]):
    """Store a wavelet waveform for a chirplet using the Taylor time method."""

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelGenericRange, *, wavelet_mode: int = 1, response_mode: int = 0) -> None:
        """Construct a binary wavelet object."""
        # get the intrinsic_waveform
        source_waveform = LinearChirpletSourceWaveformTime(
            params, nt_lim_waveform, lc, response_mode=response_mode,
        )

        super().__init__(params, wc, lc, nt_lim_waveform, source_waveform, wavelet_mode=wavelet_mode)


class LinearChirpletWaveletSparseTime(BinaryWaveletSparseTime[LinearChirpletIntrinsicParams, ExtrinsicParams]):
    """Store a wavelet waveform for a chirplet, using the Taylor time method."""

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelGenericRange, *, response_mode: int = 0) -> None:
        """Construct a binary wavelet object."""
        self._nt_lim_grid: PixelGenericRange = get_sparse_source_t_grid(wc, lc.t0)
        source_waveform = LinearChirpletSourceWaveformTime(
            params, self._nt_lim_grid, lc, response_mode=response_mode,
        )

        super().__init__(params, wc, lc, nt_lim_waveform, source_waveform)
