"""A source with linearly increasing frequency and constant amplitude."""

from typing import override

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.stationary_source_waveform import SourceParams
from LisaWaveformTools.stationary_time_source import StationarySourceWaveformTime
from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams, chirplet_time_intrinsic
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletTaylorTime
from WaveletWaveforms.wdm_config import WDMWaveletConstants


# TODO do consistency checks
class LinearChirpletSourceWaveformTime(StationarySourceWaveformTime):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant amplitude in the time domain.
    """

    @override
    def _update_intrinsic(self) -> None:
        """Update the intrinsic_waveform with respect to the intrinsic parameters."""
        self._consistent_intrinsic = False
        if not isinstance(self.params.intrinsic, LinearChirpletIntrinsicParams):
            msg = 'Intrinsic parameters must be of type LinearChirpletIntrinsicParams.'
            raise TypeError(msg)

        chirplet_time_intrinsic(self._intrinsic_waveform, self.params.intrinsic, self.wavefront_time)
        self._consistent_intrinsic = True


class LinearChirpletWaveletWaveformTime(BinaryWaveletTaylorTime):
    """Store a sparse wavelet intrinsic_waveform for a source with linearly increasing frequency and constant amplitude,
    using the Taylor time method.
    """

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelTimeRange, *, wavelet_mode: int = 1) -> None:
        """Construct a binary wavelet object."""
        # get the intrinsic_waveform
        source_waveform = LinearChirpletSourceWaveformTime(
            params, nt_lim_waveform, lc,
        )

        super().__init__(params, wc, lc, nt_lim_waveform, source_waveform, wavelet_mode=wavelet_mode)
