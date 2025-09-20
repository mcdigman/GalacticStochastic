"""A source with linearly increasing frequency and constant amplitude."""

from typing import override

from LisaWaveformTools.stationary_freq_source import StationarySourceWaveformFreq
from WaveletWaveforms.chirplet_funcs import LinearChirpletIntrinsicParams, chirplet_freq_intrinsic


# TODO do consistency checks
class LinearChirpletSourceWaveformFreq(StationarySourceWaveformFreq):
    """Store a binary intrinsic_waveform with linearly increasing frequency and constant amplitude in the time domain.
    """

    @override
    def _update_intrinsic(self) -> None:
        """Update the intrinsic_waveform with respect to the intrinsic parameters."""
        if not isinstance(self.params.intrinsic, LinearChirpletIntrinsicParams):
            msg = 'Intrinsic parameters must be of type LinearChirpletIntrinsicParams.'
            raise TypeError(msg)

        chirplet_freq_intrinsic(self._intrinsic_waveform, self.params.intrinsic, self._intrinsic_waveform.F)
        self._consistent_intrinsic: bool = True
