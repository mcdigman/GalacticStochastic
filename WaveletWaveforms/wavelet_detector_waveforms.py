"""Store a sparse binary wavelet waveform."""

from abc import ABC, abstractmethod
from typing import override

from LisaWaveformTools.linear_frequency_source import LinearFrequencyWaveformTime
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.stationary_source_waveform import SourceParams, StationarySourceWaveform
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_time_coefficients import (
    WaveletTaylorTimeCoeffs,
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class SparseWaveletSourceWaveform(ABC):
    """Abstract base class for sparse wavelet waveforms."""
    def __init__(self, params: SourceParams, wavelet_waveform: SparseWaveletWaveform, source_waveform: StationarySourceWaveform):
        """Initialize the sparse wavelet waveform."""
        self._consistent = False
        self.params = params
        self.wavelet_waveform = wavelet_waveform
        self.source_waveform = source_waveform

        # initialize to input parameters
        self.update_params(params)

    @property
    def params(self) -> SourceParams:
        """Get the current source parameters."""
        return self._params

    @params.setter
    def params(self, params_in: SourceParams) -> None:
        """Set the source parameters and update the waveform. If called directly without
        calling `update_params`, the internal representation will not be consistent.
        """
        self.consistent = False
        self._params = params_in

    @abstractmethod
    def _update_wavelet_waveform(self) -> None:
        """Update the wavelet waveform to match the current parameters."""

    def update_params(self, params_in: SourceParams) -> None:
        """Update the internal wavelet representation to match the input parameters.
        This will update both the source waveform and the wavelet waveform to match the new parameters.
        """
        # Store the new parameters
        self.params = params_in

        self.source_waveform.update_params(params_in)
        self._update_wavelet_waveform()

        # Set consistent to True after the update is complete
        self.consistent = True

    def get_unsorted_coeffs(self) -> SparseWaveletWaveform:
        """Get wavelet coefficients in the order they are generated."""
        if not self.consistent:
            self.update_params(self.params)
        return self.wavelet_waveform

    @property
    def consistent(self) -> bool:
        """Check if the internal representation is consistent with the input parameters.

        Returns
        ----------
        consistent : bool
            Whether the internal representation is consistent with the input parameters.

        """
        return self._consistent

    @consistent.setter
    def consistent(self, consistent: bool) -> None:
        """Set whether the internal representation is consistent with the input parameters.

        Parameters
        ----------
        consistent : bool
            Whether the internal representation is consistent with the input parameters.
        """
        self._consistent = consistent

    def _update_source_waveform(self) -> None:
        """Update the source waveform to match the current parameters."""
        self.source_waveform.update_params(self.params)


class BinaryWaveletAmpFreqDT(SparseWaveletSourceWaveform):
    """Store a sparse binary wavelet and update if for search."""

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelTimeRange) -> None:
        """Construct a binary wavelet object."""
        self.wc = wc
        self.lc = lc
        self.nt_lim_waveform = nt_lim_waveform

        # get the waveform
        source_waveform = LinearFrequencyWaveformTime(
            params, self.nt_lim_waveform, self.lc, self.wc,
        )

        # get a blank waveform in the sparse wavelet domain
        # when consistent is set to True, it will be the correct waveform
        wavelet_waveform_loc: SparseWaveletWaveform = get_empty_sparse_taylor_time_waveform(self.lc.nc_waveform, wc)

        # interpolation for wavelet taylor expansion
        self.taylor_time_table: WaveletTaylorTimeCoeffs = get_taylor_table_time(
            self.wc, cache_mode='skip', output_mode='skip',
        )

        super().__init__(params, wavelet_waveform_loc, source_waveform)

    @override
    def _update_wavelet_waveform(self) -> None:
        """Update the wavelet waveform to match the current parameters."""
        wavemaket(
            self.wavelet_waveform,
            self.source_waveform.get_tdi_waveform(),
            self.nt_lim_waveform,
            self.wc,
            self.taylor_time_table,
            force_nulls=False,
        )
