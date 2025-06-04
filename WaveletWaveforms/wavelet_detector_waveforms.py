"""Store a sparse binary wavelet intrinsic_waveform."""

from abc import ABC, abstractmethod
from typing import override
from warnings import warn

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.stationary_source_waveform import SourceParams, StationarySourceWaveform
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_time_coefficients import (
    WaveletTaylorTimeCoeffs,
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket, wavemaket_direct
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class SparseWaveletSourceWaveform(ABC):
    """Abstract base class for sparse wavelet waveforms."""
    def __init__(self, params: SourceParams, wavelet_waveform: SparseWaveletWaveform, source_waveform: StationarySourceWaveform):
        """Initialize the sparse wavelet intrinsic_waveform."""
        self._consistent: bool = False
        self._params: SourceParams = params
        self._wavelet_waveform: SparseWaveletWaveform = wavelet_waveform
        self._source_waveform: StationarySourceWaveform = source_waveform

        self.update_params(params)

        assert self.consistent, 'SparseWaveletSourceWaveform failed to initialize consistently. '

    @property
    def params(self) -> SourceParams:
        """Get the current source parameters."""
        return self._params

    @params.setter
    def params(self, params_in: SourceParams) -> None:
        """Set the source parameters without updating the waveform. If called directly without
        calling `update_params`, the internal representation will not be consistent.
        """
        self._consistent = False
        self._params = params_in

    @property
    def consistent_source(self) -> bool:
        """Check if the source waveform is consistent with the input parameters."""
        return self._source_waveform.consistent and self.params == self._source_waveform.params

    @property
    def source_waveform(self) -> StationarySourceWaveform:
        """Get the current source parameters."""
        if not self.consistent_source:
            msg = 'Source waveform is not consistent with the requested source parameters.'
            raise ValueError(msg)

        return self._source_waveform

    @source_waveform.setter
    def source_waveform(self, source_waveform_in: StationarySourceWaveform) -> None:
        """Set the source waveform. If called directly without
        calling `update_params`, the internal representation will not be consistent.
        """
        self._consistent = False
        if source_waveform_in.params != self.params:
            msg = 'Source waveform parameters do not match the current source parameters.'
            warn(msg, UserWarning, stacklevel=2)
        self._source_waveform = source_waveform_in
        self.update_params(self._source_waveform.params)

    @property
    def wavelet_waveform(self) -> SparseWaveletWaveform:
        """Get the current source waveform."""
        if not self.consistent:
            msg = 'Wavelet waveform is not consistent with the current source parameters.'
            raise ValueError(msg)

        return self._wavelet_waveform

    @abstractmethod
    def _update_wavelet_waveform(self) -> None:
        """Update the wavelet intrinsic_waveform to match the current parameters."""

    def _update_source_waveform(self) -> None:
        """Update the source intrinsic_waveform to match the current parameters."""
        self._source_waveform.update_params(self.params)

    def update_params(self, params_in: SourceParams) -> None:
        """Update the internal wavelet representation to match the input parameters.
        This will update both the source intrinsic_waveform and the wavelet intrinsic_waveform to match the new parameters.
        """
        # Store the new parameters
        self.params = params_in

        self._update_source_waveform()
        self._update_wavelet_waveform()

        # Set consistent to True after the update is complete
        self._consistent = True

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


class BinaryWaveletTaylorTime(SparseWaveletSourceWaveform):
    """Store a sparse binary wavelet for a time domain taylor intrinsic_waveform."""

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelTimeRange, source_waveform: StationarySourceWaveform, *, wavelet_mode: int = 1) -> None:
        """Construct a sparse binary wavelet for a time domain taylor intrinsic_waveform with interpolation."""
        self._wc: WDMWaveletConstants = wc
        self._lc: LISAConstants = lc
        self._nt_lim_waveform: PixelTimeRange = nt_lim_waveform
        self._wavelet_mode: int = wavelet_mode

        # store the intrinsic_waveform
        self._source_waveform: StationarySourceWaveform = source_waveform

        # get a blank wavelet intrinsic_waveform with the correct size for the sparse taylor time method
        # when consistent is set to True, it will be the correct intrinsic_waveform
        wavelet_waveform_loc: SparseWaveletWaveform = get_empty_sparse_taylor_time_waveform(int(self._lc.nc_waveform), wc)

        # interpolation for wavelet taylor expansion
        self._taylor_time_table: WaveletTaylorTimeCoeffs = get_taylor_table_time(
            self._wc, cache_mode='skip', output_mode='skip',
        )

        super().__init__(params, wavelet_waveform_loc, source_waveform)

    @override
    def _update_wavelet_waveform(self) -> None:
        """Update the wavelet intrinsic_waveform to match the current parameters."""
        if self._wavelet_mode == 0:
            wavemaket_direct(
                self._wavelet_waveform,
                self.source_waveform.tdi_waveform,
                self._nt_lim_waveform,
                self._wc,
                self._taylor_time_table,
            )
        elif self._wavelet_mode in (1, 2, 3):
            # including 3 for future compatibility with computing coefficients for nulls
            wavemaket(
                self._wavelet_waveform,
                self.source_waveform.tdi_waveform,
                self._nt_lim_waveform,
                self._wc,
                self._taylor_time_table,
                force_nulls=self._wavelet_mode - 1,
            )
        else:
            msg = 'Unrecognized wavelet mode: {}. Valid modes are 0, 1, 2 or 3.'.format(self._wavelet_mode)
            raise NotImplementedError(msg)
