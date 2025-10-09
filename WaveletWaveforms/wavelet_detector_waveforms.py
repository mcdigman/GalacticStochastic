"""Store a sparse binary wavelet intrinsic_waveform."""

from abc import ABC, abstractmethod
from typing import Generic, override
from warnings import warn

import h5py

from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.source_params import ExtrinsicParamsType, IntrinsicParamsType, SourceParams
from LisaWaveformTools.stationary_source_waveform import StationarySourceWaveform, StationaryWaveformTime, StationaryWaveformType
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.sparse_wavelet_time import SparseTimeCoefficientTable, get_empty_sparse_sparse_wavelet_time_waveform, get_sparse_table_helper, make_sparse_wavelet_time
from WaveletWaveforms.taylor_time_coefficients import (
    WaveletTaylorTimeCoeffs,
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket, wavemaket_direct
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class SparseWaveletSourceWaveform(Generic[StationaryWaveformType, IntrinsicParamsType, ExtrinsicParamsType], ABC):
    """Abstract base class for sparse wavelet waveforms."""

    def __init__(self, params: SourceParams, wavelet_waveform: SparseWaveletWaveform, source_waveform: StationarySourceWaveform[StationaryWaveformType, IntrinsicParamsType, ExtrinsicParamsType]) -> None:
        """Initialize the sparse wavelet intrinsic_waveform."""
        self._consistent: bool = False
        self._params: SourceParams = params
        self._wavelet_waveform: SparseWaveletWaveform = wavelet_waveform
        self._source_waveform: StationarySourceWaveform[StationaryWaveformType, IntrinsicParamsType, ExtrinsicParamsType] = source_waveform

        self.update_params(params)

        assert self.consistent, 'SparseWaveletSourceWaveform failed to initialize consistently. '

    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'wavelet_waveform', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
        if group_mode == 0:
            hf_wavelet = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_wavelet = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)

        hf_wavelet.attrs['creator_name'] = self.__class__.__name__
        hf_wavelet.attrs['params_name'] = self._params.__class__.__name__
        hf_wavelet.attrs['extrinsic_params_name'] = self._params.extrinsic.__class__.__name__
        hf_wavelet.attrs['intrinsic_params_name'] = self._params.intrinsic.__class__.__name__
        hf_wavelet.attrs['wavelet_waveform_name'] = self._wavelet_waveform.__class__.__name__
        hf_wavelet.attrs['source_waveform_name'] = self._source_waveform.__class__.__name__
        hf_wavelet.attrs['consistent'] = self._consistent
        hf_wavelet.attrs['consistent_source'] = self.consistent_source

        hf_p = hf_wavelet.create_group('params_intrinsic')
        for key in self._params.intrinsic._fields:
            hf_p.attrs[key] = getattr(self._params.intrinsic, key)

        hf_i = hf_wavelet.create_group('params_extrinsic')
        for key in self._params.extrinsic._fields:
            hf_i.attrs[key] = getattr(self._params.extrinsic, key)

        _ = hf_wavelet.create_dataset('wave_value', data=self._wavelet_waveform.wave_value, compression='gzip')
        _ = hf_wavelet.create_dataset('pixel_index', data=self._wavelet_waveform.pixel_index, compression='gzip')
        _ = hf_wavelet.create_dataset('n_set', data=self._wavelet_waveform.n_set, compression='gzip')
        _ = hf_wavelet.create_dataset('n_pixel_max', data=[self._wavelet_waveform.n_pixel_max], compression='gzip')

        _ = self._source_waveform.store_hdf5(hf_wavelet)

        return hf_wavelet

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
    def source_waveform(self) -> StationarySourceWaveform[StationaryWaveformType, IntrinsicParamsType, ExtrinsicParamsType]:
        """Get the current source parameters."""
        if not self.consistent_source:
            msg = 'Source waveform is not consistent with the requested source parameters.'
            raise ValueError(msg)

        return self._source_waveform

    @source_waveform.setter
    def source_waveform(self, source_waveform_in: StationarySourceWaveform[StationaryWaveformType, IntrinsicParamsType, ExtrinsicParamsType]) -> None:
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
        assert self.consistent, 'Waveform failed to update consistently.'

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
        return self._consistent and self.consistent_source


class BinaryWaveletTaylorTime(SparseWaveletSourceWaveform[StationaryWaveformTime, IntrinsicParamsType, ExtrinsicParamsType]):
    """Store a sparse binary wavelet for a time domain taylor waveform."""

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelGenericRange, source_waveform: StationarySourceWaveform[StationaryWaveformTime, IntrinsicParamsType, ExtrinsicParamsType], *, wavelet_mode: int = 1, storage_mode: int = 0, table_cache_mode: str = 'check', table_output_mode: str = 'skip') -> None:
        """Construct a sparse binary wavelet for a time domain taylor intrinsic_waveform with interpolation."""
        if storage_mode not in (0, 1):
            msg = 'Unrecognized option for storage_mode'
            raise ValueError(msg)

        self._wc: WDMWaveletConstants = wc
        self._lc: LISAConstants = lc
        self._nt_lim_waveform: PixelGenericRange = nt_lim_waveform
        self._wavelet_mode: int = wavelet_mode
        self._table_cache_mode = table_cache_mode
        self._table_output_mode = table_output_mode

        # get a blank wavelet intrinsic_waveform with the correct size for the sparse taylor time method
        # when consistent is set to True, it will be the correct intrinsic_waveform
        wavelet_waveform_loc: SparseWaveletWaveform = get_empty_sparse_taylor_time_waveform(int(self._lc.nc_waveform), wc)

        # interpolation for wavelet taylor expansion
        # TODO need better way of setting whether cache is checked, whether to output, and whether to store in hdf5
        self._taylor_time_table: WaveletTaylorTimeCoeffs = get_taylor_table_time(
            self._wc,
            cache_mode=self._table_cache_mode,
            output_mode=self._table_output_mode,
        )

        self._storage_mode: int = storage_mode

        super().__init__(params, wavelet_waveform_loc, source_waveform)

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'wavelet_waveform', group_mode: int = 0) -> h5py.Group:
        hf_wavelet = super().store_hdf5(hf_in, group_name=group_name, group_mode=group_mode)

        hf_wavelet.attrs['wavelet_mode'] = self._wavelet_mode
        hf_wavelet.attrs['storage_mode'] = self._storage_mode
        hf_wavelet.attrs['table_cache_mode'] = self._table_cache_mode
        hf_wavelet.attrs['table_output_mode'] = self._table_output_mode
        hf_wavelet.attrs['taylor_time_table_name'] = self._taylor_time_table.__class__.__name__

        if self._storage_mode == 0:
            hf_wavelet.create_dataset('Nfsam', data=self._taylor_time_table.Nfsam, compression='gzip')
            hf_wavelet.create_dataset('evc', data=self._taylor_time_table.evc, compression='gzip')
            hf_wavelet.create_dataset('evs', data=self._taylor_time_table.evs, compression='gzip')
            hf_wavelet.create_dataset('wavelet_norm', data=self._taylor_time_table.wavelet_norm, compression='gzip')

        hf_wavelet.attrs['nt_lim_name'] = self._nt_lim_waveform.__class__.__name__
        hf_nt = hf_wavelet.create_group('nt_lim_waveform')
        for key in self._nt_lim_waveform._fields:
            hf_nt.attrs[key] = getattr(self._nt_lim_waveform, key)

        hf_wavelet.attrs['wc_name'] = self._wc.__class__.__name__
        hf_wc = hf_wavelet.create_group('wc')
        for key in self._wc._fields:
            hf_wc.attrs[key] = getattr(self._wc, key)

        hf_lc = hf_wavelet.create_group('lc')
        hf_wavelet.attrs['lc_name'] = self._lc.__class__.__name__
        for key in self._lc._fields:
            hf_lc.attrs[key] = getattr(self._lc, key)
        return hf_wavelet

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


class BinaryWaveletSparseTime(SparseWaveletSourceWaveform[StationaryWaveformTime, IntrinsicParamsType, ExtrinsicParamsType]):
    """Store a sparse binary wavelet for a time domain sparse waveform."""

    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelGenericRange, source_waveform: StationarySourceWaveform[StationaryWaveformTime, IntrinsicParamsType, ExtrinsicParamsType], *, storage_mode: int = 0, wavelet_mode: int = 0) -> None:
        """Construct a sparse binary wavelet for a time domain taylor intrinsic_waveform with interpolation."""
        self._wc: WDMWaveletConstants = wc
        self._lc: LISAConstants = lc
        self._nt_lim_waveform: PixelGenericRange = nt_lim_waveform
        self._storage_mode: int = storage_mode
        self._wavelet_mode: int = wavelet_mode

        # get a blank wavelet intrinsic_waveform with the correct size for the sparse taylor time method
        # when consistent is set to True, it will be the correct intrinsic_waveform
        wavelet_waveform_loc: SparseWaveletWaveform = get_empty_sparse_sparse_wavelet_time_waveform(int(self._lc.nc_waveform), wc)

        # interpolation for wavelet taylor expansion
        self._sparse_table: SparseTimeCoefficientTable = get_sparse_table_helper(self._wc)

        super().__init__(params, wavelet_waveform_loc, source_waveform)

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'wavelet_waveform', group_mode: int = 0) -> h5py.Group:
        hf_wavelet = super().store_hdf5(hf_in, group_name=group_name, group_mode=group_mode)

        hf_wavelet.attrs['wavelet_mode'] = self._wavelet_mode
        hf_wavelet.attrs['storage_mode'] = self._storage_mode
        hf_wavelet.attrs['sparse_table_name'] = self._sparse_table.__class__.__name__

        if self._storage_mode == 0:
            hf_wavelet.create_dataset('cM', data=self._sparse_table.cM, compression='gzip')
            hf_wavelet.create_dataset('sM', data=self._sparse_table.sM, compression='gzip')

        hf_wavelet.attrs['nt_lim_name'] = self._nt_lim_waveform.__class__.__name__
        hf_nt = hf_wavelet.create_group('nt_lim_waveform')
        for key in self._nt_lim_waveform._fields:
            hf_nt.attrs[key] = getattr(self._nt_lim_waveform, key)

        hf_wavelet.attrs['wc_name'] = self._wc.__class__.__name__
        hf_wc = hf_wavelet.create_group('wc')
        for key in self._wc._fields:
            hf_wc.attrs[key] = getattr(self._wc, key)

        hf_lc = hf_wavelet.create_group('lc')
        hf_wavelet.attrs['lc_name'] = self._lc.__class__.__name__
        for key in self._lc._fields:
            hf_lc.attrs[key] = getattr(self._lc, key)
        return hf_wavelet

    @override
    def _update_wavelet_waveform(self) -> None:
        """Update the wavelet intrinsic_waveform to match the current parameters."""
        make_sparse_wavelet_time(
            self._source_waveform,
            self._wavelet_waveform,
            self._sparse_table,
            self._wc,
        )
