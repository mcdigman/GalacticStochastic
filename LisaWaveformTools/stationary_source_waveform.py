"""Abstract class for stationary wave approximation-based TDI intrinsic_waveform models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, cast

from LisaWaveformTools.source_params import AbstractExtrinsicParamsManager, AbstractIntrinsicParamsManager, ExtrinsicParams, ExtrinsicParamsManager, ExtrinsicParamsType, IntrinsicParamsType, SourceParams

if TYPE_CHECKING:
    import h5py
    import numpy as np
    from numpy.typing import NDArray


class StationaryWaveformTime(NamedTuple):
    T: NDArray[np.floating]
    PT: NDArray[np.floating]
    FT: NDArray[np.floating]
    FTd: NDArray[np.floating]
    AT: NDArray[np.floating]


class StationaryWaveformFreq(NamedTuple):
    F: NDArray[np.floating]
    PF: NDArray[np.floating]
    TF: NDArray[np.floating]
    TFp: NDArray[np.floating]
    AF: NDArray[np.floating]


class StationaryWaveformGeneric(NamedTuple):
    Y: NDArray[np.floating]
    P: NDArray[np.floating]
    X: NDArray[np.floating]
    Xp: NDArray[np.floating]
    A: NDArray[np.floating]


# Subclasses can be either in time or frequency domain
StationaryWaveformType = TypeVar('StationaryWaveformType', bound=StationaryWaveformTime | StationaryWaveformFreq | StationaryWaveformGeneric)

StationarySourceWaveformType = TypeVar('StationarySourceWaveformType')


class StationarySourceWaveform(Generic[StationaryWaveformType, IntrinsicParamsType, ExtrinsicParamsType], ABC):
    """Abstract base class for intrinsic_waveform models to be used in the stationary wave approximation."""

    @abstractmethod
    def _create_intrinsic_params_manager(self, params_intrinsic: IntrinsicParamsType) -> AbstractIntrinsicParamsManager[IntrinsicParamsType]:
        """Get an intrinsic parameter manager object."""

    def _create_extrinsic_params_manager(self, params_extrinsic: ExtrinsicParamsType) -> AbstractExtrinsicParamsManager[ExtrinsicParamsType] | ExtrinsicParamsManager:
        """Get an intrinsic parameter manager object."""
        if not isinstance(params_extrinsic, ExtrinsicParams):
            msg = """No implementation for input type"""
            raise NotImplementedError(msg)
        return ExtrinsicParamsManager(params_extrinsic)

    def __init__(
        self,
        params: SourceParams,
        intrinsic_waveform: StationaryWaveformType,
        tdi_waveform: StationaryWaveformType,
        response_mode: int = 0,
    ) -> None:
        """
        Initialize the intrinsic_waveform object for use in the stationary wave approximation.

        Parameters
        ----------
        params : SourceParams
            Model-specific parameters (intrinsic and extrinsic) for the source.
        """
        self._consistent: bool = False
        self._consistent_intrinsic: bool = False
        self._consistent_extrinsic: bool = False
        self._response_mode: int = response_mode

        intrinsic: IntrinsicParamsType = cast('IntrinsicParamsType', params.intrinsic)
        extrinsic: ExtrinsicParamsType = cast('ExtrinsicParamsType', params.extrinsic)
        self._intrinsic_params_manager: AbstractIntrinsicParamsManager[IntrinsicParamsType] = self._create_intrinsic_params_manager(intrinsic)
        self._extrinsic_params_manager: AbstractExtrinsicParamsManager[ExtrinsicParamsType] | ExtrinsicParamsManager = self._create_extrinsic_params_manager(extrinsic)

        self._intrinsic_waveform: StationaryWaveformType = intrinsic_waveform
        self._tdi_waveform: StationaryWaveformType = tdi_waveform

        self.update_params(params)

        assert self.consistent_instrinsic, 'StationarySourceWaveform failed to initialize consistently.'
        assert self.consistent_extrinsic, 'StationarySourceWaveform failed to initialize consistently.'
        assert self.consistent, 'StationarySourceWaveform failed to initialize consistently.'

    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'source_waveform', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
        if group_mode == 0:
            hf_source = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_source = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)

        hf_source.attrs['creator_name'] = self.__class__.__name__
        hf_source.attrs['intrinsic_waveform_name'] = self._intrinsic_waveform.__class__.__name__
        hf_source.attrs['tdi_waveform_name'] = self._tdi_waveform.__class__.__name__
        hf_source.attrs['intrinsic_param_manager_name'] = self._intrinsic_params_manager.__class__.__name__
        hf_source.attrs['extrinsic_param_manager_name'] = self._extrinsic_params_manager.__class__.__name__
        hf_source.attrs['_consistent'] = self._consistent
        hf_source.attrs['_consistent_extrinsic'] = self._consistent_extrinsic
        hf_source.attrs['_consistent_intrinsic'] = self._consistent_intrinsic
        _ = self._intrinsic_params_manager.store_hdf5(hf_source, group_name='intrinsic')
        _ = self._extrinsic_params_manager.store_hdf5(hf_source, group_name='extrinsic')
        return hf_source

    @abstractmethod
    def _update_intrinsic(self) -> None:
        """
        Update source intrinsic_waveform quantities specific to the stored intrinsic source parameters.

        Compute e.g. amplitude, phase, frequency, and frequency derivative arrays for the source signal.
        To be called by update_params.
        """

    @abstractmethod
    def _update_extrinsic(self) -> None:
        """
        Compute the TDI intrinsic_waveform channels specific to the stored extrinsic source parameters.

        Typical extrinsic parameters include the ecliptic longitude, ecliptic colatitude, inclination, and polarization.
        To be called by update_params.
        """

    def _update(self) -> None:
        """Update both intrinsic and extrinsic parts of the waveform."""
        self._update_intrinsic()
        self._update_extrinsic()
        self._consistent = True
        assert self.consistent, 'Waveform failed to update consistently.'

    def update_params(self, params: SourceParams) -> None:
        """
        Update the intrinsic_waveform to match a new set of model parameters.

        Typically should call both _update_intrinsic and _update_extrinsic.

        Parameters
        ----------
        params : SourceParams
            Updated parameters (intrinsic and extrinsic) for the source.
        """
        self._consistent_intrinsic = False
        self._consistent_extrinsic = False
        self._consistent = False
        intrinsic: IntrinsicParamsType = cast('IntrinsicParamsType', params.intrinsic)
        extrinsic: ExtrinsicParamsType = cast('ExtrinsicParamsType', params.extrinsic)
        self._intrinsic_params_manager.params = intrinsic
        self._extrinsic_params_manager.params = extrinsic
        self._update()

    @property
    def consistent(self) -> bool:
        """
        Check if the internal representation is consistent with the input parameters.

        Returns
        -------
        consistent : bool
            Whether the internal representation is consistent with the input parameters.
        """
        return self._consistent and self._consistent_intrinsic and self._consistent_extrinsic

    @property
    def consistent_instrinsic(self) -> bool:
        """
        Check if the intrinsic representation is consistent with the input parameters.

        Returns
        -------
        consistent : bool
            Whether the internal representation is consistent with the intrinsic parameters.
        """
        return self._consistent_intrinsic

    @property
    def consistent_extrinsic(self) -> bool:
        """
        Check if the extrinsic representation is consistent with the input parameters.

        Returns
        -------
        consistent : bool
            Whether the internal representation is consistent with the extrinsic parameters.
        """
        return self._consistent_extrinsic

    @property
    def params(self) -> SourceParams:
        """
        Get the current source parameters.

        Returns
        -------
        params : SourceParams
            The current source parameters.
        """
        return SourceParams(self._intrinsic_params_manager.params, self._extrinsic_params_manager.params)

    @params.setter
    def params(self, params_in: SourceParams) -> None:
        """
        Set the source parameters without updating the waveform.

        If called directly without
        calling `update_params`, the internal representation will not be consistent.
        """
        self._consistent_intrinsic = False
        self._consistent_extrinsic = False
        self._consistent = False
        intrinsic: IntrinsicParamsType = cast('IntrinsicParamsType', params_in.intrinsic)
        extrinsic: ExtrinsicParamsType = cast('ExtrinsicParamsType', params_in.extrinsic)
        self._intrinsic_params_manager.params = intrinsic
        self._extrinsic_params_manager.params = extrinsic

    @property
    def tdi_waveform(self) -> StationaryWaveformType:
        """
        Get the TDI intrinsic_waveform channels for the source.

        Returns
        -------
        tdi_waveform: StationaryWaveformType
            The TDI intrinsic_waveform channels computed from the source parameters.
        """
        if not self.consistent_extrinsic:
            msg = 'Source parameters have not been updated yet.'
            raise ValueError(msg)
        return self._tdi_waveform

    @property
    def intrinsic_waveform(self) -> StationaryWaveformType:
        """
        Get the intrinsic intrinsic_waveform channels for the source.

        Returns
        -------
        intrinsic_waveform: StationaryWaveformType
            The intrinsic intrinsic_waveform computed for the source parameters.
        """
        if not self.consistent_instrinsic:
            msg = 'Source parameters have not been updated yet.'
            raise ValueError(msg)
        return self._intrinsic_waveform

    @property
    @abstractmethod
    def nc_waveform(self) -> int:
        """Get the number of channels in the waveform.

        Returns
        -------
        nc_waveform : int
            The number of channels in the waveform.
        """

    @property
    def response_mode(self) -> int:
        """Get the current response mode."""
        return self._response_mode
