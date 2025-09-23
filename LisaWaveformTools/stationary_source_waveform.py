"""Abstract class for stationary wave approximation-based TDI intrinsic_waveform models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Generic, TypeVar

ExtrinsicParams = namedtuple('ExtrinsicParams', ['costh', 'phi', 'cosi', 'psi'])
ExtrinsicParams.__doc__ = """
Store the extrinsic parameters common to most detector intrinsic_waveform models.

Parameters
----------
costh : float
    Cosine of the source's ecliptic colatitude
phi : float
    Source's ecliptic longitude in radians
cosi : float
    Cosine of the source's inclination angle
psi : float
    Source polarization angle in radians
"""

SourceParams = namedtuple('SourceParams', ['intrinsic', 'extrinsic'])

SourceParams.__doc__ = """
Store both the intrinsic and extrinsic parameters for a source.

Parameters
----------
intrinsic: namedtuple
    The intrinsic parameters for the particular source class, e.g. LinearFrequencyIntrinsicParams
extrinsic: ExtrinsicParams
    The extrinsic parameters for the source
"""


class StationaryWaveform:
    """Abstract base class for waveforms in the stationary wave approximation."""


# class StationaryWaveformTime(NamedTuple, StationaryWaveform):
#    T: NDArray[np.floating]
#    PT: NDArray[np.floating]
#    FT: NDArray[np.floating]
#    FTd: NDArray[np.floating]
#    AT: NDArray[np.floating]


class StationaryWaveformTime(namedtuple('StationaryWaveformTime', ['T', 'PT', 'FT', 'FTd', 'AT']), StationaryWaveform):
    pass


class StationaryWaveformFreq(namedtuple('StationaryWaveformFreq', ['F', 'PF', 'TF', 'TFp', 'AF']), StationaryWaveform):
    pass


class StationaryWaveformGeneric(namedtuple('StationaryWaveformGeneric', ['Y', 'P', 'X', 'Xp', 'A']), StationaryWaveform):
    pass


# Subclasses can be either in time or frequency domain
StationaryWaveformType = TypeVar('StationaryWaveformType', bound=StationaryWaveform)


class StationarySourceWaveform(Generic[StationaryWaveformType], ABC):
    """Abstract base class for intrinsic_waveform models to be used in the stationary wave approximation."""

    def __init__(
        self,
        params: SourceParams,
        intrinsic_waveform: StationaryWaveformType,
        tdi_waveform: StationaryWaveformType,
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
        self._params: SourceParams = params

        self._intrinsic_waveform: StationaryWaveformType = intrinsic_waveform
        self._tdi_waveform: StationaryWaveformType = tdi_waveform

        self.update_params(params)

        assert self.consistent_instrinsic, 'StationarySourceWaveform failed to initialize consistently.'
        assert self.consistent_extrinsic, 'StationarySourceWaveform failed to initialize consistently.'
        assert self.consistent, 'StationarySourceWaveform failed to initialize consistently.'

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
        params : namedtuple
            Updated parameters (intrinsic and extrinsic) for the source.
        """
        self.params = params
        self._update()

    @property
    def consistent(self) -> bool:
        """Check if the internal representation is consistent with the input parameters.

        Returns
        ----------
        consistent : bool
            Whether the internal representation is consistent with the input parameters.

        """
        return self._consistent and self._consistent_intrinsic and self._consistent_extrinsic

    @property
    def consistent_instrinsic(self) -> bool:
        """Check if the intrinsic representation is consistent with the input parameters.

        Returns
        ----------
        consistent : bool
            Whether the internal representation is consistent with the intrinsic parameters.

        """
        return self._consistent_intrinsic

    @property
    def consistent_extrinsic(self) -> bool:
        """Check if the extrinsic representation is consistent with the input parameters.

        Returns
        ----------
        consistent : bool
            Whether the internal representation is consistent with the extrinsic parameters.

        """
        return self._consistent_extrinsic

    @property
    def params(self) -> SourceParams:
        """Get the current source parameters."""
        return self._params

    @params.setter
    def params(self, params_in: SourceParams) -> None:
        """Set the source parameters without updating the waveform. If called directly without
        calling `update_params`, the internal representation will not be consistent.
        """
        self._consistent_intrinsic = False
        self._consistent_extrinsic = False
        self._consistent = False
        self._params = params_in

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
        """Get the number of channels in the waveform."""
