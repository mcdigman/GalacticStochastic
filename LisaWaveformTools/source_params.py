"""Objects that store the parametrization for binary sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, override

import numpy as np

if TYPE_CHECKING:
    import h5py
    from numpy.typing import NDArray

N_EXTRINSIC_PACKED = 4


class ExtrinsicParams(NamedTuple):
    """
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
    costh: float
    phi: float
    cosi: float
    psi: float


class SourceParams(NamedTuple):
    """
    Store both the intrinsic and extrinsic parameters for a source.

    Parameters
    ----------
    intrinsic: NamedTuple
        The intrinsic parameters for the particular source class, e.g. LinearFrequencyIntrinsicParams
    extrinsic: LisaWaveformTools.source_params.ExtrinsicParams
        The extrinsic parameters for the source
    """
    intrinsic: NamedTuple
    extrinsic: ExtrinsicParams


def _load_extrinsic_from_packed_helper(params_packed: NDArray[np.floating]) -> ExtrinsicParams:
    """Load the tuple object from a packed array containing the minimum parameters required to reconstruct it."""
    costh = params_packed[0]
    phi = params_packed[1]
    cosi = params_packed[2]
    psi = params_packed[3]
    return ExtrinsicParams(costh, phi, cosi, psi)


def _packed_from_extrinsic_helper(params_extrinsic: ExtrinsicParams) -> NDArray[np.floating]:
    """Store the minimum parameters required to reconstruct the object to an array."""
    costh = params_extrinsic.costh
    phi = params_extrinsic.phi
    cosi = params_extrinsic.cosi
    psi = params_extrinsic.psi
    return np.array([costh, phi, cosi, psi])


def _validate_extrinsic_helper(params_extrinsic: ExtrinsicParams) -> bool:
    """Check if the given set of extrinsic parameters can be interpreted as valid"""
    if not -1. <= params_extrinsic.costh <= 1.:
        return False
    return -1.0 <= params_extrinsic.cosi <= 1.0


IntrinsicParamsType = TypeVar('IntrinsicParamsType', bound=NamedTuple)
ExtrinsicParamsType = TypeVar('ExtrinsicParamsType', bound=ExtrinsicParams)
SourceParamsType = TypeVar('SourceParamsType')

ParamsType = TypeVar('ParamsType', bound=NamedTuple)


class AbstractParamsManager(Generic[ParamsType], ABC):
    """
    Manage creation, translation, and handling of ExtrinsicParams objects.
    """
    def __init__(self, params: ParamsType) -> None:
        self._params: ParamsType = params

    @property
    @abstractmethod
    def n_packed(self) -> int:
        return self.params_packed.size

    @property
    def params(self) -> ParamsType:
        return self._params

    @params.setter
    def params(self, params_in: ParamsType) -> None:
        self._params = params_in

    @property
    @abstractmethod
    def params_packed(self) -> NDArray[np.floating]:
        pass

    @params_packed.setter
    @abstractmethod
    def params_packed(self, params_in: NDArray[np.floating]) -> None:
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'params', group_mode: int = 0) -> h5py.Group:
        """Store the object as a group in an hdf5 file"""
        if group_mode == 0:
            hf_params = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_params = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)
        hf_params.attrs['creator_name'] = self.__class__.__name__
        hf_params.attrs['n_packed'] = self.n_packed

        hf_params.create_dataset('packed', data=self.params_packed)

        hf_p = hf_params.create_group('params')
        for key in self.params._fields:
            hf_p.attrs[key] = getattr(self.params, key)

        return hf_params


class AbstractExtrinsicParamsManager(AbstractParamsManager[ExtrinsicParamsType], ABC):
    def __init__(self, params_load: ExtrinsicParamsType) -> None:
        super().__init__(params_load)


# TODO write default methods for AbstractIntrinsicParamsManager that reads names of packed parameter fields from a list, to make subclassing easier
class AbstractIntrinsicParamsManager(AbstractParamsManager[IntrinsicParamsType], ABC):
    """
    Manage creation, translation, and handling of IntrinsicParamsType objects.
    """
    def __init__(self, params_load: IntrinsicParamsType) -> None:
        super().__init__(params_load)


class SourceParamsManager(Generic[ExtrinsicParamsType, IntrinsicParamsType], AbstractParamsManager[SourceParams]):
    def __init__(self, intrinsic: AbstractIntrinsicParamsManager[IntrinsicParamsType], extrinsic: AbstractExtrinsicParamsManager[ExtrinsicParamsType]) -> None:
        self._intrinsic_manager: AbstractIntrinsicParamsManager[IntrinsicParamsType] = intrinsic
        self._extrinsic_manager: AbstractExtrinsicParamsManager[ExtrinsicParamsType] = extrinsic

        self._n_extrinsic: int = self._extrinsic_manager.n_packed
        self._n_intrinsic: int = self._intrinsic_manager.n_packed
        self._n_packed: int = self._n_extrinsic + self._n_intrinsic

        params_load = SourceParams(self._intrinsic_manager.params, self._extrinsic_manager.params)
        super().__init__(params_load)

    @property
    @override
    def n_packed(self) -> int:
        return self._n_packed

    @property
    @override
    def params_packed(self) -> NDArray[np.floating]:
        return np.hstack([self._extrinsic_manager.params_packed, self._intrinsic_manager.params_packed])

    @params_packed.setter
    @override
    def params_packed(self, params_in: NDArray[np.floating]) -> None:
        assert params_in.size == self.n_packed
        extrinsic_loc = params_in[:self._n_extrinsic]
        intrinsic_loc = params_in[self._n_extrinsic:self._n_extrinsic + self._n_intrinsic]
        self._extrinsic_manager.params_packed = extrinsic_loc
        self._intrinsic_manager.params_packed = intrinsic_loc
        self.params: SourceParams = SourceParams(self._intrinsic_manager.params, self._extrinsic_manager.params)

    @override
    def is_valid(self) -> bool:
        extrinsic_valid = self._extrinsic_manager.is_valid()
        intrinsic_valid = self._intrinsic_manager.is_valid()
        return extrinsic_valid and intrinsic_valid


class ExtrinsicParamsManager(AbstractExtrinsicParamsManager[ExtrinsicParams]):
    """
    Manage creation, translation, and handling of ExtrinsicParams objects.
    """
    def __init__(self, params: ExtrinsicParams) -> None:
        self._n_packed: int = N_EXTRINSIC_PACKED
        super().__init__(params)

    @property
    @override
    def n_packed(self) -> int:
        return self._n_packed

    @property
    @override
    def params_packed(self) -> NDArray[np.floating]:
        return _packed_from_extrinsic_helper(self._params)

    @params_packed.setter
    @override
    def params_packed(self, params_in: NDArray[np.floating]) -> None:
        assert params_in.size == self.n_packed
        self.params: ExtrinsicParams = _load_extrinsic_from_packed_helper(params_in)

    def is_valid(self) -> bool:
        return _validate_extrinsic_helper(self.params)
