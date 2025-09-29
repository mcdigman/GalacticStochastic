"""Objects that store the parametrization for binary sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, override

import numpy as np

if TYPE_CHECKING:
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

ParamsType = TypeVar('ParamsType')


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
    def __init__(self, intrinsic_manager: AbstractIntrinsicParamsManager[IntrinsicParamsType], extrinsic_manager: AbstractExtrinsicParamsManager[ExtrinsicParamsType]) -> None:
        self._intrinsic_manager: AbstractIntrinsicParamsManager[IntrinsicParamsType] = intrinsic_manager
        self._extrinsic_manager: AbstractExtrinsicParamsManager[ExtrinsicParamsType] = extrinsic_manager

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
        self.params = SourceParams(self._intrinsic_manager.params, self._extrinsic_manager.params)

    @override
    def is_valid(self) -> bool:
        extrinsic_valid = self._extrinsic_manager.is_valid()
        intrinsic_valid = self._intrinsic_manager.is_valid()
        return extrinsic_valid and intrinsic_valid


class ExtrinsicParamsManager(AbstractExtrinsicParamsManager[ExtrinsicParams]):
    """
    Manage creation, translation, and handling of ExtrinsicParams objects.
    """
    def __init__(self, params_packed: NDArray[np.floating]) -> None:
        assert len(params_packed.shape) == 1
        assert params_packed.size == N_EXTRINSIC_PACKED
        self._n_packed = N_EXTRINSIC_PACKED

        params_load = _load_extrinsic_from_packed_helper(params_packed)

        super().__init__(params_load)

    @property
    @override
    def n_packed(self):
        return self._n_packed

    @property
    @override
    def params_packed(self):
        return _packed_from_extrinsic_helper(self._params)

    @params_packed.setter
    @override
    def params_packed(self, params_in: NDArray[np.floating]) -> None:
        assert params_in.size == self.n_packed
        self.params = _load_extrinsic_from_packed_helper(params_in)

    def is_valid(self):
        return _validate_extrinsic_helper(self.params)
