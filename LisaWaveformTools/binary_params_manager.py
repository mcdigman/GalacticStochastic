"""Manage intrinsic parameters of black hole binaries."""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, override

import numpy as np
from numpy.testing import assert_allclose

from LisaWaveformTools.source_params import AbstractIntrinsicParamsManager

if TYPE_CHECKING:
    from numpy.typing import NDArray

M_SUN_SEC = 4.925491025543575903411922162094833998e-6  # Geometrized solar mass, s
M_SUN_KG = 1.988546954961461467461011951140572744e30  # Solar mass, kg
M_SUN_M = 1.476625061404649406193430731479084713e3  # Geometrized solar mass, m
CLIGHT_M_SEC = 2.99792458e8     # Speed of light in m/s
PC_M = 3.085677581491367278913937957796471611e16  # parsec m

N_BINARY_PACKED = 10


class BinaryIntrinsicParams(NamedTuple):
    """Black hole binary intrinsic parameters."""

    mass_total_detector_sec: float  # total mass in detector frame [s]
    mass_total_detector_kg: float  # total mass in detector frame [kg]
    mass_total_detector_solar: float  # total mass in detector frame [solar mass]
    mass_chirp_detector_sec: float  # chirp mass in detector frame [s]
    mass_chirp_detector_kg: float  # chirp mass in detector frame [kg]
    mass_chirp_detector_solar: float  # chirp mass in detector frame [solar mass]
    mass_1_detector_sec: float  # mass primary in detector frame [s]
    mass_1_detector_kg: float  # mass primary in detector frame [kg]
    mass_1_detector_solar: float  # mass primary in detector frame [solar mass]
    mass_2_detector_sec: float  # mass secondary in detector frame [s]
    mass_2_detector_kg: float  # mass secondary in detector frame [kg]
    mass_2_detector_solar: float  # mass secondary in detector frame [solar mass]
    symmetric_mass_ratio: float  # symmetric mass ratio m1*m2/(m1+m2)^2 [unitless]
    mass_ratio: float  # mass ratio m2/m1 [unitless]
    mass_delta: float  # m1-m2/(m1+m2) [unitless]
    ln_luminosity_distance_m: float  # log luminosity distance in m
    luminosity_distance_m: float  # luminosity distance in m [m]
    frequency_i_hz: float  # starting frequency [Hz]
    time_c_sec: float  # time to coalescence [s]
    phase_c: float  # phase at coalescence [rad]
    chi_s: float  # aligned spin
    chi_a: float  # antialigned spin
    chi_1z: float  # z component of spin 1 (in the direction of the orbital angular momentum)
    chi_2z: float  # z component of spin 2 (in the direction of the orbital angular momentum)
    chi_postnewtonian: float  # postnewtonian modification to effective spin
    chi_eff: float  # effective spin
    chi_postnewtonian_norm: float  # postnewtonian modification to effective spin, normalized
    chi_p: float  # precessing spin
    eccentricity_i: float  # initial eccentricity [unitless]


def _packed_from_intrinsic_binary_helper(params: BinaryIntrinsicParams) -> NDArray[np.floating]:
    return np.array([
        params.ln_luminosity_distance_m,
        params.mass_total_detector_sec,
        params.mass_chirp_detector_sec,
        params.frequency_i_hz,
        params.time_c_sec,
        params.phase_c,
        params.chi_postnewtonian_norm,
        params.chi_a,
        params.chi_p,
        params.eccentricity_i,
    ])


def _load_intrinsic_binary_from_packed_helper(params_packed: NDArray[np.floating], mass_ratio_eps: float = 1.e-14) -> BinaryIntrinsicParams:
    assert len(params_packed.shape) == 1
    assert params_packed.size == N_BINARY_PACKED
    ln_luminosity_distance_m = float(params_packed[0])
    mass_total_detector_sec = float(params_packed[1])
    mass_chirp_detector_sec = float(params_packed[2])
    frequency_i_hz = float(params_packed[3])
    time_c_sec = float(params_packed[4])
    phase_c = float(params_packed[5])
    chi_postnewtonian_norm = float(params_packed[6])
    chi_a = float(params_packed[7])
    chi_p = float(params_packed[8])
    eccentricity_i = float(params_packed[9])

    assert mass_total_detector_sec > 0.0, 'Total mass must be positive.'
    assert mass_chirp_detector_sec > 0.0, 'Chirp mass must be positive.'
    assert frequency_i_hz > 0.0, 'Initial frequency must be positive.'
    assert 0.0 <= eccentricity_i < 1.0, 'Eccentricity must be in [0, 1).'
    assert -1.0 <= chi_postnewtonian_norm <= 1.0, 'chi_postnewtonian out of range [-1, 1]'
    assert 0.0 <= chi_a <= 1.0, 'chi_a must be in [0, 1]'

    # Derived parameters
    symmetric_mass_ratio: float = float((mass_chirp_detector_sec / mass_total_detector_sec) ** (5.0 / 3.0))

    if symmetric_mass_ratio >= 0.25:
        if symmetric_mass_ratio > 0.25 + mass_ratio_eps:
            # avoid raising errors for slightly unphysical values like symmetric_mass_ratio = 0.25 + 1e-14 due to numerical precision
            msg = f'Unphysical value of symmetric_mass_ratio: {symmetric_mass_ratio}'
            raise ValueError(msg)
        # equal mass case; avoid issues due to square root in computation of mass_delta
        symmetric_mass_ratio = 0.25
        mass_delta: float = 0.0
        mass_ratio: float = 1.0
        mass_1_detector_sec: float = mass_total_detector_sec / 2.0
        mass_2_detector_sec: float = mass_1_detector_sec
    elif symmetric_mass_ratio > 0.0:
        mass_delta = float(np.sqrt(1 - 4 * symmetric_mass_ratio))
        mass_ratio = (1 - mass_delta) / (1 + mass_delta)
        mass_1_detector_sec = mass_total_detector_sec / (1.0 + mass_ratio)
        mass_2_detector_sec = mass_total_detector_sec - mass_1_detector_sec
        assert mass_1_detector_sec >= mass_2_detector_sec, 'm1 must be the larger mass.'
    else:
        msg = f'Unphysical value of symmetric_mass_ratio: {symmetric_mass_ratio}'
        raise ValueError(msg)

    luminosity_distance_m: float = float(np.exp(ln_luminosity_distance_m))

    pn_norm_loc: float = 1.0 - 76.0 / 113.0 * symmetric_mass_ratio
    chi_s: float = chi_postnewtonian_norm - chi_a * mass_delta / pn_norm_loc
    chi_postnewtonian: float = chi_postnewtonian_norm * pn_norm_loc
    chi_eff: float = chi_postnewtonian_norm - 76.0 / 113.0 * symmetric_mass_ratio * chi_a * mass_delta / pn_norm_loc
    chi_1z: float = chi_s + chi_a
    chi_2z: float = chi_s - chi_a
    assert_allclose(chi_postnewtonian, pn_norm_loc * chi_s + chi_a * mass_delta), 'Inconsistent chi_postnewtonian'
    assert -1.0 <= chi_1z <= 1.0, 'Spin 1 out of range [-1, 1]'
    assert -1.0 <= chi_2z <= 1.0, 'Spin 2 out of range [-1, 1]'
    assert -1.0 <= chi_s <= 1.0, 'chi_s out of range [-1, 1]'
    assert -1.0 <= chi_a <= 1.0, 'chi_a out of range [-1, 1]'

    # mass conversions
    mass_total_detector_solar = mass_total_detector_sec / M_SUN_SEC
    mass_chirp_detector_solar = mass_chirp_detector_sec / M_SUN_SEC
    mass_1_detector_solar = mass_1_detector_sec / M_SUN_SEC
    mass_2_detector_solar = mass_2_detector_sec / M_SUN_SEC

    mass_total_detector_kg = mass_total_detector_solar * M_SUN_KG
    mass_chirp_detector_kg = mass_chirp_detector_solar * M_SUN_KG
    mass_1_detector_kg = mass_1_detector_solar * M_SUN_KG
    mass_2_detector_kg = mass_2_detector_solar * M_SUN_KG

    # TODO these checks should be formalized as a test instaed
    assert_allclose(mass_2_detector_sec / mass_1_detector_sec, mass_ratio)
    assert_allclose(mass_2_detector_kg / mass_1_detector_kg, mass_ratio)
    assert_allclose(mass_2_detector_solar / mass_1_detector_solar, mass_ratio)
    assert_allclose(mass_2_detector_sec + mass_1_detector_sec, mass_total_detector_sec)
    assert_allclose(mass_2_detector_kg + mass_1_detector_kg, mass_total_detector_kg)
    assert_allclose(mass_2_detector_solar + mass_1_detector_solar, mass_total_detector_solar)
    assert_allclose((mass_1_detector_sec - mass_2_detector_sec) / mass_total_detector_sec, mass_delta)
    assert_allclose((mass_1_detector_kg - mass_2_detector_kg) / mass_total_detector_kg, mass_delta)
    assert_allclose((mass_1_detector_solar - mass_2_detector_solar) / mass_total_detector_solar, mass_delta)
    assert_allclose(float((mass_chirp_detector_sec / mass_total_detector_sec) ** (5.0 / 3.0)), symmetric_mass_ratio)
    assert_allclose(float((mass_chirp_detector_kg / mass_total_detector_kg) ** (5.0 / 3.0)), symmetric_mass_ratio)
    assert_allclose(float((mass_chirp_detector_solar / mass_total_detector_solar) ** (5.0 / 3.0)), symmetric_mass_ratio)
    assert_allclose((mass_1_detector_sec * mass_2_detector_sec) ** (3.0 / 5.0) / (mass_1_detector_sec + mass_2_detector_sec) ** (1. / 5.), mass_chirp_detector_sec)
    assert_allclose((mass_1_detector_kg * mass_2_detector_kg) ** (3.0 / 5.0) / (mass_1_detector_kg + mass_2_detector_kg) ** (1. / 5.), mass_chirp_detector_kg)
    assert_allclose((mass_1_detector_solar * mass_2_detector_solar) ** (3.0 / 5.0) / (mass_1_detector_solar + mass_2_detector_solar) ** (1. / 5.), mass_chirp_detector_solar)
    assert_allclose((mass_1_detector_sec * mass_2_detector_sec) / (mass_1_detector_sec + mass_2_detector_sec) ** 2, symmetric_mass_ratio)
    assert_allclose((mass_1_detector_kg * mass_2_detector_kg) / (mass_1_detector_kg + mass_2_detector_kg) ** 2, symmetric_mass_ratio)
    assert_allclose((mass_1_detector_solar * mass_2_detector_solar) / (mass_1_detector_solar + mass_2_detector_solar) ** 2, symmetric_mass_ratio)

    return BinaryIntrinsicParams(
        mass_total_detector_sec=mass_total_detector_sec,
        mass_total_detector_kg=mass_total_detector_kg,
        mass_total_detector_solar=mass_total_detector_solar,
        mass_chirp_detector_sec=mass_chirp_detector_sec,
        mass_chirp_detector_kg=mass_chirp_detector_kg,
        mass_chirp_detector_solar=mass_chirp_detector_solar,
        mass_1_detector_sec=mass_1_detector_sec,
        mass_1_detector_kg=mass_1_detector_kg,
        mass_1_detector_solar=mass_1_detector_solar,
        mass_2_detector_sec=mass_2_detector_sec,
        mass_2_detector_kg=mass_2_detector_kg,
        mass_2_detector_solar=mass_2_detector_solar,
        symmetric_mass_ratio=symmetric_mass_ratio,
        mass_ratio=mass_ratio,
        mass_delta=mass_delta,
        ln_luminosity_distance_m=ln_luminosity_distance_m,
        luminosity_distance_m=luminosity_distance_m,
        frequency_i_hz=frequency_i_hz,
        time_c_sec=time_c_sec,
        phase_c=phase_c,
        chi_s=chi_s,
        chi_a=chi_a,
        chi_1z=chi_1z,
        chi_2z=chi_2z,
        chi_postnewtonian=chi_postnewtonian,
        chi_eff=chi_eff,
        chi_postnewtonian_norm=chi_postnewtonian_norm,
        chi_p=chi_p,
        eccentricity_i=eccentricity_i,
    )


def _validate_intrinsic_binary_helper(params: BinaryIntrinsicParams) -> bool:
    """Check if the given set of intrinsic parameters can be interpreted as valid."""
    if not params.mass_total_detector_sec > 0.0:
        return False
    if not params.mass_chirp_detector_sec > 0.0:
        return False
    if not params.frequency_i_hz > 0.0:
        return False
    if not 0.0 < params.symmetric_mass_ratio <= 0.25:
        return False
    if not 0.0 < params.mass_ratio <= 1.0:
        return False
    if not 0.0 <= params.mass_delta < 1.0:
        return False
    if not params.luminosity_distance_m > 0.0:
        return False
    if not 0.0 <= params.eccentricity_i < 1.0:
        return False
    if not -1.0 <= params.chi_postnewtonian_norm <= 1.0:
        return False
    if not 0.0 <= params.chi_a <= 1.0:
        return False
    if not -1.0 <= params.chi_1z <= 1.0:
        return False
    if not -1.0 <= params.chi_2z <= 1.0:
        return False
    if not -1.0 <= params.chi_s <= 1.0:
        return False
    if not -1.0 <= params.chi_a <= 1.0:  # noqa: SIM103
        return False

    return True


class BinaryIntrinsicParamsManager(AbstractIntrinsicParamsManager[BinaryIntrinsicParams]):
    """Manage creation, translation, and handling of ExtrinsicParams objects."""

    def __init__(self, params_load_in: NDArray[np.floating] | BinaryIntrinsicParams) -> None:
        """Construct parameter from intrinsic parameters named tuple."""
        self._n_packed: int = N_BINARY_PACKED
        if isinstance(params_load_in, BinaryIntrinsicParams):
            params_load = params_load_in
        elif isinstance(params_load_in, np.ndarray):
            assert params_load_in.shape == (N_BINARY_PACKED,)
            params_load = _load_intrinsic_binary_from_packed_helper(params_load_in)
        else:
            msg = 'params_load must be of type BinaryIntrinsicParams or ndarray'
            raise TypeError(msg)
        super().__init__(params_load)

    @property
    @override
    def n_packed(self) -> int:
        """Get number of intrinsic parameters in packed representation."""
        return self._n_packed

    @property
    @override
    def params_packed(self) -> NDArray[np.floating]:
        """Get intrinsic parameters in packed representation."""
        return _packed_from_intrinsic_binary_helper(self._params)

    @params_packed.setter
    @override
    def params_packed(self, params_in: NDArray[np.floating]) -> None:
        """Set intrinsic parameters from packed representation."""
        assert params_in.size == self.n_packed
        self.params: BinaryIntrinsicParams = _load_intrinsic_binary_from_packed_helper(params_in)

    @override
    def is_valid(self) -> bool:
        """Check if the represented parameters are internally consistent."""
        return _validate_intrinsic_binary_helper(self.params)
