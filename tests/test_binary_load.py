"""test the IMRPhenomD module C 2021 Matthew Digman"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose
from PyIMRPhenomD.IMRPhenomD_internals import chiPN

from LisaWaveformTools.binary_params_manager import M_SUN_KG, M_SUN_SEC, PC_M, BinaryIntrinsicParams, BinaryIntrinsicParamsManager

if TYPE_CHECKING:
    from numpy.typing import NDArray


def setup_test_helper(m1_solar: float, m2_solar: float, chi1: float, chi2: float) -> BinaryIntrinsicParams:

    distance: float = 56.00578366287752 * 1.0e9 * PC_M
    m1_sec: float = m1_solar * M_SUN_SEC
    m2_sec: float = m2_solar * M_SUN_SEC
    Mt_sec: float = m1_sec + m2_sec
    Mt_solar: float = m1_solar + m2_solar

    tc: float = 2.496000e+07

    eta: float = m1_sec * m2_sec / Mt_sec**2
    delta: float = (m1_sec - m2_sec) / Mt_sec
    Mc_sec: float = eta**(3 / 5) * Mt_sec
    assert_allclose(eta, (Mc_sec / Mt_sec)**(5 / 3))

    chis: float = (chi1 + chi2) / 2
    chia: float = (chi1 - chi2) / 2
    phic: float = 2.848705 / 2
    FI: float = 3.4956509169372e-05
    chi: float = chiPN(eta, chis, chia)
    chi_postnewtonian_norm: float = chi / (1. - 76. / 113. * eta)
    q = m2_solar / m1_solar

    intrinsic_params_packed: NDArray[np.floating] = np.array([
        np.log(distance),  # Log luminosity distance in meters
        Mt_sec,  # Total mass in seconds
        Mc_sec,  # Chirp mass in seconds
        FI,              # Initial frequency in Hz
        tc,             # Coalescence time in seconds
        phic,               # Phase at coalescence
        chi_postnewtonian_norm,               # Normalized postnewtonian spin parameter
        chia,               # Antisymmetric component of aligned spin
        0.0,               # Precessing spin
        0.0,               # Initial eccentricity
    ])

    intrinsic_params_manager: BinaryIntrinsicParamsManager = BinaryIntrinsicParamsManager(intrinsic_params_packed)
    intrinsic: BinaryIntrinsicParams = intrinsic_params_manager.params

    # check masses are as input
    print(m1_solar, m2_solar, Mt_solar / 2.0, Mt_solar)
    print(intrinsic.mass_1_detector_solar, intrinsic.mass_2_detector_solar, intrinsic.mass_total_detector_solar / 2.0, intrinsic.mass_total_detector_solar)
    print(m1_sec, m2_sec, Mt_sec / 2.0, Mt_sec)
    print(m2_solar / m1_solar, m2_sec / m1_sec, intrinsic.mass_ratio, intrinsic.mass_2_detector_solar / intrinsic.mass_1_detector_solar)
    print(0.25 - eta, 0.25 - intrinsic.symmetric_mass_ratio, 0.25 - (Mc_sec / Mt_sec)**(5 / 3), 0.25 - np.cbrt(Mc_sec / Mt_sec)**5, 0.25 - np.cbrt((Mc_sec / Mt_sec)**5))
    print(eta, intrinsic.symmetric_mass_ratio, (Mc_sec / Mt_sec)**(5 / 3), np.cbrt(Mc_sec / Mt_sec)**5, np.cbrt((Mc_sec / Mt_sec)**5))
    print(Mc_sec / Mt_sec, eta**(3 / 5), intrinsic.symmetric_mass_ratio**(3. / 5))
    print(q, intrinsic.mass_ratio, intrinsic.mass_2_detector_sec / intrinsic.mass_1_detector_sec)
    print(eta - eta, intrinsic.symmetric_mass_ratio - eta, (Mc_sec / Mt_sec)**(5 / 3) - eta, np.cbrt(Mc_sec / Mt_sec)**5 - eta, np.cbrt((Mc_sec / Mt_sec)**5) - eta)
    print(eta - (intrinsic.mass_1_detector_sec * intrinsic.mass_2_detector_sec) / (intrinsic.mass_1_detector_sec + intrinsic.mass_2_detector_sec) ** 2, intrinsic.symmetric_mass_ratio - (intrinsic.mass_1_detector_sec * intrinsic.mass_2_detector_sec) / (intrinsic.mass_1_detector_sec + intrinsic.mass_2_detector_sec) ** 2)
    print(q - q, q - intrinsic.mass_ratio, q - intrinsic.mass_2_detector_sec / intrinsic.mass_1_detector_sec, q - (1 - intrinsic.mass_delta) / (1 + intrinsic.mass_delta))

    if delta > 0.:
        atoldelta = 1. / delta * 1.e-14
        atol2 = max(m1_solar, m2_solar) / delta * 1.e-14
        atolq = max(np.abs(8 * (6 * eta + delta - 2) / delta * 1.e-14), 1.e-14)
    elif delta < 0.:
        atoldelta = np.abs(1. / delta) * 1.e-14
        atol2 = np.abs(max(m1_solar, m2_solar) / delta) * 1.e-14
        atolq = max(np.abs(8 * (6 * eta + delta - 2) / delta * 1.e-14), 1.e-14)
    else:
        atoldelta = 1.e-100
        atol2 = 1.e-100
        atolq = 1.e-100

    if delta >= 0.:
        assert_allclose(intrinsic.mass_1_detector_solar, m1_solar, atol=atol2, rtol=2.e-14)
        assert_allclose(intrinsic.mass_2_detector_solar, m2_solar, atol=atol2, rtol=1.e-14)
        assert_allclose(delta, intrinsic.mass_delta, atol=atoldelta, rtol=1.e-14)
        assert_allclose(q, intrinsic.mass_ratio, atol=atolq, rtol=1.e-14)
        assert_allclose((1 - intrinsic.mass_delta) / (1 + intrinsic.mass_delta), q, atol=atolq, rtol=3.e-13)
    else:
        # everything flipped if delta is negative
        assert_allclose(intrinsic.mass_2_detector_solar, m1_solar, atol=atol2, rtol=2.e-14)
        assert_allclose(intrinsic.mass_1_detector_solar, m2_solar, atol=atol2, rtol=2.e-14)
        assert_allclose(-delta, intrinsic.mass_delta, atol=atoldelta, rtol=1.e-14)
        assert_allclose(1. / q, intrinsic.mass_ratio, atol=atolq, rtol=1.e-14)
        assert_allclose((1 - intrinsic.mass_delta) / (1 + intrinsic.mass_delta), 1. / q, atol=atolq, rtol=3.e-13)

    assert_allclose(intrinsic.mass_total_detector_solar, m1_solar + m2_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_total_detector_kg, (m1_solar + m2_solar) * M_SUN_KG, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_total_detector_sec, (m1_solar + m2_solar) * M_SUN_SEC, atol=1.e-100, rtol=1.e-14)

    # check relationships between masses match expectations
    assert_allclose(eta, intrinsic.symmetric_mass_ratio, atol=1.e-100, rtol=1.e-14)
    assert_allclose((1 - intrinsic.mass_delta) / (1 + intrinsic.mass_delta), intrinsic.mass_ratio, atol=1.e-14, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_2_detector_kg, intrinsic.mass_1_detector_sec / intrinsic.mass_1_detector_kg, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_2_detector_kg, intrinsic.mass_total_detector_sec / intrinsic.mass_total_detector_kg, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_2_detector_kg, intrinsic.mass_chirp_detector_sec / intrinsic.mass_chirp_detector_kg, atol=1.e-100, rtol=1.e-14)

    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_2_detector_solar, intrinsic.mass_1_detector_sec / intrinsic.mass_1_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_2_detector_solar, intrinsic.mass_total_detector_sec / intrinsic.mass_total_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_2_detector_solar, intrinsic.mass_chirp_detector_sec / intrinsic.mass_chirp_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_kg / intrinsic.mass_2_detector_solar, intrinsic.mass_1_detector_kg / intrinsic.mass_1_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_kg / intrinsic.mass_2_detector_solar, intrinsic.mass_total_detector_kg / intrinsic.mass_total_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_kg / intrinsic.mass_2_detector_solar, intrinsic.mass_chirp_detector_kg / intrinsic.mass_chirp_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_1_detector_sec, intrinsic.mass_ratio, atol=1.e-100, rtol=2.e-12)
    assert_allclose(intrinsic.mass_2_detector_kg / intrinsic.mass_1_detector_kg, intrinsic.mass_ratio, atol=1.e-100, rtol=2.e-12)
    assert_allclose(intrinsic.mass_2_detector_solar / intrinsic.mass_1_detector_solar, intrinsic.mass_ratio, atol=1.e-100, rtol=2.e-12)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_1_detector_sec, intrinsic.mass_2_detector_kg / intrinsic.mass_1_detector_kg, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec / intrinsic.mass_1_detector_sec, intrinsic.mass_2_detector_solar / intrinsic.mass_1_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_sec + intrinsic.mass_1_detector_sec, intrinsic.mass_total_detector_sec, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_kg + intrinsic.mass_1_detector_kg, intrinsic.mass_total_detector_kg, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.mass_2_detector_solar + intrinsic.mass_1_detector_solar, intrinsic.mass_total_detector_solar, atol=1.e-100, rtol=1.e-14)
    assert_allclose((intrinsic.mass_1_detector_sec - intrinsic.mass_2_detector_sec) / intrinsic.mass_total_detector_sec, intrinsic.mass_delta, atol=atoldelta, rtol=1.e-14)
    assert_allclose((intrinsic.mass_1_detector_kg - intrinsic.mass_2_detector_kg) / intrinsic.mass_total_detector_kg, intrinsic.mass_delta, atol=atoldelta, rtol=1.e-14)
    assert_allclose((intrinsic.mass_1_detector_solar - intrinsic.mass_2_detector_solar) / intrinsic.mass_total_detector_solar, intrinsic.mass_delta, atol=atoldelta, rtol=1.e-14)
    assert_allclose(float((intrinsic.mass_chirp_detector_sec / intrinsic.mass_total_detector_sec) ** (5.0 / 3.0)), intrinsic.symmetric_mass_ratio, atol=1.e-100, rtol=1.e-14)
    assert_allclose(float((intrinsic.mass_chirp_detector_kg / intrinsic.mass_total_detector_kg) ** (5.0 / 3.0)), intrinsic.symmetric_mass_ratio, atol=1.e-100, rtol=1.e-14)
    assert_allclose(float((intrinsic.mass_chirp_detector_solar / intrinsic.mass_total_detector_solar) ** (5.0 / 3.0)), intrinsic.symmetric_mass_ratio, atol=1.e-100, rtol=1.e-14)
    assert_allclose((intrinsic.mass_1_detector_sec * intrinsic.mass_2_detector_sec) ** (3.0 / 5.0) / (intrinsic.mass_1_detector_sec + intrinsic.mass_2_detector_sec) ** (1. / 5.), intrinsic.mass_chirp_detector_sec, atol=1.e-100, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_kg * intrinsic.mass_2_detector_kg) ** (3.0 / 5.0) / (intrinsic.mass_1_detector_kg + intrinsic.mass_2_detector_kg) ** (1. / 5.), intrinsic.mass_chirp_detector_kg, atol=1.e-100, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_solar * intrinsic.mass_2_detector_solar) ** (3.0 / 5.0) / (intrinsic.mass_1_detector_solar + intrinsic.mass_2_detector_solar) ** (1. / 5.), intrinsic.mass_chirp_detector_solar, atol=1.e-100, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_sec * intrinsic.mass_2_detector_sec) / (intrinsic.mass_1_detector_sec + intrinsic.mass_2_detector_sec) ** 2, intrinsic.symmetric_mass_ratio, atol=1.e-14, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_kg * intrinsic.mass_2_detector_kg) / (intrinsic.mass_1_detector_kg + intrinsic.mass_2_detector_kg) ** 2, intrinsic.symmetric_mass_ratio, atol=1.e-14, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_solar * intrinsic.mass_2_detector_solar) / (intrinsic.mass_1_detector_solar + intrinsic.mass_2_detector_solar) ** 2, intrinsic.symmetric_mass_ratio, atol=1.e-14, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_sec * intrinsic.mass_2_detector_sec) / (intrinsic.mass_1_detector_sec + intrinsic.mass_2_detector_sec) ** 2, eta, atol=1.e-14, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_kg * intrinsic.mass_2_detector_kg) / (intrinsic.mass_1_detector_kg + intrinsic.mass_2_detector_kg) ** 2, eta, atol=1.e-14, rtol=3.e-11)
    assert_allclose((intrinsic.mass_1_detector_solar * intrinsic.mass_2_detector_solar) / (intrinsic.mass_1_detector_solar + intrinsic.mass_2_detector_solar) ** 2, eta, atol=1.e-14, rtol=3.e-11)

    # check spins
    pn_norm_loc: float = 1.0 - 76.0 / 113.0 * intrinsic.symmetric_mass_ratio
    assert_allclose(intrinsic.chi_postnewtonian, pn_norm_loc * intrinsic.chi_s + intrinsic.chi_a * intrinsic.mass_delta, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.chi_s, intrinsic.chi_postnewtonian_norm - intrinsic.chi_a * intrinsic.mass_delta / pn_norm_loc, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.chi_eff, intrinsic.chi_postnewtonian_norm - 76 / 113 * intrinsic.chi_a * intrinsic.mass_delta * intrinsic.symmetric_mass_ratio / pn_norm_loc, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.chi_postnewtonian, pn_norm_loc * intrinsic.chi_postnewtonian_norm, atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.chi_s, (intrinsic.chi_1z + intrinsic.chi_2z) / 2., atol=1.e-100, rtol=1.e-14)
    assert_allclose(intrinsic.chi_a, (intrinsic.chi_1z - intrinsic.chi_2z) / 2., atol=1.e-100, rtol=1.e-14)

    return intrinsic


@pytest.mark.parametrize('m1_solar', [1., 1. + 1.e-10, 1.00001, 2., 100., 199.99, 199.99999, 199.999999, 199.999999999, 200., 200.01, 1242860.685, 2599137.035])
@pytest.mark.parametrize('m2_solar', [1., 1. + 1.e-10, 1.00001, 2., 100., 199.99, 199.99999, 199.999999, 199.999999999, 200., 200.01, 1242860.685, 2599137.035])
@pytest.mark.parametrize('chi1', [0.7534821857057837, 0.6215875279643664])
@pytest.mark.parametrize('chi2', [0.7534821857057837, 0.6215875279643664])
def test_load_consistency(m1_solar: float, m2_solar: float, chi1: float, chi2: float) -> None:
    """Various checks for internal consistency of binary parameter loading"""
    setup_test_helper(m1_solar, m2_solar, chi1, chi2)
