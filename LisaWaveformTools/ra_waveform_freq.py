"""Functions to compute rigid adiabatic detector response in the frequency domain."""

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from LisaWaveformTools.algebra_tools import stabilized_gradient_uniform_inplace
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.ra_waveform_time import (
    amp_phase_loop_helper,
    apply_edge_rise_helper,
    spacecraft_channel_deriv_helper,
)
from LisaWaveformTools.source_params import ExtrinsicParams
from LisaWaveformTools.spacecraft_objects import (
    AntennaResponseChannels,
    ComplexTransferFunction,
    DetectorAmplitudePhaseCombinations,
    DetectorPolarizationResponse,
    EdgeRiseModel,
    SpacecraftOrbits,
    SpacecraftRelativePhases,
    SpacecraftScalarPosition,
    SpacecraftSeparationVectors,
    SpacecraftSeparationWaveProjection,
    TDIComplexAntennaPattern,
    TensorBasis,
)
from LisaWaveformTools.stationary_source_waveform import (
    StationaryWaveformFreq,
    StationaryWaveformGeneric,
)
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange


@njit()
def get_tensor_basis(params_extrinsic: ExtrinsicParams) -> TensorBasis:
    """
    Compute the gravitational-wave tensor basis vectors for LISA observations.

    This function calculates:
    1. The wave propagation direction unit vector (kv)
    2. The plus polarization tensor basis (e_plus)
    3. The cross polarization tensor basis (e_cross)

    Parameters
    ----------
    params_extrinsic : ExtrinsicParams
        An ExtrinsicParams object containing the extrinsic parameters:
        - costh: Cosine of the source's ecliptic colatitude
        - phi: Source's ecliptic longitude in radians

    Returns
    -------
    TensorBasis
        An ExtrinsicParams object containing:
        - kv: ndarray of shape (3,)
            Unit vector pointing from the source toward LISA
        - e_plus: ndarray of shape (3, 3)
            Plus polarization tensor basis
        - e_cross: ndarray of shape (3, 3)
            Cross polarization tensor basis

    Raises
    ------
    ValueError
        If the provided costh value is not in the range [-1, 1].
    """
    # Calculate cos and sin of sky position, inclination, polarization
    n_space = 3  # number of spatial dimensions (must be 3)

    costh: float = params_extrinsic.costh
    phi: float = params_extrinsic.phi

    # avoid sqrt becoming complex
    if np.abs(costh) > 1.0:
        msg = 'Invalid value for cos theta: must be in [-1, 1]'
        raise ValueError(msg)

    sinth: float = float(np.sqrt(1.0 - costh**2))
    cosph: float = np.cos(phi)
    sinph: float = np.sin(phi)

    kv: NDArray[np.floating] = np.zeros(n_space)
    u: NDArray[np.floating] = np.zeros(n_space)
    v: NDArray[np.floating] = np.zeros(n_space)

    kv[0] = -sinth * cosph
    kv[1] = -sinth * sinph
    kv[2] = -costh

    u[0] = sinph
    u[1] = -cosph
    u[2] = 0.0

    v[0] = -costh * cosph
    v[1] = -costh * sinph
    v[2] = sinth

    e_plus: NDArray[np.floating] = np.zeros((n_space, n_space))
    e_cross: NDArray[np.floating] = np.zeros((n_space, n_space))

    for i in range(n_space):
        for j in range(n_space):
            e_plus[i, j] = u[i] * u[j] - v[i] * v[j]
            e_cross[i, j] = u[i] * v[j] + v[i] * u[j]
    return TensorBasis(kv, e_plus, e_cross)


@njit()
def get_oribtal_phase_constants(lc: LISAConstants, n_spacecraft: int) -> SpacecraftRelativePhases:
    """
    Calculate the initial orbital phase offsets and their sine and cosine values for each spacecraft.

    This function determines the relative phase angles of each spacecraft's orbit,
    according to the geometric configuration of the LISA constellation. These phase offsets are needed to compute the
    spacecraft positions,the light travel time, and model the detector's response to gravitational waves.
    The calculation assumes the constellation is an equilateral triangle with the specified initial orientation.

    Parameters
    ----------
    lc: LISAConstants
        An object containing the LISA configuration constants
    n_spacecraft: int
        The number of spacecraft in the constellation.

    Returns
    -------
    SpacecraftRelativePhases:
        A structure containing:
        - sin_beta: ndarray of shape (n_spacecraft,)
            The sine of each spacecraft's initial orbital phase offset times orbital radius and eccentricity.
        - cos_beta: ndarray of shape (n_spacecraft,)
            The cosine of each spacecraft's initial orbital phase offset times orbital rasius and eccentricity.
        - betas: ndarray of shape (n_spacecraft,)
            The initial orbital phase offsets (in radians) for the spacecraft.
    """
    # quantities for computing spacecraft positions
    betas: NDArray[np.floating] = np.zeros(n_spacecraft)
    sin_beta: NDArray[np.floating] = np.zeros(n_spacecraft)
    cos_beta: NDArray[np.floating] = np.zeros(n_spacecraft)
    for itrc in range(n_spacecraft):
        betas[itrc] = float(2.0 / 3.0 * np.pi) * itrc + lc.lambda0
        sin_beta[itrc] = (lc.r_orbit * lc.ec) * np.sin(betas[itrc])
        cos_beta[itrc] = (lc.r_orbit * lc.ec) * np.cos(betas[itrc])
    return SpacecraftRelativePhases(sin_beta, cos_beta, betas)


@njit()
def get_detector_amplitude_phase_combinations(params_extrinsic: ExtrinsicParams) -> DetectorAmplitudePhaseCombinations:
    """
    Compute the amplitude and phase factors required for the detector's response to a gravitational-wave signal.

    This function uses the the binary inclination and polarization angles
    to calculate the coefficients that scale the plus and cross polarizations as seen by the detector.
    It also computes the cosine and sine of twice the polarization angle, as needed for projecting the signal onto
    the detector's antenna response.

    Parameters
    ----------
    params_extrinsic: ExtrinsicParams
        Structure containing extrinsic source parameters, including
        - cosi: cosine of the inclination angle of the binary's orbital plane,
        - psi: polarization angle of the gravitational-wave signal (in radians).

    Returns
    -------
    DetectorAmplitudePhaseCombinations:
        Structure containing the following fiels:
        - A_plus: float
            Amplitude coefficient for the plus polarization.
        - A_cross: float
            Amplitude coefficient for the cross polarization.
        - cos_psi: float
            Cosine of twice the polarization angle.
        - sin_psi: float
            Sine of twice the polarization angle.
    """
    cosi: float = params_extrinsic.cosi
    psi: float = params_extrinsic.psi

    # polarization angle quantities
    cos_psi: float = np.cos(2 * psi)
    sin_psi: float = np.sin(2 * psi)

    # amplitude multipliers
    A_plus: float = (1.0 + cosi**2) / 2
    A_cross: float = -cosi

    return DetectorAmplitudePhaseCombinations(A_plus, A_cross, cos_psi, sin_psi)


@njit()
def get_aet_combinations(tdi_xyz: TDIComplexAntennaPattern, tdi_aet: TDIComplexAntennaPattern) -> None:
    """
    Convert the Michelson TDI channel responses (X, Y, Z) into the noise orthogonal A, E, and T combinations.

    This function transforms the complex antenna pattern components for the standard TDI Michelson channels into the
    noise-orthogonal AET basis. The A, E, and T channels are constructed as specific linear combinations of X, Y, and Z,
    designed to diagonalize the noise covariance matrix.

    Parameters
    ----------
    tdi_xyz: TDIComplexAntennaPattern
        Contains the complex antenna pattern responses for the X, Y, and Z Michelson TDI channels.
    tdi_aet: TDIComplexAntennaPattern
        Output structure to be populated in place with the corresponding A, E, and T channel responses.
    """
    FpR_aet: NDArray[np.floating] = tdi_aet.FpR
    FcR_aet: NDArray[np.floating] = tdi_aet.FcR
    FpI_aet: NDArray[np.floating] = tdi_aet.FpI
    FcI_aet: NDArray[np.floating] = tdi_aet.FcI

    FpR_xyz: NDArray[np.floating] = tdi_xyz.FpR
    FcR_xyz: NDArray[np.floating] = tdi_xyz.FcR
    FpI_xyz: NDArray[np.floating] = tdi_xyz.FpI
    FcI_xyz: NDArray[np.floating] = tdi_xyz.FcI

    nc_michelson: int = tdi_xyz.FpR.shape[0]
    assert nc_michelson == 3
    nc_generate: int = 3

    assert FpR_xyz.shape == (nc_michelson,)
    assert FcR_xyz.shape == (nc_michelson,)
    assert FpI_xyz.shape == (nc_michelson,)
    assert FcI_xyz.shape == (nc_michelson,)

    assert FpR_aet.shape == (nc_generate,)
    assert FcR_aet.shape == (nc_generate,)
    assert FpI_aet.shape == (nc_generate,)
    assert FcI_aet.shape == (nc_generate,)

    FpR_aet[0] = (2 * FpR_xyz[0] - FpR_xyz[1] - FpR_xyz[2]) / 3.0
    FcR_aet[0] = (2 * FcR_xyz[0] - FcR_xyz[1] - FcR_xyz[2]) / 3.0

    FpR_aet[1] = (FpR_xyz[2] - FpR_xyz[1]) / float(np.sqrt(3.0))
    FcR_aet[1] = (FcR_xyz[2] - FcR_xyz[1]) / float(np.sqrt(3.0))

    FpR_aet[2] = (FpR_xyz[0] + FpR_xyz[1] + FpR_xyz[2]) / 3.0
    FcR_aet[2] = (FcR_xyz[0] + FcR_xyz[1] + FcR_xyz[2]) / 3.0

    FpI_aet[0] = (2 * FpI_xyz[0] - FpI_xyz[1] - FpI_xyz[2]) / 3.0
    FcI_aet[0] = (2 * FcI_xyz[0] - FcI_xyz[1] - FcI_xyz[2]) / 3.0

    FpI_aet[1] = (FpI_xyz[2] - FpI_xyz[1]) / float(np.sqrt(3.0))
    FcI_aet[1] = (FcI_xyz[2] - FcI_xyz[1]) / float(np.sqrt(3.0))

    FpI_aet[2] = (FpI_xyz[0] + FpI_xyz[1] + FpI_xyz[2]) / 3.0
    FcI_aet[2] = (FcI_xyz[0] + FcI_xyz[1] + FcI_xyz[2]) / 3.0


@njit()
def get_michelson_combinations(
    polarization_response: DetectorPolarizationResponse,
    transfer_function: ComplexTransferFunction,
    A_psi: DetectorAmplitudePhaseCombinations,
    tdi_xyz: TDIComplexAntennaPattern,
) -> None:
    """
    Construct the standard Michelson Time-Delay Interferometry (TDI) X, Y, and Z channel response combinations.

    Computes the frequency-domain antenna pattern for each Michelson interferometer channel (X, Y, Z).
    It projects the polarization responses, applies arm transfer functions, and forms the tid components.

    The outputs are written in place into the provided TDIComplexAntennaPattern structure, populating the complex
    parts of the antenna pattern for different polarizations.

    Parameters
    ----------
    polarization_response: DetectorPolarizationResponse
        Contains the detector tensor projection onto plus and cross polarizations for each arm combination.
    transfer_function: ComplexTransferFunction
        Provides the complex transfer functions for each arm, representing the propagation and time-delay effects.
    A_psi: DetectorAmplitudePhaseCombinations
        Holds the intrinsic_waveform amplitude for each polarization and the polarization angle.
    tdi_xyz: TDIComplexAntennaPattern
        Output object to be filled with the resulting Michelson TDI channel antenna pattern components.
    """
    A_plus: float = A_psi.A_plus
    A_cross: float = A_psi.A_cross
    cos_psi: float = A_psi.cos_psi
    sin_psi: float = A_psi.sin_psi

    nc_michelson: int = tdi_xyz.FpR.shape[0]
    TR: NDArray[np.floating] = transfer_function.TR
    TI: NDArray[np.floating] = transfer_function.TI

    d_plus: NDArray[np.floating] = polarization_response.d_plus
    d_cross: NDArray[np.floating] = polarization_response.d_cross

    n_arm: int = d_plus.shape[0]

    assert d_plus.shape == (n_arm, n_arm)
    assert d_cross.shape == (n_arm, n_arm)
    assert TR.shape == (n_arm, n_arm)
    assert TI.shape == (n_arm, n_arm)
    assert len(TR.shape) == 2
    assert len(tdi_xyz.FpR.shape) == 1

    assert nc_michelson <= n_arm
    assert tdi_xyz.FpR.shape == (nc_michelson,)
    assert tdi_xyz.FcR.shape == (nc_michelson,)
    assert tdi_xyz.FpI.shape == (nc_michelson,)
    assert tdi_xyz.FcI.shape == (nc_michelson,)

    for i in range(nc_michelson):
        tdi_xyz.FpR[i] = (
            -(
                +(d_plus[i, (i + 1) % n_arm] * cos_psi + d_cross[i, (i + 1) % n_arm] * sin_psi) * TR[i, (i + 1) % n_arm]
                - (d_plus[i, (i + 2) % n_arm] * cos_psi + d_cross[i, (i + 2) % n_arm] * sin_psi)
                * TR[i, (i + 2) % n_arm]
            )
            * A_plus
            / 2
        )
        tdi_xyz.FcR[i] = (
            -(
                +(-d_plus[i, (i + 1) % n_arm] * sin_psi + d_cross[i, (i + 1) % n_arm] * cos_psi)
                * TR[i, (i + 1) % n_arm]
                - (-d_plus[i, (i + 2) % n_arm] * sin_psi + d_cross[i, (i + 2) % n_arm] * cos_psi)
                * TR[i, (i + 2) % n_arm]
            )
            * A_cross
            / 2
        )
        tdi_xyz.FpI[i] = (
            -(
                +(d_plus[i, (i + 1) % n_arm] * cos_psi + d_cross[i, (i + 1) % n_arm] * sin_psi) * TI[i, (i + 1) % n_arm]
                - (d_plus[i, (i + 2) % n_arm] * cos_psi + d_cross[i, (i + 2) % n_arm] * sin_psi)
                * TI[i, (i + 2) % n_arm]
            )
            * A_plus
            / 2
        )
        tdi_xyz.FcI[i] = (
            -(
                +(-d_plus[i, (i + 1) % n_arm] * sin_psi + d_cross[i, (i + 1) % n_arm] * cos_psi)
                * TI[i, (i + 1) % n_arm]
                - (-d_plus[i, (i + 2) % n_arm] * sin_psi + d_cross[i, (i + 2) % n_arm] * cos_psi)
                * TI[i, (i + 2) % n_arm]
            )
            * A_cross
            / 2
        )


@njit()
def get_projected_detector_response(
    tb: TensorBasis,
    sc_sep: SpacecraftSeparationVectors,
    polarization_response: DetectorPolarizationResponse,
) -> None:
    """
    Compute the projected detector response of the LISA constellation to a gravitational wave.

    This function evaluates the response of the detector arms to an incoming gravitational wave,
    by projecting the polarization basis tensors onto the direction of the separation vectors between spacecraft.
    The result is the sensitivity of each detector link to the polarizations (plus and cross),
    as determined by the geometry of the constellation and the incident wave direction.

    Parameters
    ----------
    tb: TensorBasis
        Contains the wave propagation unit vector and polarization basis tensors.
    sc_sep: SpacecraftSeparationVectors
        Contains the projection (dot products) of the wave vector onto the relevant separation vectors,
        as computed by `get_separation_wave_projection`.
    polarization_response: DetectorPolarizationResponse
        Output object to be populated with the detector response for each link and polarization. The
        arrays within `polarization_response` are updated in place.
    """
    d_plus: NDArray[np.floating] = polarization_response.d_plus
    d_cross: NDArray[np.floating] = polarization_response.d_cross

    e_plus: NDArray[np.floating] = tb.e_plus
    e_cross: NDArray[np.floating] = tb.e_cross

    r12: NDArray[np.floating] = sc_sep.r12
    r13: NDArray[np.floating] = sc_sep.r13
    r23: NDArray[np.floating] = sc_sep.r23

    n_space: int = r12.shape[0]

    n_arm = 3

    assert d_plus.shape == (n_arm, n_arm)
    assert d_cross.shape == (n_arm, n_arm)

    assert e_plus.shape == (n_space, n_space)
    assert e_cross.shape == (n_space, n_space)

    assert r12.shape == (n_space,)
    assert r13.shape == (n_space,)
    assert r23.shape == (n_space,)

    d_plus[:] = 0.0
    d_cross[:] = 0.0
    # Convenient quantities d+ & dx
    for i in range(n_space):
        for j in range(n_space):
            d_plus[0, 1] += r12[i] * r12[j] * e_plus[i, j]
            d_cross[0, 1] += r12[i] * r12[j] * e_cross[i, j]
            d_plus[1, 2] += r23[i] * r23[j] * e_plus[i, j]
            d_cross[1, 2] += r23[i] * r23[j] * e_cross[i, j]
            d_plus[0, 2] += r13[i] * r13[j] * e_plus[i, j]
            d_cross[0, 2] += r13[i] * r13[j] * e_cross[i, j]

    d_plus[1, 0] = d_plus[0, 1]
    d_cross[1, 0] = d_cross[0, 1]
    d_plus[2, 1] = d_plus[1, 2]
    d_cross[2, 1] = d_cross[1, 2]
    d_plus[2, 0] = d_plus[0, 2]
    d_cross[2, 0] = d_cross[0, 2]


@njit()
def get_separation_wave_projection(
    tb: TensorBasis,
    sc_sep: SpacecraftSeparationVectors,
    sc_wave_proj: SpacecraftSeparationWaveProjection,
) -> None:
    """
    Project the gravitational-wave propagation vector onto the separation vectors.

    This function calculates:
        - The dot products between the wave vector and the separation vectors connecting each
          spacecraft pair (stored in `sc_wave_proj.k_sc_sc_sep`).
        - The dot products between the wave vector and the vectors connecting each spacecraft to
          the guiding center (stored in `sc_wave_proj.k_sc_gc_sep`).
    All operations are performed in place modifying sc_wave_proj.
    kc_sc_sc_sep has indices [i, j] for each spacecraft pair,
    and kc_sc_gc_sep ahs indices [i] for each spacecraft-guiding center vector pair.

    Parameters
    ----------
    tb: TensorBasis
        Tensor basis structure containing the wave propagation direction vector (`kv`)
        and polarization tensors.
    sc_sep: SpacecraftSeparationVectors
        Structure holding the three-dimensional separation vectors between spacecraft pairs
        and between each spacecraft and the guiding center.
    sc_wave_proj: SpacecraftSeparationWaveProjection
        Output structure to be populated in place with the calculated dot products
        for spacecraft pair separations and guiding center separations.
    """
    k_sc_sc_sep: NDArray[np.floating] = sc_wave_proj.k_sc_sc_sep
    k_sc_gc_sep: NDArray[np.floating] = sc_wave_proj.k_sc_gc_sep

    kv: NDArray[np.floating] = tb.kv

    r12: NDArray[np.floating] = sc_sep.r12
    r13: NDArray[np.floating] = sc_sep.r13
    r23: NDArray[np.floating] = sc_sep.r23
    r10: NDArray[np.floating] = sc_sep.r10
    r20: NDArray[np.floating] = sc_sep.r20
    r30: NDArray[np.floating] = sc_sep.r30

    # number of spatial dimensions
    n_space = kv.shape[0]
    # number of spacecraft
    n_spacecraft = k_sc_gc_sep.shape[0]

    assert n_spacecraft == 3, 'Only implemented for 3 spacecraft'

    assert k_sc_sc_sep.shape == (n_spacecraft, n_spacecraft)
    assert k_sc_gc_sep.shape == (n_spacecraft,)

    assert kv.shape == (n_space,)
    assert r12.shape == (n_space,)
    assert r13.shape == (n_space,)
    assert r23.shape == (n_space,)
    assert r10.shape == (n_space,)
    assert r20.shape == (n_space,)
    assert r30.shape == (n_space,)

    k_sc_sc_sep[:] = 0.0
    k_sc_gc_sep[:] = 0.0
    for k in range(n_space):
        k_sc_sc_sep[0, 1] += kv[k] * r12[k]
        k_sc_sc_sep[0, 2] += kv[k] * r13[k]
        k_sc_sc_sep[1, 2] += kv[k] * r23[k]
        k_sc_gc_sep[0] += kv[k] * r10[k]
        k_sc_gc_sep[1] += kv[k] * r20[k]
        k_sc_gc_sep[2] += kv[k] * r30[k]


@njit()
def get_transfer_function(
    fr: float,
    sc_wave_proj: SpacecraftSeparationWaveProjection,
    transfer_function: ComplexTransferFunction,
) -> None:
    """
    Compute the complex transfer function for the LISA TDI channels.

    This function calculates the frequency response of the LISA constellation arms to a passing
    gravitational wave, taking into account the time-varying separations between spacecraft and
    projections of wave vectors relevant to the current geometry. The result is stored in the
    `transfer_function` argument.
    The mathematical expressions are described in [Phys. Rev. D 101, 124008 (2020), Eq. (B9)].

    Parameters
    ----------
    fr: float
        Dimensionless frequency, typically `f / f_star`, where f is the frequency
        and f_star is the LISA transfer frequency.
    sc_wave_proj: SpacecraftSeparationWaveProjection
        Precomputed inner products of the wave propagation direction with spacecraft separation vectors.
    transfer_function: ComplexTransferFunction
        Object to store the transfer function matrix for each arm pair.
    """
    k_sc_sc_sep: NDArray[np.floating] = sc_wave_proj.k_sc_sc_sep
    k_sc_gc_sep: NDArray[np.floating] = sc_wave_proj.k_sc_gc_sep

    n_spacecraft: int = k_sc_gc_sep.shape[0]

    assert n_spacecraft == 3, 'Only implemented for 3 spacecraft'
    assert k_sc_sc_sep.shape == (n_spacecraft, n_spacecraft)
    assert k_sc_gc_sep.shape == (n_spacecraft,)

    TR: NDArray[np.floating] = transfer_function.TR
    TI: NDArray[np.floating] = transfer_function.TI
    n_arm = TR.shape[0]

    assert n_arm == n_spacecraft, 'Only implemented for 3 spacecraft'
    assert TR.shape == (n_arm, n_arm)
    assert TI.shape == (n_arm, n_arm)
    assert fr > 0.0, 'Frequency must be positive'

    for i in range(n_arm - 1):
        for j in range(i + 1, n_arm):
            q1: float = fr * (1.0 - k_sc_sc_sep[i, j])
            q2: float = fr * (1.0 + k_sc_sc_sep[i, j])
            q3: float = -fr * (3.0 + k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[i])
            q4: float = -fr * (1.0 + k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[i])  # TODO missing 1/sqrt(3) on k_sc_gc_sep?
            q5: float = -fr * (3.0 - k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[j])
            q6: float = -fr * (1.0 - k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[j])  # TODO missing 1/sqrt(3) on k_sc_gc_sep?
            # sinc functions, prevent division by zero
            if q1 == 0.0:
                sincq1: float = 0.5
            else:
                sincq1 = np.sin(q1) / q1 / 2
            if q2 == 0.0:
                sincq2: float = 0.5
            else:
                sincq2 = np.sin(q2) / q2 / 2
            # Real part of T from eq B9 in PhysRevD.101.124008
            TR[i, j] = sincq1 * np.cos(q3) + sincq2 * np.cos(q4)  # goes to 1 when f/fstar small
            # imaginary part of T
            TI[i, j] = sincq1 * np.sin(q3) + sincq2 * np.sin(q4)  # goes to 0 when f/fstar small
            # save ops computing other triangle simultaneously
            TR[j, i] = sincq2 * np.cos(q5) + sincq1 * np.cos(q6)  # goes to 1 when f/fstar small
            TI[j, i] = sincq2 * np.sin(q5) + sincq1 * np.sin(q6)  # goes to 0 when f/fstar small


@njit()
def compute_separation_vectors(
    lc: LISAConstants,
    tb: TensorBasis,
    sc_pos: SpacecraftScalarPosition,
    sc_sep: SpacecraftSeparationVectors,
) -> float:
    """
    Compute the separation vectors between LISA spacecraft and between each spacecraft and the guiding center.

    Return the dot product of the gravitational-wave propagation direction with the guiding center position.

    This function calculates:
      - The three-dimensional separation vectors between all pairs of spacecraft.
      - The vector from each spacecraft to the constellation's guiding center.
      - The projection (dot product) of the constellation centroid with the wave propagation direction,
        scaled by the arm length and speed of light.
    All operations are performed in place by modifying `sc_sep`.
    The guiding center is assumed to be the average position of all 3 spacecraft.

    Parameters
    ----------
    lc: LISAConstants
        LISA configuration constants, including arm length.
    tb: TensorBasis
        Contains the gravitational-wave propagation direction vector.
    sc_pos: SpacecraftScalarPosition
        Current positions (x, y, z) of the three spacecraft.
    sc_sep: SpacecraftSeparationVectors
        Object to be populated with the computed separation vectors.

    Returns
    -------
    float: Projection of the wave propagation vector onto the guiding center
        in units of arm length over speed of light.
    """
    kv: NDArray[np.floating] = tb.kv
    x: NDArray[np.floating] = sc_pos.x
    y: NDArray[np.floating] = sc_pos.y
    z: NDArray[np.floating] = sc_pos.z

    n_spacecraft: int = x.shape[0]
    assert x.shape == (n_spacecraft,)
    assert y.shape == (n_spacecraft,)
    assert z.shape == (n_spacecraft,)

    assert n_spacecraft == 3, 'Only implemented for 3 spacecraft'

    n_space = 3  # number of spatial dimensions
    assert kv.shape == (n_space,)

    # manually average over n_spacecraft to get coordinates of the guiding center
    xa: float = (x[0] + x[1] + x[2]) / n_spacecraft
    ya: float = (y[0] + y[1] + y[2]) / n_spacecraft
    za: float = (z[0] + z[1] + z[2]) / n_spacecraft

    # manual dot product with n_spacecraft
    kdotx: float = lc.t_arm * (xa * kv[0] + ya * kv[1] + za * kv[2])

    r12 = sc_sep.r12
    r13 = sc_sep.r13
    r23 = sc_sep.r23
    r10 = sc_sep.r10
    r20 = sc_sep.r20
    r30 = sc_sep.r30

    assert r12.shape == (n_space,)
    assert r13.shape == (n_space,)
    assert r23.shape == (n_space,)
    assert r10.shape == (n_space,)
    assert r20.shape == (n_space,)
    assert r30.shape == (n_space,)

    # Separation vector from spacecraft i to j
    r12[0] = x[1] - x[0]
    r12[1] = y[1] - y[0]
    r12[2] = z[1] - z[0]
    r13[0] = x[2] - x[0]
    r13[1] = y[2] - y[0]
    r13[2] = z[2] - z[0]
    r23[0] = x[2] - x[1]
    r23[1] = y[2] - y[1]
    r23[2] = z[2] - z[1]

    # Separation vector between spacecraft and guiding center (not unit vectors)
    r10[0] = xa - x[0]
    r10[1] = ya - y[0]
    r10[2] = za - z[0]
    r20[0] = xa - x[1]
    r20[1] = ya - y[1]
    r20[2] = za - z[1]
    r30[0] = xa - x[2]
    r30[1] = ya - y[2]
    r30[2] = za - z[2]
    return kdotx


@njit()
def get_sc_scalar_pos(
    lc: LISAConstants,
    t: float,
    sc_phasing: SpacecraftRelativePhases,
    sc_pos: SpacecraftScalarPosition,
) -> None:
    """
    Compute the Cartesian positions of the LISA spacecraft at a given time.

    This function calculates the (x, y, z) positions for each spacecraft in units scaled by the LISA arm length,
    given the current time, orbital, and phasing parameters. The spacecraft orbits are constructed analytically using
    rotation matrices that describe their motion in the rotating triangular configuration.
    Positions are expressed in a heliocentric-ecliptic reference frame.
    Positions are computed in place by modifying `sc_pos`.

    Parameters
    ----------
    lc: LISAConstants
        LISA system constants and orbital parameters (e.g., mean motion, phase offset, arm length).
    t: float
        Time at which to evaluate the positions.
    sc_phasing: SpacecraftRelativePhases
        Precomputed sine and cosine of relative phase angles for each spacecraft.
    sc_pos:SpacecraftScalarPosition
        Object whose x, y, and z arrays will be populated with spacecraft positions.
    """
    sin_beta: NDArray[np.floating] = sc_phasing.sin_beta
    cos_beta: NDArray[np.floating] = sc_phasing.cos_beta

    alpha: float = 2 * np.pi * lc.fm * t + lc.kappa0
    sa: float = np.sin(alpha)
    ca: float = np.cos(alpha)

    x: NDArray[np.floating] = sc_pos.x
    y: NDArray[np.floating] = sc_pos.y
    z: NDArray[np.floating] = sc_pos.z

    n_spacecraft: int = x.shape[0]

    assert x.shape == (n_spacecraft,)
    assert y.shape == (n_spacecraft,)
    assert z.shape == (n_spacecraft,)
    assert cos_beta.shape == (n_spacecraft,)
    assert sin_beta.shape == (n_spacecraft,)

    for itrc in range(n_spacecraft):
        x[itrc] = (lc.r_orbit * ca
                   + sa * ca * sin_beta[itrc] - (1.0 + sa * sa) * cos_beta[itrc])
        y[itrc] = lc.r_orbit * sa + sa * ca * cos_beta[itrc] - (1.0 + ca * ca) * sin_beta[itrc]
        z[itrc] = -float(np.sqrt(3.0)) * (ca * cos_beta[itrc] + sa * sin_beta[itrc])


@njit(fastmath=True)
def rigid_adiabatic_antenna(
    sc_channels: AntennaResponseChannels,
    params_extrinsic: ExtrinsicParams,
    T: NDArray[np.floating],
    F: NDArray[np.floating],
    nx_lim: PixelGenericRange,
    kdotx: NDArray[np.floating],
    lc: LISAConstants,
) -> None:
    """Get the waveform for LISA given polarization angle, spacecraft, tensor basis and Fs, channel order AET."""
    RR = sc_channels.RR
    II = sc_channels.II

    nc_waveform = RR.shape[0]  # number of channels in the output intrinsic_waveform

    n_points = T.shape[0]
    assert 0 <= nx_lim.nx_min <= nx_lim.nx_max <= n_points
    assert T.shape == (n_points,)
    assert F.shape == (n_points,)
    assert kdotx.shape == (n_points,)
    assert RR.shape == (nc_waveform, n_points)
    assert II.shape == (nc_waveform, n_points)

    tb: TensorBasis = get_tensor_basis(params_extrinsic)

    n_space = 3  # number of spatial dimensions (must be 3)

    assert tb.kv.shape == (n_space,)
    assert tb.e_plus.shape == (n_space, n_space)
    assert tb.e_cross.shape == (n_space, n_space)

    n_spacecraft = 3  # number of spacecraft (currently must be 3)

    n_arm = 3  # number of arms (currently must be 3)

    nc_generate = 3  # number of combinations to generate internally (currently must be 3)
    assert nc_generate >= nc_waveform

    nc_michelson = 3  # number of michelson combinations (currently must be 3)
    assert nc_michelson >= nc_generate

    polarization_response: DetectorPolarizationResponse = DetectorPolarizationResponse(np.zeros((n_arm, n_arm)), np.zeros((n_arm, n_arm)))

    transfer_function: ComplexTransferFunction = ComplexTransferFunction(np.zeros((n_arm, n_arm)), np.zeros((n_arm, n_arm)))

    sc_pos: SpacecraftScalarPosition = SpacecraftScalarPosition(np.zeros(n_spacecraft), np.zeros(n_spacecraft), np.zeros(n_spacecraft))

    # for projecting spacecraft arm vectors into tensor basis
    sc_wave_proj: SpacecraftSeparationWaveProjection = SpacecraftSeparationWaveProjection(np.zeros((n_arm, n_arm)), np.zeros(n_arm))

    # Tuple to hold the XYZ antenna pattern
    tdi_xyz: TDIComplexAntennaPattern = TDIComplexAntennaPattern(
        np.zeros(nc_michelson),
        np.zeros(nc_michelson),
        np.zeros(nc_michelson),
        np.zeros(nc_michelson),
    )

    # Tuple to hold the AET antenna pattern
    tdi_aet: TDIComplexAntennaPattern = TDIComplexAntennaPattern(
        np.zeros(nc_generate),
        np.zeros(nc_generate),
        np.zeros(nc_generate),
        np.zeros(nc_generate),
    )

    sc_sep: SpacecraftSeparationVectors = SpacecraftSeparationVectors(
        np.zeros(n_space),
        np.zeros(n_space),
        np.zeros(n_space),
        np.zeros(n_space),
        np.zeros(n_space),
        np.zeros(n_space),
    )

    sc_phasing: SpacecraftRelativePhases = get_oribtal_phase_constants(lc, n_spacecraft)

    assert lc.fstr > 0.0, 'LISA transfer frequency must be positive'

    A_psi: DetectorAmplitudePhaseCombinations = get_detector_amplitude_phase_combinations(params_extrinsic)

    # Main Loop
    for n in prange(nx_lim.nx_min, nx_lim.nx_max):
        # get the spacecraft response for the current time step

        get_sc_scalar_pos(lc, T[n], sc_phasing, sc_pos)

        kdotx[n] = compute_separation_vectors(lc, tb, sc_pos, sc_sep)

        get_separation_wave_projection(tb, sc_sep, sc_wave_proj)

        get_projected_detector_response(tb, sc_sep, polarization_response)

        # normalized frequency
        fr: float = 1 / (2 * lc.fstr) * F[n]

        get_transfer_function(fr, sc_wave_proj, transfer_function)

        get_michelson_combinations(polarization_response, transfer_function, A_psi, tdi_xyz)

        get_aet_combinations(tdi_xyz, tdi_aet)

        for itrc in range(nc_waveform):
            RR[itrc, n] = tdi_aet.FpR[itrc] - tdi_aet.FcI[itrc]
            II[itrc, n] = tdi_aet.FcR[itrc] + tdi_aet.FpI[itrc]


@njit()
def get_wavefront_time(
    lc: LISAConstants,
    tb: TensorBasis,
    time: NDArray[np.floating],
    sv: SpacecraftOrbits,
    wavefront_time: NDArray[np.floating],
) -> None:
    """
    Compute, in place, the wavefront time coordinate for each spacecraft in the LISA constellation.

    Calculates the time at which a surface of constant gravitational-wave phase, defined by
    t - (k · x), is reached at each spacecraft.
    Accounts for the projected position of each spacecraft along the direction of wave propagation,
    normalized by the LISA arm length and the speed of light.
    The output array is overwritten with the resulting wavefront times.
    wavefront_time is called xi in [Phys. Rev. D 101, 124008 (2020), Eq. (B3)].

    Parameters
    ----------
    lc: LISAConstants
        LISA configuration constants, including arm length and geometry.
    tb: TensorBasis
        The tensor basis structure, containing the propagation direction unit vector (kv).
    time: NDArray[np.floating]
        Array of reference times at which wavefront times are calculated.
    sv: SpacecraftOrbits
        Spacecraft position data, with positional components (xas, yas, zas).
    wavefront_time: NDArray[np.floating]
        Output array to be updated in place with the wavefront times. Each value represents
        the arrival time of the gravitational-wave phase front at a given spacecraft.
    """
    n_space: int = 3
    assert tb.kv.shape == (n_space,)
    n_points = time.shape[0]

    assert sv.xa.shape == (n_points,)
    assert sv.ya.shape == (n_points,)
    assert sv.za.shape == (n_points,)
    assert wavefront_time.shape == (n_points,)

    kdotx: NDArray[np.floating] = lc.t_arm * (sv.xa * tb.kv[0] + sv.ya * tb.kv[1] + sv.za * tb.kv[2])
    wavefront_time[:] = time - kdotx


@njit()
def get_spacecraft_vec(time: NDArray[np.floating], lc: LISAConstants) -> SpacecraftOrbits:
    """Compute the time-dependent positions of the three LISA spacecraft and the guiding center.

    This function evaluates the heliocentric positions of each spacecraft across the input times.
    Each spacecraft moves in a trailing, rotating triangle with fixed eccentricity, inclination, and phase,
    following the analytic description for the nominal LISA orbit.

    Returns the coordinates of both the individual spacecraft and the constellation guiding center (average position).

    Parameters
    ----------
    time:  NDArray[np.floating]
        1D array of time values (in seconds) for which to compute the spacecraft positions; shape (n_t,).
    lc: LISAConstants
        Structure containing LISA orbit configuration parameters.

    Returns
    -------
    SpacecraftOrbits:
        Structure containing arrays for the positions of each spacecraft (xs, ys, zs) and
        arrays for the guiding center coordinates (xas, yas, zas) at all specified times.
    """
    assert len(time.shape) == 1

    n_spacecraft: int = 3  # number of spacecraft (currently must be 3)

    xs: NDArray[np.floating] = np.zeros((n_spacecraft, time.size))
    ys: NDArray[np.floating] = np.zeros((n_spacecraft, time.size))
    zs: NDArray[np.floating] = np.zeros((n_spacecraft, time.size))
    alpha: NDArray[np.floating] = float(2 * np.pi * lc.fm) * time + float(lc.kappa0)

    sa: NDArray[np.floating] = np.sin(alpha)
    ca: NDArray[np.floating] = np.cos(alpha)

    for i in range(n_spacecraft):
        beta: float = i * float(2 / 3 * np.pi) + float(lc.lambda0)
        sb: float = np.sin(beta)
        cb: float = np.cos(beta)
        xs[i] = lc.r_orbit * ca + lc.r_orbit * lc.ec * (sa * ca * sb - (1.0 + sa * sa) * cb)
        ys[i] = lc.r_orbit * sa + lc.r_orbit * lc.ec * (sa * ca * cb - (1.0 + ca * ca) * sb)
        zs[i] = -float(np.sqrt(3.0)) * lc.r_orbit * lc.ec * (ca * cb + sa * sb)

    # guiding center
    xas: NDArray[np.floating] = (xs[0] + xs[1] + xs[2]) / n_spacecraft
    yas: NDArray[np.floating] = (ys[0] + ys[1] + ys[2]) / n_spacecraft
    zas: NDArray[np.floating] = (zs[0] + zs[1] + zs[2]) / n_spacecraft
    return SpacecraftOrbits(xs, ys, zs, xas, yas, zas)


@njit(fastmath=True)
def phase_wrap_freq(
    tdi_waveform: StationaryWaveformFreq,
    waveform: StationaryWaveformFreq,
    nf_lim: PixelGenericRange,
    kdotx: NDArray[np.floating],
    wrap_thresh: float = np.pi,
) -> None:
    r"""
    Wrap the phase perturbations in the frequency domain consistently across TDI channels.

    This routine applies the perturbations from the tdi phases in from the doppler modulation and
    rotation of the antenna pattern to get the tdi phases. Results are stored in `tdi_waveform.PF`.

    The method takes into account implicit 2*pi phase shifts from the intrinsic waveform, and wraps
    the phase to the proper multiple of 2*pi as accurately as possible, for more useful plotting and
    computation of numerical derivatives, such as for fisher matrix calculations. 2*pi errors should not
    typiccaly affect the likelihood or parameter estimation directly.

    Frequency bins are processed in descending order, because typically the reference
    phase will be towards the end, and this approach minimizes cumulative wrapping errors.

    Parameters
    ----------
    tdi_waveform : StationaryWaveformFreq
        Frequency-domain TDI waveform object.
    waveform : StationaryWaveformFreq
        Source (barycenter) frequency-domain waveform.
    nf_lim : PixelGenericRange
        Frequency grid description
    kdotx : NDArray[np.floating]
        Doppler time perturbation array used to compute the barycenter
        Doppler phase term.
    wrap_thresh : float, optional
        Threshold (in radians) for deciding when to apply a \(\pm 2\pi\) wrap to a channel's
        accumulated phase perturbation. Defaults to `np.pi`.
    """
    tdi_PF: NDArray[np.floating] = tdi_waveform.PF
    tdi_TF: NDArray[np.floating] = tdi_waveform.TF
    PF: NDArray[np.floating] = waveform.PF
    TF: NDArray[np.floating] = waveform.TF

    nf_points: int = TF.shape[0]

    assert 0 <= nf_lim.nx_min <= nf_lim.nx_max <= nf_points

    assert TF.shape == (nf_points,)
    assert PF.shape == (nf_points,)
    assert kdotx.shape == (nf_points,)

    assert len(tdi_PF.shape) == 2

    n_tdi: int = tdi_PF.shape[0]

    assert tdi_PF.shape == (n_tdi, nf_points)
    assert tdi_TF.shape == (n_tdi, nf_points)

    phase_accums: NDArray[np.floating] = np.zeros(n_tdi)

    js: NDArray[np.floating] = np.zeros(n_tdi)

    for n in range(nf_lim.nx_max - 1, nf_lim.nx_min - 1, -1):
        # Barycenter time and frequency
        t: float = TF[n]
        f: float = n * nf_lim.dx + nf_lim.x_min  # FF[n]

        # the doppler-only phase perturbation (should be already unwrapped)
        phase: float = -2 * np.pi * f * kdotx[n] - PF[n]  # TODO check pi/4

        for itrc in range(n_tdi):
            # only the phase perturbation from the antenna pattern
            p: float = -tdi_PF[itrc, n]

            # adjust for the phase perturbation implicit in the time perturbation; more stable than wrapping p directly
            if n == nf_lim.nx_max - 1:
                phase_accums[itrc] = -p
            else:
                phase_accums[itrc] -= np.pi * nf_lim.dx * (tdi_TF[itrc, n] - t + tdi_TF[itrc, n + 1] - TF[n + 1])

            # wrap if the accumulated phase perturbation exceeds the threshold
            if phase_accums[itrc] + p + js[itrc] > wrap_thresh:
                js[itrc] -= 2 * np.pi
            if -p - js[itrc] - phase_accums[itrc] > wrap_thresh:
                js[itrc] += 2 * np.pi

            tdi_PF[itrc, n] = tdi_PF[itrc, n] - phase - js[itrc]


@njit(fastmath=True)
def get_freq_tdi_amp_phase(
    tdi_waveform: StationaryWaveformFreq,
    waveform: StationaryWaveformFreq,
    spacecraft_channels: AntennaResponseChannels,
    lc: LISAConstants,
    nf_lim: PixelGenericRange,
    kdotx: NDArray[np.floating],
    er: EdgeRiseModel,
) -> None:
    """Get the frequency domain TDI response."""
    F = tdi_waveform.F
    tdi_AF: NDArray[np.floating] = tdi_waveform.AF
    tdi_PF: NDArray[np.floating] = tdi_waveform.PF
    tdi_TF: NDArray[np.floating] = tdi_waveform.TF
    tdi_TFp: NDArray[np.floating] = tdi_waveform.TFp

    TF: NDArray[np.floating] = waveform.TF
    TFp: NDArray[np.floating] = waveform.TFp
    AF: NDArray[np.floating] = waveform.AF

    nf_points: int = TF.shape[0]
    assert 0 <= nf_lim.nx_min <= nf_lim.nx_max <= nf_points
    assert TF.shape == (nf_points,)
    assert TFp.shape == (nf_points,)
    assert AF.shape == (nf_points,)
    assert F.shape == (nf_points,)
    assert kdotx.shape == (nf_points,)

    assert len(tdi_TF.shape) == 2
    n_tdi: int = tdi_TF.shape[0]
    assert tdi_TF.shape == (n_tdi, nf_points)
    assert tdi_PF.shape == (n_tdi, nf_points)
    assert tdi_AF.shape == (n_tdi, nf_points)
    assert tdi_TFp.shape == (n_tdi, nf_points)

    assert nf_lim.dx != 0.0, 'Frequency spacing must be non-zero'

    # for the derivative of RR and II absorb 1/(2*nf_lim.dx) into the constant in tdi_TF
    spacecraft_channel_deriv_helper(spacecraft_channels, -1.0 / (2 * nf_lim.dx))

    tdi_waveform_generic = StationaryWaveformGeneric(F, tdi_PF, tdi_TF, tdi_TF, tdi_AF)
    # Time based method applies phase perturbation to PF, so set PF to zero here
    waveform_generic = StationaryWaveformGeneric(
        F,
        np.zeros_like(waveform.PF),
        TF,
        TFp,
        AF,
    )

    amp_phase_loop_helper(
        F,
        TF,
        waveform_generic,
        tdi_waveform_generic,
        spacecraft_channels,
        lc,
        nf_lim,
    )

    # sign is flipped relative to time
    tdi_PF[:] = -tdi_PF

    # Wrap the phase perturbations consistently across channels
    phase_wrap_freq(tdi_waveform, waveform, nf_lim, kdotx)
    # apply edge rise/fall to tdi_AF and tdi_TF
    apply_edge_rise_helper(waveform.TF, tdi_AF, er, lc, nf_lim)

    # compute tdi_TFp as perturbation on TFps
    # compute the gradient dy/dx using a second order accurate central finite difference
    # assuming constant x grid along second axis,
    # forward/backward first order accurate at boundaries, and apply a TFps base
    stabilized_gradient_uniform_inplace(TF, TFp, tdi_TF, tdi_TFp, nf_lim.dx, nf_lim.nx_min, nf_lim.nx_max)
