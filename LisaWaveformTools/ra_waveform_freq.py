"""Functions to compute rigid adiabatic detector response."""
from collections import namedtuple

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from LisaWaveformTools.lisa_config import LISAConstants

AntennaResponseChannels = namedtuple('AntennaResponseChannels', ['T', 'RR', 'II', 'dRR', 'dII'])

ExtrinsicParams = namedtuple('ExtrinsicParams', ['costh', 'phi', 'cosi', 'psi'])
ExtrinsicParams.__doc__ = """
Store the extrinsic parameters common to most detector waveform models.

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

TensorBasis = namedtuple('TensorBasis', ['kv', 'e_plus', 'e_cross'])

TensorBasis.__doc__ = """Store the tensor basis vectors.
kv : numpy.ndarray
    The unit vector in the direction of the source
e_plus : numpy.ndarray
    The plus polarization tensor basis vector
e_cross : numpy.ndarray
    The cross polarization tensor basis vector
"""

SpacecraftOrbits = namedtuple('SpacecraftOrbits', ['xs', 'ys', 'zs', 'xas', 'yas', 'zas'])
SpacecraftOrbits.__doc__= """
Store the spacecraft positions and guiding center coordinates.
Parameters
----------
xs : numpy.ndarray
    2D array of spacecraft x coordinates, shape (3, n_t)
ys : numpy.ndarray
    2D array of spacecraft y coordinates, shape (3, n_t)
zs : numpy.ndarray
    2D array of spacecraft z coordinates, shape (3, n_t)
xas : numpy.ndarray
    1D array of guiding center x coordinates, shape (n_t,)
yas : numpy.ndarray
    1D array of guiding center y coordinates, shape (n_t,)
zas : numpy.ndarray
    1D array of guiding center z coordinates, shape (n_t,)
"""

TDIComplexAntennaPattern = namedtuple('TDIComplexAntennaPattern', ['FpR', 'FcR', 'FpI', 'FcI'])
TDIComplexAntennaPattern.__doc__ = """
Store the TDI antenna pattern for complex channels.

Each field is a NumPy ndarray of shape (nc,) for a single time slice or (nc, nt) for multiple time steps,
where nc is the number of TDI channels and nt is the number of time steps.

Parameters
----------
    FpR : numpy.ndarray:
        Real part of the plus-polarization antenna pattern, shape (nc,) or (nc, nt).
    FcR : numpy.ndarray:
        Real part of the cross-polarization antenna pattern, shape (nc,) or (nc, nt).
    FpI : numpy.ndarray:
        Imaginary part of the plus-polarization antenna pattern, shape (nc,) or (nc, nt).
    FcI : numpy.ndarray:
        Imaginary part of the cross-polarization antenna pattern, shape (nc,) or (nc, nt).
"""

SpacecraftSeparationVectors = namedtuple('SpacecraftSeparationVectors', ['r12', 'r13', 'r23', 'r10', 'r20', 'r30'])

SpacecraftSeparationVectors.__doc__ = """
Store the separation vectors between pairs of spacecraft and between each spacecraft and the guiding center.

Each vector is a numpy.ndarray of shape (3,) for a single time slice or (3, nt) for multiple time steps,
where 3 is the number of spatial dimensions, x, y, z and nt is the number of time steps.

r12, r13, r23 are the separation vectors between spacecraft pairs, and are unit vectors.
r10, r20, r30 are the separation vectors from each spacecraft to the guiding center, and are not unit vectors.

Parameters
----------
r12 : numpy.ndarray
    Separation vector from spacecraft 1 to spacecraft 2, shape (3,) or (3, nt)
r13 : numpy.ndarray
    Separation vector from spacecraft 1 to spacecraft 2, shape (3,) or (3, nt)
r23 : numpy.ndarray
    Separation vector from spacecraft 1 to spacecraft 2, shape (3,) or (3, nt)
r10 : numpy.ndarray
    Separation vector from spacecraft 1 to the guiding center, shape (3,) or (3, nt)
r20 : numpy.ndarray
    Separation vector from spacecraft 2 to the guiding center, shape (3,) or (3, nt)
r30 : numpy.ndarray
    Separation vector from spacecraft 3 to the guiding center, shape (3,) or (3, nt)
"""

DetectorPolarizationResponse = namedtuple('DetectorPolarizationResponse', ['d_plus', 'd_cross'])
DetectorPolarizationResponse.__doc__ = """
Store the detector response for the plus and cross polarizations.

Parameters
----------
d_plus : numpy.ndarray
    The detector response for the plus polarization, shape (n_arm, n_arm)
d_cross : numpy.ndarray
    The detector response for the cross polarization, shape (n_arm, n_arm)
"""

ComplexTransferFunction = namedtuple('ComplexTransferFunction', ['TR', 'TI'])
ComplexTransferFunction.__doc__ = """
Store the complex transfer function for the TDI channels.
Parameters
----------
TR : numpy.ndarray
    Real part of the transfer function, shape (n_arm, n_arm)
TI : numpy.ndarray
    Imaginary part of the transfer function, shape (n_arm, n_arm)
"""

DetectorAmplitudePhaseCombinations = namedtuple('DetectorAmplitudePhaseCombinations', ['A_plus', 'A_cross', 'cos_psi', 'sin_psi'])

DetectorAmplitudePhaseCombinations.__doc__ = """
Store the amplitude and phase combinations need to compute the detector response.
Parameters
----------
A_plus : float
    Amplitude multiplier for the plus polarization
A_cross : float
    Amplitude multiplier for the cross polarization
cos_psi : float
    Cosine of the polarization angle
sin_psi : float
    Sine of the polarization angle
"""

SpacecraftScalarPosition = namedtuple('SpacecraftScalarPosition', ['x', 'y', 'z'])

SpacecraftScalarPosition.__doc__ = """
Store the positions of the spacecraft in the LISA constellation as a single time slice.
Parameters
----------
x : numpy.ndarray
    1D array of spacecraft x coordinates, shape (n_sc,)
y : numpy.ndarray
    1D array of spacecraft y coordinates, shape (n_sc,)
z : numpy.ndarray
    1D array of spacecraft z coordinates, shape (n_sc,)
"""

SpacecraftRelativePhases = namedtuple('SpacecraftRelativePhases', ['sin_beta', 'cos_beta', 'betas'])
SpacecraftRelativePhases.__doc__ = """
Store the time-independent parts of the parameters needed to compute spacecraft orbits for the LISA constellation.

Specifically, includes the fixed orbital phase offset of each spacecraft,
where beta_i = 2/3 * pi * i + lambda0, for i=0, 1, 2 determines the angluar position of each spacecraft i along
the guiding center orbit (lambda0 is the initial orientation of the constellation).
Also includes the sine and cosine of these phase offsets, which are used to compute the spacecraft positions.

Parameters
----------
sin_beta : numpy.ndarray
    1D array of sine of the spacecraft orbital phase offsets, shape (n_sc,)
cos_beta : numpy.ndarray
    1D array of cosine of the spacecraft orbital phase offsets, shape (n_sc,)
betas : numpy.ndarray
    1D array of spacecraft orbital phases offset, shape (n_sc,)
"""

SpacecraftSeparationWaveProjection = namedtuple('SpacecraftSeparationWaveProjection', ['k_sc_sc_sep', 'k_sc_gc_sep'])
SpacecraftSeparationWaveProjection.__doc__ = """
Store the dot products of the wave vector with the separation vectors.

Parameters
----------
k_sc_sc_sep : numpy.ndarray
    Dot products of the wave vector with the separation vectors between each spacecraft pair, shape (n_arm, n_arm)
k_sc_gc_sep : numpy.ndarray
    Dot products of the wave vector with the separation vectors between spacecraft and the guiding center, shape (n_arm,)
"""


@njit()
def get_tensor_basis(params_extrinsic: ExtrinsicParams) -> TensorBasis:
    """Compute the gravitational-wave tensor basis vectors for LISA observations.

    This function calculates:
    1. The wave propagation direction unit vector (kv)
    2. The plus polarization tensor basis (e_plus)
    3. The cross polarization tensor basis (e_cross)

    Parameters
    ----------
    params_extrinsic : ExtrinsicParams
        A namedtuple containing the extrinsic parameters:
        - costh: Cosine of the source's ecliptic colatitude
        - phi: Source's ecliptic longitude in radians

    Returns
    -------
    TensorBasis
        A namedtuple containing:
        - kv: ndarray of shape (3,)
            Unit vector pointing from the source toward LISA
        - e_plus: ndarray of shape (3, 3)
            Plus polarization tensor basis
        - e_cross: ndarray of shape (3, 3)
            Cross polarization tensor basis

    Notes
    -----
    The function constructs intermediate coordinate vectors u and v that are used
    to build the polarization tensors. The calculations are performed in a
    three-dimensional space (n_space=3) using spherical coordinates.

    """
    # Calculate cos and sin of sky position, inclination, polarization
    n_space = 3  # number of spatial dimensions (must be 3)

    costh = params_extrinsic.costh
    phi = params_extrinsic.phi
    sinth = np.sqrt(1.0 - costh**2)
    cosph = np.cos(phi)
    sinph = np.sin(phi)

    kv = np.zeros(n_space)
    u = np.zeros(n_space)
    v = np.zeros(n_space)

    kv[0] = -sinth * cosph
    kv[1] = -sinth * sinph
    kv[2] = -costh

    u[0] = sinph
    u[1] = -cosph
    u[2] = 0.0

    v[0] = -costh * cosph
    v[1] = -costh * sinph
    v[2] = sinth

    e_plus = np.zeros((n_space, n_space))
    e_cross = np.zeros((n_space, n_space))

    for i in range(n_space):
        for j in range(n_space):
            e_plus[i, j] = u[i] * u[j] - v[i] * v[j]
            e_cross[i, j] = u[i] * v[j] + v[i] * u[j]
    return TensorBasis(kv, e_plus, e_cross)

@njit()
def get_oribtal_phase_constants(lc: LISAConstants, n_sc: int) -> SpacecraftRelativePhases:
    """Calculate the initial orbital phase offsets and their sine and cosine values for each spacecraft in the LISA constellation.

    This function determines the relative phase angles at which each spacecraft orbits with respect to a fixed reference,
    according to the geometric configuration of the LISA constellation. These phase offsets are needed to compute the
    spacecraft positions,the light travel time, and model the detector's response to gravitational waves.

    Parameters
    ----------
        lc (LISAConstants):
            An object containing the fundamental LISA configuration constants, such as initial orientation, arm length, and
            orbital eccentricity parameters.
        n_sc (int):
            The number of spacecraft in the constellation (typically 3 for LISA).

    Returns
    -------
        SpacecraftRelativePhases:
            A named tuple containing:
                - sin_beta: ndarray of shape (n_sc,)
                    The sine of each spacecraft's initial orbital phase offset times orbital radius and eccentricity.
                - cos_beta: ndarray of shape (n_sc,)
                    The cosine of each spacecraft's initial orbital phase offset times orbital rasius and eccentricity.
                - betas: ndarray of shape (n_sc,)
                    The initial orbital phase offsets (in radians) for the spacecraft.

    Notes
    -----
        - These phase constants are used for computing spacecraft positions along their orbits and are vital
          for constructing the time-dependent response of the LISA detector.
        - The calculation assumes a triangular and equidistant spacecraft configuration with a specified initial orientation.

    """
    # quantities for computing spacecraft positions
    betas = np.zeros(n_sc)
    sin_beta = np.zeros(n_sc)
    cos_beta = np.zeros(n_sc)
    for itrc in range(n_sc):
        betas[itrc] = 2.0 / 3.0 * np.pi * itrc + lc.lambda0
        sin_beta[itrc] = (lc.r_orbit * lc.ec) * np.sin(betas[itrc])
        cos_beta[itrc] = (lc.r_orbit * lc.ec) * np.cos(betas[itrc])
    return SpacecraftRelativePhases(sin_beta, cos_beta, betas)

@njit()
def get_detector_amplitude_phase_combinations(params_extrinsic: ExtrinsicParams) -> DetectorAmplitudePhaseCombinations:
    """Compute the amplitude and phase factors required for the detector's response to a gravitational-wave signal.

    This function uses the the binary inclination and polarization angles
    to calculate the coefficients that scale the plus and cross polarizations as seen by the detector.
    It also computes the cosine and sine of twice the polarization angle, as needed for projecting the signal onto
    the detector's antenna response.

    Parameters
    ----------
        params_extrinsic (ExtrinsicParams):
            Structure containing extrinsic source parameters, including
            - cosi: cosine of the inclination angle of the binary's orbital plane,
            - psi: polarization angle of the gravitational-wave signal (in radians).

    Returns
    -------
        DetectorAmplitudePhaseCombinations:
            Named tuple with the following fields:
              - A_plus (float): Amplitude coefficient for the plus polarization.
              - A_cross (float): Amplitude coefficient for the cross polarization.
              - cos_psi (float): Cosine of twice the polarization angle.
              - sin_psi (float): Sine of twice the polarization angle.

    Notes
    -----
        - The amplitude coefficients encapsulate the dependence of the observed strain amplitude on binary inclination,
          while the trigonometric terms reflect the polarization mixing as projected by the detector.

    """
    cosi = params_extrinsic.cosi
    psi = params_extrinsic.psi

    # polarization angle quantities
    cos_psi = np.cos(2 * psi)
    sin_psi = np.sin(2 * psi)

    # amplitude multipliers
    A_plus = (1.0 + cosi**2) / 2
    A_cross = -cosi

    return DetectorAmplitudePhaseCombinations(A_plus, A_cross, cos_psi, sin_psi)

@njit()
def get_aet_combinations(tdi_xyz: TDIComplexAntennaPattern, tdi_aet: TDIComplexAntennaPattern) -> None:
    """Convert the Michelson TDI channel responses (X, Y, Z) into the noise orthogonal A, E, and T combinations.

    This function transforms the complex antenna pattern components for the standard TDI Michelson channels into the
    noise-orthogonal AET basis. The A, E, and T channels are constructed as specific linear combinations of X, Y, and Z,
    designed to diagonalize the noise covariance matrix.

    The transformation is applied separately to both real and imaginary parts, and for both plus and cross polarizations.

    Parameters
    ----------
        tdi_xyz (TDIComplexAntennaPattern):
            Contains the complex antenna pattern responses for the X, Y, and Z Michelson TDI channels.
        tdi_aet (TDIComplexAntennaPattern):
            Output structure to be populated in place with the corresponding A, E, and T channel responses.

    Returns
    -------
        None: The converted channel responses are written directly into the fields of `tdi_aet`.

    Notes
    -----
        - The transformation preserves the physical signal content while providing statistically independent channels.
        - The A, E, and T channels are commonly used in LISA data analysis to improve signal extraction and noise modeling.

    """
    FpR_aet = tdi_aet.FpR
    FcR_aet = tdi_aet.FcR
    FpI_aet = tdi_aet.FpI
    FcI_aet = tdi_aet.FcI

    FpR_xyz = tdi_xyz.FpR
    FcR_xyz = tdi_xyz.FcR
    FpI_xyz = tdi_xyz.FpI
    FcI_xyz = tdi_xyz.FcI

    FpR_aet[0] = (2 * FpR_xyz[0] - FpR_xyz[1] - FpR_xyz[2]) / 3.0
    FcR_aet[0] = (2 * FcR_xyz[0] - FcR_xyz[1] - FcR_xyz[2]) / 3.0

    FpR_aet[1] = (FpR_xyz[2] - FpR_xyz[1]) / np.sqrt(3.0)
    FcR_aet[1] = (FcR_xyz[2] - FcR_xyz[1]) / np.sqrt(3.0)

    FpR_aet[2] = (FpR_xyz[0] + FpR_xyz[1] + FpR_xyz[2]) / 3.0
    FcR_aet[2] = (FcR_xyz[0] + FcR_xyz[1] + FcR_xyz[2]) / 3.0

    FpI_aet[0] = (2 * FpI_xyz[0] - FpI_xyz[1] - FpI_xyz[2]) / 3.0
    FcI_aet[0] = (2 * FcI_xyz[0] - FcI_xyz[1] - FcI_xyz[2]) / 3.0

    FpI_aet[1] = (FpI_xyz[2] - FpI_xyz[1]) / np.sqrt(3)
    FcI_aet[1] = (FcI_xyz[2] - FcI_xyz[1]) / np.sqrt(3)

    FpI_aet[2] = (FpI_xyz[0] + FpI_xyz[1] + FpI_xyz[2]) / 3.0
    FcI_aet[2] = (FcI_xyz[0] + FcI_xyz[1] + FcI_xyz[2]) / 3.0


@njit()
def get_michelson_combinations(polarization_response: DetectorPolarizationResponse, transfer_function: ComplexTransferFunction, A_psi: DetectorAmplitudePhaseCombinations, tdi_xyz: TDIComplexAntennaPattern) -> None:
    """Construct the standard Michelson Time-Delay Interferometry (TDI) X, Y, and Z channel response combinations.

    This function computes the frequency-domain antenna pattern for each Michelson interferometer channel (X, Y, Z),
    accounting for polarization, transfer function, and amplitude-phase modulation. It projects the polarization responses,
    applies arm transfer functions, and forms the combinations necessary for synthesizing the TDI responses.

    The outputs are written in place into the provided TDIComplexAntennaPattern structure, populating the real and imaginary
    parts of the antenna pattern for different polarizations.

    Parameters
    ----------
        polarization_response (DetectorPolarizationResponse):
            Contains the detector tensor projection onto plus and cross polarizations for each arm combination.
        transfer_function (ComplexTransferFunction):
            Provides the complex (real and imaginary) transfer functions for each arm, representing the propagation and time-delay effects.
        A_psi (DetectorAmplitudePhaseCombinations):
            Holds the waveform amplitude for each polarization and the polarization angle.
        tdi_xyz (TDIComplexAntennaPattern):
            Output object to be filled with the resulting Michelson TDI channel antenna pattern components in the complex domain.

    Returns
    -------
        None: The function updates `tdi_xyz` fields in place for each of the X, Y, and Z Michelson channels and for each polarization component.

    Notes
    -----
        - All arrays are indexed so that arm/channel and polarization associations are consistent between input and output structures.
        - Suitable for use in modeling both instrument response and gravitational-wave signal synthesis in LISA and similar detectors.

    """
    A_plus = A_psi.A_plus
    A_cross = A_psi.A_cross
    cos_psi = A_psi.cos_psi
    sin_psi = A_psi.sin_psi

    nc_michelson = tdi_xyz.FpR.shape[0]
    TR = transfer_function.TR
    TI = transfer_function.TI

    d_plus = polarization_response.d_plus
    d_cross = polarization_response.d_cross

    n_arm = d_plus.shape[0]
    for i in range(nc_michelson):
        tdi_xyz.FpR[i] = (
            -(
                    +(d_plus[i, (i + 1) % n_arm] * cos_psi + d_cross[i, (i + 1) % n_arm] * sin_psi) * TR[i, (i + 1) % n_arm]
                    - (d_plus[i, (i + 2) % n_arm] * cos_psi + d_cross[i, (i + 2) % n_arm] * sin_psi) * TR[i, (i + 2) % n_arm]
            ) *
            A_plus / 2
        )
        tdi_xyz.FcR[i] = (
            -(
                    +(-d_plus[i, (i + 1) % n_arm] * sin_psi + d_cross[i, (i + 1) % n_arm] * cos_psi) * TR[i, (i + 1) % n_arm]
                    - (-d_plus[i, (i + 2) % n_arm] * sin_psi + d_cross[i, (i + 2) % n_arm] * cos_psi) * TR[i, (i + 2) % n_arm]
            ) *
            A_cross / 2
        )
        tdi_xyz.FpI[i] = (
            -(
                    +(d_plus[i, (i + 1) % n_arm] * cos_psi + d_cross[i, (i + 1) % n_arm] * sin_psi) * TI[i, (i + 1) % n_arm]
                    - (d_plus[i, (i + 2) % n_arm] * cos_psi + d_cross[i, (i + 2) % n_arm] * sin_psi) * TI[i, (i + 2) % n_arm]
            ) *
            A_plus / 2
        )
        tdi_xyz.FcI[i] = (
            -(
                    +(-d_plus[i, (i + 1) % n_arm] * sin_psi + d_cross[i, (i + 1) % n_arm] * cos_psi) * TI[i, (i + 1) % n_arm]
                    - (-d_plus[i, (i + 2) % n_arm] * sin_psi + d_cross[i, (i + 2) % n_arm] * cos_psi) * TI[i, (i + 2) % n_arm]
            ) *
            A_cross / 2
        )


@njit()
def get_projected_detector_response(tb: TensorBasis, sc_sep: SpacecraftSeparationVectors, polarization_response: DetectorPolarizationResponse) -> None:
    """Compute the projected detector response of the LISA constellation to a gravitational wave.

    This function evaluates the response of the detector arms to an incoming gravitational wave,
    by projecting the polarization basis tensors onto the direction of the separation vectors between spacecraft.
    The result is the sensitivity of each detector link to the polarizations (plus and cross),
    as determined by the geometry of the constellation and the incident wave direction.

    Parameters
    ----------
        tb (TensorBasis):
            Contains the wave propagation unit vector and polarization basis tensors.
        sc_wave_proj (SpacecraftSeparationWaveProjection):
            Contains the projection (dot products) of the wave vector onto the relevant separation vectors,
            as computed by `get_separation_wave_projection`.
        resp (ProjectedDetectorResponse):
            Output object to be populated with the detector response for each link and polarization. The
            arrays within `resp` are updated in place.

    Returns
    -------
        None: The computed response values are written in place into the provided `resp` object.

    Notes
    -----
        - The function operates in place, modifying only the fields of the `resp` output argument.
        - The projected response values are necessary for constructing the full LISA TDI response to signals.
        - Assumes spacecraft and links are ordered and indexed consistently throughout the relevant data structures.

    """
    d_plus = polarization_response.d_plus
    d_cross = polarization_response.d_cross

    e_plus = tb.e_plus
    e_cross = tb.e_cross

    r12 = sc_sep.r12
    r13 = sc_sep.r13
    r23 = sc_sep.r23

    n_space = r12.shape[0]

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
def get_separation_wave_projection(tb: TensorBasis, sc_sep: SpacecraftSeparationVectors, sc_wave_proj: SpacecraftSeparationWaveProjection) -> None:
    """Project the gravitational-wave propagation vector onto the separation vectors.

    This function calculates:
        - The dot products between the wave vector and the separation vectors connecting each
          spacecraft pair (stored in `sc_wave_proj.k_sc_sc_sep`).
        - The dot products between the wave vector and the vectors connecting each spacecraft to
          the guiding center (stored in `sc_wave_proj.k_sc_gc_sep`).

    Parameters
    ----------
        tb (TensorBasis):
            Tensor basis structure containing the wave propagation direction vector (`kv`)
            and polarization tensors.
        sc_sep (SpacecraftSeparationVectors):
            Structure holding the three-dimensional separation vectors between spacecraft pairs
            and between each spacecraft and the guiding center.
        sc_wave_proj (SpacecraftSeparationWaveProjection):
            Output structure to be populated in place with the calculated dot products
            for spacecraft pair separations and guiding center separations.

    Returns
    -------
        None: The results are written in place to the arrays in `sc_wave_proj`.

    Notes
    -----
        - All operations are performed in place; no values are returned.
        - The output arrays correspond to indices [i, j] for each spacecraft pair and [i] for each spacecraft-guiding center vector.

    """
    k_sc_sc_sep = sc_wave_proj.k_sc_sc_sep
    k_sc_gc_sep = sc_wave_proj.k_sc_gc_sep

    kv = tb.kv

    r12 = sc_sep.r12
    r13 = sc_sep.r13
    r23 = sc_sep.r23
    r10 = sc_sep.r10
    r20 = sc_sep.r20
    r30 = sc_sep.r30

    n_space = kv.shape[0]
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
def get_transfer_function(fr: float, sc_wave_proj: SpacecraftSeparationWaveProjection,
                          transfer_function: ComplexTransferFunction) -> None:
    """Compute the complex transfer function (real and imaginary parts) for the LISA TDI channels.

    This function calculates the frequency response of the LISA constellation arms to a passing
    gravitational wave, taking into account the time-varying separations between spacecraft and
    projections of wave vectors relevant to the current geometry. The result is stored in the
    `transfer_function` argument.

    Parameters
    ----------
        fr (float): Dimensionless frequency parameter, typically `f / f_star`, where f is the frequency
            and f_star is the LISA transfer frequency.
        sc_wave_proj (SpacecraftSeparationWaveProjection): Structure containing precomputed inner products
            of the wave propagation direction with spacecraft separation vectors.
        transfer_function (ComplexTransferFunction): Object to store the output real and imaginary
            components of the transfer function matrix for each arm pair.

    Returns
    -------
        None: The results are written in place to `transfer_function.TR` and `transfer_function.TI`.

    Notes
    -----
        - The function implements the mathematical expressions as described in
          [Phys. Rev. D 101, 124008 (2020), Eq. (B9)].
        - For small frequencies (f << f_star), the transfer function approaches 1 (real part) and 0 (imaginary part).

    """
    k_sc_sc_sep = sc_wave_proj.k_sc_sc_sep
    k_sc_gc_sep = sc_wave_proj.k_sc_gc_sep

    TR = transfer_function.TR
    TI = transfer_function.TI
    n_arm = TR.shape[0]
    for i in range(n_arm - 1):
        for j in range(i + 1, n_arm):
            q1 = fr * (1.0 - k_sc_sc_sep[i, j])
            q2 = fr * (1.0 + k_sc_sc_sep[i, j])
            q3 = -fr * (3.0 + k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[i])
            q4 = -fr * (1.0 + k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[i])  # TODO missing 1/sqrt(3) on k_sc_gc_sep?
            q5 = -fr * (3.0 - k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[j])
            q6 = -fr * (1.0 - k_sc_sc_sep[i, j] - 2 * k_sc_gc_sep[j])  # TODO missing 1/sqrt(3) on k_sc_gc_sep?
            sincq1 = np.sin(q1) / q1 / 2
            sincq2 = np.sin(q2) / q2 / 2
            # Real part of T from eq B9 in PhysRevD.101.124008
            TR[i, j] = sincq1 * np.cos(q3) + sincq2 * np.cos(q4)  # goes to 1 when f/fstar small
            # imaginary part of T
            TI[i, j] = sincq1 * np.sin(q3) + sincq2 * np.sin(q4)  # goes to 0 when f/fstar small
            # save ops computing other triangle simultaneously
            TR[j, i] = sincq2 * np.cos(q5) + sincq1 * np.cos(q6)  # goes to 1 when f/fstar small
            TI[j, i] = sincq2 * np.sin(q5) + sincq1 * np.sin(q6)  # goes to 0 when f/fstar small



@njit()
def compute_separation_vectors(lc: LISAConstants, tb: TensorBasis, sc_pos: SpacecraftScalarPosition, sc_sep: SpacecraftSeparationVectors) -> float:
    """Compute the separation vectors between LISA spacecraft and between each spacecraft and the guiding center.

    Return the dot product of the gravitational-wave propagation direction with the guiding center position.

    This function calculates:
      - The three-dimensional separation vectors between all pairs of spacecraft.
      - The vector from each spacecraft to the constellation's guiding center (centroid).
      - The projection (dot product) of the constellation centroid with the wave propagation direction,
        scaled by the arm length and speed of light.

    Parameters
    ----------
        lc (LISAConstants): LISA configuration constants, including arm length.
        tb (TensorBasis): Contains the gravitational-wave propagation direction vector.
        sc_pos (SpacecraftScalarPosition): Current positions (x, y, z) of the three spacecraft.
        sc_sep (SpacecraftSeparationVectors): Object to be populated with the computed separation vectors.

    Returns
    -------
        float: The projection of the wave propagation vector onto the guiding center, in units of arm length over speed of light.

    Notes
    -----
        - All operations are performed in place: the results are written directly to the fields of `sc_sep`.
        - Spacecraft are assumed to be indexed 0, 1, and 2.
        - The guiding center is computed as the average position of all three spacecraft.

    """
    kv = tb.kv
    x = sc_pos.x
    y = sc_pos.y
    z = sc_pos.z

    n_sc = x.shape[0]
    # manually average over n_sc to get coordinates of the guiding center
    xa = (x[0] + x[1] + x[2]) / n_sc
    ya = (y[0] + y[1] + y[2]) / n_sc
    za = (z[0] + z[1] + z[2]) / n_sc

    # manual dot product with n_sc
    kdotx = lc.t_arm * (xa * kv[0] + ya * kv[1] + za * kv[2])

    r12 = sc_sep.r12
    r13 = sc_sep.r13
    r23 = sc_sep.r23
    r10 = sc_sep.r10
    r20 = sc_sep.r20
    r30 = sc_sep.r30

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
def get_sc_scalar_pos(lc: LISAConstants, t: float, sc_phasing: SpacecraftRelativePhases, sc_pos: SpacecraftScalarPosition) -> None:
    """Compute the Cartesian positions of the LISA spacecraft at a given time.

    This function calculates the (x, y, z) positions for each spacecraft in units scaled by the LISA arm length,
    given the current time, orbital, and phasing parameters. The spacecraft orbits are constructed analytically using
    rotation matrices that describe their motion in the rotating triangular configuration.

    Parameters
    ----------
        lc (LISAConstants): LISA system constants and orbital parameters (e.g., mean motion, phase offset, arm length).
        t (float): Time at which to evaluate the positions.
        sc_phasing (SpacecraftRelativePhases): Precomputed sine and cosine of the relative phase angles for each spacecraft.
        sc_pos (SpacecraftScalarPosition): Object whose x, y, and z arrays will be populated with the spacecraft positions.

    Returns
    -------
        None: The spacecraft positions are written in place in the `sc_pos` object.

    Notes
    -----
        - The function assumes three spacecraft, each with a unique phase (beta) around the guiding center.
        - Positions are expressed in a heliocentric-ecliptic reference frame, normalized by the LISA arm length.
        - This analytic orbit model assumes a perfectly equilateral triangle with fixed arm length, neglecting small trailing and breathing motions.

    """
    sin_beta = sc_phasing.sin_beta
    cos_beta = sc_phasing.cos_beta

    alpha = 2 * np.pi * lc.fm * t + lc.kappa0
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    x = sc_pos.x
    y = sc_pos.y
    z = sc_pos.z

    n_sc = x.shape[0]
    for itrc in range(n_sc):
        x[itrc] = lc.r_orbit * ca + sa * ca * sin_beta[itrc] - (1.0 + sa * sa) * cos_beta[itrc]
        y[itrc] = lc.r_orbit * sa + sa * ca * cos_beta[itrc] - (1.0 + ca * ca) * sin_beta[itrc]
        z[itrc] = -np.sqrt(3.0) * (ca * cos_beta[itrc] + sa * sin_beta[itrc])


@njit(fastmath=True)
def rigid_adiabatic_antenna(
    sc_channels: AntennaResponseChannels, params_extrinsic: ExtrinsicParams, ts, FFs, nf_low, NTs, kdotx, lc: LISAConstants,
) -> None:
    """Get the waveform for LISA given polarization angle, spacecraft, tensor basis and Fs, channel order AET."""
    RRs = sc_channels.RR
    IIs = sc_channels.II

    tb: TensorBasis = get_tensor_basis(params_extrinsic)

    n_space = 3  # number of spatial dimensions (must be 3)

    assert tb.kv.shape == (n_space,)
    assert tb.e_plus.shape == (n_space, n_space)
    assert tb.e_cross.shape == (n_space, n_space)

    n_sc = 3  # number of spacecraft (currently must be 3)

    n_arm = 3  # number of arms (currently must be 3)

    nc_waveform = RRs.shape[0]  # number of channels in the output waveform
    assert IIs.shape[0] == nc_waveform

    nc_generate = 3  # number of combinations to generate internally (currently must be 3)
    assert nc_generate >= nc_waveform

    nc_michelson = 3  # number of michelson combinations (currently must be 3)
    assert nc_michelson >= nc_generate

    polarization_response = DetectorPolarizationResponse(np.zeros((n_arm, n_arm)), np.zeros((n_arm, n_arm)))

    transfer_function = ComplexTransferFunction(np.zeros((n_arm, n_arm)), np.zeros((n_arm, n_arm)))

    sc_pos = SpacecraftScalarPosition(np.zeros(n_sc), np.zeros(n_sc), np.zeros(n_sc))

    # for projecting spacecraft arm vectors into tensor basis
    sc_wave_proj = SpacecraftSeparationWaveProjection(np.zeros((n_arm, n_arm)), np.zeros(n_arm))

    # Tuple to hold the XYZ antenna pattern
    tdi_xyz = TDIComplexAntennaPattern(np.zeros(nc_michelson), np.zeros(nc_michelson), np.zeros(nc_michelson), np.zeros(nc_michelson))

    # Tuple to hold the AET antenna pattern
    tdi_aet = TDIComplexAntennaPattern(np.zeros(nc_generate), np.zeros(nc_generate), np.zeros(nc_generate), np.zeros(nc_generate))

    sc_sep = SpacecraftSeparationVectors(np.zeros(n_space), np.zeros(n_space), np.zeros(n_space), np.zeros(n_space), np.zeros(n_space), np.zeros(n_space))

    sc_phasing = get_oribtal_phase_constants(lc, n_sc)

    A_psi = get_detector_amplitude_phase_combinations(params_extrinsic)


    # Main Loop
    for n in prange(NTs):
        # get the spacecraft response for the current time step

        get_sc_scalar_pos(lc, ts[n + nf_low], sc_phasing, sc_pos)

        kdotx[n] = compute_separation_vectors(lc, tb, sc_pos, sc_sep)

        get_separation_wave_projection(tb, sc_sep, sc_wave_proj)

        get_projected_detector_response(tb, sc_sep, polarization_response)

        # normalized frequency
        fr = 1 / (2 * lc.fstr) * FFs[n + nf_low]

        get_transfer_function(fr, sc_wave_proj, transfer_function)

        get_michelson_combinations(polarization_response, transfer_function, A_psi, tdi_xyz)

        get_aet_combinations(tdi_xyz, tdi_aet)

        for itrc in range(nc_waveform):
            RRs[itrc, n] = tdi_aet.FpR[itrc] - tdi_aet.FcI[itrc]
            IIs[itrc, n] = tdi_aet.FcR[itrc] + tdi_aet.FpI[itrc]

@njit()
def get_wavefront_time(lc: LISAConstants, tb: TensorBasis, ts: NDArray[np.float64], sv: SpacecraftOrbits, wavefront_time: NDArray[np.float64]) -> None:
    """Compute, in place, the wavefront (retarded) time coordinate for each spacecraft in the LISA constellation.

    This function calculates the time at which a surface of constant gravitational-wave phase, defined by
    t - (k · x), is reached at each spacecraft. The calculation accounts for the projected position of each spacecraft
    along the direction of wave propagation, properly normalized by the LISA arm length and the speed of light.
    The output array is overwritten with the resulting wavefront times.
    wavefront_time is called xi in [Phys. Rev. D 101, 124008 (2020), Eq. (B3)].

    Parameters
    ----------
        lc (LISAConstants):
            LISA configuration constants, including arm length and geometry.
        tb (TensorBasis):
            The tensor basis structure, containing the propagation direction unit vector (kv).
        ts (NDArray[np.float64]):
            Array of reference times at which wavefront times are calculated.
        sv (SpacecraftOrbits):
            Spacecraft position data, with positional components (xas, yas, zas).
        waveform_time (NDArray[np.float64]):
            Output array to be updated in place with the wavefront (retarded) times. Each value represents
            the arrival time of the gravitational-wave phase front at a given spacecraft.

    Returns
    -------
        None. The function updates `waveform_time` in place.

    """
    kdotx = lc.t_arm * (sv.xas * tb.kv[0] + sv.yas * tb.kv[1] + sv.zas * tb.kv[2])
    wavefront_time[:] = ts - kdotx

@njit()
def get_spacecraft_vec(ts: NDArray[np.float64], lc: LISAConstants) -> SpacecraftOrbits:
    """Compute the time-dependent positions of the three LISA spacecraft and the guiding center.

    This function evaluates the heliocentric positions of each spacecraft in the LISA constellation across a series of input times.
    It models each spacecraft as moving in a trailing, rotating triangular configuration with fixed eccentricity, inclination, and phase,
    following the analytic description for the nominal LISA orbit.

    The function returns both the spacecraft coordinates and the coordinates of the constellation's guiding center (average position).

    Parameters
    ----------
        ts :  NDArray[np.float64]
            1D array of time values (in seconds) for which to compute the spacecraft positions; shape (n_t,).
        lc : LISAConstants
            Structure containing LISA orbit configuration parameters (e.g., mean motion, initial phases, eccentricity, arm length).

    Returns
    -------
        SpacecraftOrbits:
            Named tuple containing arrays for the positions of each spacecraft (xs, ys, zs) and
            arrays for the guiding center coordinates (xas, yas, zas) at all specified times.

    Notes
    -----
        - All arrays have shape (3, n_t) for spacecraft coordinates, and (n_t,) for guiding center coordinates.
        - The positions are normalized such that the arm length scaling (Larm) is explicit in the expressions.
        - The model assumes three spacecraft arranged in a near-equilateral triangle orbiting the Sun with fixed geometry.
        - This function is suitable for generating the nominal orbits for waveform simulation and TDI calculation.

    """
    n_sc = 3  # number of spacecraft (currently must be 3)

    xs = np.zeros((n_sc, ts.size))
    ys = np.zeros((n_sc, ts.size))
    zs = np.zeros((n_sc, ts.size))
    alpha = 2 * np.pi * lc.fm * ts + lc.kappa0

    sa = np.sin(alpha)
    ca = np.cos(alpha)

    for i in range(n_sc):
        beta = i * 2 / 3 * np.pi + lc.lambda0
        sb = np.sin(beta)
        cb = np.cos(beta)
        xs[i] = lc.r_orbit * ca + lc.r_orbit * lc.ec * (sa * ca * sb - (1.0 + sa * sa) * cb)
        ys[i] = lc.r_orbit * sa + lc.r_orbit * lc.ec * (sa * ca * cb - (1.0 + ca * ca) * sb)
        zs[i] = -np.sqrt(3) * lc.r_orbit * lc.ec * (ca * cb + sa * sb)

    # guiding center
    xas = (xs[0] + xs[1] + xs[2]) / n_sc
    yas = (ys[0] + ys[1] + ys[2]) / n_sc
    zas = (zs[0] + zs[1] + zs[2]) / n_sc
    return SpacecraftOrbits(xs, ys, zs, xas, yas, zas)
