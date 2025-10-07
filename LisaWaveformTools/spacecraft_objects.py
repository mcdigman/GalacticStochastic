"""Various objects for the spacecraft and detector response calculations."""
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class AntennaResponseChannels(NamedTuple):
    """
    Store the antenna response channels for the detector.

    Attributes
    ----------
    x : numpy.ndarray
        Frequency or time grid corresponding to the response values.
    RR : numpy.ndarray
        Real part of the antenna response.
    II : numpy.ndarray
        Imaginary part of the antenna response.
    dRR : numpy.ndarray
        Derivative of the real part of the antenna response.
    dII : numpy.ndarray
        Derivative of the imaginary part of the antenna response.
    """

    x: NDArray[np.floating]
    RR: NDArray[np.floating]
    II: NDArray[np.floating]
    dRR: NDArray[np.floating]
    dII: NDArray[np.floating]


class TensorBasis(NamedTuple):
    """Store the tensor basis vectors.

    Attributes
    ----------
    kv : numpy.ndarray
        The unit vector in the direction of the source
    e_plus : numpy.ndarray
        The plus polarization tensor basis vector
    e_cross : numpy.ndarray
        The cross polarization tensor basis vector
    """

    kv: NDArray[np.floating]
    e_plus: NDArray[np.floating]
    e_cross: NDArray[np.floating]


class SpacecraftOrbits(NamedTuple):
    """
    Store the spacecraft positions and guiding center coordinates.

    Attributes
    ----------
    x : numpy.ndarray
        2D array of spacecraft x coordinates, shape (3, n_t)
    y : numpy.ndarray
        2D array of spacecraft y coordinates, shape (3, n_t)
    z : numpy.ndarray
        2D array of spacecraft z coordinates, shape (3, n_t)
    xa : numpy.ndarray
        1D array of guiding center x coordinates, shape (n_t,)
    ya : numpy.ndarray
        1D array of guiding center y coordinates, shape (n_t,)
    za : numpy.ndarray
        1D array of guiding center z coordinates, shape (n_t,)
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    z: NDArray[np.floating]
    xa: NDArray[np.floating]
    ya: NDArray[np.floating]
    za: NDArray[np.floating]


class TDIComplexAntennaPattern(NamedTuple):
    """
    Store the TDI antenna pattern for complex channels.

    Each field is a NumPy ndarray of shape (nc,) for a single time slice or (nc, nt) for multiple time steps,
    where nc is the number of TDI channels and nt is the number of time steps.

    Attributes
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

    FpR: NDArray[np.floating]
    FpI: NDArray[np.floating]
    FcR: NDArray[np.floating]
    FcI: NDArray[np.floating]


class SpacecraftSeparationVectors(NamedTuple):
    """
    Store the separation vectors between pairs of spacecraft and between each spacecraft and the guiding center.

    Each vector is a numpy.ndarray of shape (3,) for a single time slice or (3, nt) for multiple time steps,
    where 3 is the number of spatial dimensions, x, y, z and nt is the number of time steps.

    r12, r13, r23 are the separation vectors between spacecraft pairs, and are unit vectors.
    r10, r20, r30 are the separation vectors from each spacecraft to the guiding center, and are not unit vectors.

    Attributes
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

    r12: NDArray[np.floating]
    r13: NDArray[np.floating]
    r23: NDArray[np.floating]
    r10: NDArray[np.floating]
    r20: NDArray[np.floating]
    r30: NDArray[np.floating]


class DetectorPolarizationResponse(NamedTuple):
    """
    Store the detector response for the plus and cross polarizations.

    Attributes
    ----------
    d_plus : numpy.ndarray
        The detector response for the plus polarization, shape (n_arm, n_arm)
    d_cross : numpy.ndarray
        The detector response for the cross polarization, shape (n_arm, n_arm)
    """

    d_plus: NDArray[np.floating]
    d_cross: NDArray[np.floating]


class ComplexTransferFunction(NamedTuple):
    """
    Store the complex transfer function for the TDI channels.

    Attributes
    ----------
    TR : numpy.ndarray
        Real part of the transfer function, shape (n_arm, n_arm)
    TI : numpy.ndarray
        Imaginary part of the transfer function, shape (n_arm, n_arm)
    """

    TR: NDArray[np.floating]
    TI: NDArray[np.floating]


class DetectorAmplitudePhaseCombinations(NamedTuple):
    """
    Store the amplitude and phase combinations need to compute the detector response.

    Attributes
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

    A_plus: float
    A_cross: float
    cos_psi: float
    sin_psi: float


class SpacecraftScalarPosition(NamedTuple):
    """
    Store the positions of the spacecraft in the LISA constellation as a single time slice.

    Attributes
    ----------
    x : numpy.ndarray
        1D array of spacecraft x coordinates, shape (n_sc,)
    y : numpy.ndarray
        1D array of spacecraft y coordinates, shape (n_sc,)
    z : numpy.ndarray
        1D array of spacecraft z coordinates, shape (n_sc,)
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    z: NDArray[np.floating]


class SpacecraftRelativePhases(NamedTuple):
    """
    Store the time-independent parts of the parameters needed to compute spacecraft orbits for the LISA constellation.

    Specifically, includes the fixed orbital phase offset of each spacecraft,
    where beta_i = 2/3 * pi * i + lambda0, for i=0, 1, 2 determines the angluar position of each spacecraft i along
    the guiding center orbit (lambda0 is the initial orientation of the constellation).
    Also includes the sine and cosine of these phase offsets, which are used to compute the spacecraft positions.

    Attributes
    ----------
    sin_beta : numpy.ndarray
        1D array of sine of the spacecraft orbital phase offsets, shape (n_sc,)
    cos_beta : numpy.ndarray
        1D array of cosine of the spacecraft orbital phase offsets, shape (n_sc,)
    beta : numpy.ndarray
        1D array of spacecraft orbital phases offset, shape (n_sc,)
    """

    sin_beta: NDArray[np.floating]
    cos_beta: NDArray[np.floating]
    beta: NDArray[np.floating]


class SpacecraftSeparationWaveProjection(NamedTuple):
    """
    Store the dot products of the wave vector with the separation vectors.

    Attributes
    ----------
    k_sc_sc_sep : numpy.ndarray
        Dot products of the wave vector with the separation vectors between each spacecraft pair, shape (n_arm, n_arm)
    k_sc_gc_sep : numpy.ndarray
        Dot products of the wave vector with the separation vectors between spacecraft and the guiding center, shape (n_arm,)
    """

    k_sc_sc_sep: NDArray[np.floating]
    k_sc_gc_sep: NDArray[np.floating]


class EdgeRiseModel(NamedTuple):
    """Store the parameters for the model to use at the edges of the time series.

    Attributes
    ----------
    t_start : float
        Start time of the model.
    t_end : float
        End time of the model.
    """

    t_start: float
    t_end: float
