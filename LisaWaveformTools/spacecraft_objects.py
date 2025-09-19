"""Various objects for the spacecraft and detector response calculations."""
from collections import namedtuple

AntennaResponseChannels = namedtuple('AntennaResponseChannels', ['x', 'RR', 'II', 'dRR', 'dII'])

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
SpacecraftOrbits.__doc__ = """
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
