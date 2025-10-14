"""Functions for computing gradients of arrays using numerical finite differences."""

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit()
def gradient_uniform_inplace(ys: NDArray[np.floating], result: NDArray[np.floating], dx: float) -> None:
    """Compute the numerical gradient of a 2D array using finite differences.

    This function computes dy/dx using second-order accurate central finite differences
    for interior points and first-order accurate forward/backward differences at the edges.
    Results are stored in-place in the provided result array.

    Parameters
    ----------
    ys : numpy.ndarray
        Input 2D array of shape (nc_loc, n_ys) containing y-values to differentiate.
        Array is assumed to be on a uniform grid along the second axis
    result : numpy.ndarray
        Output array of same shape as ys where gradient values will be stored
    dx : float
        Step size between x-values (uniform grid spacing)

    Notes
    -----
    - Uses second-order central differences: (y[i+1] - y[i-1])/(2*dx) for interior points
    - Uses first-order differences at boundaries:
      - Forward difference at left edge: (y[1] - y[0])/dx
      - Backward difference at right edge: (y[n] - y[n-1])/dx
    - Input array must have at least 2 points along second axis for gradient computation
    - Function is optimized using Numba's @njit decorator for performance
    - If compiled with parallel=True, parallel computation is used for the main loop over interior points

    Raises
    ------
    AssertionError
        If result and ys arrays have different shapes
        If input array has less than 2 points along second axis

    """
    assert ys.shape == result.shape, 'Incompatible shape for result'
    assert len(ys.shape) == 2, 'Input ys must be a 2D array'
    nc_loc = ys.shape[0]
    n_ys = ys.shape[1]
    assert n_ys > 1, 'Insufficient length to compute gradient'

    # handle the edge cases to first order
    for itrc in range(nc_loc):
        result[itrc, 0] = (ys[itrc, 1] - ys[itrc, 0]) / dx
        result[itrc, n_ys - 1] = (ys[itrc, n_ys - 1] - ys[itrc, n_ys - 2]) / dx

    # main loop to handle the rest to second order
    for i in prange(1, n_ys - 1):
        for itrc in range(nc_loc):
            result[itrc, i] = (ys[itrc, i + 1] - ys[itrc, i - 1]) / (2 * dx)


@njit()
def stabilized_gradient_uniform_inplace(
    x: NDArray[np.floating],
    dxdt: NDArray[np.floating],
    y: NDArray[np.floating],
    dydt: NDArray[np.floating],
    dt: float,
    nx_min: int = 0,
    nx_max: int = -1,
) -> None:
    """Get a second-order central stabilized gradient of y and store it in dydt.

    Uses x and it's already-known derivative dxdt as a reference value
    for greater numerical stability, assuming the curve y is a perturbation on top of x.
    Then using the idea that y is a perturbed curve, we can compute
    dydt = d/dt (y-x) + dxdt
    which is significantly more numerically accurate if dxdt is already well-known.

    Parameters
    ----------
    x : numpy.ndarray
        1D array of reference values of length n_t
    dxdt : numpy.ndarray
        1D array containing the known derivative of x, must be same shape as x
    y : numpy.ndarray
        2D array of shape (nc_channel, n_t) containing the perturbed curves
    dydt : numpy.ndarray
        2D array of same shape as y where the computed gradients will be stored
    dt : float
        Time step size for gradient calculation
    nx_min : int
        Minimum index along second axis to compute gradient (inclusive, default=0)
    nx_max : int
        Maximum index along second axis to compute gradient (exclusive, default=-1 means n_t)

    Notes
    -----
    All input arrays must satisfy these conditions:
    - y and dydt must be 2D arrays of same shape (nc_channel, n_t)
    - x and dxdt must be 1D arrays of length n_t
    - n_t must be greater than 1 for gradient computation

    """
    # input dimension validation
    assert len(y.shape) == 2, 'Input y must be a 2D array'
    assert len(x.shape) == 1, 'Input x must be a 1D array'
    assert dydt.shape == y.shape, 'Input dydt must have the same shape as y'
    assert x.shape == dxdt.shape, 'Input x and dxdt must have the same shape'
    nc_channel = y.shape[0]

    assert x.shape[0] == y.shape[1], 'Input x and dxdt must have the same length'
    assert y.shape[1] > 1, 'Insuficient Length to Compute Gradient'
    assert dt != 0.0, 'Time step dt cannot be zero'

    if nx_max == -1:
        nx_max = x.shape[0]

    assert 0 <= nx_min < nx_max <= x.shape[0], 'Invalid range for x'

    n_points = nx_max - nx_min

    assert n_points > 1, 'Insufficient Length to Compute Gradient'

    for itrc in range(nc_channel):
        dydt[itrc, nx_min] = (y[itrc, nx_min + 1] - y[itrc, nx_min] - x[nx_min + 1] + x[nx_min]) / dt + dxdt[nx_min]
        dydt[itrc, nx_max - 1] = (
            y[itrc, nx_max - 1] - y[itrc, nx_max - 2] - x[nx_max - 1] + x[nx_max - 2]
        ) / dt + dxdt[nx_max - 1]

    for n in prange(nx_min + 1, nx_max - 1):
        x_shift = -x[n + 1] + x[n - 1]
        dxdt_shift = dxdt[n]
        for itrc in range(nc_channel):
            dydt[itrc, n] = (y[itrc, n + 1] - y[itrc, n - 1] + x_shift) / (2 * dt) + dxdt_shift
