"""some helper algebra tools using numba"""

from numba import njit, prange
from numpy.typing import NDArray


@njit()
def gradient_homog_2d_inplace(ys: NDArray[float], result: NDArray[float], dx: float) -> None:
    """Compute the gradient dy/dx using a second order accurate central finite difference.
    First order accurate method at the edges of the array..
    Assumes constant x grid along second axis.
    """
    assert ys.shape == result.shape, 'Incompatible shape for result'
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
def stabilized_gradient_inplace(
    x: NDArray[float], dxdt: NDArray[float], y: NDArray[float], dydt: NDArray[float], dt: float
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

    Returns
    -------
    None
        Results are stored in-place in dydt array

    Notes
    -----
    All input arrays must satisfy these conditions:
    - y and dydt must be 2D arrays of same shape (nc_channel, n_t)
    - x and dxdt must be 1D arrays of length n_t
    - n_t must be greater than 1 for gradient computation

    """
    nc_channel = y.shape[0]
    n_t = y.shape[1]

    # input dimension validation
    assert len(y.shape) == 2
    assert len(x.shape) == 1
    assert dydt.shape == y.shape
    assert x.shape == dxdt.shape
    assert x.shape[0] == n_t
    assert n_t > 1, 'Insuficient Length to Compute Gradient'

    for itrc in range(nc_channel):
        dydt[itrc, 0] = (y[itrc, 1] - y[itrc, 0] - x[1] + x[0]) / dt + dxdt[0]
        dydt[itrc, n_t - 1] = (y[itrc, n_t - 1] - y[itrc, n_t - 2] - x[n_t - 1] + x[n_t - 2]) / dt + dxdt[n_t - 1]

    for n in prange(1, n_t - 1):
        x_shift = -x[n + 1] + x[n - 1]
        dxdt_shift = dxdt[n]
        for itrc in range(nc_channel):
            dydt[itrc, n] = (y[itrc, n + 1] - y[itrc, n - 1] + x_shift) / (2 * dt) + dxdt_shift
