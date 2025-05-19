"""some helper algebra tools using numba"""
from numba import njit
from numpy.typing import NDArray


@njit()
def gradient_homog_2d_inplace(ys: NDArray[float], result: NDArray[float], dx: float) -> None:
    """Compute the gradient dy/dx using a second order accurate central finite difference assuming constant x grid along second axis, forward/backward first order accurate at boundaries, speedup is trivial"""
    nc_loc = ys.shape[0]
    n_ys = ys.shape[1]
    for itrc in range(nc_loc):
        result[itrc, 0] = (ys[itrc, 1] - ys[itrc, 0]) / dx
        result[itrc, n_ys - 1] = (ys[itrc, n_ys - 1] - ys[itrc, n_ys - 2]) / dx
        for i in range(1, n_ys - 1):
            result[itrc, i] = (ys[itrc, i + 1] - ys[itrc, i - 1]) / (2 * dx)
