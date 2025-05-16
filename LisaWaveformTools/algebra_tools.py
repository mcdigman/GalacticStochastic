"""some helper algebra tools using numba"""
from numba import njit


@njit()
def gradient_homog_2d_inplace(ys, dx, n_ys, NC, result):
    """Compute the gradient dy/dx using a second order accurate central finite difference assuming constant x grid along second axis, forward/backward first order accurate at boundaries, speedup is trivial"""
    for itrc in range(NC):
        result[itrc, 0] = (ys[itrc, 1] - ys[itrc, 0]) / dx
        result[itrc, n_ys - 1] = (ys[itrc, n_ys - 1] - ys[itrc, n_ys - 2]) / dx
        for i in range(1, n_ys - 1):
            result[itrc, i] = (ys[itrc, i + 1] - ys[itrc, i - 1]) / (2 * dx)
