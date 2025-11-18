"""read wavelet transform constants in from config file and compute derived parameters"""

from typing import Any, NamedTuple
from warnings import warn

import numpy as np
from numpy.testing import assert_allclose


class WDMWaveletConstants(NamedTuple):
    Nf: int
    Nt: int
    dt: float
    mult: int
    Nsf: int
    Nfd: int
    dfdot: float
    Nfd_negative: int
    Nst: int
    Tobs: float
    DF: float
    DT: float
    nx: float
    dfd: float
    df_bw: float
    BW: float
    Tw: float
    K: int
    A: float
    B: float
    dom: float
    DOM: float
    insDOM: float
    L: int
    n_f_null_extend: int
    taylor_time_interpolation_target_precision: float
    max_freq_tol_time_interpolation: float


def get_wavelet_model(config: dict[str, Any], assert_mode: int = 1) -> WDMWaveletConstants:
    config_wc: dict[str, int | float] = config['wavelet_constants']
    # number of time pixels (should be even)
    Nf = int(config_wc['Nf'])
    assert Nf & 1 == 0  # check even
    assert Nf > 0

    # number of frequency pixels (should be even)
    Nt = int(config_wc['Nt'])
    assert Nt & 1 == 0  # check even
    assert Nt > 0

    # time sampling cadence (units of seconds)
    dt = float(config_wc['dt'])
    assert dt > 0.0

    # over sampling
    mult = int(config_wc['mult'])
    assert mult > 0, 'filter width must be strictly positive'
    if mult < 4:
        warn('Mult is unusually small', stacklevel=2)

    # number of frequency steps in interpolation table
    Nsf = int(config_wc['Nsf'])
    assert Nsf > 0

    # number of fdots in interpolation table
    Nfd = int(config_wc['Nfd'])
    assert Nfd > 0

    # fractional fdot increment in interpolation table
    dfdot = float(config_wc['dfdot'])
    assert dfdot > 0.0

    # number of fdot increments which are less than zero in the interpolation table
    Nfd_negative = int(config_wc['Nfd_negative'])
    assert Nfd_negative >= 0
    assert Nfd_negative < Nfd, 'Must be some positive values'
    if Nfd_negative > Nfd // 2:
        warn('Interpolation table is larger for negative frequencies than positive', stacklevel=2)

    # number of time steps used to compute the interpolation table; must be an integer times mult
    Nst = int(config_wc['Nst'])
    assert Nst > 0

    dkstep = int(Nst // mult)
    if dkstep * mult != Nst:
        msg = 'ratio of Nst and mult must be an integer'
        raise ValueError(msg)

    # filter steepness of wavelet transform
    nx = float(config_wc['nx'])
    assert nx > 0.0

    # reduced filter length; must be a power of 2
    L = int(config_wc['L'])
    assert L > 0, 'L must be a positive power of two'
    assert (L & (L - 1)) == 0, 'L must be a power of two'

    # number of extra time-frequency pixels to allow in the taylor time method to account for nulls
    # usually not really necessary but should algorithmically guarantee no segfault
    n_f_null_extend: int = int(config_wc.get('n_f_null_extend', 200))
    assert n_f_null_extend >= 0

    # target precision of taylor time interpolation table
    taylor_time_interpolation_target_precision: float = float(config_wc.get('taylor_time_interpolation_target_precision', 1.e-3))
    assert taylor_time_interpolation_target_precision >= 0.

    # tolerance on maximum frequency allowed by the time interpolation grid
    max_freq_tol_time_interpolation: float = float(config_wc.get('max_freq_tol_time_interpolation', 1.e-5))

    # derived constants

    # total number of points
    N = int(Nt * Nf)

    # total observation duration (same units as dt)
    Tobs = float(dt * N)

    # width of wavelet pixel in time (units of time, same as dt)
    DT = float(dt * Nf)

    # width of wavelet pixel in frequency (cycles/time)
    DF = float(1.0 / (2 * dt * Nf))

    # dimensionless filter length
    K = int(mult * 2 * Nf)
    assert K > 0
    if K <= L:
        msg = 'K = %d should be than L = %d' % (K, L)
        raise ValueError(msg)
    if K % L != 0:
        msg = 'K = %d should be an integer multiple of L = %d' % (K, L)
        if assert_mode:
            raise ValueError(msg)
        warn(msg, stacklevel=2)

    # filter duration (time; same units as dt)
    Tw = float(dt * K)
    assert Tw > 0.0
    assert Tw <= Tobs, 'Probably do not want wavelet window longer than observing time'

    # angular frequency spacing (radians per time)
    dom = float(2.0 * np.pi / Tw)

    # Nyquist angular frequency (Radians per time)
    OM = float(np.pi / dt)
    assert OM > 0.0

    # 2 pi times DF (radians/time)
    DOM = float(OM / Nf)

    # inverse square root of DOM (sqrt(time/radian))
    insDOM = float(1.0 / np.sqrt(DOM))

    # wavelet parameter A
    B = OM / (2 * Nf)

    # wavelet parameter B
    A = (DOM - B) / 2

    # total width of wavelet in frequency
    BW = (A + B) / np.pi

    # nonzero terms in phi transform (only need 0 and positive)
    df_bw = BW / Nsf

    # step size in FTd
    dfd = DF / Tw * dfdot

    # double check some known relationships between the parameters hold
    # note that some of these could be allclose, but the tests pass right now
    assert_allclose(DF**2, DF / (2 * DT))
    assert_allclose(B, 2 * A)
    assert_allclose(DOM, 2 * B)
    assert_allclose(DOM, np.pi / (dt * Nf))
    assert_allclose(B, np.pi / (2 * dt * Nf))
    assert_allclose(A, np.pi / (4 * dt * Nf))
    assert_allclose(BW, 3 / (4 * dt * Nf))
    assert_allclose(dfd, dfdot / (4 * dt**2 * mult * Nf**2))
    assert_allclose(Tw, 2 * dt * mult * Nf)
    assert_allclose(df_bw, 3 / (4 * dt * Nf * Nsf))

    # Check the specifications for the interpolation grid makes sense
    df_max_time = 8 * DF / Tw  # or 2 / (dt **2 * mult * Nf **2) or 8 * dfd / dfdot or 2 * Nt ** 2 / (Tobs ** 2 * mult)
    df_max_time_grid = dfd * (Nfd - Nfd_negative)
    df_min_time_grid = dfd * (- Nfd_negative)
    # allow a slight excess to account for numerical inexactness
    if df_max_time_grid > df_max_time * (1. + max_freq_tol_time_interpolation):
        msg = f'Maximum frequency of interpolation grid {df_max_time_grid} is larger than limit of reliability {df_max_time}, consider increasing Nt and decreasing Nf'
        if assert_mode:
            raise ValueError(msg)
        warn(msg, stacklevel=2)
    if df_min_time_grid < -df_max_time * (1. + max_freq_tol_time_interpolation):
        msg = f'Minimum frequency of interpolation grid {df_min_time_grid} is smaller than limit of reliability {-df_max_time}, consider increasing Nt and decreasing Nf'
        if assert_mode:
            raise ValueError(msg)
        warn(msg, stacklevel=2)

    return WDMWaveletConstants(
        Nf,
        Nt,
        dt,
        mult,
        Nsf,
        Nfd,
        dfdot,
        Nfd_negative,
        Nst,
        Tobs,
        DF,
        DT,
        nx,
        dfd,
        df_bw,
        BW,
        Tw,
        K,
        A,
        B,
        dom,
        DOM,
        insDOM,
        L,
        n_f_null_extend,
        taylor_time_interpolation_target_precision,
        max_freq_tol_time_interpolation,
    )
