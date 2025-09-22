"""read wavelet transform constants in from config file and compute derived parameters"""

# from collections import namedtuple
from typing import NamedTuple

import numpy as np


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


def get_wavelet_model(config: dict) -> WDMWaveletConstants:
    # number of time pixels (should be even)
    Nf = int(config['wavelet_constants']['Nf'])
    assert Nf & 1 == 0  # check even

    # number of frequency pixels (should be even)
    Nt = int(config['wavelet_constants']['Nt'])
    assert Nt & 1 == 0  # check even

    # time sampling cadence (units of seconds)
    dt = float(config['wavelet_constants']['dt'])
    assert dt > 0.0

    # over sampling
    mult = int(config['wavelet_constants']['mult'])

    # number of frequency steps in interpolation table
    Nsf = int(config['wavelet_constants']['Nsf'])

    # number of fdots in interpolation table
    Nfd = int(config['wavelet_constants']['Nfd'])

    # fractional fdot increment in interpolation table
    dfdot = float(config['wavelet_constants']['dfdot'])

    # number of fdot increments which are less than zero in the interpolation table
    Nfd_negative = int(config['wavelet_constants']['Nfd_negative'])
    assert Nfd_negative < Nfd

    # number of time steps used to compute the interpolation table; must be an integer times mult
    Nst = int(config['wavelet_constants']['Nst'])

    dkstep = int(Nst // mult)
    if dkstep * mult != Nst:
        msg = 'ratio of Nst and mult must be an integer'
        raise ValueError(msg)

    # filter steepness of wavelet transform
    nx = float(config['wavelet_constants']['nx'])

    # reduced filter length; must be a power of 2
    L = int(config['wavelet_constants']['L'])
    assert L > 0
    assert (L & (L - 1)) == 0  # check power of 2

    # derived constants

    # total number of points
    N = Nt * Nf

    # total observation duration (same units as dt)
    Tobs = dt * N

    # width of wavelet pixel in time (units of time, same as dt)
    DT = dt * Nf

    # width of wavelet pixel in frequency (cycles/time)
    DF = 1.0 / (2 * dt * Nf)

    # dimensionless filter legnth
    K = int(mult * 2 * Nf)

    # filter duration (time; same units as dt)
    Tw = dt * K

    # angular frequency spacing (radians per time)
    dom = 2.0 * np.pi / Tw

    # Nyquist angular frequency (Radians per time)
    OM = np.pi / dt

    # 2 pi times DF (radians/time)
    DOM = OM / Nf

    # inverse square root of DOM (sqrt(time/radian))
    insDOM = 1.0 / np.sqrt(DOM)

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
    assert DF**2 == DF / (2 * DT)

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
    )
