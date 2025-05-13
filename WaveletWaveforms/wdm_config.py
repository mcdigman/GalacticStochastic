"""read wavelet transform constants in from config file and compute derived parameters"""

import ast
from collections import namedtuple

import numpy as np

WDMWaveletConstants = namedtuple(
        'WDMWaveletConstants',
        [
            'Nf', 'Nt', 'dt', 'mult', 'Nsf', 'Nfd', 'dfdot', 'Nfd_negative', 'Nst', 'Tobs', 'NC', 'DF', 'DT', 'nx', 'dfd', 'df', 'BW', 'Tw', 'K', 'A', 'B', 'dom', 'DOM', 'insDOM'
        ]
        )


def get_wavelet_model(config):
    # number of time pixels (should be even)
    Nf = int(ast.literal_eval(config['wavelet constants']['Nf']))
    assert Nf & 1 == 0  # check even

    # number of frequency pixels (should be even)
    Nt = int(ast.literal_eval(config['wavelet constants']['Nt']))
    assert Nt & 1 == 0  # check even

    # time sampling cadence (units of seconds)
    dt = float(ast.literal_eval(config['wavelet constants']['dt']))
    assert dt > 0.

    # over sampling
    mult = int(ast.literal_eval(config['wavelet constants']['mult']))

    # number of frequency steps in interpolation table
    Nsf = int(ast.literal_eval(config['wavelet constants']['Nsf']))

    # number of fdots in interpolation table
    Nfd = int(ast.literal_eval(config['wavelet constants']['Nfd']))

    # fractional fdot increment in interpolation table
    dfdot = float(ast.literal_eval(config['wavelet constants']['dfdot']))

    # number of fdot increments which are less than zero in the interpolation table
    Nfd_negative = int(ast.literal_eval(config['wavelet constants']['Nfd_negative']))
    assert Nfd_negative < Nfd

    # number of time steps used to compute the interpolation table; must be an integer times mult
    Nst = int(ast.literal_eval(config['wavelet constants']['Nst']))

    dkstep = int(Nst//mult)
    if dkstep*mult != Nst:
        raise ValueError('ratio of Nst and mult must be an integer')

    # filter steepness of wavelet transform
    nx = float(ast.literal_eval(config['wavelet constants']['nx']))

    # reduced filter length; must be a power of 2
    L = int(ast.literal_eval(config['wavelet constants']['L']))
    assert L > 0 and (L & (L - 1)) == 0  # check power of 2

    # derived constants

    # total number of points
    N = Nt*Nf

    # total observation duration (same units as dt)
    Tobs = dt*N

    # width of wavelet pixel in time (units of time, same as dt)
    DT = dt*Nf

    # width of wavelet pixel in frequency (cycles/time)
    DF = 1./(2*dt*Nf)

    # dimensionless filter legnth
    K = mult*2*Nf

    # filter duration (time; same units as dt)
    Tw = dt*K

    # angular frequency spacing (radians per time)
    dom = 2.*np.pi/Tw

    # Nyquist angular frequency (Radians per time)
    OM = np.pi/dt

    # 2 pi times DF (radians/time)
    DOM = OM/Nf

    # inverse square root of DOM (sqrt(time/radian))
    insDOM = 1./np.sqrt(DOM)

    # wavelet parameter A
    B = OM/(2*Nf)

    # wavelet parameter B
    A = (DOM-B)/2

    # total width of wavelet in frequency
    BW = (A+B)/np.pi

    # nonzero terms in phi transform (only need 0 and positive)
    df = BW/Nsf

    # step size in FTd
    dfd = DF/Tw*dfdot

    # number of TDI channels to use
    # TODO move this some place else
    NC = 3

    return WDMWaveletConstants(Nf, Nt, dt, mult, Nsf, Nfd, dfdot, Nfd_negative, Nst, Tobs, NC, DF, DT, nx, dfd, df, BW, Tw, K, A, B, dom, DOM, insDOM)
