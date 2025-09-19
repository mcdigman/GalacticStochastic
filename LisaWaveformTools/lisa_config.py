"""read wavelet transform constants in from config file and compute derived parameters"""

from collections import namedtuple
from warnings import warn

import numpy as np

import GalacticStochastic.global_const as gc

LISAConstants = namedtuple(
    'LISAConstants', ['Larm', 'Sps', 'Sacc', 'kappa0', 'lambda0', 'fstr', 't_arm', 'r_orbit', 'ec', 'fm', 'nc_waveform', 'nc_snr', 't0', 't_rise', 'rise_mode'],
)


def get_lisa_constants(config: dict) -> LISAConstants:
    """Get lisa constant object from config file"""
    # Mean arm length of the LISA detector (meters)
    Larm = float(config['lisa_constants']['Larm'])
    assert Larm > 0.0

    # Photon shot noise power
    Sps = float(config['lisa_constants']['Sps'])
    assert Sps >= 0.0

    # Acceleration noise power
    Sacc = float(config['lisa_constants']['Sacc'])
    assert Sacc >= 0.0

    # Initial azimuthal position of the guiding center
    kappa0 = float(config['lisa_constants']['kappa0'])

    # Initial orientation of the LISA constellation
    lambda0 = float(config['lisa_constants']['lambda0'])

    # Transfer frequency should be fixed to c/(2*pi*Larm)?
    fstr = float(config['lisa_constants']['fstr'])
    assert fstr > 0.0
    if not np.isclose(fstr, gc.CLIGHT / (2 * np.pi * Larm), atol=1.0e-10, rtol=1.0e-3):
        warn(
            'expected fstr=gc.CLIGHT/(2*np.pi*Larm)=%+.6e, got %+.6e' % (gc.CLIGHT / (2 * np.pi * Larm), fstr),
            stacklevel=2,
        )

    # Light travel time across arm length
    t_arm = float(Larm / gc.CLIGHT)

    # Lisa orbital radius in units of AU
    r_au = float(config['lisa_constants']['r_au'])

    # Lisa orbital radius in meters
    r_m = float(r_au * gc.AU)

    # Lisa orbital radius in units of arm lengths
    r_orbit = float(r_m / Larm)

    # LISA orbital eccentricity; should be fixed to Larm/(2*r_m*np.sqrt(3))
    ec = float(config['lisa_constants']['ec'])
    if not np.isclose(ec, Larm / (2 * r_m * np.sqrt(3)), atol=1.0e-10, rtol=1.0e-3):
        warn(
            'expected ec=Larm/(2*r_m*np.sqrt(3))=%+.6e, got %+.6e' % (Larm / (2 * r_m * np.sqrt(3)), ec),
            stacklevel=2,
        )

    # LISA orbital modulation frequency (i.e. 1/orbital period) (Hz)
    fm = float(config['lisa_constants']['fm'])
    assert fm > 0.0

    # number of intrinsic_waveform channels that must be evaluated
    nc_waveform = int(config['lisa_constants']['nc_waveform'])
    assert nc_waveform >= 0
    assert nc_waveform <= 3

    nc_snr = int(config['lisa_constants']['nc_snr'])
    assert nc_snr >= 0
    assert nc_snr <= 3
    assert nc_snr <= nc_waveform

    # Global time offset of start of lisa observations
    t0 = float(config['lisa_constants'].get('t0', 0.0))
    assert t0 == 0.0, 'Not all methods currently support nonzero t0'

    # Rise time for antenna pattern in frequency domain
    t_rise = float(config['lisa_constants'].get('t_rise', 0.0))

    assert t_rise == 0.0, 'Some methods may not support nonzero t_rise'

    # Mode to use for rise time calculation
    rise_mode = int(config['lisa_constants'].get('rise_mode', 3))

    return LISAConstants(Larm, Sps, Sacc, kappa0, lambda0, fstr, t_arm, r_orbit, ec, fm, nc_waveform, nc_snr, t0, t_rise, rise_mode)
