"""read wavelet transform constants in from config file and compute derived parameters"""

from collections import namedtuple
from warnings import warn

import numpy as np

import GalacticStochastic.global_const as gc

LISAConstants = namedtuple(
    'LISAConstants', ['Larm', 'Sps', 'Sacc', 'kappa0', 'lambda0', 'fstr', 'ec', 'fm', 'nc_waveform', 'nc_snr'],
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

    # LISA orbital eccentricity; should be fixed to Larm/(2*AU*np.sqrt(3))?
    ec = float(config['lisa_constants']['ec'])
    if not np.isclose(ec, Larm / (2 * gc.AU * np.sqrt(3)), atol=1.0e-10, rtol=1.0e-3):
        warn(
            'expected ec=Larm/(2*AU*np.sqrt(3))=%+.6e, got %+.6e' % (Larm / (2 * gc.AU * np.sqrt(3)), fstr),
            stacklevel=2,
        )

    # LISA modulation frequency
    fm = float(config['lisa_constants']['fm'])
    assert fm > 0.0

    # number of waveform channels that must be evaluated
    nc_waveform = int(config['lisa_constants']['nc_waveform'])
    assert nc_waveform >= 0
    assert nc_waveform <= 3

    nc_snr = int(config['lisa_constants']['nc_snr'])
    assert nc_snr >= 0
    assert nc_snr <= 3
    assert nc_snr <= nc_waveform

    return LISAConstants(Larm, Sps, Sacc, kappa0, lambda0, fstr, ec, fm, nc_waveform, nc_snr)
