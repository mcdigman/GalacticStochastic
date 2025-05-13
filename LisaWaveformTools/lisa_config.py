"""read wavelet transform constants in from config file and compute derived parameters"""

import ast
from collections import namedtuple
from warnings import warn

import numpy as np

import GalacticStochastic.global_const as gc

LISAConstants = namedtuple('LISAConstants', ['Larm', 'Sps', 'Sacc', 'kappa0', 'lambda0', 'fstr', 'ec', 'fm'])


def get_lisa_constants(config):
    """Get lisa constant object from config file"""

    # Mean arm length of the LISA detector (meters)
    Larm = float(ast.literal_eval(config['lisa constants']['Larm']))
    assert Larm > 0.

    # Photon shot noise power
    Sps = float(ast.literal_eval(config['lisa constants']['Sps']))
    assert Sps >= 0.

    # Acceleration noise power
    Sacc = float(ast.literal_eval(config['lisa constants']['Sacc']))
    assert Sacc >= 0.

    # Initial azimuthal position of the guiding center
    kappa0 = float(ast.literal_eval(config['lisa constants']['kappa0']))

    # Initial orientation of the LISA constellation
    lambda0 = float(ast.literal_eval(config['lisa constants']['lambda0']))

    # Transfer frequency should be fixed to c/(2*pi*Larm)?
    fstr = float(ast.literal_eval(config['lisa constants']['fstr']))
    assert fstr > 0.
    if not np.isclose(fstr, gc.CLIGHT/(2*np.pi*Larm), atol=1.e-10, rtol=1.e-3):
        warn('expected fstr=gc.CLIGHT/(2*np.pi*Larm)=%+.6e, got %+.6e' % (gc.CLIGHT/(2*np.pi*Larm), fstr))

    # LISA orbital eccentricity; should be fixed to Larm/(2*AU*np.sqrt(3))?
    ec = float(ast.literal_eval(config['lisa constants']['ec']))
    if not np.isclose(ec,  Larm/(2*gc.AU*np.sqrt(3)), atol=1.e-10, rtol=1.e-3):
        warn('expected ec=Larm/(2*AU*np.sqrt(3))=%+.6e, got %+.6e' % (Larm/(2*gc.AU*np.sqrt(3)), fstr))

    # LISA modulation frequency
    fm = float(ast.literal_eval(config['lisa constants']['fm']))
    assert fm > 0.

    return LISAConstants(Larm, Sps, Sacc, kappa0, lambda0, fstr, ec, fm)
