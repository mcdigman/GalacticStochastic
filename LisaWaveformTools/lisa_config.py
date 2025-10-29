"""Read wavelet transform constants in from config file and compute derived parameters."""

from typing import Any, NamedTuple
from warnings import warn

import numpy as np

import GalacticStochastic.global_const as gc


class LISAConstants(NamedTuple):
    Larm: float
    Sps: float
    Sacc: float
    kappa0: float
    lambda0: float
    fstr: float
    t_arm: float
    r_orbit: float
    ec: float
    fm: float
    nc_waveform: int
    nc_snr: int
    t0: float
    t_rise: float
    rise_mode: int
    noise_curve_mode: int
    f_roll_acc_f_inv: float
    f_roll_acc_f: float
    f_roll_ps_f_inv: float


def get_lisa_constants(config: dict[str, Any]) -> LISAConstants:
    """Get lisa constant object from config file."""
    config_lc: dict[str, float | int] = config['lisa_constants']

    # Mean arm length of the LISA detector (meters)
    Larm = float(config_lc['Larm'])
    assert Larm > 0.0

    # Photon shot noise power
    Sps = float(config_lc['Sps'])
    assert Sps >= 0.0

    # Acceleration noise power
    Sacc = float(config_lc['Sacc'])
    assert Sacc >= 0.0

    # Initial azimuthal position of the guiding center
    kappa0 = float(config_lc['kappa0'])

    # Initial orientation of the LISA constellation
    lambda0 = float(config_lc['lambda0'])

    # Transfer frequency should be fixed to c/(2*pi*Larm)?
    fstr = float(config_lc['fstr'])
    assert fstr > 0.0
    if not np.isclose(fstr, gc.CLIGHT / (2 * np.pi * Larm), atol=1.0e-10, rtol=1.0e-3):
        warn(
            'expected fstr=gc.CLIGHT/(2*np.pi*Larm)=%+.6e, got %+.6e' % (gc.CLIGHT / (2 * np.pi * Larm), fstr),
            stacklevel=2,
        )

    # Light travel time across arm length
    t_arm = float(Larm / gc.CLIGHT)

    # Lisa orbital radius in units of AU
    r_au = float(config_lc['r_au'])

    # Lisa orbital radius in meters
    r_m = float(r_au * gc.AU)

    # Lisa orbital radius in units of arm lengths
    r_orbit = float(r_m / Larm)

    # LISA orbital eccentricity; should be fixed to Larm/(2*r_m*np.sqrt(3))
    ec = float(config_lc['ec'])
    ec_exp = float(Larm / (2 * r_m * np.sqrt(3)))
    if not np.isclose(ec, ec_exp, atol=1.0e-10, rtol=1.0e-3):
        warn(
            'expected ec=Larm/(2*r_m*np.sqrt(3))=%+.6e, got %+.6e' % (Larm / (2 * r_m * np.sqrt(3)), ec),
            stacklevel=2,
        )

    # LISA orbital modulation frequency (i.e. 1/orbital period) (Hz)
    fm = float(config_lc['fm'])
    assert fm > 0.0

    # number of intrinsic_waveform channels that must be evaluated
    nc_waveform = int(config_lc['nc_waveform'])
    assert nc_waveform >= 0
    assert nc_waveform <= 3

    nc_snr = int(config_lc['nc_snr'])
    assert nc_snr >= 0
    assert nc_snr <= 3
    assert nc_snr <= nc_waveform

    # Global time offset of start of lisa observations
    t0 = float(config_lc.get('t0', 0.0))
    assert t0 == 0.0, 'Not all methods currently support nonzero t0'

    # Rise time for antenna pattern in frequency domain
    t_rise = float(config_lc.get('t_rise', 0.0))

    assert t_rise == 0.0, 'Some methods may not support nonzero t_rise'

    # Mode to use for rise time calculation
    rise_mode = int(config_lc.get('rise_mode', 3))

    # Mode to use for instrument noise curve calculation
    noise_curve_mode = int(config_lc.get('noise_curve_mode', 0))

    f_roll_acc_f_inv = float(config_lc.get('f_roll_acc_f_inv', 4.0e-4))
    f_roll_acc_f = float(config_lc.get('f_roll_acc_f', 8.0e-3))
    f_roll_ps_f_inv = float(config_lc.get('f_roll_ps_f_inv', 2.0e-3))

    return LISAConstants(
        Larm,
        Sps,
        Sacc,
        kappa0,
        lambda0,
        fstr,
        t_arm,
        r_orbit,
        ec,
        fm,
        nc_waveform,
        nc_snr,
        t0,
        t_rise,
        rise_mode,
        noise_curve_mode,
        f_roll_acc_f_inv,
        f_roll_acc_f,
        f_roll_ps_f_inv,
    )
