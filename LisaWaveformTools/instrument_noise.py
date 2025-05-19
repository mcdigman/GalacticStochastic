"""get the instrument noise profile"""


import numpy as np
import numpy.typing as npt

# import numba as nb
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.wdm_config import WDMWaveletConstants


def instrument_noise1(f, lc: LISAConstants):
    # Power spectral density of the detector noise and transfer frequency
    SAE = np.zeros(f.size)
    Sps = 9.e-24     # should match sangria v2? Should it be backlinknoise or readoutnoise?
    Sacc = 5.76e-30  # from sangria v2
    fonfs = f / lc.fstr
    # To match the LDC power spectra need a factor of 2 here. No idea why... (one sided/two sided?)
    LC = 2.0 * fonfs * fonfs
    # roll-offs
    rolla = (1.0 + pow((4.0e-4 / f), 2.0)) * (1.0 + pow((f / 8.0e-3), 4.0))
    rollw = 1.0 + pow((2.0e-3 / f), 4.0)
    # Calculate the power spectral density of the detector noise at the given frequency
    # not and exact match to the LDC, but within 10%
    SAE = LC * 16.0 / 3.0 * pow(np.sin(fonfs), 2.0) * (
            (2.0 + np.cos(fonfs)) * Sps * rollw + 2.0 * (3.0 + 2.0 * np.cos(fonfs) + np.cos(2.0 * fonfs))
            *
            (Sacc / pow(2.0 * np.pi * f, 4.0) * rolla)) / pow(2.0 * lc.Larm, 2.0)
    return SAE


# @njit()
def instrument_noise_AET(f, lc: LISAConstants, wc: WDMWaveletConstants):
    """Get power spectral density in all 3 channels, assuming identical in all arms"""
    # see arXiv:2005.03610
    # see arXiv:1002.1291
    fonfs = f / lc.fstr

    LC = 64 / (3 * lc.Larm**2)
    mult_all = LC * fonfs**2 * np.sin(fonfs)**2
    mult_sa = (4 * lc.Sacc / (2 * np.pi)**4) * (1 + 16.e-8 / f**2) * (1.0 + (f / 8.0e-3)**4.) / f**4
    mult_sp = lc.Sps * (1.0 + (2.0e-3 / f)**4.)

    cosfonfs = np.cos(fonfs)

    S_inst = np.zeros((f.size, wc.NC))

    S_inst[:, 0] = instrument_noise1(f, lc)  # TODO make this all self consistent
    S_inst[:, 1] = S_inst[:, 0]
    S_inst[:, 2] = mult_all * (mult_sa / 2 * (1 - 2 * cosfonfs + cosfonfs ** 2) + mult_sp * (1 - cosfonfs))
    return S_inst


# @njit()
def instrument_noise_AET_wdm_loop(phif: npt.NDArray[np.float64], lc: LISAConstants, wc: WDMWaveletConstants):
    """Helper to get the instrument noise for wdm"""
    # realistically this really only needs run once and is fast enough without jit
    # TODO check normalization
    # TODO get first and last bins correct
    nrm: float = np.sqrt(12318. / wc.Nf) * np.linalg.norm(phif)
    print('nrm instrument', nrm)
    phif /= nrm
    phif2 = phif**2

    S_inst_m = np.zeros((wc.Nf, wc.NC))
    half_Nt = wc.Nt // 2
    fs_long = np.arange(-half_Nt, half_Nt + wc.Nf * half_Nt) / wc.Tobs
    # prevent division by 0
    fs_long[half_Nt] = fs_long[half_Nt + 1]
    S_inst_long = instrument_noise_AET(fs_long, lc, wc)
    # excise the f=0 point
    S_inst_long[half_Nt, :] = 0.
    # apply window in loop
    for m in range(wc.Nf):
        S_inst_m[m] = np.dot(phif2, S_inst_long[m * half_Nt:(m + 2) * half_Nt])

    return S_inst_m


# @njit()
def instrument_noise_AET_wdm_m(lc: LISAConstants, wc: WDMWaveletConstants):
    """Get the instrument noise curve as a function of frequency for the wdm
    wavelet decomposition

    Parameters
    ----------
    lc : namedtuple
        constants for LISA constellation specified in lisa_config.py
    wc : namedtuple
        constants for WDM wavelet basis also from wdm_config.py

    Returns
    -------
    S_stat_m : numpy.ndarray (Nf x NC)
        array of the instrument noise curve for each TDI channel
        array shape is (freq. layers x number of TDI channels)
    """
    # TODO why no plus 1?
    ls = np.arange(-wc.Nt // 2, wc.Nt // 2)
    fs = ls / wc.Tobs
    phif = np.sqrt(wc.dt) * phitilde_vec(2 * np.pi * fs * wc.dt, wc.Nf, wc.nx)

    # TODO check ad hoc normalization factor
    S_inst_m = instrument_noise_AET_wdm_loop(phif, lc, wc)
    return S_inst_m
