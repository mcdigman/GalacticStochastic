"""LISA instrument noise curves."""

from warnings import warn

import numpy as np
from numpy.typing import NDArray
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.wdm_config import WDMWaveletConstants


def instrument_noise1(f: NDArray[np.floating], lc: LISAConstants) -> NDArray[np.floating]:
    # Power spectral density of the detector noise and transfer frequency
    # Sps: float = 9.0e-24  # should match sangria v2? Should it be backlinknoise or readoutnoise?
    # Sacc: float = 5.76e-30  # from sangria v2
    Sps = lc.Sps
    Sacc = lc.Sacc
    f_on_f: NDArray[np.floating] = f / lc.fstr
    # To match the LDC power spectra need a factor of 2 here. No idea why... (one sided/two sided?)
    LC: NDArray[np.floating] = 8.0 * 2.0 * f_on_f**2
    # roll-offs
    roll1 = lc.f_roll_acc_f_inv / f
    roll2 = f / lc.f_roll_acc_f
    roll3 = lc.f_roll_ps_f_inv / f
    rolla: NDArray[np.floating] = (1.0 + roll1**2) * (1.0 + roll2**4)
    rollw: NDArray[np.floating] = 1.0 + roll3**4
    scale_part: NDArray[np.floating] = LC * 16.0 / 3.0 * np.sin(f_on_f) ** 2 / (2.0 * lc.Larm) ** 2
    add_part: NDArray[np.floating] = (2.0 + np.cos(f_on_f)) * Sps * rollw + 2.0 * (3.0 + 2.0 * np.cos(f_on_f) + np.cos(2.0 * f_on_f)) * (Sacc / (2.0 * np.pi * f) ** 4 * rolla)
    # Calculate the power spectral density of the detector noise at the given frequency
    # not and exact match to the LDC, but within 10%
    return scale_part * add_part


def instrument_noise_AET(f: NDArray[np.floating], lc: LISAConstants, tdi_mode: str = 'aet_equal', diagonal_mode: int = 0) -> NDArray[np.floating]:
    """Get power spectral density in all 3 channels, assuming identical in all arms."""
    # see arXiv:2005.03610
    # see arXiv:1002.1291

    nc_aet: int = 3  # the three TDI channels

    if diagonal_mode == 0:
        S_inst: NDArray[np.floating] = np.zeros((f.size, nc_aet))
    elif diagonal_mode == 1:
        S_inst = np.zeros((f.size, nc_aet, nc_aet))
    else:
        msg = f'Unrecognized option {diagonal_mode} for diagonal mode'
        raise ValueError(msg)

    if lc.noise_curve_mode == 0:
        # standard mode
        pass
    elif lc.noise_curve_mode == 1:
        # flat unit spectrum for testing
        S_inst[:] = 1.0
        return S_inst
    else:
        msg = 'Unrecognized option for noise curve mode'
        raise ValueError(msg)

    f_on_f: NDArray[np.floating] = f / lc.fstr
    cos_f_on_f: NDArray[np.floating] = np.cos(f_on_f)
    sin_f_on_f: NDArray[np.floating] = np.sin(f_on_f)
    sin_2f_on_f: NDArray[np.floating] = np.sin(2 * f_on_f)
    cos_2f_on_f: NDArray[np.floating] = np.cos(2 * f_on_f)

    LC: float = 64 / (3 * lc.Larm**2)
    mult_all_diag: NDArray[np.floating] = LC * f_on_f**2 * sin_f_on_f ** 2
    mult_all_offdiag: NDArray[np.floating] = -LC / 4 * f_on_f**2 * sin_f_on_f * sin_2f_on_f

    # roll-offs
    roll1: NDArray[np.floating] = lc.f_roll_acc_f_inv / f
    roll2: NDArray[np.floating] = f / lc.f_roll_acc_f
    roll3: NDArray[np.floating] = lc.f_roll_ps_f_inv / f
    rolla: NDArray[np.floating] = (1.0 + roll1**2) * (1.0 + roll2**4)
    rollw: NDArray[np.floating] = 1.0 + roll3**4

    # common multipliers
    mult_sa: NDArray[np.floating] = (2 * lc.Sacc / float(2 * np.pi) ** 4) * rolla / f**4
    mult_sp: NDArray[np.floating] = lc.Sps * rollw

    # overall component scaling
    scale_part_AE_diag: NDArray[np.floating] = mult_all_diag
    scale_part_xyz_diag: NDArray[np.floating] = 3.0 / 2.0 * mult_all_diag
    scale_part_xyz_offdiag: NDArray[np.floating] = 3.0 / 2.0 * mult_all_offdiag
    scale_part_T: NDArray[np.floating] = mult_all_diag

    # different shapes
    cos_part_sa_AE = 3 + 2 * cos_f_on_f + cos_2f_on_f
    cos_part_sp_AE = 2 + cos_f_on_f
    cos_part_sa_T = 1 - 2 * cos_f_on_f + cos_f_on_f**2
    cos_part_sp_T = 1 - cos_f_on_f

    cos_part_sa_xyz_diag = 3.0 + cos_2f_on_f
    cos_part_sp_xyz_diag = 2.0

    cos_part_sa_xyz_offdiag = 4.0
    cos_part_sp_xyz_offdiag = 2.0

    # combine parts of shapes
    add_part_AE: NDArray[np.floating] = mult_sa * cos_part_sa_AE + mult_sp * cos_part_sp_AE
    add_part_T: NDArray[np.floating] = mult_sa * cos_part_sa_T + mult_sp * cos_part_sp_T
    add_part_xyz_diag: NDArray[np.floating] = mult_sa * cos_part_sa_xyz_diag + mult_sp * cos_part_sp_xyz_diag
    add_part_xyz_offdiag: NDArray[np.floating] = mult_sa * cos_part_sa_xyz_offdiag + mult_sp * cos_part_sp_xyz_offdiag

    if tdi_mode == 'aet_equal':
        # store result
        if diagonal_mode == 0:
            S_inst[:, 0] = np.abs(scale_part_AE_diag * add_part_AE)  # TODO make this all self consistent
            S_inst[:, 1] = S_inst[:, 0]
            S_inst[:, 2] = np.abs(scale_part_T * add_part_T)
        elif diagonal_mode == 1:
            S_inst[:, 0, 0] = np.abs(scale_part_AE_diag * add_part_AE)
            S_inst[:, 1, 1] = S_inst[:, 0, 0]
            S_inst[:, 2, 2] = np.abs(scale_part_T * add_part_T)
            # off diagonal components are zero for equal arm length
    elif tdi_mode == 'xyz_equal':
        if diagonal_mode == 0:
            S_inst[:, 0] = np.abs(scale_part_xyz_diag * add_part_xyz_diag)
            S_inst[:, 1] = S_inst[:, 0]
            S_inst[:, 2] = S_inst[:, 0]
            warn('Instrument noise xyz_equal has off-diagonal terms, but diagonal noise requested', stacklevel=2)
        else:
            S_inst[:, 0, 0] = np.abs(scale_part_xyz_diag * add_part_xyz_diag)
            S_inst[:, 1, 1] = S_inst[:, 0, 0]
            S_inst[:, 2, 2] = S_inst[:, 0, 0]
            S_inst[:, 0, 1] = scale_part_xyz_offdiag * add_part_xyz_offdiag
            S_inst[:, 0, 2] = S_inst[:, 0, 1]
            S_inst[:, 1, 2] = S_inst[:, 0, 1]
            # symmetric components
            S_inst[:, 1, 0] = S_inst[:, 0, 1]
            S_inst[:, 2, 0] = S_inst[:, 0, 2]
            S_inst[:, 2, 1] = S_inst[:, 1, 2]
    else:
        msg = f'Unrecognized tdi mode {tdi_mode} for instrument noise AET/XYZ'
        raise ValueError(msg)
    return S_inst


def instrument_noise_AET_wdm_loop(
    phif: NDArray[np.floating],
    lc: LISAConstants,
    wc: WDMWaveletConstants,
    tdi_mode: str = 'aet_equal',
    diagonal_mode: int = 0,
) -> NDArray[np.floating]:
    """Help get the the instrument noise in the wdm wavelet basis."""
    # realistically this really only needs run once and is fast enough without jit
    # TODO get first and last bins correct
    # nrm: float = float(np.sqrt(12318.0 / wc.Nf)) * float(np.linalg.norm(phif))

    phif2 = phif**2

    half_Nt = wc.Nt // 2
    fs_long = np.arange(-half_Nt, half_Nt + wc.Nf * half_Nt) / wc.Tobs
    # prevent division by 0
    fs_long[half_Nt] = fs_long[half_Nt + 1]
    S_inst_long = instrument_noise_AET(fs_long, lc, tdi_mode=tdi_mode, diagonal_mode=diagonal_mode)

    # excise the f=0 point
    S_inst_long[half_Nt, :] = 0.0

    if diagonal_mode == 0:
        assert len(S_inst_long.shape) == 2
        S_inst_m: NDArray[np.floating] = np.zeros((wc.Nf, S_inst_long.shape[-1]))
    elif diagonal_mode == 1:
        assert len(S_inst_long.shape) == 3
        assert S_inst_long.shape[1] == S_inst_long.shape[2]
        S_inst_m = np.zeros((wc.Nf, S_inst_long.shape[-2], S_inst_long.shape[-1]))
    else:
        msg = f'Unrecognized diagonal mode {diagonal_mode}'
        raise ValueError(msg)

    # apply window in loop
    for m in range(wc.Nf):
        S_inst_m[m] = np.sum(phif2 * S_inst_long[m * half_Nt:(m + 2) * half_Nt].T, axis=-1)

    return S_inst_m


def instrument_noise_AET_wdm_m(lc: LISAConstants, wc: WDMWaveletConstants, tdi_mode: str = 'aet_equal', diagonal_mode: int = 0) -> NDArray[np.floating]:
    """Get the tdi instrument noise curve as a function of frequency for the wdm wavelet basis.

    Parameters
    ----------
    lc : LISAConstants
        constants for LISA constellation specified in lisa_config.py
    wc : WDMWaveletConstants
        constants for WDM wavelet basis also from wdm_config.py

    Returns
    -------
    S_stat_m : numpy.ndarray (Nf x 3)
        array of the instrument noise curve for each TDI channel
        array shape is (freq. layers x number of TDI channels)

    """
    # TODO why no plus 1?
    ls: NDArray[np.integer] = np.arange(-wc.Nt // 2, wc.Nt // 2)
    fs: NDArray[np.floating] = ls / wc.Tobs
    phif: NDArray[np.floating] = phitilde_vec(2 * np.pi * fs * wc.dt, wc.Nf, wc.nx)
    nrm: float = float(np.sqrt(2 * wc.dt)) * float(np.linalg.norm(phif))
    print('nrm instrument', nrm)
    phif = phif / nrm

    return instrument_noise_AET_wdm_loop(phif, lc, wc, tdi_mode=tdi_mode, diagonal_mode=diagonal_mode)
