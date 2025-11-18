"""test comparison of signal for sangria v1 verification binaries"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import tomllib
import WDMWaveletTransforms.fft_funcs as fft
from numpy.testing import assert_allclose, assert_array_less
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import dual_annealing
from scipy.signal import butter, csd, filtfilt, hilbert, resample, welch
from WDMWaveletTransforms.transform_freq_funcs import tukey
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq, inverse_wavelet_time, transform_wavelet_freq

from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop
from LisaWaveformTools.instrument_noise import instrument_noise_AET, instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants, get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel
from tests.test_tdi_diagonal_noise_consistency import get_csd_helper, tdi_aet_to_xyz_helper, tdi_xyz_to_aet_helper
from WaveletWaveforms.wdm_config import get_wavelet_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_sangria_tdi(verification_only: int = 0, t_min: float = 0., t_max: float = np.inf) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load sangria tdi from a file."""
    full_galactic_params_filename = 'LDC/LDC2_sangria_training_v2.h5'
    sangria_verification_filename = 'LDC/LDC2_sangria_vgb-tdi_v1_sgsEVXb.h5'
    if verification_only == 0:
        hf_in = h5py.File(full_galactic_params_filename, 'r')
        hf_obs = hf_in['obs']
        assert isinstance(hf_obs, h5py.Group)
        hf_tdi = hf_obs['tdi']
        assert isinstance(hf_tdi, h5py.Dataset)
        x_tdi: NDArray[np.floating] = np.asarray(hf_tdi['X']).flatten()  # type: ignore[reportArgumentType, index]
        y_tdi: NDArray[np.floating] = np.asarray(hf_tdi['Y']).flatten()  # type: ignore[reportArgumentType, index]
        z_tdi: NDArray[np.floating] = np.asarray(hf_tdi['Z']).flatten()  # type: ignore[reportArgumentType, index]
        time_tdi: NDArray[np.floating] = np.asarray(hf_tdi['t']).flatten()  # type: ignore[reportArgumentType, index]
        hf_in.close()
    elif verification_only == 1:
        hf_in = h5py.File(sangria_verification_filename, 'r')
        assert isinstance(hf_in, h5py.Group)
        hf_x = hf_in['X']
        assert isinstance(hf_x, h5py.Dataset)
        hf_y = hf_in['Y']
        assert isinstance(hf_y, h5py.Dataset)
        hf_z = hf_in['Z']
        assert isinstance(hf_z, h5py.Dataset)
        x_tdi = np.asarray(hf_x[:, 1]).flatten()
        y_tdi = np.asarray(hf_y[:, 1]).flatten()
        z_tdi = np.asarray(hf_z[:, 1]).flatten()
        time_tdi = np.asarray(hf_z[:, 0]).flatten()
        hf_in.close()
    else:
        msg = f'Unrecognized load option {verification_only}'
        raise ValueError(msg)

    t_mask = (time_tdi >= t_min) & (time_tdi <= t_max)
    x_tdi = x_tdi[t_mask]
    y_tdi = y_tdi[t_mask]
    z_tdi = z_tdi[t_mask]
    time_tdi = time_tdi[t_mask]

    assert x_tdi.shape == y_tdi.shape
    assert x_tdi.shape == z_tdi.shape
    assert x_tdi.shape == time_tdi.shape

    return np.array([x_tdi, y_tdi, z_tdi]), time_tdi


def parseval_rfft(sig_freq: NDArray[np.floating] | NDArray[np.complexfloating], ND: int) -> float:
    """Sum for parseval theorem in freq domain"""
    if ND % 2:
        pars2 = 1 / ND * (np.abs(sig_freq[0])**2 + 2 * np.sum(np.abs(sig_freq[1:ND // 2 + 1])**2))
    else:
        pars2 = 1 / ND * (np.abs(sig_freq[0])**2 + np.abs(sig_freq[ND // 2])**2 + 2 * np.sum(np.abs(sig_freq[1:ND // 2])**2))
    return pars2


def parseval_time(sig_time: NDArray[np.floating]) -> float:
    """Sum for parseval theorem in time domain"""
    return np.sum(np.abs(sig_time)**2)


def parseval_wavelet(wave: NDArray[np.floating]) -> float:
    """Sum for parseval theorem in wavelet domain"""
    return np.sum(np.abs(wave)**2)


def match_sangria_curve(toml_filename_in: str, fcsd: NDArray[np.floating], psd_aet: NDArray[np.floating], psd_xyz: NDArray[np.floating], re_csd_xyz: NDArray[np.floating], nf_min: int, nf_max: int, f_mask_low: float = 1.e-4, f_mask_high: float = 2.4e-2) -> LISAConstants:
    with Path(toml_filename_in).open('rb') as f:
        config_in3 = tomllib.load(f)

    log_f = np.linspace(np.log10(fcsd[0]), np.log10(fcsd[-1] * 0.88), 1000)
    f_log_space = 10**log_f
    mask_spect = ((f_log_space < f_mask_low) | (f_log_space > f_mask_high))
    log_f_masked = log_f[mask_spect]
    f_log_space_masked = f_log_space[mask_spect]
    log_psd_goal_A = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[0, nf_min:nf_max]), k=3, ext=2)(log_f_masked)
    log_psd_goal_E = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[1, nf_min:nf_max]), k=3, ext=2)(log_f_masked)
    log_psd_goal_T = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[2, nf_min:nf_max]), k=3, ext=2)(log_f)
    psd_goal_01 = InterpolatedUnivariateSpline(np.log10(fcsd), re_csd_xyz[:, 0, 1], k=3, ext=2)(log_f_masked)
    psd_goal_02 = InterpolatedUnivariateSpline(np.log10(fcsd), re_csd_xyz[:, 0, 2], k=3, ext=2)(log_f_masked)
    psd_goal_12 = InterpolatedUnivariateSpline(np.log10(fcsd), re_csd_xyz[:, 1, 2], k=3, ext=2)(log_f_masked)
    log_psd_goal_00 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[0, nf_min:nf_max])), k=3, ext=2)(log_f_masked)
    log_psd_goal_11 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[1, nf_min:nf_max])), k=3, ext=2)(log_f_masked)
    log_psd_goal_22 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[2, nf_min:nf_max])), k=3, ext=2)(log_f_masked)

    def S_func_temp(tpl: NDArray[np.floating]) -> float:
        config_in3['lisa_constants']['Sps'] = 10**tpl[0]
        config_in3['lisa_constants']['Sacc'] = 10**tpl[1]
        config_in3['lisa_constants']['f_roll_acc_f2_inv'] = tpl[2]
        config_in3['lisa_constants']['f_roll_acc_f4'] = tpl[3]
        config_in3['lisa_constants']['f_roll_ps_f4_inv'] = tpl[4]
        # config_in3['lisa_constants']['f_roll_acc_f_inv'] = tpl[5]
        lc3 = get_lisa_constants(config_in3)
        psd_xyz_expect3 = instrument_noise_AET(f_log_space_masked, lc3, tdi_mode='xyz_equal', diagonal_mode=1)
        psd_aet_expect3 = instrument_noise_AET(f_log_space_masked, lc3, tdi_mode='aet_equal', diagonal_mode=0)
        contrib_A = np.linalg.norm(np.log10(psd_aet_expect3[mask_spect, 0]) - log_psd_goal_A)
        contrib_E = np.linalg.norm(np.log10(psd_aet_expect3[mask_spect, 1]) - log_psd_goal_E)
        contrib_T = np.linalg.norm(np.log10(psd_aet_expect3[f_log_space_masked > 1.e-4, 2]) - log_psd_goal_T[f_log_space_masked > 1.e-4])
        # contrib_T = 0.
        contrib_01 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 1])) - np.log10(np.abs(psd_goal_01)))
        contrib_02 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 2])) - np.log10(np.abs(psd_goal_02)))
        contrib_12 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 1, 2])) - np.log10(np.abs(psd_goal_12)))
        contrib_00 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 0])) - log_psd_goal_00)
        contrib_11 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 1, 1])) - log_psd_goal_11)
        contrib_22 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 2, 2])) - log_psd_goal_22)
        return np.sqrt(contrib_00**2 + contrib_11**2 + contrib_22**2 + contrib_01**2 + contrib_02**2 + contrib_12**2 + contrib_A**2 + contrib_E**2 + contrib_T**2)

    bounds = np.zeros((5, 2))
    bounds[0, 0] = np.log10(1.e-3 * 2.25e-22)
    bounds[0, 1] = np.log10(1.e3 * 2.25e-22)
    bounds[1, 0] = np.log10(1.e-3 * 9.0e-30)
    bounds[1, 1] = np.log10(1.e3 * 9.0e-30)
    bounds[2, 0] = 0.0
    bounds[2, 1] = 1.0e-3
    bounds[3, 0] = 4.0e-3
    bounds[3, 1] = 1.6e-2
    bounds[4, 0] = 1.0e-3
    bounds[4, 1] = 4.0e-3
    # bounds[5, 0] = 0.0
    # bounds[5, 1] = 1.0e-3

    res_found = dual_annealing(S_func_temp, bounds, maxiter=5000)
    res = res_found['x']
    print(res_found)
    config_in3['lisa_constants']['Sps'] = 10**res[0]
    config_in3['lisa_constants']['Sacc'] = 10**res[1]
    config_in3['lisa_constants']['f_roll_acc_f2_inv'] = res[2]
    config_in3['lisa_constants']['f_roll_acc_f4'] = res[3]
    config_in3['lisa_constants']['f_roll_ps_f4_inv'] = res[4]
    # config_in3['lisa_constants']['f_roll_acc_f_inv'] = res[5]
    lc3 = get_lisa_constants(config_in3)
    psd_xyz_expect3 = instrument_noise_AET(f_log_space_masked, lc3, tdi_mode='xyz_equal', diagonal_mode=1)
    psd_aet_expect3 = instrument_noise_AET(f_log_space_masked, lc3, tdi_mode='aet_equal', diagonal_mode=0)
    plt.loglog(f_log_space_masked, psd_xyz_expect3[:, 0, 0])
    plt.loglog(f_log_space_masked, 10**log_psd_goal_00)
    plt.loglog(f_log_space_masked, np.abs(psd_xyz_expect3[:, 0, 1]))
    plt.loglog(f_log_space_masked, np.abs(psd_goal_01))
    plt.loglog(f_log_space_masked, np.abs(psd_xyz_expect3[:, 0, 2]))
    plt.loglog(f_log_space_masked, np.abs(psd_goal_02))
    plt.loglog(f_log_space_masked, np.abs(psd_xyz_expect3[:, 1, 2]))
    plt.loglog(f_log_space_masked, np.abs(psd_goal_12))
    plt.show()
    plt.semilogx(f_log_space_masked, psd_xyz_expect3[:, 0, 1] / np.sqrt(psd_xyz_expect3[:, 0, 0] * psd_xyz_expect3[:, 1, 1]))
    plt.semilogx(f_log_space_masked, psd_goal_01 / np.sqrt(psd_xyz_expect3[:, 0, 0] * psd_xyz_expect3[:, 1, 1]))
    plt.semilogx(f_log_space_masked, psd_xyz_expect3[:, 0, 2] / np.sqrt(psd_xyz_expect3[:, 0, 0] * psd_xyz_expect3[:, 2, 2]))
    plt.semilogx(f_log_space_masked, psd_goal_02 / np.sqrt(psd_xyz_expect3[:, 0, 0] * psd_xyz_expect3[:, 2, 2]))
    plt.semilogx(f_log_space_masked, psd_xyz_expect3[:, 1, 2] / np.sqrt(psd_xyz_expect3[:, 1, 1] * psd_xyz_expect3[:, 2, 2]))
    plt.semilogx(f_log_space_masked, psd_goal_12 / np.sqrt(psd_xyz_expect3[:, 1, 1] * psd_xyz_expect3[:, 2, 2]))
    plt.show()
    plt.loglog(f_log_space_masked, psd_aet_expect3[:, 0])
    plt.loglog(f_log_space_masked, 10**log_psd_goal_A)
    plt.loglog(f_log_space_masked, psd_aet_expect3[:, 1])
    plt.loglog(f_log_space_masked, 10**log_psd_goal_E)
    plt.loglog(f_log_space_masked, psd_aet_expect3[:, 2])
    plt.loglog(f_log_space_masked, 10**log_psd_goal_T)
    plt.show()
    return lc3


# def match_sangria_curve(toml_filename_in: str, fcsd: NDArray[np.floating], psd_aet: NDArray[np.floating], psd_xyz: NDArray[np.floating], re_csd_xyz: NDArray[np.floating], nf_min: int, nf_max: int, f_mask_low: float = 1.e-4, f_mask_high: float = 2.4e-2) -> LISAConstants:
#    with Path(toml_filename_in).open('rb') as f:
#        config_in3 = tomllib.load(f)
#
#    log_f = np.linspace(np.log10(fcsd[0]), np.log10(fcsd[-1] * 0.92), 1000)
#    f_log_space = 10**log_f
#    log_psd_goal_T = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[2, nf_min:nf_max]), k=3, ext=2)(log_f)
#    mask_spect = ((f_log_space < f_mask_low) | (f_log_space > f_mask_high))
#    log_f_masked = log_f[mask_spect]
#    f_log_space_masked = f_log_space[mask_spect]
#    log_psd_goal_A = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[0, nf_min:nf_max]), k=3, ext=2)(log_f_masked)
#    log_psd_goal_E = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[1, nf_min:nf_max]), k=3, ext=2)(log_f_masked)
#    log_psd_goal_01 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(re_csd_xyz[:, 0, 1])), k=3, ext=2)(log_f_masked)
#    log_psd_goal_02 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(re_csd_xyz[:, 0, 2])), k=3, ext=2)(log_f_masked)
#    log_psd_goal_12 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(re_csd_xyz[:, 1, 2])), k=3, ext=2)(log_f_masked)
#    log_psd_goal_00 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[0, nf_min:nf_max])), k=3, ext=2)(log_f_masked)
#    log_psd_goal_11 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[1, nf_min:nf_max])), k=3, ext=2)(log_f_masked)
#    log_psd_goal_22 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[2, nf_min:nf_max])), k=3, ext=2)(log_f_masked)
#
#    def S_func_temp(tpl: NDArray[np.floating]) -> float:
#        config_in3['lisa_constants']['Sps'] = 10**tpl[0]
#        config_in3['lisa_constants']['Sacc'] = 10**tpl[1]
#        config_in3['lisa_constants']['f_roll_acc_f2_inv'] = tpl[2]
#        config_in3['lisa_constants']['f_roll_acc_f4'] = tpl[3]
#        config_in3['lisa_constants']['f_roll_ps_f4_inv'] = tpl[4]
#        lc3 = get_lisa_constants(config_in3)
#        psd_aet_expect3 = instrument_noise_AET(f_log_space, lc3, tdi_mode='aet_equal', diagonal_mode=0)
#        psd_xyz_expect3 = instrument_noise_AET(f_log_space_masked, lc3, tdi_mode='xyz_equal', diagonal_mode=1)
#        contrib_T = np.linalg.norm(np.log10(psd_aet_expect3[:, 2]) - log_psd_goal_T)
#        contrib_A = np.linalg.norm(np.log10(psd_aet_expect3[mask_spect, 0]) - log_psd_goal_A)
#        contrib_E = np.linalg.norm(np.log10(psd_aet_expect3[mask_spect, 1]) - log_psd_goal_E)
#        contrib_01 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 1])) - log_psd_goal_01)
#        contrib_02 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 2])) - log_psd_goal_02)
#        contrib_12 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 1, 2])) - log_psd_goal_12)
#        contrib_00 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 0])) - log_psd_goal_00)
#        contrib_11 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 1, 1])) - log_psd_goal_11)
#        contrib_22 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 2, 2])) - log_psd_goal_22)
#        return np.sqrt(contrib_A**2 + contrib_E**2 + contrib_T**2 + contrib_01**2 + contrib_02**2 + contrib_12**2 + contrib_00**2 + contrib_11**2 + contrib_22**2)
#
#    bounds = np.zeros((5, 2))
#    bounds[0, 0] = np.log10(1.e-3 * 9.0e-24)
#    bounds[0, 1] = np.log10(1.e3 * 9.0e-24)
#    bounds[1, 0] = np.log10(1.e-3 * 5.76e-30)
#    bounds[1, 1] = np.log10(1.e3 * 5.76e-30)
#    bounds[2, 0] = 2.0e-4
#    bounds[2, 1] = 8.0e-4
#    bounds[3, 0] = 4.0e-3
#    bounds[3, 1] = 1.6e-2
#    bounds[4, 0] = 1.0e-3
#    bounds[4, 1] = 4.0e-3
#
#    res_found = dual_annealing(S_func_temp, bounds, maxiter=5000)
#    res = res_found['x']
#    print(res_found)
#    config_in3['lisa_constants']['Sps'] = 10**res[0]
#    config_in3['lisa_constants']['Sacc'] = 10**res[1]
#    config_in3['lisa_constants']['f_roll_acc_f2_inv'] = res[2]
#    config_in3['lisa_constants']['f_roll_acc_f4'] = res[3]
#    config_in3['lisa_constants']['f_roll_ps_f4_inv'] = res[4]
#    lc3 = get_lisa_constants(config_in3)
#    return lc3


if __name__ == '__main__':
    # toml_filename_in = 'Galaxies/GalaxyFullLDC/run_old_parameters.toml'
    verification_only = False
    if verification_only:
        toml_filename_in = 'Galaxies/GalaxyVerification2LDC/run_verification_parameters.toml'
    else:
        toml_filename_in = 'Galaxies/GalaxyFullLDC/run_match_parameters_fit2.toml'
        toml_filename_in2 = 'parameters_5m.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    with Path(toml_filename_in2).open('rb') as f:
        config_in2 = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    wc2 = get_wavelet_model(config_in2)
    lc2 = get_lisa_constants(config_in2)

    ts_bg = np.arange(0., wc.Nt * wc.Nf) * wc.dt

    xyz_tdi_time, time_tdi = load_sangria_tdi(verification_only=verification_only, t_max=ts_bg[-1])
    n_samples_in = time_tdi.size
    dt_in = time_tdi[1] - time_tdi[0]
    assert_allclose(np.diff(time_tdi), dt_in, atol=1.e-100, rtol=1.e-14)

    AET_tdi_time = tdi_xyz_to_aet_helper(xyz_tdi_time, axis=0)

    cyclo_mode = 1

    # get the computed total galactic background
    ifm = fetch_or_run_iterative_loop(config_in, cyclo_mode=cyclo_mode, fetch_mode=3, output_mode=0, preprocess_mode=1)
    galactic_bg = ifm.noise_manager.bgd.get_galactic_total(shape_mode=1)
    snrs_got = ifm.bis.get_final_snrs_tot_upper()

    SAET_m = instrument_noise_AET_wdm_m(lc2, wc)
    noise_AET_dense_pure = DiagonalStationaryDenseNoiseModel(SAET_m, wc, prune=0, nc_snr=lc.nc_snr)
    if not verification_only:
        galactic_bg += noise_AET_dense_pure.generate_dense_noise()

    AET_tdi_time_rec_temp = np.zeros((galactic_bg.shape[-1], wc.Nt * wc.Nf))
    for itrc in range(galactic_bg.shape[-1]):
        AET_tdi_time_rec_temp[itrc, :] = inverse_wavelet_time(galactic_bg[:, :, itrc], wc.Nf, wc.Nt)

    assert ts_bg[-1] >= time_tdi[-1]
    nt_full_max = int(np.argmax(ts_bg >= time_tdi[-1])) + 1
    AET_tdi_time_rec = np.zeros((galactic_bg.shape[-1], n_samples_in))
    for itrc in range(galactic_bg.shape[-1]):
        if np.isclose(wc.dt, dt_in, atol=1.e-100, rtol=1.e-14):
            AET_tdi_time_rec[itrc, :] = AET_tdi_time_rec_temp[itrc, :nt_full_max]
        else:
            AET_tdi_time_rec[itrc, :], t_tdi_mbh_alt = resample(AET_tdi_time_rec_temp[itrc, :nt_full_max], num=n_samples_in, t=ts_bg[:nt_full_max])
            assert_allclose(t_tdi_mbh_alt, time_tdi, atol=1.e-100, rtol=1.e-14)

    AET_tdi_freq_rec = np.fft.rfft(AET_tdi_time_rec, axis=-1)
    AET_tdi_freq = np.fft.rfft(AET_tdi_time, axis=-1)
    AET_tdi_freq_resid = AET_tdi_freq - AET_tdi_freq_rec

    fs_fft = np.arange(0, (wc.Nt * wc.Nf) // 2 + 1) / (wc.dt * wc.Nt * wc.Nf)
    psd_aet_expect_full = instrument_noise_AET(fs_fft[1:], lc2, tdi_mode='aet_equal', diagonal_mode=1)
    plt.loglog(fs_fft, np.abs(AET_tdi_freq_rec[1]))
    plt.loglog(fs_fft, np.abs(AET_tdi_freq[1]))
    plt.loglog(fs_fft, np.abs(AET_tdi_freq_resid[1]))
    plt.loglog(fs_fft[1:], np.sqrt(psd_aet_expect_full[:, 1, 1] * wc.Nf * wc.Nf / wc.dt / 2))
    plt.plot([wc.DF, wc.DF], [0., 1.])
    plt.plot([1. / wc.Tw, 1. / wc.Tw], [0., 1.])
    plt.show()

    fs = 1.0 / dt_in
    nperseg = int(np.round(((1. / dt_in) / (1. / wc.dt)) * wc.Nf))
    fpsd, psd_aet_rec = welch(AET_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    fpsd, psd_aet = welch(AET_tdi_time, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    fpsd, psd_aet_resid = welch(AET_tdi_time - AET_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)

    ND = AET_tdi_time_rec.shape[1]
    nt_max = int(ND // wc.Nf)
    assert nt_max * wc.Nf == ND

    t0 = perf_counter()

    tf = perf_counter()
    print('got inverse in ', tf - t0, 's')

    xyz_tdi_freq_rec = tdi_aet_to_xyz_helper(AET_tdi_freq_rec, axis=0)

    xyz_tdi_time_rec = tdi_aet_to_xyz_helper(AET_tdi_time_rec, axis=0)

    xyz_tdi_time = xyz_tdi_time[:, :ND]

    xyz_tdi_freq = fft.rfft(xyz_tdi_time, axis=-1)
    time_tdi = time_tdi[:ND]

    fpsd, psd_xyz_rec = welch(xyz_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    fpsd, psd_xyz = welch(xyz_tdi_time, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    fpsd, psd_xyz_resid = welch(xyz_tdi_time - xyz_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)

    nf_min = max(1, int(np.argmax(fpsd > 4. / wc.DT)), int(np.argmax(fpsd > 4. / wc.Tw)))
    nf_max = fpsd.size - 2

    fcsd, csd_xyz = get_csd_helper(xyz_tdi_time, fs, nperseg, nf_min, nf_max, axis=0)
    fcsd, csd_aet = get_csd_helper(AET_tdi_time, fs, nperseg, nf_min, nf_max, axis=0)

    fcsd, csd_xyz_rec = get_csd_helper(xyz_tdi_time_rec, fs, nperseg, nf_min, nf_max, axis=0)
    fcsd, csd_aet_rec = get_csd_helper(AET_tdi_time_rec, fs, nperseg, nf_min, nf_max, axis=0)

    _fcsd, xyz_cross_csd = csd(xyz_tdi_time, xyz_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    xyz_cross_csd = xyz_cross_csd[:, nf_min:nf_max]

    _fcsd, aet_cross_csd = csd(AET_tdi_time, AET_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    aet_cross_csd = aet_cross_csd[:, nf_min:nf_max]

    xyz_cross_cohere = np.zeros(((nf_max - nf_min), lc.nc_snr))
    aet_cross_cohere = np.zeros(((nf_max - nf_min), lc.nc_snr))
    for itrc in range(lc.nc_snr):
        xyz_cross_cohere[:, itrc] = np.abs(xyz_cross_csd[itrc])**2 / (psd_xyz_rec[itrc, nf_min:nf_max] * psd_xyz[itrc, nf_min:nf_max])
        aet_cross_cohere[:, itrc] = np.abs(aet_cross_csd[itrc])**2 / (psd_aet_rec[itrc, nf_min:nf_max] * psd_aet[itrc, nf_min:nf_max])

    re_csd_xyz = np.real(csd_xyz)
    im_csd_xyz = np.imag(csd_xyz)
    angle_csd_xyz = np.angle(csd_xyz)
    abs_csd_xyz = np.abs(csd_xyz)

    re_csd_aet = np.real(csd_aet)
    im_csd_aet = np.imag(csd_aet)
    angle_csd_aet = np.angle(csd_aet)
    abs_csd_aet = np.abs(csd_aet)

    re_csd_xyz_rec = np.real(csd_xyz_rec)
    im_csd_xyz_rec = np.imag(csd_xyz_rec)
    angle_csd_xyz_rec = np.angle(csd_xyz_rec)
    abs_csd_xyz_rec = np.abs(csd_xyz_rec)

    re_csd_aet_rec = np.real(csd_aet_rec)
    im_csd_aet_rec = np.imag(csd_aet_rec)
    angle_csd_aet_rec = np.angle(csd_aet_rec)
    abs_csd_aet_rec = np.abs(csd_aet_rec)

    psd_xyz_expect = instrument_noise_AET(fpsd, lc, tdi_mode='xyz_equal', diagonal_mode=1)
    psd_xyz_expect2 = instrument_noise_AET(fpsd, lc2, tdi_mode='xyz_equal', diagonal_mode=1)

    psd_aet_expect = instrument_noise_AET(fpsd, lc, tdi_mode='aet_equal', diagonal_mode=1)
    psd_aet_expect2 = instrument_noise_AET(fpsd, lc2, tdi_mode='aet_equal', diagonal_mode=1)

    do_match = False
    if do_match:
        lc3 = match_sangria_curve(toml_filename_in, fcsd, psd_aet, psd_xyz, re_csd_xyz, nf_min, nf_max)
    else:
        lc3 = lc

    psd_aet_expect3 = instrument_noise_AET(fpsd, lc3, tdi_mode='aet_equal', diagonal_mode=1)
    psd_xyz_expect3 = instrument_noise_AET(fpsd, lc3, tdi_mode='xyz_equal', diagonal_mode=1)

    plt.loglog(fpsd, psd_aet_rec[0])
    plt.loglog(fpsd, psd_aet[0])
    plt.loglog(fpsd, psd_aet_expect[:, 0, 0])
    plt.loglog(fpsd, psd_aet_expect2[:, 0, 0])
    plt.loglog(fpsd, psd_aet_resid[0])
    plt.title('A channel comparison')
    plt.show()

    # plt.loglog(fpsd, psd_aet_rec[2])
    # plt.loglog(fpsd, instrument_noise_AET(fpsd, lc)[:,2])
    plt.loglog(fpsd, psd_aet_rec[2])
    plt.loglog(fpsd, psd_aet[2])
    plt.loglog(fpsd, psd_aet_expect[:, 2, 2])
    plt.loglog(fpsd, psd_aet_expect2[:, 2, 2])
    plt.loglog(fpsd, psd_aet_resid[2])
    plt.title('T channel comparison')
    plt.show()

    plt.loglog(fpsd, np.mean(psd_xyz, axis=0))
    plt.loglog(fpsd, np.mean(psd_xyz_rec, axis=0))
    plt.loglog(fpsd, psd_xyz_expect[:, 0, 0])
    plt.loglog(fpsd, psd_xyz_expect2[:, 0, 0])
    plt.loglog(fpsd, np.mean(psd_xyz_resid, axis=0))
    plt.title('xyz spectrum')
    plt.show()

    plt.loglog(fcsd, np.abs(abs_csd_xyz[:, 0, 1]))
    plt.loglog(fcsd, np.abs(abs_csd_xyz[:, 0, 2]))
    plt.loglog(fcsd, np.abs(abs_csd_xyz[:, 1, 2]))
    plt.loglog(fcsd, np.abs(abs_csd_xyz_rec[:, 0, 1]))
    plt.loglog(fcsd, np.abs(abs_csd_xyz_rec[:, 0, 2]))
    plt.loglog(fcsd, np.abs(abs_csd_xyz_rec[:, 1, 2]))
    plt.loglog(fpsd, np.abs(psd_xyz_expect[:, 0, 1]))
    plt.loglog(fpsd, np.abs(psd_xyz_expect3[:, 0, 1]))
    plt.loglog(fpsd, np.abs(psd_xyz_expect2[:, 0, 1]))
    plt.title('xyz cross spectrum')
    plt.show()

    plt.loglog(fcsd, np.abs(re_csd_aet[:, 0, 1]))
    plt.loglog(fcsd, np.abs(re_csd_aet[:, 0, 2]))
    plt.loglog(fcsd, np.abs(re_csd_aet[:, 1, 2]))
    plt.loglog(fcsd, np.abs(re_csd_aet_rec[:, 0, 1]))
    plt.loglog(fcsd, np.abs(re_csd_aet_rec[:, 0, 2]))
    plt.loglog(fcsd, np.abs(re_csd_aet_rec[:, 1, 2]))
    plt.loglog(fpsd, np.abs(psd_aet_expect[:, 0, 1]))
    plt.loglog(fpsd, np.abs(psd_aet_expect3[:, 0, 1]))
    plt.loglog(fpsd, np.abs(psd_aet_expect2[:, 0, 1]))
    plt.title('aet cross spectrum')
    plt.show()

    import sys
    sys.exit()

    coh_xyz = np.zeros_like(re_csd_aet)
    coh_aet = np.zeros_like(re_csd_aet)
    coh_xyz_rec = np.zeros_like(re_csd_aet)
    coh_aet_rec = np.zeros_like(re_csd_aet)
    for itr1 in range(lc.nc_snr - 1):
        for itr2 in range(itr1 + 1, lc.nc_snr):
            coh_xyz[:, itr1, itr2] = np.abs(csd_xyz[:, itr1, itr2])**2 / (psd_xyz[itr1, nf_min:nf_max] * psd_xyz[itr2, nf_min:nf_max])
            coh_xyz_rec[:, itr1, itr2] = np.abs(csd_xyz_rec[:, itr1, itr2])**2 / (psd_xyz_rec[itr1, nf_min:nf_max] * psd_xyz_rec[itr2, nf_min:nf_max])
            coh_aet[:, itr1, itr2] = np.abs(csd_aet[:, itr1, itr2])**2 / (psd_aet[itr1, nf_min:nf_max] * psd_aet[itr2, nf_min:nf_max])
            coh_aet_rec[:, itr1, itr2] = np.abs(csd_aet_rec[:, itr1, itr2])**2 / (psd_aet_rec[itr1, nf_min:nf_max] * psd_aet_rec[itr2, nf_min:nf_max])

    coh_xyz_min = np.min(np.array([coh_xyz, coh_xyz_rec]), axis=0)
    coh_aet_min = np.min(np.array([coh_aet, coh_aet_rec]), axis=0)
    coh_xyz_min_overall = np.min(np.array([coh_xyz_min[:, 0, 1], coh_xyz_min[:, 0, 2], coh_xyz_min[:, 1, 2]]), axis=0)
    coh_aet_min_overall = np.min(np.array([coh_aet_min[:, 0, 1], coh_aet_min[:, 0, 2], coh_aet_min[:, 1, 2]]), axis=0)
    coh_xyz_min_mask = 1. * (coh_xyz_min > 0.5)
    coh_aet_min_mask = 1. * (coh_aet_min > 0.5)
    nf_min_show = np.argmin(np.abs(0.95 * ifm.bis.params_gb[:, 3].min() - fcsd))
    nf_max_show = np.argmin(np.abs(1.05 * ifm.bis.params_gb[:, 3].max() - fcsd))

    # test expected doppler modulation of one of the signals
    idx_f_select = int(np.argmin(np.abs(ifm.bis.params_gb[:, 3] - 0.02146691497723561)))
    f_select = ifm.bis.params_gb[idx_f_select, 3]
    cos_select = np.cos(2 * np.pi * f_select * time_tdi)
    filter_band = (f_select * 0.99, f_select * 1.01)
    b_band, a_band = butter(4, filter_band, fs=1.0 / dt_in, btype='bandpass', analog=False)

    aet_tdi_time_filter = AET_tdi_time.copy()
    aet_tdi_time_filter_rec = AET_tdi_time_rec.copy()

    tukey_alpha = 0.05
    tukey(aet_tdi_time_filter.T, tukey_alpha, aet_tdi_time_filter.shape[-1])
    tukey(aet_tdi_time_filter_rec.T, tukey_alpha, aet_tdi_time_filter_rec.shape[-1])

    aet_tdi_time_filter = filtfilt(b_band, a_band, aet_tdi_time_filter, axis=-1)
    aet_tdi_time_filter_rec = filtfilt(b_band, a_band, aet_tdi_time_filter_rec, axis=-1)
    # multiply by cos of test signal
    aet_select_proj = aet_tdi_time_filter * cos_select
    aet_select_proj_rec = aet_tdi_time_filter_rec * cos_select
# low pass filter multiplied signals

    f_lowpass = 3.e-5
    b, a = butter(4, f_lowpass, fs=1.0 / dt_in, btype='lowpass', analog=False)
    aet_select_proj = filtfilt(b, a, aet_select_proj, axis=-1)
    aet_select_proj_rec = filtfilt(b, a, aet_select_proj_rec, axis=-1)

    aet_select_proj_analytic = hilbert(aet_select_proj, axis=1)
    aet_select_proj_analytic_rec = hilbert(aet_select_proj_rec, axis=1)

    aet_proj_angle = np.unwrap(np.angle(aet_select_proj_analytic), axis=1)
    aet_proj_angle_rec = np.unwrap(np.angle(aet_select_proj_analytic_rec), axis=1)

    aet_proj_dangle = aet_proj_angle - aet_proj_angle_rec
    plt.plot(aet_proj_dangle.T)
    plt.show()
    tukey(aet_proj_dangle.T, tukey_alpha, aet_proj_dangle.shape[-1])
    aet_proj_dangle = filtfilt(b, a, aet_proj_dangle, axis=-1)

# tukey(aet_proj_angle.T, tukey_alpha, aet_proj_angle.shape[-1])
# tukey(aet_proj_angle_rec.T, tukey_alpha, aet_proj_angle_rec.shape[-1])
    aet_proj_angle = filtfilt(b, a, aet_proj_angle, axis=-1)
    aet_proj_angle_rec = filtfilt(b, a, aet_proj_angle_rec, axis=-1)

    aet_proj_mag = np.abs(aet_select_proj_analytic)
    aet_proj_mag_rec = np.abs(aet_select_proj_analytic_rec)
# tukey(aet_proj_mag.T, tukey_alpha, aet_proj_mag.shape[-1])
# tukey(aet_proj_mag_rec.T, tukey_alpha, aet_proj_mag_rec.shape[-1])
    aet_proj_mag = filtfilt(b, a, aet_proj_mag, axis=-1)
    aet_proj_mag_rec = filtfilt(b, a, aet_proj_mag_rec, axis=-1)

    _, psd_aet_proj = welch(aet_select_proj, fs=fs, nperseg=nperseg * 8, scaling='density', window='tukey', axis=-1)
    _, psd_aet_proj_rec = welch(aet_select_proj_rec, fs=fs, nperseg=nperseg * 8, scaling='density', window='tukey', axis=-1)

    plt.semilogx(fcsd[nf_min_show:nf_max_show], coh_xyz[nf_min_show:nf_max_show, 0, 1])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], coh_xyz[nf_min_show:nf_max_show, 0, 2])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], coh_xyz[nf_min_show:nf_max_show, 1, 2])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], coh_xyz_min[nf_min_show:nf_max_show, 0, 1])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], coh_xyz_min[nf_min_show:nf_max_show, 0, 2])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], coh_xyz_min[nf_min_show:nf_max_show, 1, 2])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], coh_xyz_min_overall[nf_min_show:nf_max_show])
    plt.title('xyz coherence')
    plt.show()

    plt.semilogx(fcsd, np.unwrap(angle_csd_xyz[:, 0, 1]))
    plt.semilogx(fcsd, np.unwrap(angle_csd_xyz[:, 0, 2]))
    plt.semilogx(fcsd, np.unwrap(angle_csd_xyz[:, 1, 2]))
    plt.semilogx(fcsd, np.unwrap(angle_csd_xyz_rec[:, 0, 1]))
    plt.semilogx(fcsd, np.unwrap(angle_csd_xyz_rec[:, 0, 2]))
    plt.semilogx(fcsd, np.unwrap(angle_csd_xyz_rec[:, 1, 2]))
    plt.title('xyz cross spectrum angle')
    plt.show()

    plt.semilogx(fcsd[nf_min_show:nf_max_show], xyz_cross_cohere[nf_min_show:nf_max_show])
    plt.title('xyz cross method coherence')
    plt.show()

    plt.semilogx(fcsd[nf_min_show:nf_max_show], aet_cross_cohere[nf_min_show:nf_max_show])
    plt.title('aet cross method coherence')
    plt.show()

    plt.semilogx(fcsd[nf_min_show:nf_max_show], np.unwrap(angle_csd_xyz[nf_min_show:nf_max_show, 0, 1]) - np.unwrap(angle_csd_xyz_rec[nf_min_show:nf_max_show, 0, 1]))
    plt.semilogx(fcsd[nf_min_show:nf_max_show], np.unwrap(angle_csd_xyz[nf_min_show:nf_max_show, 0, 2]) - np.unwrap(angle_csd_xyz_rec[nf_min_show:nf_max_show, 0, 2]))
    plt.semilogx(fcsd[nf_min_show:nf_max_show], np.unwrap(angle_csd_xyz[nf_min_show:nf_max_show, 1, 2]) - np.unwrap(angle_csd_xyz_rec[nf_min_show:nf_max_show, 1, 2]))
    plt.show()

    plt.semilogx(fcsd[nf_min_show:nf_max_show], (np.unwrap(angle_csd_xyz[nf_min_show:nf_max_show, 0, 1]) - np.unwrap(angle_csd_xyz_rec[nf_min_show:nf_max_show, 0, 1])) * coh_xyz_min_mask[nf_min_show:nf_max_show, 0, 1])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], (np.unwrap(angle_csd_xyz[nf_min_show:nf_max_show, 0, 2]) - np.unwrap(angle_csd_xyz_rec[nf_min_show:nf_max_show, 0, 2])) * coh_xyz_min_mask[nf_min_show:nf_max_show, 0, 2])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], (np.unwrap(angle_csd_xyz[nf_min_show:nf_max_show, 1, 2]) - np.unwrap(angle_csd_xyz_rec[nf_min_show:nf_max_show, 1, 2])) * coh_xyz_min_mask[nf_min_show:nf_max_show, 1, 2])
    plt.show()

    angle_diff_xyz = ((np.unwrap(angle_csd_xyz[nf_min_show:nf_max_show]) - np.unwrap(angle_csd_xyz_rec[nf_min_show:nf_max_show])) % (2 * np.pi) + np.pi) % (2 * np.pi) - np.pi
    angle_diff_xyz_mask = angle_diff_xyz * coh_xyz_min_mask[nf_min_show:nf_max_show]

    angle_diff_aet = ((np.unwrap(angle_csd_aet[nf_min_show:nf_max_show]) - np.unwrap(angle_csd_aet_rec[nf_min_show:nf_max_show])) % (2 * np.pi) + np.pi) % (2 * np.pi) - np.pi
    angle_diff_aet_mask = angle_diff_aet * coh_aet_min_mask[nf_min_show:nf_max_show]

    angle_aet_cross = np.unwrap(np.angle(aet_cross_csd[:, nf_min_show:nf_max_show].T))
    angle_xyz_cross = np.unwrap(np.angle(xyz_cross_csd[:, nf_min_show:nf_max_show].T))

    # there might be some frequency dependent offset but it should be the same in all channels
    # to test this, use a smooth monotonic mean angle
    angle_aet_mean = np.maximum.accumulate(scipy.ndimage.gaussian_filter(np.mean(angle_aet_cross[:, :2], axis=1), sigma=2))
    angle_xyz_mean = np.maximum.accumulate(scipy.ndimage.gaussian_filter(np.mean(angle_xyz_cross, axis=1), sigma=2))
    dangle_aet_cross = (angle_aet_cross.T - angle_aet_mean).T
    dangle_xyz_cross = (angle_xyz_cross.T - angle_xyz_mean).T

    plt.semilogx(fcsd[nf_min_show:nf_max_show], angle_diff_xyz[:, 0, 1])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], angle_diff_xyz[:, 0, 2])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], angle_diff_xyz[:, 1, 2])
    plt.title('xyz cross spectrum angle diff')
    plt.show()

    plt.semilogx(fcsd[nf_min_show:nf_max_show], angle_diff_aet_mask[:, 0, 1])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], angle_diff_aet_mask[:, 0, 2])
    plt.semilogx(fcsd[nf_min_show:nf_max_show], angle_diff_aet_mask[:, 1, 2])
    plt.title('aet cross spectrum angle diff')
    plt.show()

    plt.semilogx(fcsd[nf_min_show:nf_max_show], dangle_aet_cross)
    plt.title('aet method delta angle')
    plt.show()

    plt.semilogx(fcsd[nf_min_show:nf_max_show], dangle_xyz_cross)
    plt.title('xyz method delta angle')
    plt.show()

    # check xyz psds match each other and expectation
    if verification_only:
        # check absolute errors
        assert_allclose(psd_xyz[:, nf_min:nf_max], psd_xyz_rec[:, nf_min:nf_max], rtol=1.e-14, atol=6.0e-44)
        assert_allclose(psd_aet[:2, nf_min:nf_max], psd_aet_rec[:2, nf_min:nf_max], rtol=1.e-14, atol=6.5e-44)
        assert_allclose(psd_aet[2, nf_min:nf_max], psd_aet_rec[2, nf_min:nf_max], rtol=1.e-14, atol=1.5e-50)

        # check relative errors tight for most points
        assert_array_less(np.sum(~np.isclose(psd_xyz[:, nf_min:nf_max], psd_xyz_rec[:, nf_min:nf_max], rtol=1.e-1, atol=1.e-51)), 25)
        assert_array_less(np.sum(~np.isclose(psd_aet[:2, nf_min:nf_max], psd_aet_rec[:2, nf_min:nf_max], rtol=1.e-1, atol=1.e-51)), 18)
        assert_array_less(np.sum(~np.isclose(psd_aet[2, nf_min:nf_max], psd_aet_rec[2, nf_min:nf_max], rtol=1.e-1, atol=1.e-56)), 18)

        # check signal power agreement
        assert_allclose(np.trapezoid(psd_xyz[:, nf_min:nf_max], fpsd[nf_min:nf_max], axis=-1), np.trapezoid(psd_xyz_rec[:, nf_min:nf_max], fpsd[nf_min:nf_max], axis=-1), atol=1.e-100, rtol=4.e-3)
        assert_allclose(np.trapezoid(psd_aet[:, nf_min:nf_max], fpsd[nf_min:nf_max], axis=-1), np.trapezoid(psd_aet_rec[:, nf_min:nf_max], fpsd[nf_min:nf_max], axis=-1), atol=1.e-100, rtol=4.e-3)

        # check cross spectrum agreement
        assert_allclose(re_csd_aet[fcsd > 0.001, 0, 1], re_csd_aet_rec[fcsd > 0.001, 0, 1], rtol=1.e-14, atol=2.e-44)
        assert_allclose(re_csd_aet[fcsd > 0.001, 0, 2], re_csd_aet_rec[fcsd > 0.001, 0, 2], rtol=1.e-14, atol=3.e-47)
        assert_allclose(re_csd_aet[fcsd > 0.001, 1, 2], re_csd_aet_rec[fcsd > 0.001, 1, 2], rtol=1.e-14, atol=3.e-47)
        assert_allclose(re_csd_xyz[fcsd > 0.001, :, :], re_csd_xyz_rec[fcsd > 0.001, :, :], rtol=1.e-14, atol=5.e-44)

        assert_allclose(abs_csd_aet[fcsd > 0.001, 0, 1], abs_csd_aet_rec[fcsd > 0.001, 0, 1], rtol=1.e-14, atol=4.e-44)
        assert_allclose(abs_csd_aet[fcsd > 0.001, 0, 2], abs_csd_aet_rec[fcsd > 0.001, 0, 2], rtol=1.e-14, atol=4.e-48)
        assert_allclose(abs_csd_aet[fcsd > 0.001, 1, 2], abs_csd_aet_rec[fcsd > 0.001, 1, 2], rtol=1.e-14, atol=2.e-47)
        assert_allclose(abs_csd_xyz[fcsd > 0.001, :, :], abs_csd_xyz_rec[fcsd > 0.001, :, :], rtol=1.e-14, atol=4.e-44)

        assert_allclose(im_csd_aet[fcsd > 0.001, 0, 1], im_csd_aet_rec[fcsd > 0.001, 0, 1], rtol=1.e-14, atol=4.e-44)
        assert_allclose(im_csd_aet[fcsd > 0.001, 0, 2], im_csd_aet_rec[fcsd > 0.001, 0, 2], rtol=1.e-14, atol=4.e-48)
        assert_allclose(im_csd_aet[fcsd > 0.001, 1, 2], im_csd_aet_rec[fcsd > 0.001, 1, 2], rtol=1.e-14, atol=2.e-47)
        assert_allclose(im_csd_xyz[fcsd > 0.001, :, :], im_csd_xyz_rec[fcsd > 0.001, :, :], rtol=1.e-14, atol=4.e-44)

        # check coherence agreement
        assert_array_less(~np.isclose(coh_xyz[nf_min_show:nf_max_show], coh_xyz_rec[nf_min_show:nf_max_show], atol=1.e-1, rtol=1.e-2), 13)
        assert_array_less(~np.isclose(coh_aet[nf_min_show:nf_max_show, 0, 1], coh_aet_rec[nf_min_show:nf_max_show, 0, 1], atol=1.e-1, rtol=1.e-2), 4)
        assert_array_less(~np.isclose(coh_aet[nf_min_show:nf_max_show], coh_aet_rec[nf_min_show:nf_max_show], atol=1.e-1, rtol=1.e-2), 51)

        # test channel relative phase angle agreement
        assert_array_less(np.abs(angle_diff_xyz_mask), 0.09)
        assert_array_less(np.abs(np.mean(angle_diff_xyz_mask, axis=0)), 2.e-3)
        assert_array_less(np.abs(np.mean(angle_diff_xyz, axis=0)), 5.e-3)

        assert_array_less(np.abs(angle_diff_aet_mask[:, 0, 1]), 2.e-3)
        assert_array_less(np.abs(angle_diff_aet_mask), 0.1)
        assert_array_less(np.abs(np.mean(angle_diff_aet_mask[:, 0, 1], axis=0)), 4.e-6)
        assert_array_less(np.abs(np.mean(angle_diff_aet[:, 0, 1], axis=0)), 7.e-3)

        # coherence between methods
        assert_array_less(0.5, xyz_cross_cohere[nf_min_show:nf_max_show])
        assert_array_less(0.994, np.mean(xyz_cross_cohere[nf_min_show:nf_max_show], axis=0))
        assert_array_less(0.994, np.mean(aet_cross_cohere[nf_min_show:nf_max_show, :2], axis=0))
        assert_array_less(0.7, aet_cross_cohere[nf_min_show:nf_max_show, :2])
        assert_array_less(0.6, aet_cross_cohere[max(nf_max_show // 2, nf_min_show):nf_max_show, 2])

        # angle between methods
        assert_array_less(np.abs(dangle_xyz_cross), 0.05)
        assert_array_less(np.abs(np.mean(dangle_xyz_cross, axis=0)), 5.e-4)
        assert_array_less(np.abs(dangle_aet_cross), 0.24)
        assert_array_less(np.abs(dangle_aet_cross[:, :2]), 0.05)
        assert_array_less(np.abs(np.mean(dangle_aet_cross[:, :2], axis=0)), 5.e-4)
        assert_array_less(np.abs(dangle_aet_cross[dangle_aet_cross.shape[0] // 2:, 2]), 0.09)
        assert_array_less(np.abs(np.mean(dangle_aet_cross[dangle_aet_cross.shape[0] // 2:, 2])), 5.e-3)
        # check the absolute angle matches at low frequency
        assert_array_less(angle_aet_mean[0], 3.e-2)
        assert_array_less(angle_xyz_mean[0], 3.e-2)

        # check envelope modulation matches for selected source
        assert_allclose(psd_aet_proj[:2], psd_aet_proj_rec[:2], atol=1.e-10 * np.max(psd_aet_proj[:2]), rtol=4.e-3)
        assert_allclose(psd_aet_proj[2], psd_aet_proj_rec[2], atol=1.e-10 * np.max(psd_aet_proj[2]), rtol=5.e-3)
        assert_array_less(np.abs(aet_proj_dangle), 0.6)
        assert_array_less(np.abs(np.mean(aet_proj_dangle, axis=1)), 1.4e-2)
        assert_array_less(np.abs(np.mean(aet_proj_angle[:, :] - aet_proj_angle_rec[:, :], axis=1)), 1.1e-2)
        for itrc in range(lc.nc_snr):
            assert_array_less(1 - 1.e-4, np.corrcoef(aet_proj_angle[itrc, :], aet_proj_angle_rec[itrc, :])[0, 1])
            assert_array_less(0.85, np.corrcoef(aet_proj_mag[itrc, :], aet_proj_mag_rec[itrc, :])[0, 1])

        assert_allclose(np.mean(aet_proj_mag, axis=1), np.mean(aet_proj_mag_rec, axis=1), atol=1.e-100, rtol=3.e-3)

    else:
        assert_allclose(psd_xyz_expect[:, 0, 0], psd_xyz_rec[0, nf_min:nf_max], rtol=1.e-1, atol=2.e-38)
        assert_allclose(psd_xyz_expect[:, 1, 1], psd_xyz_rec[1, nf_min:nf_max], rtol=1.e-1, atol=2.e-38)
        assert_allclose(psd_xyz_expect[:, 2, 2], psd_xyz_rec[2, nf_min:nf_max], rtol=1.e-1, atol=2.e-38)

        assert_allclose(psd_xyz_expect[:, 0, 0], psd_xyz[0, nf_min:nf_max], rtol=1.e-1, atol=2.e-38)
        assert_allclose(psd_xyz_expect[:, 1, 1], psd_xyz[1, nf_min:nf_max], rtol=1.e-1, atol=2.e-38)
        assert_allclose(psd_xyz_expect[:, 2, 2], psd_xyz[2, nf_min:nf_max], rtol=1.e-1, atol=2.e-38)

        assert_allclose(psd_xyz[0, nf_min:nf_max], psd_xyz_rec[0, nf_min:nf_max], rtol=1.e-1, atol=1.e-38)
        assert_allclose(psd_xyz[1, nf_min:nf_max], psd_xyz_rec[1, nf_min:nf_max], rtol=1.e-1, atol=1.e-38)
        assert_allclose(psd_xyz[2, nf_min:nf_max], psd_xyz_rec[2, nf_min:nf_max], rtol=1.e-1, atol=1.e-38)

    xyz_powers = np.sum(xyz_tdi_time**2, axis=-1)
    xyz_powers_rec = np.sum(xyz_tdi_time_rec**2, axis=-1)

    xyz_powers_rat = xyz_powers / xyz_powers_rec

    print('x power', xyz_powers[0], xyz_powers_rec[0], xyz_powers_rat[0])
    print('y power', xyz_powers[1], xyz_powers_rec[1], xyz_powers_rat[1])
    print('z power', xyz_powers[2], xyz_powers_rec[2], xyz_powers_rat[2])
    print('AET total power', np.sum(xyz_powers), np.sum(xyz_powers_rec), np.sum(xyz_powers) / np.sum(xyz_powers_rec))

    AET_powers = np.sum(AET_tdi_time**2, axis=-1)
    AET_powers_rec = np.sum(AET_tdi_time_rec**2, axis=-1)

    AET_powers_rat = AET_powers / AET_powers_rec
    print('A power', AET_powers[0], AET_powers_rec[0], AET_powers_rat[0])
    print('E power', AET_powers[1], AET_powers_rec[1], AET_powers_rat[1])
    print('T power', AET_powers[2], AET_powers_rec[2], AET_powers_rat[2])
    print('AET total power', np.sum(AET_powers), np.sum(AET_powers_rec), np.sum(AET_powers) / np.sum(AET_powers_rec))

    AET_tdi_freq = fft.rfft(AET_tdi_time, axis=-1)

    ts = time_tdi.copy()  # np.asarray(hf['ts'])
    dt = ts[1] - ts[0]
    ND = ts.size
    Nf = wc.Nf
    Nt = int(np.int64(ND / Nf))

    fs_fft = np.arange(0, ND // 2 + 1) / (dt * ND)
    fs_fft2 = np.arange(0, (wc.Nf * nt_max) // 2 + 1) / (wc.DT * nt_max)
    print('got fft')

    wave_sangria = transform_wavelet_freq(AET_tdi_freq[0], Nf, Nt)

    plt.loglog(np.mean(wave_sangria**2, axis=0))
    plt.loglog(((galactic_bg[:, :, 0])**2).mean(axis=0))
    plt.show()

    plt.loglog(fs_fft[1:], 2 * dt_in * np.abs(AET_tdi_freq[0, 1:]))
    plt.loglog(fs_fft2[1:], 2 * wc.dt * np.abs(AET_tdi_freq_rec[0, 1:]))
    plt.xlim([3.e-4, 1.e-2])
    plt.show()

    fpsd1, psd1 = welch(AET_tdi_time[0], fs=fs, nperseg=2 * Nf, scaling='density', window='tukey')
    fpsd2, psd2 = welch(AET_tdi_time_rec[0], fs=fs, nperseg=2 * Nf, scaling='density', window='tukey')
    plt.loglog(fpsd1, psd1)
    plt.loglog(fpsd2, psd2)
    plt.show()


def test_sangria_tdi_match() -> None:
    """Test agreement with tdi data generate by sangria, only checks ampltidues and total power but not phase"""
    # toml_filename_in = 'Galaxies/GalaxyFullLDC/run_old_parameters.toml'
    toml_filename_in = 'Galaxies/GalaxyVerification2LDC/run_verification_parameters.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    # lc = get_lisa_constants(config_in)

    cyclo_mode = 1

    ifm = fetch_or_run_iterative_loop(config_in, cyclo_mode=cyclo_mode, fetch_mode=3, output_mode=0, preprocess_mode=1)
    galactic_bg = ifm.noise_manager.bgd.get_galactic_total(shape_mode=1)

    ND = 6306816

    nt_min = 0  # 64*itrm+48#wc.Nt//4
    nt_max = nt_min + wc.Nt  # wc.Nt-1*wc.Nt//4
    print(nt_min, nt_max, wc.Nt, wc.Nf, cyclo_mode)

    xyz_tdi, time_tdi = load_sangria_tdi(verification_only=True)

    tdis_freq_rec = np.zeros((3, ND // 2 + 1), dtype=np.complex128)
    tdis_time_rec = np.zeros((3, ND))
    for itrc in range(3):
        bg_hold = galactic_bg[:, :, itrc]
        tdis_freq_rec[itrc] = inverse_wavelet_freq(bg_hold, wc.Nf, wc.Nt)
        tdis_time_rec[itrc] = inverse_wavelet_time(bg_hold, wc.Nf, wc.Nt)

    xyz_tdis_freq_rec = tdi_aet_to_xyz_helper(tdis_freq_rec, axis=0)

    xyz_tdi = xyz_tdi[:, :ND]
    xyz_tdis_freq = fft.rfft(xyz_tdi, axis=1)
    time_tdi = time_tdi[:ND]

    tdis_time = tdi_xyz_to_aet_helper(xyz_tdi, axis=0)

    tdis_freq = np.zeros((3, ND // 2 + 1), dtype=np.complex128)
    for itrc in range(3):
        tdis_freq[itrc] = fft.rfft(tdis_time[itrc])

    ts = time_tdi.copy()  # np.asarray(hf['ts'])
    ND = ts.size
    # Nf = wc.Nf
    # Nt = wc.Nt
    # ND = wc.Nf*wc.Nt

    print('got fft')
    tdis_freq_power = np.linalg.norm(tdis_freq, axis=0)
    tdis_freq_rec_power = np.linalg.norm(tdis_freq_rec, axis=0)

    tdis_xyz_freq_power = np.linalg.norm(xyz_tdis_freq, axis=0)
    tdis_xyz_freq_rec_power = np.linalg.norm(xyz_tdis_freq_rec, axis=0)

    # check match in total power across all channels
    assert_allclose(tdis_freq_power, tdis_freq_rec_power, atol=7.e-19, rtol=1.e-9)
    assert_allclose(tdis_xyz_freq_power, tdis_xyz_freq_rec_power, atol=9.e-19, rtol=1.e-9)

    power_mask = tdis_freq_rec_power > 1.e-2 * float(np.max(tdis_freq_rec_power))
    power_brightest = int(np.argmax(tdis_freq_rec_power))
    # check the phase matches at the brightest frequency
    plt.loglog(tdis_freq_power)
    plt.plot(tdis_freq_rec_power)
    plt.plot(np.full(tdis_freq_power.size, 1.e-2 * np.max(tdis_freq_rec_power)))
    plt.show()

    plt.semilogx(np.abs(tdis_freq_rec[1, power_mask]) / tdis_freq_rec_power[power_mask])
    plt.semilogx(np.abs(tdis_freq[1, power_mask]) / tdis_freq_rec_power[power_mask])
    plt.show()

    plt.semilogx(np.angle(tdis_freq[0, power_mask]))
    plt.semilogx(np.angle(tdis_freq_rec[0, power_mask]))
    plt.show()

    plt.semilogx(np.angle(tdis_freq[0, power_mask]))
    plt.semilogx(np.angle(tdis_freq_rec[0, power_mask]))
    plt.show()

    assert_allclose(np.angle(tdis_freq[0, power_mask]), np.angle(tdis_freq_rec[0, power_mask]), atol=1.e-10, rtol=1.e-10)
    assert_allclose(np.angle(tdis_freq[0, power_brightest]), np.angle(tdis_freq_rec[0, power_brightest]), atol=1.e-10, rtol=1.e-10)

    assert_allclose(np.abs(tdis_freq), np.abs(tdis_freq_rec), atol=1.e-18, rtol=1.e-3)

    assert_allclose(np.abs(xyz_tdi_freq), np.abs(xyz_tdi_freq_rec), atol=1.e-18, rtol=1.e-3)

    for itrc in range(3):
        pars_wave = parseval_wavelet(galactic_bg[:, :, itrc])
        pars_freq1 = parseval_rfft(tdis_freq_rec[itrc], ND)
        pars_time0 = parseval_time(tdis_time[itrc])
        assert_allclose(pars_wave, pars_freq1, atol=1.e-80, rtol=1.e-14)
        assert_allclose(pars_wave, pars_time0, atol=1.e-80, rtol=1.e-2)

    pars_freq1 = parseval_rfft(xyz_tdi_freq_rec[0], ND)
    pars_freq0 = parseval_rfft(xyz_tdi_freq[0], ND)
    assert_allclose(pars_freq0, pars_freq1, atol=1.e-60, rtol=1.e-4)

    pars_freq1 = parseval_rfft(xyz_tdi_freq_rec[1], ND)
    pars_freq0 = parseval_rfft(xyz_tdi_freq[1], ND)
    assert_allclose(pars_freq0, pars_freq1, atol=1.e-60, rtol=1.e-2)

    pars_freq1 = parseval_rfft(xyz_tdi_freq_rec[2], ND)
    pars_freq0 = parseval_rfft(xyz_tdi_freq[2], ND)
    assert_allclose(pars_freq0, pars_freq1, atol=1.e-60, rtol=1.e-2)

    # check correlation
    assert np.corrcoef(tdis_time[0], tdis_time_rec[0])[0, 1] > 0.8
    assert np.corrcoef(tdis_time[1], tdis_time_rec[1])[0, 1] > 0.8
    assert np.corrcoef(tdis_time[2], tdis_time_rec[2])[0, 1] > 0.75
