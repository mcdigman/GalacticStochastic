"""test comparison of signal for sangria v1 verification binaries"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tomllib
import WDMWaveletTransforms.fft_funcs as fft
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import dual_annealing
from scipy.signal import resample, welch
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq, inverse_wavelet_time, transform_wavelet_freq

from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop
from LisaWaveformTools.instrument_noise import instrument_noise_AET, instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants, get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel
from tests.test_tdi_diagonal_noise_consistency import get_csd_helper, tdi_aet_to_xyz_helper, tdi_xyz_to_aet_helper
from WaveletWaveforms.wdm_config import get_wavelet_model

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_sangria_tdi(verification_only: int = 0) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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
        hf_y = hf_in['X']
        assert isinstance(hf_y, h5py.Dataset)
        hf_z = hf_in['X']
        assert isinstance(hf_z, h5py.Dataset)
        x_tdi = np.asarray(hf_x[:, 1]).flatten()
        y_tdi = np.asarray(hf_y[:, 1]).flatten()
        z_tdi = np.asarray(hf_z[:, 1]).flatten()
        time_tdi = np.asarray(hf_z[:, 0]).flatten()
        hf_in.close()
    else:
        msg = f'Unrecognized load option {verification_only}'
        raise ValueError(msg)

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

    log_f = np.linspace(np.log10(fcsd[0]), np.log10(fcsd[-1] * 0.92), 1000)
    f_log_space = 10**log_f
    log_psd_goal_T = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[2, nf_min:nf_max]), k=3, ext=2)(log_f)
    mask_spect = ((f_log_space < f_mask_low) | (f_log_space > f_mask_high))
    log_f_masked = log_f[mask_spect]
    f_log_space_masked = f_log_space[mask_spect]
    log_psd_goal_A = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[0, nf_min:nf_max]), k=3, ext=2)(log_f_masked)
    log_psd_goal_E = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(psd_aet[1, nf_min:nf_max]), k=3, ext=2)(log_f_masked)
    log_psd_goal_01 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(re_csd_xyz[:, 0, 1])), k=3, ext=2)(log_f_masked)
    log_psd_goal_02 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(re_csd_xyz[:, 0, 2])), k=3, ext=2)(log_f_masked)
    log_psd_goal_12 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(re_csd_xyz[:, 1, 2])), k=3, ext=2)(log_f_masked)
    log_psd_goal_00 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[0, nf_min:nf_max])), k=3, ext=2)(log_f_masked)
    log_psd_goal_11 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[1, nf_min:nf_max])), k=3, ext=2)(log_f_masked)
    log_psd_goal_22 = InterpolatedUnivariateSpline(np.log10(fcsd), np.log10(np.abs(psd_xyz[2, nf_min:nf_max])), k=3, ext=2)(log_f_masked)

    def S_func_temp(tpl: NDArray[np.floating]) -> float:
        config_in3['lisa_constants']['Sps'] = 10**tpl[0]
        config_in3['lisa_constants']['Sacc'] = 10**tpl[1]
        config_in3['lisa_constants']['f_roll_acc_f_inv'] = tpl[2]
        config_in3['lisa_constants']['f_roll_acc_f'] = tpl[3]
        config_in3['lisa_constants']['f_roll_ps_f_inv'] = tpl[4]
        lc3 = get_lisa_constants(config_in3)
        psd_aet_expect3 = instrument_noise_AET(f_log_space, lc3, tdi_mode='aet_equal', diagonal_mode=0)
        psd_xyz_expect3 = instrument_noise_AET(f_log_space_masked, lc3, tdi_mode='xyz_equal', diagonal_mode=1)
        contrib_T = np.linalg.norm(np.log10(psd_aet_expect3[:, 2]) - log_psd_goal_T)
        contrib_A = np.linalg.norm(np.log10(psd_aet_expect3[mask_spect, 0]) - log_psd_goal_A)
        contrib_E = np.linalg.norm(np.log10(psd_aet_expect3[mask_spect, 1]) - log_psd_goal_E)
        contrib_01 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 1])) - log_psd_goal_01)
        contrib_02 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 2])) - log_psd_goal_02)
        contrib_12 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 1, 2])) - log_psd_goal_12)
        contrib_00 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 0, 0])) - log_psd_goal_00)
        contrib_11 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 1, 1])) - log_psd_goal_11)
        contrib_22 = np.linalg.norm(np.log10(np.abs(psd_xyz_expect3[:, 2, 2])) - log_psd_goal_22)
        return np.sqrt(contrib_A**2 + contrib_E**2 + contrib_T**2 + contrib_01**2 + contrib_02**2 + contrib_12**2 + contrib_00**2 + contrib_11**2 + contrib_22**2)

    bounds = np.zeros((5, 2))
    bounds[0, 0] = np.log10(1.e-3 * 9.0e-24)
    bounds[0, 1] = np.log10(1.e3 * 9.0e-24)
    bounds[1, 0] = np.log10(1.e-3 * 5.76e-30)
    bounds[1, 1] = np.log10(1.e3 * 5.76e-30)
    bounds[2, 0] = 2.0e-4
    bounds[2, 1] = 8.0e-4
    bounds[3, 0] = 4.0e-3
    bounds[3, 1] = 1.6e-2
    bounds[4, 0] = 1.0e-3
    bounds[4, 1] = 4.0e-3

    res_found = dual_annealing(S_func_temp, bounds, maxiter=5000)
    res = res_found['x']
    print(res_found)
    config_in3['lisa_constants']['Sps'] = 10**res[0]
    config_in3['lisa_constants']['Sacc'] = 10**res[1]
    config_in3['lisa_constants']['f_roll_acc_f_inv'] = res[2]
    config_in3['lisa_constants']['f_roll_acc_f'] = res[3]
    config_in3['lisa_constants']['f_roll_ps_f_inv'] = res[4]
    lc3 = get_lisa_constants(config_in3)
    return lc3


if __name__ == '__main__':
    # pytest.cmdline.main(['test_sangria_v1_verification_match.py'])
    # toml_filename_in = 'Galaxies/GalaxyFullLDC/run_old_parameters.toml'
    toml_filename_in = 'Galaxies/GalaxyFullLDC/run_match_parameters_fit.toml'
    # toml_filename_in = 'Galaxies/GalaxyVerification2LDC/run_verification_parameters.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    with Path(toml_filename_in).open('rb') as f:
        config_in2 = tomllib.load(f)

    # config_in2['lisa_constants']['Sps'] = 2.25e-22
    # config_in2['lisa_constants']['Sps'] = 1.25e-22
    # config_in2['lisa_constants']['Sacc'] = 1.25e-29

    # config_in2['lisa_constants']['Sps'] = 9.0e-24
    # config_in2['lisa_constants']['Sacc'] = 5.76e-30

    config_in2['lisa_constants']['Sps'] = 9.0e-24
    config_in2['lisa_constants']['Sacc'] = 1. / 4. * 5.76e-30

    lc2 = get_lisa_constants(config_in2)

    xyz_tdi_time, time_tdi = load_sangria_tdi(verification_only=False)
    n_samples_in = time_tdi.size
    dt_in = time_tdi[1] - time_tdi[0]
    assert_allclose(np.diff(time_tdi), dt_in, atol=1.e-100, rtol=1.e-14)

    AET_tdi_time = tdi_xyz_to_aet_helper(xyz_tdi_time, axis=0)

    cyclo_mode = 1

    # get the computed total galactic background
    ifm = fetch_or_run_iterative_loop(config_in, cyclo_mode=cyclo_mode, fetch_mode=3, output_mode=0, preprocess_mode=1)
    galactic_bg = ifm.noise_manager.bgd.get_galactic_total(shape_mode=1)
    snrs_got = ifm.bis.get_final_snrs_tot_upper()

    SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    noise_AET_dense_pure = DiagonalStationaryDenseNoiseModel(SAET_m, wc, prune=0, nc_snr=lc.nc_snr)
    galactic_bg += noise_AET_dense_pure.generate_dense_noise()

    ts_bg = np.arange(0., wc.Nt * wc.Nf) * wc.dt

    AET_tdi_time_rec_temp = np.zeros((galactic_bg.shape[-1], wc.Nt * wc.Nf))
    for itrc in range(galactic_bg.shape[-1]):
        AET_tdi_time_rec_temp[itrc, :] = inverse_wavelet_time(galactic_bg[:, :, itrc], wc.Nf, wc.Nt)

    nt_full_max = int(np.argmax(ts_bg >= time_tdi[-1])) + 1
    AET_tdi_time_rec = np.zeros((galactic_bg.shape[-1], n_samples_in))
    for itrc in range(galactic_bg.shape[-1]):
        AET_tdi_time_rec[itrc, :], t_tdi_mbh_alt = resample(AET_tdi_time_rec_temp[itrc, :nt_full_max], num=n_samples_in, t=ts_bg[:nt_full_max])
        assert_allclose(t_tdi_mbh_alt, time_tdi, atol=1.e-100, rtol=1.e-14)

    AET_tdi_freq_rec = np.fft.rfft(AET_tdi_time_rec, axis=-1)

    fs = 1.0 / dt_in
    nperseg = int(np.round(((1. / dt_in) / (1. / wc.dt)) * wc.Nf)) * 2
    fpsd, psd_aet_rec = welch(AET_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    fpsd, psd_aet = welch(AET_tdi_time, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)

    ND = AET_tdi_time_rec.shape[1]
    nt_max = int(ND // wc.Nf)
    assert nt_max * wc.Nf == ND

    t0 = perf_counter()

    tf = perf_counter()
    print('got inverse in ', tf - t0, 's')

    xyz_tdi_freq_rec = tdi_aet_to_xyz_helper(AET_tdi_freq_rec, axis=0)

    xyz_tdi_time_rec = tdi_aet_to_xyz_helper(AET_tdi_time_rec, axis=0)

    xyz_tdi_time = xyz_tdi_time[:ND]

    xyz_tdi_freq = fft.rfft(xyz_tdi_time, axis=-1)
    time_tdi = time_tdi[:ND]

    fpsd, psd_xyz_rec = welch(xyz_tdi_time_rec, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)
    fpsd, psd_xyz = welch(xyz_tdi_time, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=-1)

    nf_min = max(1, int(np.argmax(fpsd > 4. / wc.DT)), int(np.argmax(fpsd > 4. / wc.Tw)))
    nf_max = fpsd.size - 2

    fcsd, csd_xyz = get_csd_helper(xyz_tdi_time, fs, nperseg, nf_min, nf_max, axis=0)
    fcsd, csd_aet = get_csd_helper(AET_tdi_time, fs, nperseg, nf_min, nf_max, axis=0)

    fcsd, csd_xyz_rec = get_csd_helper(xyz_tdi_time_rec, fs, nperseg, nf_min, nf_max, axis=0)
    fcsd, csd_aet_rec = get_csd_helper(AET_tdi_time_rec, fs, nperseg, nf_min, nf_max, axis=0)

    csd_scale: NDArray[np.floating] = (lc.fstr / fcsd) ** 2

    re_csd_xyz = np.real(csd_xyz)
    re_csd_xyz_scale = (csd_scale * re_csd_xyz.T).T

    re_csd_aet = np.real(csd_aet)
    re_csd_aet_scale = (csd_scale * re_csd_aet.T).T

    re_csd_xyz_rec = np.real(csd_xyz_rec)
    re_csd_xyz_rec_scale = (csd_scale * re_csd_xyz_rec.T).T

    re_csd_aet_rec = np.real(csd_aet_rec)
    re_csd_aet_rec_scale = (csd_scale * re_csd_aet_rec.T).T

    psd_xyz_expect = instrument_noise_AET(fcsd, lc2, tdi_mode='xyz_equal', diagonal_mode=1)
    psd_xyz_expect_scale = (csd_scale * psd_xyz_expect.T).T

    psd_aet_expect = instrument_noise_AET(fcsd, lc2, tdi_mode='aet_equal', diagonal_mode=1)
    psd_aet_expect_scale = (csd_scale * psd_aet_expect.T).T

    lc3 = match_sangria_curve(toml_filename_in, fcsd, psd_aet, psd_xyz, re_csd_xyz, nf_min, nf_max)

    psd_aet_expect3 = instrument_noise_AET(fcsd, lc3, tdi_mode='aet_equal', diagonal_mode=1)
    psd_xyz_expect3 = instrument_noise_AET(fcsd, lc3, tdi_mode='xyz_equal', diagonal_mode=1)

    plt.loglog(fpsd, psd_aet_rec[0])
    plt.loglog(fpsd, psd_aet[0])
    # plt.loglog(fpsd, instrument_noise_AET(fpsd, lc)[:,0])
    # plt.loglog(fpsd, instrument_noise_AET(fpsd, lc2)[:,0])
    plt.loglog(fcsd, psd_aet_expect3[:, 0, 0])
    plt.title('A channel comparison')
    plt.show()

    # plt.loglog(fpsd, psd_aet_rec[2])
    # plt.loglog(fpsd, instrument_noise_AET(fpsd, lc)[:,2])
    plt.loglog(fcsd, psd_aet_rec[2, nf_min:nf_max])
    plt.loglog(fcsd, psd_aet[2, nf_min:nf_max])
    plt.loglog(fcsd, psd_aet_expect[:, 2, 2])
    plt.loglog(fcsd, psd_aet_expect3[:, 2, 2])
    plt.title('T channel comparison')
    plt.show()

    plt.loglog(fpsd[nf_min:nf_max], psd_xyz[:, nf_min:nf_max].T)
    plt.loglog(fpsd[nf_min:nf_max], psd_xyz_rec[:, nf_min:nf_max].T)
    plt.loglog(fpsd[nf_min:nf_max], psd_xyz_expect[:, 0, 0])
    plt.loglog(fpsd[nf_min:nf_max], psd_xyz_expect3[:, 0, 0])
    plt.title('xyz spectrum')
    plt.show()

    plt.loglog(fcsd, np.abs(re_csd_xyz[:, 0, 1]))
    plt.loglog(fcsd, np.abs(re_csd_xyz[:, 0, 2]))
    plt.loglog(fcsd, np.abs(re_csd_xyz[:, 1, 2]))
    plt.loglog(fcsd, np.abs(re_csd_xyz_rec[:, 0, 1]))
    plt.loglog(fcsd, np.abs(re_csd_xyz_rec[:, 0, 2]))
    plt.loglog(fcsd, np.abs(re_csd_xyz_rec[:, 1, 2]))
    plt.loglog(fcsd, np.abs(psd_xyz_expect[:, 0, 1]))
    plt.loglog(fcsd, np.abs(psd_xyz_expect3[:, 0, 1]))
    plt.title('xyz cross spectrum')
    plt.show()

    plt.loglog(fcsd, np.abs(re_csd_aet[:, 0, 1]))
    plt.loglog(fcsd, np.abs(re_csd_aet[:, 0, 2]))
    plt.loglog(fcsd, np.abs(re_csd_aet[:, 1, 2]))
    plt.loglog(fcsd, np.abs(re_csd_aet_rec[:, 0, 1]))
    plt.loglog(fcsd, np.abs(re_csd_aet_rec[:, 0, 2]))
    plt.loglog(fcsd, np.abs(re_csd_aet_rec[:, 1, 2]))
    plt.loglog(fcsd, np.abs(psd_aet_expect[:, 0, 1]))
    plt.loglog(fcsd, np.abs(psd_aet_expect3[:, 0, 1]))
    plt.title('aet cross spectrum')
    plt.show()

    # check xyz psds match each other and expectation
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
